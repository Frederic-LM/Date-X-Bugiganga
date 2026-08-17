# brain.py (Version 10.2)
# The measuring engine: reading Tucson files, standardising series, cross-dating, and the
# reporting helpers that state what a number was computed over. gogo.py drives it and owns
# the CLI, the plots and the searches; nothing here imports gogo, so this file can be used
# on its own.
__version__ = "10.2"
APP_NAME = "Date-X"
import os, re, json, hashlib, textwrap, warnings
from datetime import datetime, timezone
from typing import Tuple, Dict, Optional
import pandas as pd, numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.linalg import solveh_banded

warnings.filterwarnings("ignore", message="The maximal number of iterations")

# --- Single source of truth for analysis defaults -----------------------------------
# Every entry point (CLI, desktop GUI, Pennyscope/finotserv, batch) reads these. They
# used to be hardcoded separately in each one, so the same sample run against the same
# category gave different answers depending on how it was launched: detective defaulted
# to 80 years of overlap from the CLI and GUI but 50 from Pennyscope, and the internal
# bass-to-treble comparison was pinned at 40 with no way to change it.
DEFAULTS = {
    # Violin plates commonly yield 60-100 measurable rings, so 80 rejected usable
    # references outright while 50 is loose enough to let short chance alignments in.
    'min_overlap': 60,
    # Bass and treble halves of one top are compared to each other, not to a master;
    # they are the same length as the sample, so they get the same floor.
    'min_overlap_internal': 60,
    'spline_stiffness_pct': 67,
    'detrend_wavelength': 32,   # fixed-wavelength spline cutoff, in years (see detrend)
    'detrend_mode': 'fixed',    # 'fixed' | 'percent'
    # All three indices are computed; these two settings only decide which one a single
    # column or a single file has to be.
    #
    # lead_index fills the one t_value / overlap_n pair a CSV row can hold, and heads a
    # report. Baillie-Pilcher, because that is the index the laboratory reports a reader is
    # holding were computed on. Every row states which index it came from.
    'lead_index': 'bp',
    # master_index is what 'build' and 'create' standardise into a master's 'value' column.
    # Spline, because a master is built by standardising each series and taking a biweight
    # mean, which is the higher-quality order of operations. The master also stores the raw
    # mean ring width per year, so the other two indices can be derived from it at
    # comparison time and one master serves all three.
    'master_index': 'spline',
    # Added to every width before the log so a zero-width ring gives a finite number. In mm.
    'index_log_epsilon': 0.001,
    'min_end_year': 1500,
    'top_n': 10,
    'min_series_depth': 5,      # trees required at a year for a master to report it
    # Below this many measured series, a country prefix's best t means nothing either way.
    'min_country_series': 5,
}

# --- Run provenance -----------------------------------------------------------------
#
# A date is only a result if someone else can reproduce it. Every artefact this program
# emits carries the conditions it was produced under, because a t-value quoted without
# its detrending settings, its overlap floor and the exact reference it was measured
# against cannot be checked by anyone -- including the person who produced it a month
# later. One implementation, called from every entry point, so the artefacts agree.

MANIFEST_FIELDS = (
    'tool', 'version', 'generated_utc', 'master_file', 'master_sha256_12',
    # Which index the t-value was computed on, and the constant added before the log.
    'index', 'index_log_epsilon',
    'detrend_mode', 'detrend_wavelength', 'spline_stiffness_pct',
    'min_overlap', 'min_depth', 'n_series',
    # The holdout: which series was removed from the reference, and the depth that cost.
    'held_out_series', 'holdout_performed', 'holdout_depth_year',
    'depth_before_holdout', 'depth_after_holdout',
)

def file_sha256_12(path):
    """First 12 hex characters of a file's SHA-256, or None if unreadable.

    Enough to pin which reference was used without bloating every header line. The
    master file name alone is not enough: masters get rebuilt."""
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()[:12]
    except (OSError, TypeError):
        return None

def run_manifest(**overrides):
    """The conditions of one analysis run, as a flat dict of primitives.

    Pass whatever the caller knows (master_file, min_overlap, detrend_mode, ...); the
    rest falls back to DEFAULTS so a field is never silently absent. master_sha256_12 is
    derived from master_file unless given explicitly."""
    manifest = {
        'tool': APP_NAME,
        'version': __version__,
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'master_file': None,
        'master_sha256_12': None,
        'index': DEFAULTS['lead_index'],
        'index_log_epsilon': DEFAULTS['index_log_epsilon'],
        'detrend_mode': DEFAULTS['detrend_mode'],
        'detrend_wavelength': DEFAULTS['detrend_wavelength'],
        'spline_stiffness_pct': DEFAULTS['spline_stiffness_pct'],
        'min_overlap': DEFAULTS['min_overlap'],
        'min_depth': DEFAULTS['min_series_depth'],
        'n_series': None,
        'held_out_series': None,
        'holdout_performed': None,
        'holdout_depth_year': None,
        'depth_before_holdout': None,
        'depth_after_holdout': None,
    }
    for key, value in overrides.items():
        if value is not None or key not in manifest:
            manifest[key] = value
    if manifest.get('master_file') and not manifest.get('master_sha256_12'):
        manifest['master_sha256_12'] = file_sha256_12(manifest['master_file'])
    if manifest.get('master_file'):
        manifest['master_file'] = os.path.basename(str(manifest['master_file']))
    return manifest

def manifest_comment_lines(manifest, prefix='# '):
    """The manifest as leading '# key: value' lines for the head of a CSV."""
    return [f"{prefix}{k}: {'' if manifest.get(k) is None else manifest[k]}"
            for k in MANIFEST_FIELDS if k in manifest] + \
           [f"{prefix}{k}: {manifest[k]}" for k in manifest if k not in MANIFEST_FIELDS]

def write_manifest_sidecar(manifest, out_prefix):
    """Write <out_prefix>_run.json beside the CSVs it describes."""
    path = f"{out_prefix}_run.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    return path

def format_run_manifest(manifest, title="RUN PROVENANCE"):
    """The manifest as a readable block for a printed or exported report."""
    if not manifest:
        return f"{title}\n  (not recorded)"
    order = [k for k in MANIFEST_FIELDS if k in manifest] + \
            [k for k in manifest if k not in MANIFEST_FIELDS]
    width = max(len(k) for k in order)
    lines = [title]
    for key in order:
        value = manifest[key]
        lines.append(f"  {key.ljust(width)} : {'-' if value is None else value}")
    return '\n'.join(lines)

# The single wording for what a last ring does and does not establish. Held here so the
# CLI, the GUI report and the server cannot drift into saying different things.
def terminus_post_quem_note(end_year=None):
    """What a last measured ring establishes -- and what it does not.

    States one thing: a lower bound. The date of felling is deliberately not discussed,
    in either direction -- an unknown amount of wood was removed in working the piece, so
    the measurement supports no statement about when the tree came down, and raising the
    subject only invites the reader to fill the gap with an assumed allowance."""
    if end_year is None:
        first = "Each end year above is the date of that series' last measured ring."
        bound = "  The object cannot have been made before that year.\n"
    else:
        year = int(end_year)
        first = f"The last measured ring dates to {year}."
        bound = f"  The object cannot have been made before {year}.\n"
    return (
        "WHAT THIS DATE ESTABLISHES\n"
        f"  {first}\n"
        + bound +
        "  This is a lower bound and nothing else. An unknown number of outer rings was\n"
        "  removed in working the wood, and that amount varies from piece to piece:\n"
        "  it must not be modelled as a constant, and no upper bound follows from it.\n"
        "  The result carries no information about maker, workshop or origin."
    )

def write_csv_with_manifest(df, path, manifest):
    """Write a DataFrame to CSV behind '# key: value' provenance lines."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        for line in manifest_comment_lines(manifest):
            f.write(line + '\n')
        df.to_csv(f, index=False)
    return path

def filter_index(index_df: pd.DataFrame, params: dict, min_end_year: int) -> pd.DataFrame:
    """Apply one category's rules to the site index."""
    missing = {'species', 'filename', 'length', 'start_year', 'end_year'} - set(index_df.columns)
    if missing:
        raise ValueError(f"Index is missing columns {sorted(missing)}. Run 'python gogo.py reindex'.")
    mask = (index_df['species'].isin(params['species'])
            & index_df['filename'].str.lower().str.startswith(tuple(params['countries']))
            & (index_df['length'] >= params['min_len'])
            & (index_df['start_year'] < params['min_start'])
            & (index_df['end_year'] >= min_end_year))
    if params.get('cembra_only'):
        if 'is_cembra' not in index_df.columns:
            raise ValueError(
                "This index carries no species-level identification, so 'alpine_pine' cannot "
                "tell Pinus cembra from Aleppo, maritime or Weymouth pine. "
                "Run 'python gogo.py reindex' to rebuild it.")
        mask &= index_df['is_cembra'].fillna(False).astype(bool)
    return index_df[mask]

# Tiers are named by the criteria they apply. An adjective would be an unattributed judgement
# in the voice of the measurement, and quotable out of context; "tier t7/n80/g70" is not.
CLASSIFICATION_TIERS = (
    {'handle': 't7/n80/g70', 't': 7.0, 'n': 80, 'glk': 70.0,
     'reading': 'a strong dating candidate, to be checked against competing alignments'},
    {'handle': 't6/n70/g65', 't': 6.0, 'n': 70, 'glk': 65.0,
     'reading': 'plausible, but in need of independent corroboration'},
    {'handle': 't5/n50/g60', 't': 5.0, 'n': 50, 'glk': 60.0,
     'reading': 'weak, and meaningful only with independent support'},
)

def tier_criteria_text(tier: dict) -> str:
    """A tier's criteria as they are applied: 't>=7.0, n>=80, GLK>=70'."""
    return f"t>={tier['t']:.1f}, n>={tier['n']}, GLK>={tier['glk']:.0f}"

def shared_variation(t_value, overlap_years) -> Optional[float]:
    """r2 = t2/(t2+n-2): the share of year-to-year variation the two series hold in common.

    Depends on the overlap as well as on t, which is why n travels with every t printed."""
    try:
        t, n = float(t_value), int(overlap_years)
    except (TypeError, ValueError):
        return None
    if n < 3:
        return None
    denominator = t * t + n - 2
    if denominator <= 0:
        return None
    return (t * t) / denominator

def _classify_dendro_match(t_value, overlap_years, gleich_percent) -> dict:
    """Which tier's criteria the measured values meet. Returns values, not a verdict."""
    try:
        t, n, glk = float(t_value), int(overlap_years), float(gleich_percent)
    except (TypeError, ValueError):
        t, n, glk = 0.0, 0, 0.0
    values = {'t_value': t, 'overlap_n': n, 'glk': glk, 'r2': shared_variation(t, n)}
    for position, tier in enumerate(CLASSIFICATION_TIERS):
        if t >= tier['t'] and n >= tier['n'] and glk >= tier['glk']:
            return {'tier_index': position, 'handle': tier['handle'],
                    'criteria': tier_criteria_text(tier), 'reading': tier['reading'],
                    'meets_any': True, 'values': values}
    lowest = CLASSIFICATION_TIERS[-1]
    return {'tier_index': None, 'handle': None,
            'criteria': tier_criteria_text(lowest), 'reading': None,
            'meets_any': False, 'values': values}

# The convention's provenance, stated wherever a reading is offered. Published violin-report
# thresholds were calibrated on the two log indices, so a spline t compared against them is
# an approximation and says so.
_CONVENTION_SOURCE = ("The convention is not a derived constant and published thresholds vary "
                      "between laboratories")
_INDEX_CALIBRATION = {
    'spline': (", and were calibrated on Baillie-Pilcher and Hollstein indices, not on spline "
               "output -- read the comparison as approximate."),
    'bp': ". The convention applies directly to this index.",
    'hollstein': ". The convention applies directly to this index.",
}

def convention_note(index: str = None) -> str:
    """Who owns the reading of these numbers, and whether it applies to this index."""
    return _CONVENTION_SOURCE + _INDEX_CALIBRATION[resolve_index(index)]

def format_classification(classification: dict, index: str = None, width: int = 88) -> str:
    """The measured values, the criteria they meet, and who says what that means."""
    if not classification:
        return ""
    values = classification.get('values', {})
    t, n = values.get('t_value', 0.0), values.get('overlap_n', 0)
    glk, r2 = values.get('glk', 0.0), values.get('r2')
    convention = convention_note(index)
    lines = [f"Measured: t = {t:.2f} over n = {n} compared rings, GLK = {glk:.1f}%"]
    if r2 is not None:
        lines.append(f"Shared year-to-year variation: r2 = t2/(t2+n-2) = {r2:.2f} "
                     f"({r2 * 100:.0f}% of the variation is common to the two series)")
    if classification.get('meets_any'):
        lines.append(f"Criteria met: tier {classification['handle']} "
                     f"({classification['criteria']})")
        reading = (f"Values in this range are conventionally read in the literature as "
                   f"{classification['reading']}. {convention}")
    else:
        lines.append(f"Criteria met: none of the three tiers "
                     f"({classification['criteria']} not met)")
        reading = (f"Values in this range are not conventionally treated as evidential on "
                   f"their own. {convention}")
    lines.append(reading)
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width, subsequent_indent='  ') or [''])
    return '\n'.join(wrapped)

def classification_handle(classification: dict) -> str:
    """Short handle for a table cell: the criteria met, or that none were."""
    if not classification:
        return ''
    return f"tier {classification['handle']}" if classification.get('meets_any') else 'no tier met'

def _parse_rwl_header(file_path: str) -> dict:
    """Parses the first 3 lines of an RWL file to extract metadata."""
    # The Tucson header puts its content in column 10 onward (index 9): 'VIG    1 Fodara
    # Vedla Alm'. Reading from index 12 truncated every field -- 'Fodara' arrived as 'ara',
    # 'Italy' as 'ly' -- which is why these fields used to look like noise.
    header_info = {'site_name': 'N/A', 'location': 'N/A', 'pi': 'N/A', 'collector': 'N/A'}
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(3)]
        if len(lines) >= 3:
            header_info['site_name'] = lines[0][9:61].strip()
            # Line 2 begins with the region/country in a 13-character field, followed by the
            # species name and the numeric fields, which are not part of the location.
            header_info['location'] = lines[1][9:22].strip()
            # Line 3 is one investigator field. Splitting it into 'pi' and 'collector' on the
            # first space invented two people out of one name, so it is kept whole.
            header_info['pi'] = lines[2][9:72].strip()
            header_info['collector'] = ''
    except Exception:
        pass # Fail silently, return defaults
    return header_info

_CONTROL_CHARS_RE = re.compile(r'[\x00-\x1f\x7f]+')
_WHITESPACE_RE = re.compile(r'\s+')

def _normalize_location(raw_location) -> Optional[str]:
    """Clean a location string parsed from an RWL header, or return None if unusable.

    ITRDB-style headers are inconsistent and sometimes truncated or corrupted; a raw
    'location' field cannot be trusted as-is. Returns None for anything empty, purely
    numeric, or too short to be a meaningful place name (e.g. stray header noise)."""
    if raw_location is None:
        return None
    cleaned = _CONTROL_CHARS_RE.sub(' ', str(raw_location))
    cleaned = _WHITESPACE_RE.sub(' ', cleaned).strip()
    if not cleaned or cleaned.upper() == 'N/A':
        return None
    if len(cleaned) < 3:
        return None
    if cleaned.replace('.', '').replace('-', '').replace(' ', '').isdigit():
        return None
    return cleaned

# --- Tucson (.rwl) decadal-format reader -------------------------------------------
#
# The ITRDB Tucson format is COLUMN-positional, not whitespace-delimited: characters
# 1-8 hold the series ID and 9-12 the decade year, with no guarantee of a space
# between them. Splitting on whitespace silently misreads every file whose series ID
# fills all 8 characters -- "GArPC03B1910   143" splits as id="GArPC03B1910",
# year="143" -- so the first ring width is consumed as the year and the whole row is
# filed under a nonsense date. 848 of the 1943 cached ITRDB files (44%) have such IDs.
#
# BC-dated collections are the one documented exception: a year like -3660 needs five
# characters, so it overflows one column left into the ID field. A '-' at column 8 is
# unambiguous (no ITRDB series ID ends in a hyphen at that position) and marks that case.
_RWL_INT_RE = re.compile(r'-?\d+$')
_RWL_YEAR_RE = re.compile(r'-?\d{1,5}$')

# Sentinels that terminate a series or mark an unmeasured ring. Which pair is in use
# also tells us the measurement precision: files written to 0.001 mm close with -9999,
# files written to 0.01 mm close with 999.
_RWL_STOP_COARSE = (999, -999)
_RWL_STOP_FINE = (9999, -9999)

def _parse_rwl_data_line(line: str):
    """Split one Tucson data row into (series_id, first_year, [raw values]).

    Returns None for header lines, blanks and anything that is not a data row."""
    if len(line) < 13:
        return None
    if line[7] == '-':
        series_id, year_text = line[:7].strip(), line[7:12].strip()
    else:
        series_id, year_text = line[:8].strip(), line[8:12].strip()
    if not series_id or not _RWL_YEAR_RE.fullmatch(year_text):
        return None
    values = []
    for token in line[12:].split():
        if _RWL_INT_RE.fullmatch(token):
            values.append(int(token))
        else:
            break  # trailing free-text column ("gap", "Simpl", ...) ends the numbers
    if not values:
        return None
    return series_id, int(year_text), values

def read_rwl_series(file_path: str) -> Dict[str, pd.Series]:
    """Read a Tucson .rwl into {series_id: Series of ring widths in mm, indexed by year}.

    Every measured series in the file is returned separately; nothing is merged here.
    Stop markers become series terminators and unmeasured rings become NaN rather than
    being dropped, so a gap never silently shifts the years that follow it."""
    raw: Dict[str, Dict[int, float]] = {}
    uses_fine_precision = False
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: could not read {file_path}: {e}")
        return {}

    parsed = []
    for line in lines:
        stripped = line.rstrip('\n').rstrip('\r')
        if len(stripped.strip()) < 5:
            continue
        record = _parse_rwl_data_line(stripped)
        if record is None:
            continue
        parsed.append(record)
        if any(v in _RWL_STOP_FINE for v in record[2]):
            uses_fine_precision = True

    scale = 1000.0 if uses_fine_precision else 100.0
    stops = _RWL_STOP_FINE if uses_fine_precision else _RWL_STOP_COARSE
    for series_id, year, values in parsed:
        bucket = raw.setdefault(series_id, {})
        for offset, value in enumerate(values):
            if value in stops:
                break  # end of this series; nothing meaningful follows on the row
            bucket[year + offset] = np.nan if value <= 0 else value / scale

    out: Dict[str, pd.Series] = {}
    for series_id, bucket in raw.items():
        if not bucket:
            continue
        s = pd.Series(bucket, name=series_id).sort_index()
        s.index.name = 'year'
        if s.notna().sum() >= 5:
            out[series_id] = s
    return out

def parse_as_floating_series(file_path: str) -> pd.Series:
    """Read a measured sample as an undated (floating) ring-width sequence.

    A sample file is expected to hold one measured series. If it holds several the
    longest is used and the rest are named in a warning -- concatenating them into one
    sequence, as this function used to do, invents ring-to-ring transitions that were
    never measured and quietly corrupts the sample being dated."""
    series_map = read_rwl_series(file_path)
    if not series_map:
        return pd.Series(dtype=np.float64)
    if len(series_map) > 1:
        chosen = max(series_map, key=lambda k: series_map[k].notna().sum())
        others = ', '.join(sorted(k for k in series_map if k != chosen))
        print(f"WARNING: '{os.path.basename(file_path)}' contains {len(series_map)} measured series. "
              f"Using the longest ('{chosen}'); ignoring: {others}. "
              f"Measure one plate per file, or use the two-piece tool to combine them.")
    else:
        chosen = next(iter(series_map))
    widths = series_map[chosen].dropna().tolist()
    if not widths:
        return pd.Series(dtype=np.float64)
    return pd.Series(widths, index=pd.RangeIndex(start=1, stop=len(widths) + 1, name='ring_number'))

def _build_master_from_rwl_file(file_path: str, exclude_series=None) -> pd.Series:
    """Mean ring width per year across every measured series in a dated .rwl file.

    This averages the trees. The previous implementation kept whichever tree happened
    to appear first in the file for each year and discarded the rest, which spliced one
    arbitrary core to another at arbitrary years instead of producing a site mean."""
    series_map = read_rwl_series(file_path)
    if exclude_series:
        drop = {_normalize_series_id(x) for x in exclude_series}
        series_map = {k: v for k, v in series_map.items()
                      if _normalize_series_id(k) not in drop}
    if not series_map:
        return pd.Series(dtype=np.float64)
    combined = pd.concat(series_map.values(), axis=1).sort_index()
    return combined.mean(axis=1).dropna()

def _whittaker_bands(n: int, lam: float) -> np.ndarray:
    """A = I + lam * D'D as solveh_banded's upper form, where ab[2 + i - j, j] == A[i, j].

    D is the (n-2) x n second-difference operator, so A is symmetric pentadiagonal. The
    bands are accumulated from D's own entries -- 1, -2, 1 on row r at columns r, r+1, r+2 --
    rather than written out as literals, which keeps the short-series cases (n = 3, 4) right
    without special-casing them."""
    diagonal = np.ones(n, dtype=np.float64)
    first = np.zeros(n, dtype=np.float64)
    second = np.zeros(n, dtype=np.float64)
    rows = np.arange(n - 2)
    np.add.at(diagonal, rows, lam)              # A[r, r]     from D[r, r]^2
    np.add.at(diagonal, rows + 1, 4.0 * lam)    # A[r+1, r+1] from D[r, r+1]^2
    np.add.at(diagonal, rows + 2, lam)          # A[r+2, r+2] from D[r, r+2]^2
    np.add.at(first, rows + 1, -2.0 * lam)      # A[r, r+1]
    np.add.at(first, rows + 2, -2.0 * lam)      # A[r+1, r+2]
    np.add.at(second, rows + 2, lam)            # A[r, r+2]
    return np.vstack([second, first, diagonal])

def _whittaker_smooth(y: np.ndarray, lam: float) -> np.ndarray:
    """Order-2 Whittaker-Henderson smoother. For equally spaced points this is the
    discrete cubic smoothing spline, so it lets us set a real frequency response.

    Solved as a symmetric positive-definite banded system: A is pentadiagonal, so a
    Cholesky band solve is O(n) and allocates three rows instead of a sparse matrix. It
    agrees with the general sparse solve to ~1e-12 at the default 32-year cutoff, and to
    ~1e-8 in the stiffest percent-mode cases, where lambda reaches 1e8 and both solvers are
    working near the limit of double precision."""
    n = len(y)
    if n < 3: return y.astype(float)
    return solveh_banded(_whittaker_bands(n, lam), y.astype(np.float64), lower=False)

def detrend(series: pd.Series, spline_stiffness_pct: int = None, mode: str = None,
            wavelength: int = None) -> Tuple[pd.Series, pd.Series]:
    """Cook & Peters (1981) cubic smoothing spline detrending.

    The spline has a 50% frequency response at a wavelength of `nyrs`, set either way:

    mode='fixed'   (default) nyrs = `wavelength` years, the same for every series.
    mode='percent'           nyrs = `spline_stiffness_pct` % of this series' length.

    'percent' is the classic Cook & Peters formulation and is right when detrending a
    set of comparable series, but it is wrong for cross-dating, because the two sides of
    the comparison are not comparable: an 80-ring violin plate at 67% gets a 54-year
    cutoff while a 450-year reference chronology at 67% gets a 300-year one, so the
    sample is high-pass filtered far harder than the reference and the two are then
    correlated across different frequency bands. That depresses r and biases the search
    toward references of similar length to the sample. A fixed cutoff -- 32 years is the
    usual choice for timber and instrument work -- puts both sides in the same band.

    Implemented via the order-2 Whittaker-Henderson smoother, which for equally spaced
    points is the cubic smoothing spline."""
    mode = mode or DEFAULTS['detrend_mode']
    series = series.dropna()
    if len(series) < 15:
        return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    y = series.values.astype(float)
    if mode == 'percent':
        pct = DEFAULTS['spline_stiffness_pct'] if spline_stiffness_pct is None else spline_stiffness_pct
        nyrs = max(4.0, (pct / 100.0) * len(series))
    else:
        nyrs = float(wavelength or DEFAULTS['detrend_wavelength'])
        # A cutoff longer than the series leaves nothing to detrend against; fall back
        # to a proportion of what is actually there rather than returning a flat line.
        nyrs = max(4.0, min(nyrs, 0.9 * len(series)))
    # 50% amplitude of the order-2 smoother's transfer function at f0 = 1/nyrs.
    lam = 1.0 / (16.0 * np.sin(np.pi / nyrs) ** 4)
    fit = np.clip(_whittaker_smooth(y, lam), 1e-6, None)
    spline_fit = pd.Series(fit, index=series.index)
    detrended_series = series / spline_fit
    detrended_series.attrs['index_method'] = 'spline'
    return detrended_series, spline_fit

# --- The three indices ---------------------------------------------------------------
#   Hollstein (1980):         d(y) = ln( w(y) / w(y-1) )
#   Baillie & Pilcher (1973): d(y) = ln( 5*w(y) / (w(y-2)+w(y-1)+w(y)+w(y+1)+w(y+2)) )
# Both on RAW ring widths, as alternatives to the spline rather than additions to it. Both
# lose rings at the ends, so n is counted from the survivors: a drift of up to four years
# would corrupt every r2 quoted from it.
INDEX_METHODS = ('spline', 'bp', 'hollstein')

# (rings lost at the start, rings lost at the end) for a gap-free series.
INDEX_EDGE_LOSS = {'spline': (0, 0), 'bp': (2, 2), 'hollstein': (1, 0)}

INDEX_LABELS = {
    'spline': "cubic smoothing spline (Cook & Peters 1981)",
    'bp': "Baillie-Pilcher log index (Baillie & Pilcher 1973), on raw ring widths",
    'hollstein': "Hollstein log index (Hollstein 1980), on raw ring widths",
}

def resolve_index(index: str = None) -> str:
    """Validate an index name, falling back to master_index.

    The fallback is the build default, not the lead: 'no index given' means 'standardise the
    way a master is standardised'. Which index HEADS a report is a separate decision, made by
    the callers that have a single row to fill (DEFAULTS['lead_index'])."""
    index = (index or DEFAULTS['master_index']).strip().lower()
    if index not in INDEX_METHODS:
        raise ValueError(f"Unknown index '{index}'. Valid indices: {', '.join(INDEX_METHODS)}.")
    return index

def index_edge_loss(index: str = None) -> Tuple[int, int]:
    """(leading, trailing) rings a given index cannot compute."""
    return INDEX_EDGE_LOSS[resolve_index(index)]

def _log_index(series: pd.Series, index: str, epsilon: float) -> pd.Series:
    """Baillie-Pilcher or Hollstein index of a raw ring-width series.

    Computed over a gap-filled year/ring axis so that an unmeasured ring makes its own
    neighbourhood undefined rather than silently pulling a later ring into a window it
    does not belong to. Undefined positions -- the edges, and anything touching a gap --
    are dropped, so what comes back is exactly the set of rings the index exists for."""
    s = series.astype(float)
    axis = pd.RangeIndex(int(s.index.min()), int(s.index.max()) + 1)
    w = s.reindex(axis) + epsilon
    if index == 'hollstein':
        d = np.log(w / w.shift(1))
    else:
        window = w.shift(2) + w.shift(1) + w + w.shift(-1) + w.shift(-2)
        d = np.log(5.0 * w / window)
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d.index.name = series.index.name
    return d

def index_series(series: pd.Series, index: str = None, spline_stiffness_pct: int = None,
                 mode: str = None, wavelength: int = None,
                 epsilon: float = None) -> Tuple[pd.Series, pd.Series]:
    """Standardise one series under the chosen index.

    Returns (indexed_series, spline_fit); the fit is empty for the log indices. The result
    is tagged with its index setting (see assert_same_index). An empty result means the
    series is too short to survive the transform, which callers state as an exclusion."""
    index = resolve_index(index)
    if index == 'spline':
        return detrend(series, spline_stiffness_pct=spline_stiffness_pct, mode=mode,
                       wavelength=wavelength)
    epsilon = DEFAULTS['index_log_epsilon'] if epsilon is None else epsilon
    series = series.dropna()
    if len(series) < 15:
        return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    indexed = _log_index(series, index, epsilon)
    if len(indexed) < 15:
        return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    indexed.attrs['index_method'] = index
    return indexed, pd.Series(dtype=np.float64)

def index_of(series) -> str:
    """The index setting a series carries. Untagged counts as 'spline', which is what every
    earlier version produced."""
    tag = getattr(series, 'attrs', {}).get('index_method') if series is not None else None
    return tag or 'spline'

def tag_index(series: pd.Series, index: str = None) -> pd.Series:
    """Mark a series with the index setting it was produced under."""
    if series is not None:
        series.attrs['index_method'] = resolve_index(index)
    return series

def index_survival(n_input: int, index: str = None) -> int:
    """How many rings of a gap-free series of `n_input` survive an index."""
    lead, trail = index_edge_loss(index)
    return max(0, int(n_input) - lead - trail)

def describe_index(index: str = None, epsilon: float = None) -> str:
    """One line stating the index in use and what it costs at the edges."""
    index = resolve_index(index)
    lead, trail = INDEX_EDGE_LOSS[index]
    line = f"Index: {index} -- {INDEX_LABELS[index]}"
    if lead or trail:
        epsilon = DEFAULTS['index_log_epsilon'] if epsilon is None else epsilon
        line += (f"; {lead} ring(s) lost at the start and {trail} at the end, so the overlap n "
                 f"counts surviving rings; {epsilon:g} mm added before the log")
    return line

def read_master_csv(path) -> Tuple[pd.DataFrame, dict]:
    """A master .csv and the '# key: value' provenance header above it, which is how a
    reference's own index setting is known when it is read back. Headerless files still
    read; their header is empty."""
    header = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if ':' in line:
                key, value = line[1:].split(':', 1)
                header[key.strip()] = value.strip()
    table = pd.read_csv(path, index_col='year', comment='#')
    return table, header

def master_declared_index(header: dict) -> str:
    """The index a master .csv says it was built under; 'spline' when it does not say, which
    is what every master written before the setting existed is."""
    value = (header or {}).get('index', '')
    value = str(value).strip().lower()
    return value if value in INDEX_METHODS else 'spline'

def load_master_csv_series(master_file, index: str = None, mode: str = None,
                           wavelength: int = None, spline_stiffness_pct: int = None):
    """(stored series, indexed series, depth per year or None) for a master .csv.

    Raises when the master is a finished index built under a different setting: no log index
    can be recovered from a spline index, so rebuilding is the only answer."""
    index = resolve_index(index)
    table, header = read_master_csv(master_file)
    declared = master_declared_index(header)
    stored = (table['value'] if 'value' in table.columns else table.squeeze("columns")).dropna()
    depth = table['n_sites'] if 'n_sites' in table.columns else None
    if stored.empty:
        return stored, stored, depth
    if declared == index:
        return stored, tag_index(stored.copy(), index), depth
    # A different index than the file was built under can still be had, from the raw mean
    # ring width per year that masters store beside the index. Standardising that mean is a
    # step behind standardising each series and averaging (which is what 'value' holds), so
    # reference_metadata says which of the two a run used.
    raw = (table['raw_mean'].dropna() if 'raw_mean' in table.columns else None)
    if raw is None or raw.empty:
        # A file of raw widths (a two-column export from elsewhere) can be standardised any
        # way asked; a finished index with no raw column cannot.
        if float(np.nanmedian(stored.values)) > 5.0:
            raw = stored
        else:
            raise ValueError(
                f"'{os.path.basename(str(master_file))}' is a finished {declared} index and "
                f"carries no 'raw_mean' column, so a '{index}' index cannot be derived from it "
                f"-- the log indices are computed on raw ring widths. Rebuild it with this "
                f"version (which stores the raw mean), or run with --index {declared}.")
    indexed, _ = index_series(raw, index=index, mode=mode, wavelength=wavelength,
                              spline_stiffness_pct=spline_stiffness_pct)
    if indexed.empty:
        raise ValueError(f"'{os.path.basename(str(master_file))}' is too short to standardise "
                         f"under the {index} index.")
    return stored, indexed, depth

def _biweight_mean(values: np.ndarray, c: float = 9.0) -> float:
    """Tukey biweight robust mean of one year's indices across trees.

    A plain mean lets one anomalous core (a scar, a compression-wood year, a
    mismeasurement) pull a site's index for that year. The biweight is the standard
    choice for building tree-ring chronologies for the same reason."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan
    if v.size < 3:
        return float(np.mean(v))
    median = np.median(v)
    mad = np.median(np.abs(v - median))
    if mad <= 1e-12:
        return float(median)
    u = (v - median) / (c * mad)
    mask = np.abs(u) < 1.0
    if not mask.any():
        return float(median)
    w = (1.0 - u[mask] ** 2) ** 2
    return float(np.sum(w * v[mask]) / np.sum(w))

# --- Holding a series out of its own master ------------------------------------------
#
# COFECHA (Holmes 1983) removes each series from the master before testing that series
# against it: a series scored against a mean containing it cannot fail. Performed wherever
# the master's constituent series IDs can be recovered, and stated where they cannot.

def _normalize_series_id(series_id) -> str:
    """Compare series IDs case- and whitespace-insensitively; ITRDB IDs are written
    inconsistently, and a holdout that silently fails to match is worse than none."""
    return re.sub(r'\s+', '', str(series_id)).upper()


def series_ids_in_file(file_path: str) -> set:
    """Normalised IDs of every measured series in a .rwl file."""
    return {_normalize_series_id(k) for k in read_rwl_series(file_path)}


def sample_series_id(file_path: str) -> Optional[str]:
    """The ID of the series parse_as_floating_series() would use: the longest in the file."""
    series_map = read_rwl_series(file_path)
    if not series_map:
        return None
    if len(series_map) == 1:
        return next(iter(series_map))
    return max(series_map, key=lambda k: series_map[k].notna().sum())


def build_site_chronology(file_path: str, mode: str = None, wavelength: int = None,
                          spline_stiffness_pct: int = None,
                          exclude_series=None, index: str = None) -> Tuple[pd.Series, pd.Series]:
    """Standardised site chronology from a dated .rwl, plus its sample depth per year.

    Each measured series is detrended on its own and the resulting indices are averaged
    with a biweight mean -- the standard order of operations. Averaging raw widths first
    and detrending the mean afterwards, as this code used to do, leaves an age trend in
    the mean whose shape is dictated by which trees happen to be present in each year.

    `exclude_series` names series IDs to leave out. Both the mean and the depth are
    recomputed from what remains: a biweight mean cannot have a member subtracted out."""
    series_map = read_rwl_series(file_path)
    if not series_map:
        return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    if exclude_series:
        drop = {_normalize_series_id(x) for x in exclude_series}
        series_map = {k: v for k, v in series_map.items()
                      if _normalize_series_id(k) not in drop}
        if not series_map:
            return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    index = resolve_index(index)
    indices = []
    for series_id, s in series_map.items():
        # The identical transform is applied to every reference series and, by the callers
        # below, to the sample: two series standardised differently cannot be correlated.
        indexed, _ = index_series(s, index=index, spline_stiffness_pct=spline_stiffness_pct,
                                  mode=mode, wavelength=wavelength)
        if not indexed.empty:
            indices.append(indexed.rename(series_id))
    if not indices:
        return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    frame = pd.concat(indices, axis=1).sort_index()
    chronology = pd.Series(
        [_biweight_mean(row) for row in frame.to_numpy()],
        index=frame.index, name='index')
    keep = chronology.notna()
    # Sample depth is counted from the MEASURED rings, not from the transformed ones: how many
    # series a year holds is a property of the collection, and a year must not appear
    # undocumented just because bp cannot compute an index for the last two rings.
    depth = pd.concat(series_map.values(), axis=1).sort_index().notna().sum(axis=1)
    return tag_index(chronology[keep], index), depth

def master_series_ids(master_file) -> Tuple[set, bool]:
    """(normalised series IDs inside a reference, whether they are recoverable at all).

    An .rwl carries its IDs directly; a master .csv only via the '.sources.json' sidecar
    written beside every master this program builds."""
    path = str(master_file)
    if path.lower().endswith('.rwl'):
        return series_ids_in_file(path), True
    sources = read_master_sources(path)
    if not sources:
        return set(), False
    ids = set()
    for entry in sources['source_files']:
        ids |= {_normalize_series_id(x) for x in entry.get('series_ids', [])}
    return ids, True

def rebuild_master_without(master_file, series_id, mode: str = None, wavelength: int = None,
                           spline_stiffness_pct: int = None, index: str = None):
    """Rebuild a reference chronology without one series.

    Returns (chronology, depth_per_year, depth_floor, depth_label). A master .csv is rebuilt
    from the source files in its sidecar, with the settings it was built with; raises if any
    are missing, since a partial rebuild would be a different master under the same name."""
    path = str(master_file)
    if path.lower().endswith('.rwl'):
        chrono, depth = build_site_chronology(
            path, mode=mode, wavelength=wavelength,
            spline_stiffness_pct=spline_stiffness_pct, exclude_series=[series_id],
            index=index)
        if chrono.empty:
            raise ValueError(f"Holding '{series_id}' out of '{os.path.basename(path)}' left no "
                             f"usable chronology.")
        return chrono, depth, DEFAULTS['min_series_depth'], 'series'
    sources = read_master_sources(path)
    if not sources:
        raise ValueError(f"No source record beside '{os.path.basename(path)}'.")
    present = [e['path'] for e in sources['source_files'] if os.path.exists(e['path'])]
    missing = [os.path.basename(e['path']) for e in sources['source_files']
               if not os.path.exists(e['path'])]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(sources['source_files'])} files this master was built from "
            f"are no longer at their recorded location (e.g. {missing[0]}), so it cannot be "
            f"rebuilt without the sample's series.")
    min_depth = int(sources.get('min_depth', DEFAULTS['min_series_depth']))
    master, _site_count, _tree_count, _kept, _raw, site_depth = _combine_site_chronologies(
        present, f"Rebuilding reference without {series_id}", min_depth,
        detrend_mode=sources.get('detrend_mode'),
        detrend_wavelength=sources.get('detrend_wavelength'),
        spline_stiffness_pct=sources.get('spline_stiffness_pct'),
        exclude_series=[series_id], progress=False,
        index=index or sources.get('index'))
    return master, site_depth, min_depth, 'site'

def holdout_note(holdout: dict) -> str:
    """One plain sentence on the holdout, for printing before any result. Every branch
    produces one, including those where nothing could be done."""
    sid = holdout.get('held_out_series') or holdout.get('sample_series_id')
    if holdout.get('performed'):
        return (f"HOLDOUT: the sample's series ID '{sid}' is one of the series in this reference, "
                f"so the reference was rebuilt without it before dating and its mean and depth "
                f"were recomputed from the remaining series.")
    if holdout.get('recoverable') and holdout.get('n_reference_series') is not None:
        return (f"HOLDOUT: not needed -- the sample's series ID "
                f"{'(' + str(sid) + ') ' if sid else ''}is not among the "
                f"{holdout['n_reference_series']} series that make up this reference.")
    reason = holdout.get('reason') or "the reference's constituent series IDs are not recoverable"
    return (f"HOLDOUT: NOT PERFORMED -- {reason}. This result may include the sample's own "
            f"measurements in the reference it was scored against; it cannot be ruled out here.")

def apply_master_holdout(master_file, sample_id, master_detrended, reference_depth,
                         mode=None, wavelength=None, spline_stiffness_pct=None, index=None):
    """Hold the sample's own series out of its reference where possible.

    Returns (master_detrended, reference_depth, holdout), the reference rebuilt when a
    holdout was called for. `holdout` always carries a sentence for the report."""
    holdout = {'sample_series_id': sample_id, 'performed': False, 'recoverable': False,
               'held_out_series': None, 'n_reference_series': None, 'reason': None,
               'depth_floor': None, 'depth_label': None,
               'depth_before_at_year': None, 'depth_after_at_year': None,
               'depth_year': None, 'unusable': False}
    if not sample_id:
        holdout['reason'] = "the sample's own series ID could not be read from its file"
        holdout['note'] = holdout_note(holdout)
        return master_detrended, reference_depth, holdout
    try:
        ids, recoverable = master_series_ids(master_file)
    except Exception as e:
        ids, recoverable = set(), False
        holdout['reason'] = f"the reference's series IDs could not be read ({e})"
    holdout['recoverable'] = bool(recoverable)
    if not recoverable:
        holdout['reason'] = holdout['reason'] or (
            f"'{os.path.basename(str(master_file))}' is a prebuilt reference whose constituent "
            f"series IDs are not recorded (no '{os.path.basename(master_sources_path(master_file))}' "
            f"beside it)")
        holdout['note'] = holdout_note(holdout)
        return master_detrended, reference_depth, holdout
    holdout['n_reference_series'] = len(ids)
    if _normalize_series_id(sample_id) not in ids:
        holdout['note'] = holdout_note(holdout)
        return master_detrended, reference_depth, holdout
    try:
        chrono, depth, floor, label = rebuild_master_without(
            master_file, sample_id, mode=mode, wavelength=wavelength,
            spline_stiffness_pct=spline_stiffness_pct, index=index)
    except Exception as e:
        holdout['recoverable'] = False
        holdout['reason'] = (f"the sample's series is inside this reference but the reference "
                             f"could not be rebuilt without it ({e})")
        holdout['note'] = holdout_note(holdout)
        return master_detrended, reference_depth, holdout
    holdout.update({'performed': True, 'held_out_series': sample_id,
                    'depth_floor': int(floor), 'depth_label': label})
    holdout['note'] = holdout_note(holdout)
    holdout['_depth_before'] = reference_depth
    holdout['_depth_after'] = depth
    return chrono, depth, holdout

def _depth_lookup(depth, year) -> Optional[int]:
    """Depth at one year from a per-year depth series, or None if not covered."""
    if depth is None or year is None:
        return None
    try:
        value = depth.get(int(year))
    except (AttributeError, TypeError, ValueError):
        return None
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)

def record_holdout_depths(holdout: dict, year) -> dict:
    """Depth at the reported year, before and after the holdout. A holdout that leaves it
    under the floor removes the basis for the result, so that case is marked unusable."""
    if not holdout or not holdout.get('performed') or year is None:
        return holdout
    holdout['depth_year'] = int(year)
    holdout['depth_before_at_year'] = _depth_lookup(holdout.get('_depth_before'), year)
    holdout['depth_after_at_year'] = _depth_lookup(holdout.get('_depth_after'), year)
    floor = holdout.get('depth_floor')
    before, after = holdout['depth_before_at_year'], holdout['depth_after_at_year']
    # Under the floor, or the year gone from the reference entirely. A year the reference
    # never covered is not the holdout's doing.
    too_thin = after is not None and floor is not None and after < floor
    year_lost = after is None and before is not None
    if too_thin or year_lost:
        holdout['unusable'] = True
        holdout['unusable_reason'] = (
            f"RESULT UNUSABLE: after holding out '{holdout.get('held_out_series')}', the reference "
            f"has {0 if after is None else after} {holdout.get('depth_label', 'series')}(s) at "
            f"{int(year)}, below the minimum depth of {floor}. The reported year rests partly on "
            f"the sample's own measurements and there is not enough independent reference left at "
            f"that year to test it.")
    return holdout

def holdout_manifest_fields(holdout: dict) -> dict:
    """The holdout, as flat manifest fields."""
    if not holdout:
        return {}
    return {
        'held_out_series': holdout.get('held_out_series'),
        'holdout_performed': bool(holdout.get('performed')),
        'holdout_depth_year': holdout.get('depth_year'),
        # Sites for a master .csv, measured series for a single .rwl site.
        'holdout_depth_basis': holdout.get('depth_label'),
        'depth_before_holdout': holdout.get('depth_before_at_year'),
        'depth_after_holdout': holdout.get('depth_after_at_year'),
        'holdout_note': holdout.get('note'),
    }

# --- Reference-set composition ------------------------------------------------------

_COUNTRY_PREFIX_RE = re.compile(r'^([a-z]{2,4})[0-9]')

def _country_prefix(filename: str) -> str:
    """ITRDB country prefix from a site filename ('aust003.rwl' -> 'aust')."""
    m = _COUNTRY_PREFIX_RE.match(os.path.basename(filename).lower())
    return m.group(1) if m else 'other'

def describe_reference_set(path: str, files=None, mode: str = None, wavelength: int = None,
                           spline_stiffness_pct: int = None) -> dict:
    """What a search could possibly have matched against.

    A "best match" carries no information without this. A set that is 80% Alpine will
    produce an Alpine best match for almost any Alpine-climate sample, simply because
    that is where the chances are; the ranking then describes the reference set rather
    than the wood. Reports the number of sites and series, the geographic composition,
    the year span, and the depth at each year so a result can be read against how much
    reference actually existed at the year claimed.

    `path` may be a folder of .rwl sites or a master .csv written by this program.
    `files` restricts a folder scan to the sites actually searched -- a category search
    filters the cache down to a handful of sites, and describing the whole cache instead
    would overstate the reference set by two orders of magnitude."""
    info = {'path': path, 'kind': None, 'n_sites': 0, 'n_series': 0,
            'countries': {}, 'country_series': {}, 'year_min': None, 'year_max': None,
            'depth_by_year': {}}

    if os.path.isdir(path):
        info['kind'] = 'folder'
        names = (sorted(os.path.basename(f) for f in files) if files is not None
                 else sorted(f for f in os.listdir(path) if f.lower().endswith('.rwl')))
        depth_total = None
        for name in names:
            series_map = read_rwl_series(os.path.join(path, name))
            if not series_map:
                continue
            info['n_sites'] += 1
            info['n_series'] += len(series_map)
            key = _country_prefix(name)
            info['countries'][key] = info['countries'].get(key, 0) + 1
            info['country_series'][key] = info['country_series'].get(key, 0) + len(series_map)
            frame = pd.concat(series_map.values(), axis=1)
            per_year = frame.notna().sum(axis=1)
            depth_total = per_year if depth_total is None else depth_total.add(per_year, fill_value=0)
        if depth_total is not None and not depth_total.empty:
            depth_total = depth_total.astype(int).sort_index()
            info['year_min'] = int(depth_total.index.min())
            info['year_max'] = int(depth_total.index.max())
            info['depth_by_year'] = {int(y): int(n) for y, n in depth_total.items()}

    elif os.path.isfile(path) and path.lower().endswith('.csv'):
        info['kind'] = 'master_csv'
        # Masters written by this program carry their manifest as '# key: value' lines,
        # which the plain reader here used to choke on.
        table, header = read_master_csv(path)
        info['index'] = master_declared_index(header)
        if 'n_sites' in table.columns:
            depth = table['n_sites'].astype(int)
            info['n_sites'] = int(depth.max())
            info['depth_by_year'] = {int(y): int(n) for y, n in depth.items()}
        if 'n_trees' in table.columns:
            info['n_series'] = int(table['n_trees'].max())
        info['year_min'] = int(table.index.min())
        info['year_max'] = int(table.index.max())

    elif os.path.isfile(path) and path.lower().endswith('.rwl'):
        info['kind'] = 'single_rwl'
        series_map = read_rwl_series(path)
        info['n_sites'] = 1 if series_map else 0
        info['n_series'] = len(series_map)
        info['countries'] = {_country_prefix(path): 1} if series_map else {}
        info['country_series'] = ({_country_prefix(path): len(series_map)} if series_map else {})
        if series_map:
            frame = pd.concat(series_map.values(), axis=1)
            depth = frame.notna().sum(axis=1).astype(int).sort_index()
            info['year_min'] = int(depth.index.min())
            info['year_max'] = int(depth.index.max())
            info['depth_by_year'] = {int(y): int(n) for y, n in depth.items()}
    return info

def reference_depth_at(info: dict, year) -> Optional[int]:
    """Reference depth at one year, or None if that year is outside the set."""
    if year is None:
        return None
    return info.get('depth_by_year', {}).get(int(year))

def reference_metadata(path, depth_at_year=None) -> dict:
    """What the reference actually is, read from the file the result was measured against.

    A filename identifies nothing on its own. For an .rwl this returns what the ITRDB header
    states -- site, location, investigator, species, span, number of measured series -- and
    for a master .csv the conditions it was built under, from its own '# key: value' header
    and its .sources.json sidecar. Fields the file does not carry come back as None and are
    printed as 'not recorded' rather than guessed at."""
    path = str(path)
    meta = {'file': os.path.basename(path), 'sha256_12': file_sha256_12(path),
            'kind': None, 'site_name': None, 'location': None, 'investigator': None,
            'collector': None, 'species': None, 'species_code': None,
            'first_year': None, 'last_year': None, 'n_series': None, 'n_sites': None,
            'index': None, 'built_utc': None, 'built_by': None, 'min_depth': None,
            'detrend': None, 'source_file_count': None, 'sources_record': None,
            'depth_at_reported_year': depth_at_year}
    if path.lower().endswith('.rwl'):
        meta['kind'] = 'rwl'
        header = _parse_rwl_header(path)
        meta['site_name'] = _normalize_location(header.get('site_name'))
        meta['location'] = _normalize_location(header.get('location'))
        for key, source in (('investigator', 'pi'), ('collector', 'collector')):
            value = str(header.get(source, '') or '').strip()
            meta[key] = value if value and value.upper() != 'N/A' else None
        try:
            index_meta = get_metadata_from_rwl(path) or {}
        except Exception:
            index_meta = {}
        meta['species'] = index_meta.get('species')
        meta['species_code'] = index_meta.get('species_code')
        meta['first_year'] = index_meta.get('start_year')
        meta['last_year'] = index_meta.get('end_year')
        meta['n_series'] = len(read_rwl_series(path)) or None
        meta['n_sites'] = 1
        meta['index'] = 'raw ring widths (standardised by this run)'
    elif path.lower().endswith('.csv'):
        meta['kind'] = 'master_csv'
        try:
            table, header = read_master_csv(path)
        except Exception:
            table, header = None, {}
        meta['index'] = master_declared_index(header)
        meta['built_utc'] = header.get('generated_utc')
        meta['built_by'] = (f"{header.get('tool')} {header.get('version')}"
                            if header.get('tool') else None)
        meta['min_depth'] = header.get('min_depth')
        if header.get('detrend_mode'):
            meta['detrend'] = (f"{header.get('detrend_mode')} / {header.get('detrend_wavelength')} yr"
                               f" / {header.get('spline_stiffness_pct')}%")
        if table is not None and not table.empty:
            meta['first_year'] = int(table.index.min())
            meta['last_year'] = int(table.index.max())
            if 'n_sites' in table.columns:
                meta['n_sites'] = int(table['n_sites'].max())
            if 'n_trees' in table.columns:
                meta['n_series'] = int(table['n_trees'].max())
        sources = read_master_sources(path)
        if sources:
            meta['sources_record'] = os.path.basename(master_sources_path(path))
            meta['source_file_count'] = len(sources.get('source_files') or [])
    return meta

_METADATA_LABELS = (
    ('file', 'file'), ('sha256_12', 'sha-256 (first 12)'), ('kind', 'kind'),
    ('site_name', 'site name'), ('location', 'location'), ('species', 'species'),
    ('species_code', 'species code'), ('investigator', 'investigator'),
    ('collector', 'collector'), ('first_year', 'first year'), ('last_year', 'last year'),
    ('n_sites', 'sites'), ('n_series', 'measured series'), ('index', 'index'),
    ('detrend', 'detrending'), ('min_depth', 'depth floor when built'),
    ('built_by', 'built by'), ('built_utc', 'built (UTC)'),
    ('source_file_count', 'files it was built from'), ('sources_record', 'source record'),
    ('depth_at_reported_year', 'depth at the reported year'),
)

def format_reference_metadata(meta: dict, title="REFERENCE FILE") -> str:
    """The reference's own metadata as a readable block. Absent fields say so."""
    if not meta:
        return f"{title}\n  (not recorded)"
    width = max(len(label) for _, label in _METADATA_LABELS)
    lines = [title]
    for key, label in _METADATA_LABELS:
        value = meta.get(key)
        if value is None and key not in ('site_name', 'location', 'species', 'investigator'):
            continue
        lines.append(f"  {label.ljust(width)} : {'not recorded' if value is None else value}")
    if meta.get('kind') == 'rwl':
        lines.append("  Header fields are as the depositor wrote them; ITRDB headers are often "
                     "abbreviated or approximate.")
    return '\n'.join(lines)

def format_reference_set(info: dict, max_countries: int = 8) -> str:
    """One compact block describing a reference set, for printing above a ranked table."""
    if not info or not info.get('n_sites'):
        return "REFERENCE SET: (empty or unreadable)"
    span = (f"{info['year_min']}-{info['year_max']}"
            if info.get('year_min') is not None else "unknown span")
    depths = list(info.get('depth_by_year', {}).values())
    lines = [
        "REFERENCE SET",
        f"  {info['n_sites']} sites / {info['n_series']} measured series, {span}",
    ]
    if depths:
        lines.append(f"  depth per year: min {min(depths)}, median {int(np.median(depths))}, max {max(depths)}")
    if info.get('countries'):
        ordered = sorted(info['countries'].items(), key=lambda kv: -kv[1])
        shown = ', '.join(f"{k}:{v}" for k, v in ordered[:max_countries])
        extra = '' if len(ordered) <= max_countries else f" (+{len(ordered)-max_countries} more)"
        lines.append(f"  composition: {shown}{extra}")
        top_key, top_n = ordered[0]
        share = top_n / max(1, info['n_sites'])
        if share >= 0.5:
            lines.append(f"  NOTE: {share:.0%} of this set is '{top_key}'. A best match here largely "
                         f"reflects that composition, not the origin of the wood.")
    return '\n'.join(lines)

# --- What was searched, per country prefix ------------------------------------------
# "No Austrian match" reads as a finding about the wood when it may only mean the set holds
# four Austrian series. This reports what was searched in each part of the set and the best
# t obtained there. It describes the SEARCH, not the wood, and ranks no origin.

COUNTRY_COVERAGE_COLUMNS = ('country', 'n_sites', 'n_series', 'best_t', 'overlap_n',
                            'best_site', 'too_thin')

def country_coverage(info: dict, results=None, min_country_series: int = None) -> pd.DataFrame:
    """Per-country-prefix coverage of a reference set, with the best t achieved in each.

    `info` is a describe_reference_set() dict; `results` holds one dict per reference tested
    with 'source_file', 't_value' and 'overlap_n'. Every country in the set gets a row,
    matched or not; below `min_country_series` the row is flagged too_thin. Columns are
    COUNTRY_COVERAGE_COLUMNS, the same set the CSV exports use."""
    floor = DEFAULTS['min_country_series'] if min_country_series is None else min_country_series
    sites = dict(info.get('countries') or {})
    series = dict(info.get('country_series') or {})
    best = {}
    for row in (results or []):
        source = row.get('source_file')
        if not source:
            continue
        key = _country_prefix(source)
        t = row.get('t_value')
        if t is None or (isinstance(t, float) and np.isnan(t)):
            continue
        current = best.get(key)
        if current is None or float(t) > float(current['t_value']):
            best[key] = {'t_value': float(t), 'overlap_n': row.get('overlap_n'),
                         'source_file': source}
    rows = []
    # Rows come from the set's composition, never from the results: a merged master has no
    # per-country breakdown and must not grow one from its own filename.
    for key in sorted(set(sites) | set(series)):
        hit = best.get(key)
        n_series = int(series.get(key, 0))
        rows.append({
            'country': key,
            'n_sites': int(sites.get(key, 0)),
            'n_series': n_series,
            'best_t': None if hit is None else round(float(hit['t_value']), 2),
            'overlap_n': None if hit is None else (None if hit['overlap_n'] is None
                                                   else int(hit['overlap_n'])),
            'best_site': None if hit is None else os.path.basename(str(hit['source_file'])),
            'too_thin': 'yes' if n_series < floor else 'no',
        })
    df = pd.DataFrame(rows, columns=list(COUNTRY_COVERAGE_COLUMNS))
    if not df.empty:
        # Sorted by best t descending; countries that produced no alignment sort last.
        df = (df.assign(_rank=df['best_t'].astype(float))
                .sort_values('_rank', ascending=False, na_position='last')
                .drop(columns='_rank').reset_index(drop=True))
    df.attrs['min_country_series'] = int(floor)
    return df

def format_country_coverage(df: pd.DataFrame, min_country_series: int = None,
                            min_overlap=None) -> str:
    """The per-country coverage table as a readable block."""
    floor = (df.attrs.get('min_country_series') if min_country_series is None
             else min_country_series)
    floor = DEFAULTS['min_country_series'] if floor is None else floor
    lines = ["WHAT WAS SEARCHED, PER ITRDB COUNTRY PREFIX"]
    if df is None or df.empty:
        lines.append("  (the reference set carries no recoverable country prefixes)")
        return '\n'.join(lines)
    shown = df.copy()
    for column in ('best_t', 'overlap_n', 'best_site'):
        shown[column] = shown[column].apply(lambda v: '-' if v is None or (isinstance(v, float) and np.isnan(v)) else v)
    lines.append(shown.to_string(index=False))
    thin = [r['country'] for _, r in df.iterrows() if r['too_thin'] == 'yes']
    lines.append(f"  best_t is the highest t obtained anywhere in that country's series, over the "
                 f"overlap_n stated beside it"
                 + (f"; alignments needed at least {min_overlap} years of overlap to be considered."
                    if min_overlap is not None else "."))
    lines.append("  A '-' means no alignment in that country met the overlap floor, so that "
                 "country produced no t at all.")
    if thin:
        lines.append(f"  TOO THIN (fewer than {floor} measured series, so neither a high best_t nor "
                     f"the absence of one means anything either way): {', '.join(thin)}")
    lines.append("  This table describes the reference set that was searched. It is not a ranking "
                 "of candidate origins, and a high best_t in one country is a statement about that "
                 "part of the reference set, not about where the wood grew.")
    return '\n'.join(lines)

# --- Sensitivity of a date to the detrending setting --------------------------------

# Filter length is a free parameter with documented effects on cross-dating statistics
# (Holmes 1983), and no cutoff is canonical. Exposing the setting without testing
# sensitivity to it invites parameter shopping: try cutoffs until the t-value is good,
# report that one. Re-running the same comparison across several filter lengths costs
# little and turns the setting from a lever into a stated check.
STABILITY_SETTINGS = [('fixed', 20), ('fixed', 32), ('fixed', 64), ('percent', 67)]

def _one_stability_row(sample_chrono, master, mode, value, index, min_overlap, kind):
    """One re-dating of the sample under one (filter, index) setting."""
    wavelength = value if mode == 'fixed' else None
    pct = value if mode == 'percent' else None
    row = {'kind': kind, 'index': index, 'detrend_mode': mode, 'setting': value}
    try:
        if isinstance(master, pd.Series):
            master_det = master
            if index_of(master_det) != index:
                raise ValueError(f"the reference supplied is a {index_of(master_det)} index and "
                                 f"cannot be re-expressed as {index}")
        elif str(master).lower().endswith('.rwl'):
            master_det, _ = build_site_chronology(master, mode=mode, wavelength=wavelength,
                                                  spline_stiffness_pct=pct, index=index)
        else:
            _stored, master_det, _depth = load_master_csv_series(
                master, index=index, mode=mode, wavelength=wavelength,
                spline_stiffness_pct=pct)
        sample_det, _ = index_series(sample_chrono, index=index, spline_stiffness_pct=pct,
                                     mode=mode, wavelength=wavelength)
        if sample_det.empty or master_det.empty:
            raise ValueError(f"series too short to survive the {index} index")
        result = cross_date_indexed(sample_det, master_det, min_overlap=min_overlap, index=index)
        if 'error' in result:
            raise ValueError(result['error'])
        best = result['best_match']
        row.update({
            'end_year': int(best['end_year']),
            't_value': round(float(best['t_value']), 2),
            'glk': round(float(best.get('glk', 0.0)), 1),
            'overlap_n': int(best['overlap_n']),
            't_zscore': round(float(best.get('t_zscore', 0.0)), 2),
        })
    except Exception as e:
        row.update({'end_year': None, 't_value': None, 'glk': None, 'overlap_n': None,
                    't_zscore': None, 'error': str(e)})
    return row

def stability_check(sample, master, settings_list=None, min_overlap=None, index=None,
                    index_list=None) -> pd.DataFrame:
    """Re-date one sample under several detrending settings, and under each index.

    Rows carry `kind`: 'filter' rows vary the spline cutoff under the run's own index,
    'index' rows re-date under each index at the run's own filter setting. Each row's
    `overlap_n` is its own surviving compared rings, so it is not comparable across rows.

    `sample` and `master` are paths; `master` may be a master .csv or an .rwl site."""
    settings_list = STABILITY_SETTINGS if settings_list is None else settings_list
    index = resolve_index(index)
    index_list = list(INDEX_METHODS) if index_list is None else list(index_list)
    min_overlap = DEFAULTS['min_overlap'] if min_overlap is None else min_overlap
    sample_chrono = (sample if isinstance(sample, pd.Series)
                     else parse_as_floating_series(sample))
    if sample_chrono.empty:
        raise ValueError(f"Could not read sample: {sample}")

    rows = [_one_stability_row(sample_chrono, master, mode, value, index, min_overlap, 'filter')
            for mode, value in settings_list]
    # The same sample under each index, at this run's filter setting.
    base_mode = DEFAULTS['detrend_mode']
    base_value = DEFAULTS['detrend_wavelength']
    for candidate in index_list:
        rows.append(_one_stability_row(sample_chrono, master, base_mode, base_value,
                                       candidate, min_overlap, 'index'))
    return pd.DataFrame(rows)

def _year_verdict(rows: pd.DataFrame, what: str) -> dict:
    """Did the end year hold across a set of re-datings? A row that produced nothing arrives
    as None or as NaN; neither counts as a year."""
    years = [y for y in rows.get('end_year', []) if y is not None and not pd.isna(y)]
    n = len(rows)
    if not years:
        return {'stable': False, 'n_settings': n, 'n_ok': 0, 'end_years': [],
                'summary': f"No {what} produced a usable alignment."}
    unique = sorted(set(int(y) for y in years))
    stable = len(unique) == 1 and len(years) == n
    # The log indices have no filter parameter, so varying the spline cutoff cannot move
    # their result: saying the year "held" would claim a check that did not happen.
    caveat = ''
    if what == "detrending setting" and 'index' in rows.columns:
        used = set(rows['index'])
        if used and not (used & {'spline'}):
            caveat = (f" The {'/'.join(sorted(used))} index has no filter parameter, so these rows "
                      f"are identical by construction and this is not an independent check.")
    if stable:
        summary = f"End year {unique[0]} held under all {n} {what}s."
    elif len(unique) == 1:
        summary = (f"End year {unique[0]} held wherever a result was produced, but "
                   f"{n - len(years)} of {n} {what}s produced none.")
    else:
        summary = (f"End year CHANGED with the {what}: "
                   + ', '.join(str(u) for u in unique)
                   + ". This date depends on a parameter choice and is not stable.")
    return {'stable': bool(stable), 'n_settings': n, 'n_ok': len(years),
            'end_years': unique, 'summary': summary + caveat}

def stability_verdict(table: pd.DataFrame) -> dict:
    """Did the end year survive every detrending setting? Reads the 'filter' rows only; the
    index dimension is a separate question (index_stability_verdict)."""
    rows = table[table['kind'] == 'filter'] if 'kind' in table.columns else table
    return _year_verdict(rows, "detrending setting")

def index_stability_verdict(table: pd.DataFrame) -> dict:
    """Did the end year survive a change of index? A row that raised is a check that could
    not be run, not a failed date, and the verdict says so."""
    if 'kind' not in table.columns:
        return {'stable': None, 'determinable': False, 'indices': {},
                'summary': "Index stability was not tested."}
    rows = table[table['kind'] == 'index']
    if rows.empty:
        return {'stable': None, 'determinable': False, 'indices': {},
                'summary': "Index stability was not tested."}
    def _year(value):
        return None if value is None or pd.isna(value) else int(value)

    years = {r['index']: _year(r['end_year']) for _, r in rows.iterrows()}
    errors = {r['index']: (r.get('error') or 'no usable alignment')
              for _, r in rows.iterrows() if _year(r['end_year']) is None}
    produced = [y for y in years.values() if y is not None]
    if not produced:
        return {'stable': None, 'determinable': False, 'indices': years, 'errors': errors,
                'summary': ("Index stability could not be determined: no index produced a "
                            "usable alignment against this reference "
                            f"({'; '.join(f'{k}: {v}' for k, v in errors.items())}).")}
    unique = sorted(set(produced))
    if errors:
        return {'stable': None, 'determinable': False, 'indices': years, 'errors': errors,
                'summary': ("Index stability could not be determined: "
                            + '; '.join(f"{k} could not be computed ({v})" for k, v in errors.items())
                            + f". Where a result was produced the end year was "
                              f"{', '.join(str(u) for u in unique)}. To run this check, build the "
                              f"reference under each index (gogo.py build/create --index ...) and "
                              f"re-date against each, or use a .rwl reference, which can be "
                              f"standardised any way.")}
    stable = len(unique) == 1
    detail = ', '.join(f"{k}: {v}" for k, v in years.items())
    summary = (f"End year {unique[0]} held under all {len(years)} indices ({detail})."
               if stable else
               f"End year CHANGED with the index ({detail}). The date depends on which index "
               f"was used and is not stable.")
    return {'stable': bool(stable), 'determinable': True, 'indices': years,
            'errors': errors, 'summary': summary}

def index_stability_flag(verdict: dict) -> str:
    """'yes' / 'no' / a stated reason -- never a bare 'no' for a check that could not run."""
    if not verdict or not verdict.get('determinable'):
        return 'not determinable'
    return 'yes' if verdict.get('stable') else 'no'

CORRELATION_CAP = 1.0 - 1e-12   # beyond this, r is indistinguishable from perfect

def calculate_t_value(r: float, n: int) -> float:
    """Student's t from a correlation, with the degenerate end capped rather than infinite.

    The cap already existed for |r| == 1; it now covers |r| within 1e-12 of 1, because that
    is where t stops meaning anything and starts depending on the last bit of the
    correlation: 1.0 gives the 999 sentinel while 1 - 4e-16 gives 3e8. Only a series compared
    against a copy of itself reaches that range, and both readings say the same thing."""
    if n < 3: return 0.0
    if abs(r) >= CORRELATION_CAP: return float(np.sign(r) * 999.0)
    return r * np.sqrt((n - 2) / (1 - r**2))

def calculate_glk(series1: pd.Series, series2: pd.Series) -> float:
    """Gleichlaeufigkeit: the share of year-to-year steps that move the same way.

    Ties are counted strictly: two steps agree when np.sign() gives the same value, so a
    year where BOTH series are flat counts as agreement and a year where only one is flat
    counts as disagreement. Part of the literature instead scores a half-agreement for a
    single flat step ("semi-Gleichlaeufigkeit", after Huber 1943). This program does not:
    the classification tiers documented in the README were set against the strict rule, and
    switching would move every reported GLK. Exact ties are uncommon anyway, since widths
    are measured to 1/100 mm."""
    diff1 = series1.diff().dropna(); diff2 = series2.diff().dropna()
    common_index = diff1.index.intersection(diff2.index)
    if len(common_index) < 2: return 0.0
    agreements = np.sum(np.sign(diff1.loc[common_index]) == np.sign(diff2.loc[common_index]))
    return (agreements / len(common_index)) * 100

def assert_same_index(sample_series, master_series) -> str:
    """Refuse to correlate two series standardised under different indices: the result would
    not be a weak number but a meaningless one. Raises rather than warning."""
    a, b = index_of(sample_series), index_of(master_series)
    if a != b:
        raise ValueError(
            f"Refusing to correlate series standardised differently: the sample carries "
            f"index '{a}' and the reference carries index '{b}'. Both sides of a comparison "
            f"must be transformed identically. Re-run with --index {b}, or rebuild the "
            f"reference with --index {a}.")
    return a

def _contiguous_grid(series: pd.Series):
    """(values, first label) if the series sits on a gap-free integer grid, else None.

    The fast offset scan below works on positions rather than labels, which is only the same
    thing when there are no missing years and nothing to skip over."""
    index = series.index.to_numpy()
    if index.size < 2 or not np.issubdtype(index.dtype, np.integer):
        return None
    first = int(index[0])
    if not np.array_equal(index, np.arange(first, first + index.size)):
        return None
    values = series.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        return None
    return values, first

def _offset_rows_fast(sample_series, master_series, min_overlap):
    """Every offset's r, t, GLK and overlap, computed on plain arrays. None if not possible.

    Same offsets and same arithmetic as the reference loop below, without building a pandas
    object per offset: on contiguous grids the overlap at each offset is a plain slice, so
    the sums come from two array views. Both sides are centred on their own overall mean
    first -- correlation is shift-invariant, and centring keeps the sums of squares away from
    the cancellation that the raw computational formula suffers on values near 1.0."""
    sample_grid = _contiguous_grid(sample_series)
    master_grid = _contiguous_grid(master_series)
    if sample_grid is None or master_grid is None:
        return None
    x, s_first = sample_grid
    y, m_start = master_grid
    ns, nm = x.size, y.size
    xc = x - x.mean()
    yc = y - y.mean()
    sign_x = np.sign(np.diff(x))
    sign_y = np.sign(np.diff(y))
    floor = max(int(min_overlap), 3)
    rows = []
    # k aligns sample position p with master position p + k; L is the resulting overlap.
    for k in range(floor - ns, nm - floor + 1):
        p0, p1 = max(0, -k), min(ns, nm - k)
        length = p1 - p0
        if length < floor:
            continue
        xw = xc[p0:p1]
        yw = yc[p0 + k:p1 + k]
        sum_x, sum_y = xw.sum(), yw.sum()
        cov = xw @ yw - sum_x * sum_y / length
        var_x = xw @ xw - sum_x * sum_x / length
        var_y = yw @ yw - sum_y * sum_y / length
        spread = var_x * var_y
        r = float(cov / np.sqrt(spread)) if spread > 0 else np.nan
        agreements = int(np.count_nonzero(sign_x[p0:p1 - 1] == sign_y[p0 + k:p1 + k - 1]))
        rows.append({"end_year": int(m_start + k + ns - 1),
                     "correlation": r,
                     "t_value": calculate_t_value(r, length),
                     "glk": (agreements / (length - 1)) * 100 if length > 2 else 0.0,
                     "overlap_n": length})
    return rows

def _offset_rows_reference(sample_series, master_series, min_overlap):
    """The straightforward offset scan: one pandas alignment per offset. Correct for any
    index, including series with gaps, and the yardstick the fast path is tested against."""
    s_first, s_last = sample_series.index.min(), sample_series.index.max()
    s_span = s_last - s_first + 1
    m_start, m_end = master_series.index.min(), master_series.index.max()
    corrs = []
    search_range = range(int(m_start - s_span + min_overlap), int(m_end + s_span))
    for end_year in search_range:
        offset = end_year - s_last
        shifted_idx = sample_series.index + offset
        overlap_idx = master_series.index.intersection(shifted_idx)
        if len(overlap_idx) >= min_overlap:
            master_seg = master_series.loc[overlap_idx]
            sample_seg = sample_series.loc[overlap_idx - offset]
            if len(master_seg) < 3 or len(sample_seg) < 3: continue
            glk_sample = pd.Series(sample_seg.values, index=overlap_idx)
            try: r, _ = pearsonr(sample_seg.values, master_seg.values)
            except ValueError: continue
            t = calculate_t_value(r, len(overlap_idx))
            glk = calculate_glk(glk_sample, master_seg)
            corrs.append({"end_year": end_year, "correlation": r, "t_value": t, "glk": glk, "overlap_n": len(overlap_idx)})
    return corrs

def cross_date(sample_series: pd.Series, master_series: pd.Series, min_overlap: int = None) -> dict:
    min_overlap = DEFAULTS['min_overlap'] if min_overlap is None else min_overlap
    assert_same_index(sample_series, master_series)
    if sample_series.empty or master_series.empty or len(sample_series) < min_overlap:
        return {"error": "Input series is empty or shorter than minimum overlap."}
    corrs = _offset_rows_fast(sample_series, master_series, min_overlap)
    if corrs is None:
        corrs = _offset_rows_reference(sample_series, master_series, min_overlap)
    if not corrs: return {"error": f"No suitable overlap found (min_overlap = {min_overlap} years)."}
    rdf = pd.DataFrame(corrs)
    if rdf['t_value'].isnull().all(): return {"error": "Correlation calculation failed for all overlaps."}
    best_match = rdf.loc[rdf['t_value'].idxmax()].to_dict()
    # Multiple-testing safeguard: a genuine date should stand out from the crowd of
    # candidate offsets, not just clear an absolute t threshold. Report how many
    # standard deviations the winner sits above the population of all tested offsets,
    # and the gap to the runner-up. (Baillie & Pilcher 1973 emphasised this "standing out".)
    t_pop = rdf['t_value'].replace([np.inf, -np.inf], np.nan).dropna()
    best_t = best_match.get('t_value', 0.0)
    if len(t_pop) > 2 and t_pop.std(ddof=0) > 1e-9:
        best_match['t_zscore'] = float((best_t - t_pop.mean()) / t_pop.std(ddof=0))
    else:
        best_match['t_zscore'] = 0.0
    others = t_pop[t_pop < best_t]
    best_match['second_best_t'] = float(others.max()) if not others.empty else 0.0
    return {"best_match": best_match, "all_correlations": rdf.set_index('end_year')}

def cross_date_indexed(sample_series: pd.Series, master_series: pd.Series, min_overlap: int = None,
                       index: str = None) -> dict:
    """cross_date(), with the end years re-expressed as the date of the last MEASURED ring.

    cross_date reports the year of the sample's last *supplied* value, which under bp is two
    rings short of the last measured one. Pairing, correlation and overlap n are untouched;
    only the label on the year moves, so the reported year means the same under every index."""
    result = cross_date(sample_series, master_series, min_overlap=min_overlap)
    _lead, trailing = index_edge_loss(index)
    if 'error' in result or not trailing:
        return result
    result['best_match']['end_year'] = int(result['best_match']['end_year']) + trailing
    corr = result['all_correlations']
    corr.index = corr.index + trailing
    result['index_trailing_shift'] = trailing
    return result
    
# ITRDB header line 1 ends with a 4-character species code (PICE = Pinus cembra,
# PISY = Pinus sylvestris, PCAB = Picea abies, ABAL = Abies alba, ...). The index used
# to record only the genus, which is too coarse to be useful here: filtering Alpine
# countries on genus PINUS returns Swiss stone pine alongside Aleppo pine, maritime
# pine and Weymouth pine, which share neither the climate signal nor the use as tonewood.
_SPECIES_CODE_RE = re.compile(r'\b([A-Z]{2}[A-Z0-9]{2})\s*$')

# Pinus cembra, the Alpine stone pine used as a secondary violin tonewood alongside
# spruce. Some older collections carry no code and are identified by common name.
CEMBRA_CODES = ('PICE',)
CEMBRA_NAME_HINTS = ('swiss stone pine', 'stone pine', 'arolla', 'cembra', 'zirbe', 'arve')

def _species_code_from_header(first_line: str, header_blob_lower: str) -> str:
    m = _SPECIES_CODE_RE.search(first_line.rstrip())
    if m:
        return m.group(1)
    return ""

def get_metadata_from_rwl(file_path):
    series = _build_master_from_rwl_file(file_path)
    if series.empty: return None
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header_lines = [f.readline() for _ in range(5)]
    header_content = "".join(header_lines).lower()
    species_code = _species_code_from_header(header_lines[0] if header_lines else "", header_content)
    species_map = {"picea": "PICEA", "spruce": "PICEA", "pinus": "PINUS", "pine": "PINUS", "abies": "ABIES", "fir": "ABIES", "larix": "LARIX", "larch": "LARIX"}
    genus = "UNKNOWN"
    for key, val in species_map.items():
        if key in header_content:
            genus = val
            break
    # Cembra is worth resolving explicitly because it is the one pine that belongs in an
    # Alpine instrument-wood reference set.
    is_cembra = (species_code in CEMBRA_CODES) or any(h in header_content for h in CEMBRA_NAME_HINTS)
    if is_cembra and genus == "PINUS":
        species_code = species_code or "PICE"
    return {"species": genus, "species_code": species_code or "UNKNOWN",
            "is_cembra": bool(is_cembra and genus == "PINUS"),
            "start_year": int(series.index.min()), "end_year": int(series.index.max()),
            "length": len(series)}

def index_runs_table(runs) -> pd.DataFrame:
    """One row per index: end year, t, the n that t was computed over, GLK, r2."""
    rows = []
    for run in runs:
        rows.append({'index': run['index'], 'end_year': run.get('end_year'),
                     't_value': run.get('t_value'), 'overlap_n': run.get('overlap_n'),
                     'glk': run.get('glk'), 'r2': run.get('r2'),
                     'criteria_met': run.get('criteria_met'), 'note': run.get('error', '')})
    return pd.DataFrame(rows, columns=['index', 'end_year', 't_value', 'overlap_n', 'glk', 'r2',
                                       'criteria_met', 'note'])

_INDEX_SCALE_NOTE = ("Each t is stated with the n it was computed over. The three indices are "
                     "not on a common scale: do not compare these t-values with one another, and "
                     "do not carry a threshold from one index to another.")

def index_agreement(runs) -> dict:
    """Did the end year survive the choice of index? The answer leads every report."""
    ok = [r for r in runs if r.get('end_year') is not None]
    failed = [r for r in runs if r.get('end_year') is None]
    years = {r['index']: int(r['end_year']) for r in ok}
    unique = sorted(set(years.values()))
    detail = ', '.join(f"{k} {v}" for k, v in years.items())
    if not ok:
        agree, headline = None, "No index produced a usable alignment, so no year is reported."
    elif len(ok) == 1 and len(runs) == 1:
        agree = None
        headline = (f"Only the {ok[0]['index']} index was computed, so this year has not been "
                    f"tested against the choice of index. Run without --index to date under all "
                    f"{len(INDEX_METHODS)}.")
    elif len(ok) == 1:
        # One index out of several is not an agreement: nothing was compared.
        agree = None
        headline = (f"Only the {ok[0]['index']} index could be computed, so the end year "
                    f"{unique[0]} has NOT been tested against the choice of index ("
                    + '; '.join(f"{r['index']}: {r.get('error', 'no result')}" for r in failed)
                    + ").")
    elif len(unique) == 1 and not failed:
        agree = True
        headline = (f"All {len(ok)} indices date the last measured ring to {unique[0]}: the year "
                    f"survived the choice of index.")
    elif len(unique) == 1:
        agree = True
        headline = (f"The end year {unique[0]} held under the {len(ok)} indices that could be "
                    f"computed ({detail}); {', '.join(r['index'] for r in failed)} produced none.")
    else:
        agree = False
        headline = (f"THE END YEAR CHANGES WITH THE INDEX: {detail}. A date that moves with the "
                    f"index was produced partly by the index, not only by the wood. Treat it as "
                    f"unresolved until one alignment can be justified on other grounds.")
    return {'agree': agree, 'years': years, 'headline': headline,
            'excluded': [{'index': r['index'], 'reason': r.get('error', '')} for r in failed],
            'n_ok': len(ok), 'n_indices': len(runs)}

def candidate_agreement(sweeps: dict, errors: dict = None, restricted: bool = False) -> dict:
    """Does the same reference win under every index?

    A year that survives the choice of index is one finding; a WINNER that survives it is a
    different and more useful one. If spline ranks one chronology first and Hollstein ranks
    another, no single best reference can be reported, and that has to lead the report rather
    than sit inside a table nobody reaches."""
    winners, years = {}, {}
    for name, context in (sweeps or {}).items():
        table = context.get('top_results_with_depth')
        if table is None or table.empty:
            continue
        winners[name] = str(table.iloc[0]['source_file'])
        years[name] = int(table.iloc[0]['end_year'])
    distinct = sorted(set(winners.values()))
    detail = ', '.join(f"{k} {v}" for k, v in winners.items())
    if not winners:
        agree, headline = None, "No index produced a ranking, so there is no candidate to compare."
    elif restricted or len(winners) == 1:
        agree = None
        headline = (f"Only the {next(iter(winners))} index was searched, so the candidate list has "
                    f"NOT been tested against the choice of index: whether another index would "
                    f"rank a different reference first is unknown here.")
    elif len(distinct) == 1:
        agree = True
        headline = (f"The same reference ranks first under all {len(winners)} indices "
                    f"({distinct[0]}): the candidate survived the choice of index.")
    else:
        agree = False
        headline = (f"DIFFERENT REFERENCES RANK FIRST UNDER DIFFERENT INDICES: {detail}. The "
                    f"ranking depends on the index, so no single best-matching reference can be "
                    f"reported. Read the candidate lists below, not a winner.")
    return {'agree': agree, 'winners': winners, 'end_years': years,
            'distinct_winners': distinct, 'headline': headline,
            'errors': dict(errors or {}), 'n_indices': len(winners)}

def format_candidate_agreement(agreement: dict, sweeps: dict = None, top_n: int = 5) -> str:
    """The candidate-agreement headline, then each index's own top of the ranking."""
    if not agreement:
        return ""
    lines = ["CANDIDATE AGREEMENT ACROSS INDICES", agreement['headline']]
    for name, reason in (agreement.get('errors') or {}).items():
        lines.append(f"  {name}: not searched -- {reason}")
    for name, context in (sweeps or {}).items():
        table = context.get('top_results_with_depth')
        if table is None or table.empty:
            continue
        lines.append("")
        lines.append(f"  top {min(top_n, len(table))} under {name}:")
        for _, row in table.head(top_n).iterrows():
            r2 = row.get('r2')
            lines.append(f"    {row['source_file']:<16} end {int(row['end_year'])}  "
                         f"t {row['t_value']:.2f} over n {int(row['overlap_n'])}  "
                         f"r2 {'-' if r2 is None or pd.isna(r2) else f'{r2:.2f}'}")
    lines.append("  Rankings from different indices are not on a common scale; compare which "
                 "reference each puts first, not the t-values against each other.")
    return '\n'.join(lines)

def format_index_comparison(runs, agreement=None) -> str:
    """The agreement headline, the per-index numbers, and the not-a-common-scale warning."""
    if not runs:
        return ""
    agreement = agreement or index_agreement(runs)
    table = index_runs_table(runs).copy()
    for column in ('t_value', 'glk', 'r2'):
        table[column] = table[column].map(lambda v: '-' if v is None or pd.isna(v) else f"{v:.2f}")
    table['end_year'] = table['end_year'].map(lambda v: '-' if v is None or pd.isna(v) else int(v))
    table['overlap_n'] = table['overlap_n'].map(lambda v: '-' if v is None or pd.isna(v) else int(v))
    lines = ["INDEX AGREEMENT", agreement['headline'], "", table.to_string(index=False)]
    for item in agreement['excluded']:
        lines.append(f"  {item['index']}: no result -- {item['reason']}")
    lines.append(_INDEX_SCALE_NOTE)
    return '\n'.join(lines)

def _write_floating_rwl(series: pd.Series, path: str, series_id: str = "SAMPLE") -> None:
    """Write a floating ring-width series as a column-correct Tucson file.

    ID is padded to exactly 8 characters and the year right-justified in 4, because
    that is what read_rwl_series (and every other Tucson reader) expects. The previous
    inline writer emitted 'MEAN      <year> ...', which puts the year in the wrong
    columns and is misread on the way back in."""
    values = series.dropna()
    if values.empty:
        raise ValueError("Refusing to write an empty chronology.")
    sid = (series_id[:8]).ljust(8)
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(0, len(values), 10):
            block = values.iloc[i:i + 10]
            year = int(block.index[0])
            vals = "".join(f"{int(round(v * 100)):>6}" for v in block)
            f.write(f"{sid}{year:>4}{vals}\n")
        f.write(f"{sid}{int(values.index[-1]) + 1:>4}{999:>6}\n")

# A book-matched top is sawn from one wedge, so its two halves are the same tree and
# must cross-match strongly. In roughly 30% of instruments the halves come from
# different logs, in which case there is no single "sample" to date -- and forcing a
# mean of two unrelated trees destroys the signal in both.
PLATE_SAME_WEDGE_T = 6.0
PLATE_SAME_WEDGE_GLK = 62.0

def classify_plate_relationship(internal_best: dict) -> dict:
    """Decide whether two plate halves came from one wedge, from the internal match."""
    t = float(internal_best.get('t_value', 0.0))
    glk = float(internal_best.get('glk', 0.0))
    overlap = int(internal_best.get('overlap_n', 0))
    if t >= PLATE_SAME_WEDGE_T and glk >= PLATE_SAME_WEDGE_GLK:
        verdict, same_wedge = 'same_wedge', True
        note = ("The two halves cross-match strongly, consistent with a book-matched top "
                "sawn from a single wedge. They can be averaged into one chronology.")
    elif t >= 4.0:
        verdict, same_wedge = 'inconclusive', False
        note = ("The two halves show a moderate match. This is not enough to treat them as "
                "one tree; they are dated separately and no mean is formed.")
    else:
        verdict, same_wedge = 'different_logs', False
        note = ("The two halves do not cross-match. They are most likely from different logs "
                "(around 30% of instruments), so each is dated on its own.")
    return {'verdict': verdict, 'same_wedge': same_wedge, 'note': note,
            't_value': t, 'glk': glk, 'overlap_n': overlap}

def _combine_site_chronologies(paths, description, min_depth, detrend_mode=None,
                               detrend_wavelength=None, spline_stiffness_pct=None,
                               exclude_series=None, progress=True, index=None):
    """Average a set of standardised site chronologies into one master index.

    Each site is standardised on its own first (build_site_chronology), so what is being
    averaged here are dimensionless indices that already sit around 1.0. The old code
    averaged raw widths per site, divided each site by its own overall mean, and only
    then detrended -- which leaves the age trend of whichever trees happened to be
    present in a given year baked into the master.

    `exclude_series` is forwarded to every site: the biweight mean and the sample depth
    per year are both recomputed from the series that remain."""
    index = resolve_index(index)
    chronologies, depths, raw_means, kept = [], [], [], []
    iterator = tqdm(paths, desc=description) if progress else paths
    for path in iterator:
        chrono, depth = build_site_chronology(
            path, mode=detrend_mode, wavelength=detrend_wavelength,
            spline_stiffness_pct=spline_stiffness_pct, exclude_series=exclude_series,
            index=index)
        if chrono.empty:
            continue
        name = os.path.basename(path)
        chronologies.append(chrono.rename(name))
        depths.append(depth.rename(name))
        # The site's mean RAW width per year travels with the index, so a master is not
        # locked to the index it was built under: the log indices can be derived from raw
        # widths at comparison time (see load_master_csv_series).
        raw_means.append(_build_master_from_rwl_file(path, exclude_series=exclude_series).rename(name))
        kept.append(name)
    if not chronologies:
        raise ValueError("Failed to process any files into a chronology.")
    frame = pd.concat(chronologies, axis=1).sort_index()
    depth_frame = pd.concat(depths, axis=1).sort_index()
    raw_frame = pd.concat(raw_means, axis=1).sort_index() if raw_means else None
    site_count = frame.notna().sum(axis=1)
    # Depths come from the measured rings (build_site_chronology), so they cover years the
    # transform cannot reach. The master's own columns are aligned to its rows; the
    # measured-year version is returned separately, for depth lookups at the reported year.
    tree_count = depth_frame.reindex(frame.index).fillna(0).sum(axis=1).astype(int)
    site_depth_measured = depth_frame.gt(0).sum(axis=1)
    master = pd.Series([_biweight_mean(row) for row in frame.to_numpy()], index=frame.index)
    raw_master = (pd.Series([_biweight_mean(row) for row in raw_frame.to_numpy()],
                            index=raw_frame.index).reindex(frame.index)
                  if raw_frame is not None else None)
    keep = master.notna() & (site_count >= min_depth)
    master, site_count, tree_count = master[keep], site_count[keep], tree_count[keep]
    raw_master = None if raw_master is None else raw_master[keep]
    if master.empty:
        raise ValueError(
            f"No year had at least {min_depth} contributing sites, so no master was produced. "
            f"Lower min_depth or widen the selection ({len(kept)} sites were available).")
    # Tagged with the index it was built under: this series is handed straight to
    # cross_date by the holdout path, and an untagged series counts as spline, which would
    # have the mixed-index check refuse a perfectly valid bp master rebuilt in memory.
    return tag_index(master, index), site_count, tree_count, kept, raw_master, site_depth_measured

def master_sources_path(master_file: str) -> str:
    """Path of the sidecar that records which files and series a master was built from."""
    return os.path.splitext(str(master_file))[0] + '.sources.json'

def write_master_sources(output_filename, source_paths, min_depth, exclude_series=None,
                         detrend_mode=None, detrend_wavelength=None,
                         spline_stiffness_pct=None, index=None):
    """Record what a master was built from, so it can be rebuilt without one series.

    Without this a master .csv is opaque: the index it contains cannot be traced back to
    the series that produced it, so a sample that is itself one of those series cannot be
    held out and the report has to say the check was impossible. The sidecar makes the
    holdout performable for every master this program writes."""
    payload = {
        'master_file': os.path.basename(str(output_filename)),
        'tool': APP_NAME, 'version': __version__,
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'min_depth': int(min_depth),
        'index': resolve_index(index),
        'detrend_mode': detrend_mode or DEFAULTS['detrend_mode'],
        'detrend_wavelength': int(detrend_wavelength or DEFAULTS['detrend_wavelength']),
        'spline_stiffness_pct': int(spline_stiffness_pct or DEFAULTS['spline_stiffness_pct']),
        'excluded_series': sorted(exclude_series) if exclude_series else [],
        'source_files': [],
    }
    for path in source_paths:
        payload['source_files'].append({
            'path': os.path.abspath(path),
            'series_ids': sorted(read_rwl_series(path)),
        })
    path = master_sources_path(output_filename)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except OSError as e:
        print(f"Warning: could not write master source record '{path}': {e}")
        return None
    return path

def read_master_sources(master_file) -> Optional[dict]:
    """The sidecar for a master .csv, or None when it does not exist / cannot be read."""
    path = master_sources_path(master_file)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    if not payload.get('source_files'):
        return None
    return payload

def _save_master(master, site_count, tree_count, output_filename, label,
                 source_paths=None, min_depth=None, exclude_series=None,
                 detrend_mode=None, detrend_wavelength=None, spline_stiffness_pct=None,
                 index=None, raw_mean=None):
    """Write a master chronology: the standardised index, the raw mean width, and the depth.

    'value' is the index named in the header; 'raw_mean' is the mean raw ring width per year,
    which is what lets the other two indices be derived from this file at comparison time
    instead of the file being locked to one. The run manifest goes in as '# key: value' lines
    above the table, so the file states which index 'value' is."""
    index = resolve_index(index)
    columns = {'value': master}
    if raw_mean is not None and not raw_mean.dropna().empty:
        columns['raw_mean'] = raw_mean
    columns.update({'n_sites': site_count, 'n_trees': tree_count})
    out = pd.DataFrame(columns)
    out.index.name = 'year'
    manifest = run_manifest(
        master_file=None, index=index, min_depth=min_depth,
        detrend_mode=detrend_mode, detrend_wavelength=detrend_wavelength,
        spline_stiffness_pct=spline_stiffness_pct,
        n_series=len(source_paths) if source_paths else None,
        held_out_series=(sorted(exclude_series) if exclude_series else None),
        master_label=label)
    with open(output_filename, 'w', encoding='utf-8', newline='') as f:
        for line in manifest_comment_lines(manifest):
            f.write(line + '\n')
        out.to_csv(f)
    print(f"--- SUCCESS! '{label}' saved to '{output_filename}' ---")
    print(f"    span {int(master.index.min())}-{int(master.index.max())} "
          f"({len(master)} years), {int(site_count.min())}-{int(site_count.max())} sites per year")
    if source_paths:
        sidecar = write_master_sources(
            output_filename, source_paths,
            DEFAULTS['min_series_depth'] if min_depth is None else min_depth,
            exclude_series=exclude_series, detrend_mode=detrend_mode,
            detrend_wavelength=detrend_wavelength, spline_stiffness_pct=spline_stiffness_pct,
            index=index)
        if sidecar:
            print(f"    constituent series recorded in '{os.path.basename(sidecar)}' "
                  f"(needed to hold a series out of this master)")
