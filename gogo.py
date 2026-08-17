# gogo.py (Version 10.2)
# Orchestration, searches, plots and the command line. The maths and the file formats live
# in brain.py; the version is defined there and re-exported here so every component still
# reads gogo.__version__.
import os, ftplib, argparse, textwrap, multiprocessing, shutil, re
import urllib.request
from typing import Dict
import pandas as pd, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from brain import *                                    # noqa: F401,F403  (the engine)
from brain import (__version__, APP_NAME, _classify_dendro_match, _build_master_from_rwl_file,
                   _write_floating_rwl, _parse_rwl_header, _normalize_location, _biweight_mean,
                   _normalize_series_id, _depth_lookup, _country_prefix, _species_code_from_header,
                   _combine_site_chronologies, _save_master, _whittaker_smooth,
                   _offset_rows_fast, _offset_rows_reference)

# ITRDB forest site codes that the dendrochronological literature on instrument wood
# cites as relevant (Cybis wiki). Forest chronologies, not instrument measurements.
VIOLIN_FILES = [
    "fran7.rwl","fran039.rwl","swit204.rwl","swit203.rwl","swit189.rwl","swit193.rwl",
    "swit177.rwl","swit169.rwl","swit215.rwl","swit184.rwl","swit173.rwl","swit181.rwl",
    "aust003.rwl","aust007.rwl","germ12.rwl","germ11.rwl","ital007.rwl","ital006.rwl",
    "ital025.rwl","germ036.rwl","germ4.rwl","germ5.rwl","germ14.rwl","germ040.rwl",
    "germ033.rwl","germ020.rwl","czec001.rwl","czec002.rwl","czec3.rwl","czec.rwl",
    "pola022.rwl","pola019.rwl","pola020.rwl","roma002.rwl","roma005.rwl","yugo001.rwl",
    "slov001.rwl","ital022.rwl","swed311.rwl","finl012.rwl","swed312.rwl","swed011.rwl"
]

ALPINE_COUNTRIES = ['aust', 'fran', 'germ', 'ital', 'swit', 'slov']
BALTIC_COUNTRIES = ['finl', 'germ', 'lith', 'norw', 'pola', 'swed']

# Reference categories. One definition, shared by the master builder and the detective
# search, so a category cannot mean two different things depending on which one you ask.
#
# 'alpine_pine' exists because species and country were filtered independently: no
# combination of the old categories could reach Alpine Pinus. Violin tops are usually
# spruce, but Alpine stone pine (Pinus cembra) is a documented secondary tonewood, and
# the Obergurgl and Fodara Vedla Alm chronologies for it are already in the index. Under
# 'alpine' they were excluded outright; under 'baltic' or 'all' they were compared
# against Baltic lowland pine from a different climate zone. It is kept separate from
# 'alpine' rather than merged into it so a cembra match is never silently averaged into
# a spruce/fir result.
CATEGORY_PARAMS = {
    'alpine': {
        'label': 'Alpine Instrument Wood',
        'species': ['PICEA', 'ABIES'], 'countries': ALPINE_COUNTRIES,
        'min_len': 150, 'min_start': 1750,
    },
    'alpine_pine': {
        'label': 'Alpine Stone Pine',
        'species': ['PINUS'], 'countries': ALPINE_COUNTRIES,
        'cembra_only': True,
        'min_len': 150, 'min_start': 1750,
    },
    'baltic': {
        'label': 'Baltic Northern Timber',
        'species': ['PINUS', 'PICEA'], 'countries': BALTIC_COUNTRIES,
        'min_len': 150, 'min_start': 1750,
    },
    'all': {
        'label': 'All European Conifer',
        'species': ['PICEA', 'ABIES', 'PINUS', 'LARIX'],
        'countries': sorted(set(ALPINE_COUNTRIES + BALTIC_COUNTRIES)),
        'min_len': 100, 'min_start': 1800,
    },
}
CATEGORY_NAMES = list(CATEGORY_PARAMS)
# Categories offered as a detective target, plus the curated tonewood folder.
DETECTIVE_TARGETS = ['violin'] + CATEGORY_NAMES

# The 'violin' target is a curated set of ITRDB *forest* chronologies that the
# instrument-dating literature cites as relevant to tonewood. Calling it "violin
# references" reads as though it were a master built from measured instruments, which it
# is not -- nothing here was measured from a violin. The key stays 'violin' so existing
# commands and saved settings keep working; only what the user reads changes.
VIOLIN_TARGET_KEY = 'violin'
VIOLIN_REFERENCE_DIR = "tonewood_references"
VIOLIN_MASTER_FILENAME = "master_tonewood_forest_references.csv"
VIOLIN_REFERENCE_LABEL = "Tonewood forest references"
VIOLIN_REFERENCE_BLURB = (
    "Living-forest site chronologies from the ITRDB, selected because the violin-dating "
    "literature cites them as relevant to instrument wood. These are reference chronologies "
    "from standing trees -- not measurements taken from instruments."
)
# Human-readable name for any detective target, for use in menus and reports.
TARGET_LABELS = {VIOLIN_TARGET_KEY: VIOLIN_REFERENCE_LABEL}
TARGET_LABELS.update({k: v['label'] for k, v in CATEGORY_PARAMS.items()})

def tonewood_master_path(base_dir="."):
    """Path to the tonewood forest master chronology."""
    return os.path.join(base_dir, VIOLIN_REFERENCE_DIR, VIOLIN_MASTER_FILENAME)

# Cap on how many search t-values are retained/exported per detective run (Brief 2, Task 1).
# Above this, a random reservoir sample stands in for the full array so memory stays bounded
# on very large reference sets; percentiles are then computed from the sample and the report
# says so. Module-level so it can be overridden in tests without re-running a huge search.
SEARCH_T_VALUES_RESERVOIR_CAP = 100_000

def _generate_plot_report_text(analysis_dict: Dict) -> str:
    """Generates the narrative report string for embedding in the plot with specific line breaks."""
    if not analysis_dict:
        return "Analysis data not available."

    res = analysis_dict
    report_lines = []
    width = 45  # Define a consistent wrap width for long lines

    # --- Add a blank line after the title/separator for spacing ---
    report_lines.append("") 

    # --- Physical Description broken into individual lines ---
    is_two_piece = res.get('analysis_mode') == 'two_piece'
    line_belly = f"The belly is constructed from {'two sections' if is_two_piece else 'one section'}."
    report_lines.append(textwrap.fill(line_belly, width=width))

    if is_two_piece:
        bass_rev, treble_rev = res.get('reverse_bass', False), res.get('reverse_treble', False)
        if bass_rev and treble_rev: orientation_desc = "Both halves measured from centre joint outwards."
        elif bass_rev: orientation_desc = "Bass measured from centre; treble from edge."
        elif treble_rev: orientation_desc = "Treble measured from centre; bass from edge."
        else: orientation_desc = "Both halves measured from outer edge inwards."
    else:
        orientation_desc = "Measured from centre joint outwards." if res.get('reverse_sample', False) else "Measured from outer edge inwards."
    report_lines.append(textwrap.fill(orientation_desc, width=width))

    ring_count = res.get('mean_series_length') if is_two_piece else len(res.get('raw_sample', []))
    line_rings = f"The {'mean chronology' if is_two_piece else 'sample'} contains {ring_count} rings."
    report_lines.append(textwrap.fill(line_rings, width=width))

    # --- A blank line for spacing ---
    report_lines.append("")

    # Whether the year survived the choice of index leads, before any statistic.
    index_used = res.get('index', DEFAULTS['lead_index'])
    agreement = res.get('index_agreement')
    if agreement and agreement.get('headline'):
        report_lines.append(textwrap.fill(agreement['headline'], width=width))
        report_lines.append("")

    best_match = res.get('results', {}).get('best_match', {})
    if best_match:
        t_value, overlap, glk = best_match.get('t_value', 0.0), best_match.get('overlap_n', 0), best_match.get('glk', 0.0)
        classification = _classify_dendro_match(t_value, overlap, glk)
        end_year = int(best_match.get('end_year', 0))
        report_lines.append(textwrap.fill(
            f"The last measured ring dates to {end_year}. The object cannot have been made "
            f"before {end_year}; how long after cannot be read from the wood.", width=width))
        report_lines.append("")
        report_lines.append(format_classification(classification, index=index_used, width=width))
    else:
        report_lines.append("No alignment met the overlap floor, so no year is reported.")

    if res.get('analysis_type') == 'detective':
        df = res.get('enriched_results_df')
        if df is not None and not df.empty:
            top_match = df.iloc[0].to_dict()
            top_location = top_match.get('location', 'N/A').strip()
            if top_location != 'N/A' and top_location:
                report_lines.append("")
                report_lines.append(textwrap.fill(
                    f"The strongest alignment in the tested set is with the reference site "
                    f"'{top_location}' (read from the file header, often approximate). That is a "
                    f"statement about this reference set, not about where the wood grew.",
                    width=width))

    return "\n".join(report_lines)

def plot_results(analysis_dict: Dict, show: bool = True):
    """Draw the 2x2 diagnostic figure. Returns the figure, or None if it could not be drawn.

    `show=False` draws without opening a window, for callers that want the figure itself
    (the Pennyscope server saves it to PNG). They own the figure and must close it."""
    print("Generating enhanced diagnostic plot...")
    # Extract data from the main dictionary
    raw_sample = analysis_dict.get('raw_sample')
    master_detrended = analysis_dict.get('master_detrended')
    detrended_sample = analysis_dict.get('detrended_sample')
    results = analysis_dict.get('results')
    sample_filename = analysis_dict.get('sample_filename')
    master_filename = analysis_dict.get('master_filename')
    reference_is_rwl = analysis_dict.get('reference_is_rwl', False)
    raw_master = analysis_dict.get('raw_master')
    sample_spline_fit = analysis_dict.get('sample_spline_fit')
    index_used = analysis_dict.get('index', DEFAULTS['lead_index'])

    if "error" in results:
        print(f"Cannot plot: {results['error']}")
        return None

    best_match = results['best_match']
    all_correlations = results['all_correlations']
    best_end_year = int(best_match['end_year'])
    r_val, t_val, n_val, glk_val = best_match['correlation'], best_match['t_value'], int(best_match['overlap_n']), best_match.get('glk', 0.0)
    z_val, second_t = best_match.get('t_zscore', 0.0), best_match.get('second_best_t', 0.0)
    best_start_year = best_end_year - (raw_sample.index.max() - raw_sample.index.min())
    
    # Main 2x2 GridSpec with manual spacing for stability
    fig = plt.figure(figsize=(20, 11))
    gs_main = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1],
                               left=0.05, right=0.97, top=0.93, bottom=0.08,
                               wspace=0.15, hspace=0.25)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    sample_label = os.path.basename(sample_filename)
    master_label = os.path.basename(master_filename)
    
    # Graph 1: T-Value plot (Top-Left)
    ax1 = fig.add_subplot(gs_main[0, 0])
    ax1.plot(all_correlations.index, all_correlations['t_value'], color='gray', zorder=1, label=f'All Offsets (Best t={t_val:.2f})')
    ax1.scatter(best_end_year, t_val, color='red', s=120, zorder=2, ec='black', label=f'Best Match Year: {best_end_year}')
    # Lines are drawn at tier criteria, named by those criteria.
    ax1.axhline(5.0, color='orange', linestyle='--', linewidth=1,
                label=f"t=5.0 (tier {CLASSIFICATION_TIERS[-1]['handle']})")
    ax1.axhline(7.0, color='firebrick', linestyle='--', linewidth=1,
                label=f"t=7.0 (tier {CLASSIFICATION_TIERS[0]['handle']})")
    ax1.set_xlabel("Potential End Year"); ax1.set_ylabel("T-Value"); ax1.set_title("1. Cross-Dating Significance"); ax1.legend()

    # Graph 2: Aligned Detrended (Top-Right)
    ax2 = fig.add_subplot(gs_main[0, 1])
    aligned_sample_detrended = detrended_sample.copy(); aligned_sample_detrended.index += (best_start_year - 1)
    ax2.plot(master_detrended.index, master_detrended.values, label=f'Ref: {master_label}', color='blue', alpha=0.8)
    ax2.plot(aligned_sample_detrended.index, aligned_sample_detrended.values, label=f'Sample: {sample_label}', color='red', linestyle='--')
    overlap_index = master_detrended.index.intersection(aligned_sample_detrended.index)
    if not overlap_index.empty: ax2.axvspan(overlap_index.min(), overlap_index.max(), color='gray', alpha=0.2, label=f'Overlap (n={n_val})')
    ax2.set_xlim(overlap_index.min() - 20, overlap_index.max() + 20) if not overlap_index.empty else None
    ax2.set_xlabel("Year"); ax2.set_ylabel("Detrended Index"); ax2.set_title(f"2. Aligned Detrended (r={r_val:.3f})"); ax2.legend()

    # Graph 3: Raw Data Visual Match (Bottom-Left)
    ax3 = fig.add_subplot(gs_main[1, 0], sharex=ax2)
    aligned_raw_sample = raw_sample.copy(); aligned_raw_sample.index += (best_start_year - 1)
    ax3.plot(aligned_raw_sample.index, aligned_raw_sample.values, label=f'Sample: {sample_label}', color='green')
    if sample_spline_fit is not None and not sample_spline_fit.empty:
        aligned_spline = sample_spline_fit.copy(); aligned_spline.index += (best_start_year - 1)
        ax3.plot(aligned_spline.index, aligned_spline.values, color='green', linestyle='--', label='Detrending Spline')
    if reference_is_rwl and raw_master is not None: ax3.plot(raw_master.index, raw_master.values, label=f'Ref: {master_label}', color='black', alpha=0.7)
    else: rescaled_master_for_plot = master_detrended * raw_sample.mean(); ax3.plot(rescaled_master_for_plot.index, rescaled_master_for_plot.values, label=f'Ref (scaled): {master_label}', color='black', alpha=0.7)
    if not overlap_index.empty: ax3.axvspan(overlap_index.min(), overlap_index.max(), color='gray', alpha=0.2)
    ax3.set_xlabel("Year"); ax3.set_ylabel("Ring Width (mm)"); ax3.set_title("3. Raw Data Visual Match"); ax3.legend()

    # Nested GridSpec with a small, fixed space
    gs_nested = gs_main[1, 1].subgridspec(1, 2, wspace=0.1, hspace=0)

    # Prepare text for both boxes
    lead_loss, trail_loss = analysis_dict.get('index_edge_loss', index_edge_loss(index_used))
    rings_in = analysis_dict.get('rings_measured')
    rings_out = analysis_dict.get('rings_after_index')
    index_line = f"Index: {index_used}"
    if lead_loss or trail_loss:
        index_line += (f" (loses {lead_loss} ring(s) at the start, {trail_loss} at the end)"
                       f"\n      Rings: {rings_in} measured -> {rings_out} indexed")
    r2_val = shared_variation(t_val, n_val)
    tier_line = classification_handle(_classify_dendro_match(t_val, n_val, glk_val))
    summary_text_body = textwrap.dedent(f"""
    Sample: {sample_label}
    Reference: {master_label}
    {index_line}

    Most Likely End Year: {best_end_year}
    (Start Year: {best_start_year})

    Best Match Statistics:
      T-Value: {t_val:.2f}
      Overlap (n): {n_val}
      Shared variation r2: {'-' if r2_val is None else f'{r2_val:.2f}'}
      Correlation (r): {r_val:.2f}
      GLK (%): {glk_val:.1f}
      Stands out: {z_val:.1f} SD (2nd best t={second_t:.2f})
      Criteria: {tier_line}
    """)
    summary_text_full = "Statistical Summary\n" + "-"*25 + "\n" + summary_text_body
    
    narrative_text_body = _generate_plot_report_text(analysis_dict)
    narrative_text_full = "Analysis Report\n" + "-"*25 + "\n" + narrative_text_body
    
    # --- CHANGE 1: Force both text boxes to have the same height by padding the shorter one. ---
    summary_lines = summary_text_full.count('\n')
    narrative_lines = narrative_text_full.count('\n')

    if summary_lines > narrative_lines:
        padding = '\n' * (summary_lines - narrative_lines)
        narrative_text_full += padding
    elif narrative_lines > summary_lines:
        padding = '\n' * (narrative_lines - summary_lines)
        summary_text_full += padding

    # Box 4: Statistics Summary (Nested Left)
    ax4 = fig.add_subplot(gs_nested[0, 0]); ax4.axis('off')
    ax4.text(0.0, 1.0, summary_text_full, ha='left', va='top', fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", fc='whitesmoke', ec='grey', lw=1))
    
    # Box 5: Narrative Report (Nested Right)
    ax5 = fig.add_subplot(gs_nested[0, 1]); ax5.axis('off')
    ax5.text(0.0, 1.0, narrative_text_full, ha='left', va='top', fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", fc='aliceblue', ec='lightsteelblue', lw=1.5))
    
    fig.suptitle(f"Cross-Dating Analysis: {sample_label} vs. {master_label}", fontsize=16, fontweight='bold');
    if show:
        plt.show()
    return fig

def _date_one_index(sample_file, master_file, min_overlap=None, is_internal_test=False, reverse_sample=False,
                    spline_stiffness_pct=None, detrend_mode=None, detrend_wavelength=None,
                    sample_series_id_override=None, index=None, verbose=True):
    """Date one sample against one reference under ONE index. See run_date_analysis."""
    min_overlap = DEFAULTS['min_overlap'] if min_overlap is None else min_overlap
    index = resolve_index(index)
    if verbose and not is_internal_test:
        print(f"\n--- Running Analysis ({index}): {os.path.basename(sample_file)} vs {os.path.basename(master_file)} ---")
    sample_chrono = parse_as_floating_series(sample_file)
    if sample_chrono.empty: raise ValueError(f"Could not read data from sample file: {sample_file}")
    if reverse_sample:
        if verbose: print(f"-> Reversing data for sample: {os.path.basename(sample_file)}")
        sample_chrono = sample_chrono.iloc[::-1].reset_index(drop=True)
        sample_chrono.index = pd.RangeIndex(start=1, stop=len(sample_chrono) + 1, name='ring_number')
    reference_is_rwl = master_file.lower().endswith('.rwl')
    reference_depth = None
    if is_internal_test:
        master_chrono = parse_as_floating_series(master_file)
        master_detrended, _ = index_series(master_chrono, index=index,
                                           spline_stiffness_pct=spline_stiffness_pct,
                                           mode=detrend_mode, wavelength=detrend_wavelength)
    elif reference_is_rwl:
        # Standardise each tree in the reference, then average the indices (see
        # build_site_chronology); the raw site mean is kept only for plotting.
        master_chrono = _build_master_from_rwl_file(master_file)
        master_detrended, reference_depth = build_site_chronology(
            master_file, mode=detrend_mode, wavelength=detrend_wavelength,
            spline_stiffness_pct=spline_stiffness_pct, index=index)
    else:
        # A saved master is already an index; only raw widths need standardising. A master
        # built under a different index is refused here, not warned about.
        master_chrono, master_detrended, reference_depth = load_master_csv_series(
            master_file, index=index, mode=detrend_mode, wavelength=detrend_wavelength,
            spline_stiffness_pct=spline_stiffness_pct)
    if master_chrono.empty: raise ValueError(f"Could not read data from reference file: {master_file}")
    if master_detrended.empty: raise ValueError(f"Reference produced no usable chronology: {master_file}")
    if min_overlap > len(sample_chrono): raise ValueError(f"CONFIG ERROR: min_overlap ({min_overlap}) > sample length ({len(sample_chrono)}).")

    # COFECHA-style holdout, stated in one sentence before the result in every case.
    holdout = None
    if not is_internal_test:
        sid = sample_series_id_override or sample_series_id(sample_file)
        master_detrended, reference_depth, holdout = apply_master_holdout(
            master_file, sid, master_detrended, reference_depth,
            mode=detrend_mode, wavelength=detrend_wavelength,
            spline_stiffness_pct=spline_stiffness_pct, index=index)
        if verbose: print(holdout['note'])
        if master_detrended.empty:
            raise ValueError(f"Reference produced no usable chronology after the holdout: {master_file}")

    lead_loss, trail_loss = index_edge_loss(index)
    sample_detrended, sample_spline_fit = index_series(
        sample_chrono, index=index, spline_stiffness_pct=spline_stiffness_pct,
        mode=detrend_mode, wavelength=detrend_wavelength)
    if sample_detrended.empty:
        raise ValueError(
            f"Sample excluded: {len(sample_chrono)} measured rings do not survive the {index} "
            f"index, which needs at least 15 usable rings after losing {lead_loss} at the start "
            f"and {trail_loss} at the end: {sample_file}")
    if len(sample_detrended) < min_overlap:
        raise ValueError(
            f"Sample excluded: {len(sample_detrended)} rings survive the {index} index "
            f"({len(sample_chrono)} measured, {lead_loss} lost at the start and {trail_loss} at "
            f"the end), below the {min_overlap}-year minimum overlap. A t-value over a shorter "
            f"overlap is not comparable to one over {min_overlap}: {sample_file}")
    if verbose and not is_internal_test and (lead_loss or trail_loss):
        print(f"-> {describe_index(index)}")
        print(f"-> {len(sample_detrended)} of {len(sample_chrono)} measured rings survive the "
              f"transform and are what the overlap n is counted from.")

    analysis_results = cross_date_indexed(sample_detrended, master_detrended,
                                          min_overlap=min_overlap, index=index)
    if "error" in analysis_results: raise ValueError(analysis_results['error'])
    if verbose and not is_internal_test:
        print(f"--- Cross-dating complete ({index}) --- end year "
              f"{int(analysis_results['best_match']['end_year'])}")

    composition = {}
    if not is_internal_test:
        try:
            composition = describe_reference_set(master_file)
        except Exception:
            composition = {}
    manifest = run_manifest(
        master_file=master_file, min_overlap=min_overlap, index=index,
        index_edge_loss=f"{lead_loss}+{trail_loss}",
        rings_measured=len(sample_chrono), rings_after_index=len(sample_detrended),
        detrend_mode=detrend_mode or DEFAULTS['detrend_mode'],
        detrend_wavelength=detrend_wavelength or DEFAULTS['detrend_wavelength'],
        spline_stiffness_pct=spline_stiffness_pct or DEFAULTS['spline_stiffness_pct'],
        n_series=1,
        reference_n_sites=composition.get('n_sites'),
        reference_n_series=composition.get('n_series'),
        reference_year_span=(f"{composition['year_min']}-{composition['year_max']}"
                             if composition.get('year_min') is not None else None))
    # How much reference actually existed at the year being claimed. A t-value against a
    # master that has two sites at that year is not the same claim as one against forty.
    best_year = analysis_results.get('best_match', {}).get('end_year')
    depth_at_match = None
    if reference_depth is not None and best_year is not None:
        try:
            depth_at_match = int(reference_depth.get(int(best_year)))
        except (TypeError, ValueError):
            depth_at_match = None
    # What the reference actually is, read from the file the result was measured against.
    reference_meta = ({} if is_internal_test
                      else reference_metadata(master_file, depth_at_year=depth_at_match))
    if holdout is not None:
        record_holdout_depths(holdout, best_year)
        manifest.update(holdout_manifest_fields(holdout))
        if verbose and holdout.get('unusable'):
            print(holdout['unusable_reason'])

    return {'raw_sample': sample_chrono, 'master_detrended': master_detrended, 'detrended_sample': sample_detrended,
            'results': analysis_results, 'sample_filename': sample_file, 'master_filename': master_file,
            'reference_is_rwl': reference_is_rwl, 'raw_master': master_chrono if not is_internal_test else None,
            'sample_spline_fit': sample_spline_fit, 'reverse_sample': reverse_sample, 'analysis_mode': 'single',
            'spline_stiffness_pct': spline_stiffness_pct, 'min_overlap': min_overlap,
            'index': index, 'index_edge_loss': (lead_loss, trail_loss),
            'rings_measured': len(sample_chrono), 'rings_after_index': len(sample_detrended),
            'detrend_mode': detrend_mode or DEFAULTS['detrend_mode'],
            'detrend_wavelength': detrend_wavelength or DEFAULTS['detrend_wavelength'],
            'reference_metadata': reference_meta,
            'reference_metadata_text': format_reference_metadata(reference_meta),
            'reference_depth': reference_depth, 'depth_at_match': depth_at_match,
            'holdout': holdout, 'holdout_note': (holdout or {}).get('note', ''),
            'unusable_reason': (holdout or {}).get('unusable_reason'),
            'run_manifest': manifest, 'reference_composition': composition,
            'reference_composition_text': format_reference_set(composition) if composition else ''}

def run_date_analysis(sample_file, master_file, min_overlap=None, is_internal_test=False,
                      reverse_sample=False, spline_stiffness_pct=None, detrend_mode=None,
                      detrend_wavelength=None, sample_series_id_override=None, index=None,
                      lead_index=None, verbose=True):
    """Date a sample against a reference under every index, unless one is named.

    With `index=None` all three indices are computed, so the index cannot be chosen after
    seeing which one flatters the t-value. `index='bp'` restricts the run to one. The
    returned dict is the result under the lead index, carrying `index_runs` (one entry per
    index, each with its own overlap) and `index_agreement` (whether the end year held).

    An index that a series is too short to survive is excluded with a stated reason; the
    others still run. Only if none produce a result does this raise."""
    wanted = [resolve_index(index)] if index else list(INDEX_METHODS)
    # The lead index fills the single t / n pair a row or a headline can hold.
    lead = resolve_index(lead_index) if lead_index else (
        resolve_index(index) if index else DEFAULTS['lead_index'])
    if lead not in wanted:
        wanted = [lead] + [i for i in wanted if i != lead]
    if is_internal_test:
        return _date_one_index(sample_file, master_file, min_overlap, True, reverse_sample,
                               spline_stiffness_pct, detrend_mode, detrend_wavelength,
                               sample_series_id_override, lead, verbose)
    results, runs = {}, []
    for position, name in enumerate(wanted):
        try:
            result = _date_one_index(sample_file, master_file, min_overlap, False, reverse_sample,
                                     spline_stiffness_pct, detrend_mode, detrend_wavelength,
                                     sample_series_id_override, name,
                                     verbose and position == 0)
            best = result['results']['best_match']
            t, n = float(best['t_value']), int(best['overlap_n'])
            results[name] = result
            runs.append({'index': name, 'end_year': int(best['end_year']), 't_value': t,
                         'overlap_n': n, 'glk': float(best.get('glk', 0.0)),
                         'r2': shared_variation(t, n),
                         'criteria_met': classification_handle(
                             _classify_dendro_match(t, n, best.get('glk', 0.0)))})
        except Exception as e:
            runs.append({'index': name, 'end_year': None, 't_value': None, 'overlap_n': None,
                         'glk': None, 'r2': None, 'criteria_met': None, 'error': str(e)})
    if not results:
        raise ValueError("No index produced a usable result. "
                         + '; '.join(f"{r['index']}: {r.get('error')}" for r in runs))
    primary = results.get(lead) or results[next(iter(results))]
    agreement = index_agreement(runs)
    primary.update({'index_runs': runs, 'index_agreement': agreement,
                    'index_comparison_text': format_index_comparison(runs, agreement),
                    'indices_reported': [r['index'] for r in runs],
                    'index_lead': primary['index']})
    primary['run_manifest'].update({
        'indices_reported': '+'.join(r['index'] for r in runs),
        'index_restricted_to': resolve_index(index) if index else None,
        'end_year_by_index': ', '.join(f"{k}:{v}" for k, v in agreement['years'].items()),
        'end_year_agrees_across_index': agreement['agree']})
    if verbose:
        print()
        print(format_index_comparison(runs, agreement))
    return primary

def process_single_file(args):
    (filename, sample_detrended, sample_basename, base_path, min_overlap, spline_stiffness_pct,
     detrend_mode, detrend_wavelength, sample_sid, index) = args
    if filename == sample_basename: return None
    index = resolve_index(index)
    master_path = os.path.join(base_path, filename)
    # If the sample is one of this site's own series, the chronology is rebuilt without it.
    held_out = None
    depth_before = None
    if sample_sid and _normalize_series_id(sample_sid) in series_ids_in_file(master_path):
        held_out = sample_sid
        _chrono_before, depth_before = build_site_chronology(
            master_path, mode=detrend_mode, wavelength=detrend_wavelength,
            spline_stiffness_pct=spline_stiffness_pct, index=index)
    master_detrended, _depth = build_site_chronology(
        master_path, mode=detrend_mode, wavelength=detrend_wavelength,
        spline_stiffness_pct=spline_stiffness_pct,
        exclude_series=[held_out] if held_out else None, index=index)
    if master_detrended.empty or len(master_detrended) < min_overlap:
        # An exclusion with a reason, not a silent drop.
        return {'excluded': {
            'source_file': filename,
            'reason': (f"too short under the {index} index: "
                       f"{len(master_detrended)} usable years, below the {min_overlap}-year "
                       f"minimum overlap" if not master_detrended.empty else
                       f"no usable chronology under the {index} index")},
            'source_file': filename, 'n_positions': 0, 'passing_alignments': [],
            't_values': np.empty(0, dtype=np.float32)}
    analysis_results = cross_date_indexed(sample_detrended, master_detrended,
                                          min_overlap=min_overlap, index=index)
    if "error" in analysis_results: return None
    best_match = analysis_results['best_match']
    if best_match['correlation'] >= 0.9999: return None  # drop a reference that is (near) identical to the sample itself
    holdout_info = None
    if held_out:
        year = int(best_match['end_year'])
        floor = DEFAULTS['min_series_depth']
        after = _depth_lookup(_depth, year)
        before = _depth_lookup(depth_before, year)
        holdout_info = {
            'source_file': filename, 'held_out_series': held_out, 'depth_year': year,
            'depth_before_at_year': before,
            'depth_after_at_year': after, 'depth_floor': floor, 'depth_label': 'series',
            'unusable': (after is not None and after < floor) or (after is None and before is not None),
        }
        if holdout_info['unusable']:
            holdout_info['unusable_reason'] = (
                f"{filename}: after holding out '{held_out}', only "
                f"{0 if after is None else after} series remain at {year}, below the minimum "
                f"depth of {floor} -- no usable result from this reference.")
            return {'unusable_holdout': holdout_info, 'source_file': filename,
                    'n_positions': 0, 'passing_alignments': [],
                    't_values': np.empty(0, dtype=np.float32)}
    best_match['source_file'] = filename
    # Every offset tested against this reference is another comparison; keep every
    # alignment that meets the same classification threshold used for the winner,
    # not only the single best-t offset, so the search's full result set can be audited.
    rdf = analysis_results['all_correlations'].reset_index()
    passing_alignments = []
    for _, row in rdf.iterrows():
        t, o, g = row['t_value'], row['overlap_n'], row['glk']
        if _classify_dendro_match(t, o, g)['meets_any']:
            passing_alignments.append({'end_year': int(row['end_year']), 't_value': float(t),
                                        'overlap_n': int(o), 'glk': float(g), 'source_file': filename})
    # Every t-value produced, not only those passing threshold, is retained (as float32,
    # not a list of dicts) so the search's background distribution can be characterised
    # later without re-running the search. len(t_values) == len(rdf) == n_positions always,
    # by construction — nothing is filtered out here.
    t_values = rdf['t_value'].to_numpy(dtype=np.float32)
    return {'best_match': best_match, 'n_positions': len(rdf), 'passing_alignments': passing_alignments,
            'source_file': filename, 't_values': t_values, 'holdout': holdout_info}

def run_detective_analysis(sample_file, target, top_n=None, min_overlap=None, min_end_year=None, reverse_sample=False,
                           spline_stiffness_pct=None, detrend_mode=None, detrend_wavelength=None,
                           sample_series_id_override=None, min_country_series=None, index=None):
    """Rank a sample against a reference set under every index, unless one is named.

    `index` restricts the sweep (the CLI's --search-index). Restricting is faster and the
    report then states that the cross-index comparison of candidates was not performed."""
    search_indices = [resolve_index(index)] if index else list(INDEX_METHODS)
    lead_index = resolve_index(index) if index else DEFAULTS['lead_index']
    top_n = DEFAULTS['top_n'] if top_n is None else top_n
    min_overlap = DEFAULTS['min_overlap'] if min_overlap is None else min_overlap
    min_end_year = DEFAULTS['min_end_year'] if min_end_year is None else min_end_year
    min_country_series = (DEFAULTS['min_country_series'] if min_country_series is None
                          else min_country_series)
    print(f"\n--- Running Detective Analysis on {os.path.basename(sample_file)} ---")
    sample_chrono = parse_as_floating_series(sample_file)
    if sample_chrono.empty: raise ValueError("Could not process sample file.")
    if reverse_sample:
        print(f"-> Reversing data for sample: {os.path.basename(sample_file)}")
        sample_chrono = sample_chrono.iloc[::-1].reset_index(drop=True)
        sample_chrono.index = pd.RangeIndex(start=1, stop=len(sample_chrono) + 1, name='ring_number')
    if min_overlap > len(sample_chrono): raise ValueError(f"CONFIG ERROR: min_overlap ({min_overlap}) > sample length ({len(sample_chrono)}).")
    file_list, base_path_for_masters = [], ""
    ref_start_years, ref_end_years = [], []

    if target == VIOLIN_TARGET_KEY:
        base_path_for_masters = VIOLIN_REFERENCE_DIR
        if not os.path.exists(base_path_for_masters) or not os.listdir(base_path_for_masters):
             raise ValueError(f"'{VIOLIN_REFERENCE_LABEL}' needs reference files in '{base_path_for_masters}'.\nRun the 'Fetch Tonewood Forest References' tool in the Setup tab first.")
        file_list = [f for f in os.listdir(base_path_for_masters) if f.lower().endswith('.rwl')]
    elif os.path.isdir(target):
        base_path_for_masters = target
        file_list = [f for f in os.listdir(target) if f.lower().endswith('.rwl')]
    else:
        base_path_for_masters = "full_rwl_cache"
        index_filename = "noaa_europe_index.csv"
        if not os.path.exists(index_filename): raise ValueError("Index file missing. Run 'gogo index' first.")
        if target not in CATEGORY_PARAMS:
            raise ValueError(f"Invalid category '{target}'. Valid targets: {', '.join(DETECTIVE_TARGETS)}, or a folder path.")
        params = CATEGORY_PARAMS[target]
        index_df = pd.read_csv(index_filename)
        df_filtered = filter_index(index_df, params, min_end_year)
        file_list = df_filtered['filename'].tolist()
        ref_start_years = df_filtered['start_year'].tolist()
        ref_end_years = df_filtered['end_year'].tolist()

    if not file_list: raise ValueError(f"No reference files found for target '{target}' (including min_end_year={min_end_year}).")

    # Describe the reference set that will actually be searched (Task 5 disclosure):
    # date span, when not already known from the index, and geographic coverage from headers.
    if not ref_start_years:
        metas = [get_metadata_from_rwl(os.path.join(base_path_for_masters, f)) for f in file_list]
        metas = [m for m in metas if m]
        ref_start_years = [m['start_year'] for m in metas]
        ref_end_years = [m['end_year'] for m in metas]
    # Location strings from real-world headers are inconsistent, truncated, or outright junk
    # (Task 2 hardening); normalize/dedupe and track how many chronologies actually yielded
    # something usable so the report can flag an unreliable geographic-coverage list instead
    # of silently printing garbage.
    seen_normalized, ref_locations, n_location_ok = set(), [], 0
    for f in file_list:
        raw_loc = _parse_rwl_header(os.path.join(base_path_for_masters, f)).get('location', 'N/A')
        loc = _normalize_location(raw_loc)
        if loc:
            n_location_ok += 1
            key = loc.casefold()
            if key not in seen_normalized:
                seen_normalized.add(key)
                ref_locations.append(loc)
    ref_set_info = {
        'ref_set_n': len(file_list), 'ref_set_dir': base_path_for_masters,
        'ref_set_locations': sorted(ref_locations),
        'ref_set_location_n_ok': n_location_ok, 'ref_set_location_n_total': len(file_list),
        'ref_set_start': int(min(ref_start_years)) if ref_start_years else None,
        'ref_set_end': int(max(ref_end_years)) if ref_end_years else None,
    }

    # Describe what the search could match against, BEFORE any ranking is printed, so a
    # reader meets the composition of the reference set before meeting a winner from it.
    # Only the sites this run will actually search. For a category target the cache holds
    # ~1900 sites while the filter selects a few dozen; describing the folder would
    # describe a reference set that was never consulted.
    composition = describe_reference_set(base_path_for_masters, files=file_list)
    manifest = run_manifest(
        master_file=None, min_overlap=min_overlap,
        detrend_mode=detrend_mode or DEFAULTS['detrend_mode'],
        detrend_wavelength=detrend_wavelength or DEFAULTS['detrend_wavelength'],
        spline_stiffness_pct=spline_stiffness_pct or DEFAULTS['spline_stiffness_pct'],
        n_series=composition.get('n_series'), target=target, index=lead_index,
        rings_measured=len(sample_chrono),
        reference_set=base_path_for_masters, n_reference_sites=composition.get('n_sites'),
        min_country_series=min_country_series)
    print()
    print(format_reference_set(composition))

    # Every reference here is an .rwl, so its series IDs are always recoverable and the
    # holdout can always be performed. Index-independent, so it is resolved once.
    sample_sid = sample_series_id_override or sample_series_id(sample_file)

    def sweep(index):
        """The whole search under one index: ranking, coverage and search statistics."""
        index = resolve_index(index)
        sweep_manifest = dict(manifest, index=index)
        lead_loss, trail_loss = index_edge_loss(index)
        sample_detrended, _ = index_series(sample_chrono, index=index,
                                           spline_stiffness_pct=spline_stiffness_pct,
                                           mode=detrend_mode, wavelength=detrend_wavelength)
        surviving = len(sample_detrended)
        if sample_detrended.empty or surviving < min_overlap:
            raise ValueError(
                f"Sample excluded from the {index} index: {surviving} of {len(sample_chrono)} "
                f"measured rings survive it ({lead_loss} lost at the start, {trail_loss} at the "
                f"end), below the {min_overlap}-year minimum overlap.")
        print(f"\n=== Searching under the {index} index ===")
        if lead_loss or trail_loss:
            print(f"-> {describe_index(index)}")
            print(f"-> {surviving} of {len(sample_chrono)} measured rings survive the transform "
                  f"and are what every overlap n below is counted from.")
        tasks = [(filename, sample_detrended, os.path.basename(sample_file), base_path_for_masters, min_overlap,
                  spline_stiffness_pct, detrend_mode, detrend_wavelength, sample_sid, index)
                 for filename in file_list]
        print(f"\nTesting against {len(file_list)} sites using {multiprocessing.cpu_count()} CPU cores...")
        with multiprocessing.Pool() as pool:
            pool_results = [res for res in tqdm(pool.imap(process_single_file, tasks), total=len(tasks)) if res is not None]

        # References left under the depth floor by the holdout are reported, not quietly ranked.
        unusable_holdouts = [r['unusable_holdout'] for r in pool_results if 'unusable_holdout' in r]
        excluded_refs = [r['excluded'] for r in pool_results if 'excluded' in r]
        all_file_results = [r for r in pool_results if 'best_match' in r]
        if excluded_refs:
            print(f"\n{len(excluded_refs)} reference(s) excluded before ranking:")
            for info in excluded_refs[:10]:
                print(f"  {info['source_file']}: {info['reason']}")
            if len(excluded_refs) > 10:
                print(f"  ...and {len(excluded_refs) - 10} more, all for the same kind of reason.")
        held_out_files = [r['holdout'] for r in all_file_results if r.get('holdout')]
        if sample_sid is None:
            holdout_summary = ("HOLDOUT: NOT PERFORMED -- the sample's own series ID could not be read "
                               "from its file, so whether the sample sits inside any reference searched "
                               "could not be checked.")
        elif held_out_files or unusable_holdouts:
            n_held = len(held_out_files) + len(unusable_holdouts)
            holdout_summary = (f"HOLDOUT: the sample's series ID '{sample_sid}' is one of the measured "
                               f"series in {n_held} of the {len(file_list)} references searched; each of "
                               f"those was rebuilt without it before dating.")
        else:
            holdout_summary = (f"HOLDOUT: not needed -- the sample's series ID '{sample_sid}' is not one "
                               f"of the measured series in any of the {len(file_list)} references searched.")
        print("\n" + holdout_summary)
        for info in unusable_holdouts:
            print("  " + info['unusable_reason'])

        if not all_file_results: print("\nAnalysis complete, no significant correlations found."); return None

        all_best_results = [r['best_match'] for r in all_file_results]
        results_df = pd.DataFrame(all_best_results).sort_values(by='t_value', ascending=False)
        top_results = results_df.head(top_n)

        # The manifest records the holdout for the reference the reported result comes from.
        holdout_by_file = {h['source_file']: h for h in held_out_files}
        top_holdout = holdout_by_file.get(top_results.iloc[0]['source_file'])
        sweep_manifest.update({
            'held_out_series': sample_sid if (held_out_files or unusable_holdouts) else None,
            'holdout_performed': bool(held_out_files or unusable_holdouts),
            'holdout_references_rebuilt': len(held_out_files) + len(unusable_holdouts),
            'holdout_references_unusable': len(unusable_holdouts),
            'holdout_note': holdout_summary,
        })
        if top_holdout:
            sweep_manifest.update({'holdout_depth_year': top_holdout['depth_year'],
                             'depth_before_holdout': top_holdout['depth_before_at_year'],
                             'depth_after_holdout': top_holdout['depth_after_at_year']})

        # Every offset tested (across every reference chronology) is another comparison;
        # a threshold calibrated for a single test does not carry its nominal error rate
        # across a search. Retain every alignment that met the threshold, not only the
        # best site's best offset, so a reader can see the full set of candidates.
        n_positions_total = sum(r['n_positions'] for r in all_file_results)
        all_passing = []
        for r in all_file_results: all_passing.extend(r['passing_alignments'])
        n_passing = len(all_passing)
        if all_passing:
            passing_df = pd.DataFrame(all_passing).sort_values(by='t_value', ascending=False).reset_index(drop=True)
            unique_files = passing_df['source_file'].unique()
            hdr_cache = {f: _parse_rwl_header(os.path.join(base_path_for_masters, f)) for f in unique_files}
            passing_df['site_name'] = passing_df['source_file'].map(lambda f: hdr_cache[f]['site_name'])
            passing_df['location'] = passing_df['source_file'].map(lambda f: hdr_cache[f]['location'])
        else:
            passing_df = pd.DataFrame(columns=['end_year', 't_value', 'overlap_n', 'glk', 'source_file', 'site_name', 'location'])

        header_infos = [_parse_rwl_header(os.path.join(base_path_for_masters, row.source_file)) for _, row in top_results.iterrows()]
        header_df = pd.DataFrame(header_infos, index=top_results.index)
        enriched_results_df = pd.concat([top_results, header_df], axis=1)

        # Reference depth at each reported end year: a top-ranked hit at a year the reference
        # set barely covers is a weaker claim than the same t-value where coverage is deep.
        top_display = top_results[['end_year', 't_value', 'glk', 'correlation', 'overlap_n', 'source_file']].copy()
        top_display['r2'] = [round(shared_variation(t, n), 2) for t, n
                             in zip(top_display['t_value'], top_display['overlap_n'])]
        top_display['criteria_met'] = [
            classification_handle(_classify_dendro_match(t, n, g)) for t, n, g
            in zip(top_display['t_value'], top_display['overlap_n'], top_display['glk'])]
        top_display['ref_depth_at_year'] = [
            reference_depth_at(composition, y) for y in top_display['end_year']]

        print(f"\n--- Top {top_n} references by t-value, searched under the {index} index ---")
        print(top_display.to_string(index=False))
        print(f"\n{n_positions_total} alignment positions were evaluated across {len(file_list)} reference chronologies. {n_passing} met the stated threshold.")

        # What was searched per country prefix, including the countries nothing matched in.
        coverage_df = country_coverage(composition, all_best_results,
                                      min_country_series=min_country_series)
        print()
        print(format_country_coverage(coverage_df, min_overlap=min_overlap))

        # --- Search-context statistics (Task 1): a t-value alone doesn't say how many other
        # positions were tried to find it. Summarise the FULL distribution of t-values produced
        # by the search (not only the ones that passed threshold), and where the winner ranks
        # within it. This is pure summarisation of data already computed above; no new search
        # is performed and cross_date/detrend/_classify_dendro_match are untouched.
        all_t_values = (np.concatenate([r['t_values'] for r in all_file_results])
                         if all_file_results else np.empty(0, dtype=np.float32))
        search_n_alignments = int(n_positions_total)
        best_t = float(results_df.iloc[0]['t_value'])
        search_stats_reliable = search_n_alignments >= 100

        if search_n_alignments > 0:
            # Exact, computed from the full distribution regardless of what is later retained.
            best_t_rank = int(np.sum(all_t_values > best_t)) + 1
            best_t_percentile = float(np.sum(all_t_values < best_t) / search_n_alignments)
            n_above_t5 = int(np.sum(all_t_values > 5.0))
            n_above_t6 = int(np.sum(all_t_values > 6.0))
            n_above_t8 = int(np.sum(all_t_values > 8.0))
        else:
            best_t_rank, best_t_percentile, n_above_t5, n_above_t6, n_above_t8 = 0, 0.0, 0, 0, 0

        # Retention is capped for memory safety on very large reference sets; a random reservoir
        # sample stands in for the full array rather than truncating the search itself. When
        # sampled, the percentiles below are computed from the sample and the report says so.
        search_stats_sampled = search_n_alignments > SEARCH_T_VALUES_RESERVOIR_CAP
        if search_stats_sampled:
            search_t_values = np.random.default_rng().choice(all_t_values, size=SEARCH_T_VALUES_RESERVOIR_CAP, replace=False).astype(np.float32)
        else:
            search_t_values = all_t_values

        if search_stats_reliable and len(search_t_values) > 0:
            t_p50, t_p95, t_p99 = (float(v) for v in np.nanpercentile(search_t_values, [50, 95, 99]))
        else:
            t_p50 = t_p95 = t_p99 = 0.0

        # Sanity check: catches an offset-loop counting bug before the report ships. When
        # reservoir-sampled, the retained array's length is the sample size by design, not
        # the full count, so the equality check only applies in the (typical) unsampled case.
        if not search_stats_sampled:
            assert search_n_alignments == len(search_t_values), "alignment count and t-value array disagree"
        else:
            assert len(search_t_values) == SEARCH_T_VALUES_RESERVOIR_CAP, "reservoir sample size mismatch"
        print(f"INFO: search_n_alignments = {search_n_alignments} (across {len(file_list)} reference chronologies)")

        detective_context = {
            "analysis_type": "detective", "enriched_results_df": enriched_results_df,
            "min_end_year": min_end_year, "target": target, "min_overlap": min_overlap, "top_n": top_n,
            "n_positions_total": n_positions_total, "n_passing": n_passing, "candidate_alignments_df": passing_df,
            "search_n_alignments": search_n_alignments, "search_t_values": search_t_values,
            "search_stats_sampled": search_stats_sampled, "search_stats_reliable": search_stats_reliable,
            "best_t_rank": best_t_rank, "best_t_percentile": best_t_percentile,
            "n_above_t5": n_above_t5, "n_above_t6": n_above_t6, "n_above_t8": n_above_t8,
            "t_p50": t_p50, "t_p95": t_p95, "t_p99": t_p99,
            "reference_composition": composition,
            "reference_composition_text": format_reference_set(composition),
            "top_results_with_depth": top_display,
            "country_coverage_df": coverage_df,
            "country_coverage_text": format_country_coverage(coverage_df, min_overlap=min_overlap),
            "min_country_series": min_country_series,
            "run_manifest": sweep_manifest,
            "holdout_note": holdout_summary,
            "holdout_unusable_references": unusable_holdouts,
            "holdout_references_rebuilt": len(held_out_files) + len(unusable_holdouts),
            "excluded_references": excluded_refs,
            "index": index, "index_edge_loss": (lead_loss, trail_loss),
            "rings_measured": len(sample_chrono), "rings_after_index": len(sample_detrended),
            **ref_set_info,
        }
        return detective_context

    # The search runs under every index, not one. Re-dating a single winner under the others
    # tests whether the YEAR holds; it cannot test whether the WINNER holds, and if two
    # indices rank different references first that is the finding.
    sweeps, sweep_errors = {}, {}
    for name in search_indices:
        try:
            result = sweep(name)
        except Exception as e:
            sweep_errors[name] = str(e)
            print(f"\n{name}: search not performed -- {e}")
            continue
        if result is not None:
            sweeps[name] = result
    if not sweeps:
        print("\nAnalysis complete, no index produced a usable ranking.")
        return None

    candidates = candidate_agreement(sweeps, sweep_errors, restricted=bool(index))
    print()
    print(format_candidate_agreement(candidates, sweeps))

    lead = lead_index if lead_index in sweeps else next(iter(sweeps))
    detective_context = dict(sweeps[lead])
    detective_context.update({
        'search_indices': list(sweeps),
        'search_index_errors': sweep_errors,
        'candidate_agreement': candidates,
        'candidate_agreement_text': format_candidate_agreement(candidates, sweeps),
        'per_index_top': {name: ctx['top_results_with_depth'] for name, ctx in sweeps.items()},
        'per_index_winner': candidates['winners'],
    })
    detective_context['run_manifest'] = dict(
        detective_context.get('run_manifest', {}),
        search_indices='+'.join(sweeps),
        search_index_restricted_to=resolve_index(index) if index else None,
        same_reference_wins_across_index=candidates['agree'])

    top_match_file_name = sweeps[lead]['top_results_with_depth'].iloc[0]['source_file']
    top_match_full_path = os.path.join(base_path_for_masters, top_match_file_name)
    print(f"\nGenerating plot for the {lead} winner: {top_match_file_name}")
    plot_data_dict = run_date_analysis(sample_file, top_match_full_path, min_overlap=min_overlap, reverse_sample=reverse_sample,
                                       spline_stiffness_pct=spline_stiffness_pct, detrend_mode=detrend_mode,
                                       detrend_wavelength=detrend_wavelength,
                                       sample_series_id_override=sample_sid,
                                       index=index, lead_index=lead)
    if not plot_data_dict:
        print("Could not generate plot for the top match.")
        return {"analysis_mode": "single", "sample_file": sample_file, **detective_context}

    # Update dictionary with all detective mode context
    plot_data_dict.update(detective_context)
    return plot_data_dict

def run_two_piece_mean_analysis(bass_file, treble_file, date_one, reverse_bass=False,
                                reverse_treble=False, spline_stiffness_pct=None,
                                detrend_mode=None, detrend_wavelength=None, min_overlap_internal=None,
                                index=None):
    """Date a two-piece plate: each half on its own, plus the halves against each other.

    `date_one(sample_path)` dates one floating series against whatever the caller chose --
    a master (run_date_analysis) or a whole category (run_detective_analysis) -- so build it
    with functools.partial and every argument stays named:

        date_one = partial(run_date_analysis, master_file=master, min_overlap=60, index=None)

    Both halves are always dated separately, so a result is produced whether or not they
    share a wedge. The averaged chronology is only formed when the bass-to-treble comparison
    says the halves are one tree; around 30% of instruments have halves from different logs,
    and forcing a mean of two unrelated trees destroys the signal in both."""
    min_overlap_internal = DEFAULTS['min_overlap_internal'] if min_overlap_internal is None else min_overlap_internal
    index = resolve_index(index)
    print("\n--- Starting Two-Piece Plate Analysis ---")
    bass_chrono = parse_as_floating_series(bass_file)
    treble_chrono = parse_as_floating_series(treble_file)
    if bass_chrono.empty or treble_chrono.empty: raise ValueError("Could not process two-piece sample files.")
    if reverse_bass:
        print(f"-> Reversing data for Bass sample: {os.path.basename(bass_file)}")
        bass_chrono = bass_chrono.iloc[::-1].reset_index(drop=True)
        bass_chrono.index = pd.RangeIndex(start=1, stop=len(bass_chrono) + 1, name='ring_number')
    if reverse_treble:
        print(f"-> Reversing data for Treble sample: {os.path.basename(treble_file)}")
        treble_chrono = treble_chrono.iloc[::-1].reset_index(drop=True)
        treble_chrono.index = pd.RangeIndex(start=1, stop=len(treble_chrono) + 1, name='ring_number')
    raw_bass_for_report = bass_chrono.copy()
    raw_treble_for_report = treble_chrono.copy()

    print("--- Internal Cross-Match (Bass vs. Treble) ---")
    bass_detrended, _ = index_series(bass_chrono, index=index, spline_stiffness_pct=spline_stiffness_pct,
                                     mode=detrend_mode, wavelength=detrend_wavelength)
    treble_detrended, _ = index_series(treble_chrono, index=index, spline_stiffness_pct=spline_stiffness_pct,
                                       mode=detrend_mode, wavelength=detrend_wavelength)
    if bass_detrended.empty or treble_detrended.empty:
        lead_loss, trail_loss = index_edge_loss(index)
        raise ValueError(
            f"Excluded: one or both halves do not survive the {index} index, which needs at least "
            f"15 usable rings after losing {lead_loss} at the start and {trail_loss} at the end "
            f"(bass {len(bass_chrono)} rings, treble {len(treble_chrono)} rings).")
    internal_overlap_used = min(min_overlap_internal, len(bass_detrended), len(treble_detrended))
    internal_results = cross_date_indexed(bass_detrended, treble_detrended,
                                          min_overlap=internal_overlap_used, index=index)
    if "error" in internal_results:
        internal_best = {'t_value': 0.0, 'glk': 0.0, 'overlap_n': 0, 'end_year': 0}
        print(f"Internal cross-match could not be computed: {internal_results['error']}")
    else:
        internal_best = internal_results['best_match']
    plate_relationship = classify_plate_relationship(internal_best)
    internal_t = plate_relationship['t_value']
    internal_glk = plate_relationship['glk']
    internal_overlap = plate_relationship['overlap_n']
    print(f"Internal Match Stats: t-value = {internal_t:.2f}, Glk = {internal_glk:.1f}%, Overlap = {internal_overlap} yrs")
    print(f"Plate relationship: {plate_relationship['verdict'].replace('_', ' ').upper()} -- {plate_relationship['note']}")

    def _run_on(series: pd.Series, label: str, tag: str):
        """Run the caller's analysis on one floating series."""
        temp_path = f"_temp_{tag}_chrono.rwl"
        try:
            _write_floating_rwl(series, temp_path, series_id=tag.upper())
            print(f"\n--- Running Analysis on {label} ---")
            return date_one(temp_path)
        except Exception as e:
            print(f"Analysis of {label} failed: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Each half is dated independently, always.
    bass_results = _run_on(bass_chrono, "Bass side (alone)", "bass")
    treble_results = _run_on(treble_chrono, "Treble side (alone)", "treble")

    mean_results, mean_chrono_series = None, None
    if plate_relationship['same_wedge']:
        aligned_bass = bass_chrono.copy()
        aligned_bass.index = aligned_bass.index + int(internal_best['end_year'] - bass_chrono.index.max())
        mean_chrono_series = pd.concat([aligned_bass, treble_chrono], axis=1).mean(axis=1).dropna()
        mean_chrono_series.index = pd.RangeIndex(start=1, stop=len(mean_chrono_series) + 1, name='ring_number')
        mean_results = _run_on(mean_chrono_series, "Mean chronology (Bass+Treble)", "mean")

    # The mean is the best estimate when the halves are one tree; otherwise the halves
    # are separate evidence and the stronger one leads, with the other reported beside it.
    primary = mean_results or bass_results or treble_results
    if primary is None:
        raise ValueError("No usable result: neither plate half could be dated.")
    if mean_results is None and bass_results and treble_results:
        bass_t = bass_results.get('results', {}).get('best_match', {}).get('t_value', 0.0)
        treble_t = treble_results.get('results', {}).get('best_match', {}).get('t_value', 0.0)
        primary = bass_results if bass_t >= treble_t else treble_results

    primary.update({
        'analysis_mode': 'two_piece',
        'internal_stats': {'t_value': internal_t, 'glk': internal_glk, 'overlap_n': internal_overlap},
        'plate_relationship': plate_relationship,
        'bass_file': bass_file, 'treble_file': treble_file,
        'sample_filename': ("Mean Chronology (Bass+Treble)" if mean_results is primary
                            else ("Bass side" if primary is bass_results else "Treble side")),
        'mean_series_length': len(mean_chrono_series) if mean_chrono_series is not None else 0,
        'reverse_bass': reverse_bass, 'reverse_treble': reverse_treble,
        'raw_bass_series': raw_bass_for_report, 'raw_treble_series': raw_treble_for_report,
        'bass_result': bass_results, 'treble_result': treble_results, 'mean_result': mean_results,
    })
    return primary

# --- COMMAND LOGIC ---

# NOAA retired the anonymous NCDC FTP host; the paleo tree-ring archive now lives
# on NCEI over HTTPS. We try FTP first (it may still be reachable) and fall back to
# HTTPS automatically so setup keeps working either way.
NOAA_FTP_HOST = "ftp.ncdc.noaa.gov"
NOAA_FTP_DIR = "/pub/data/paleo/treering/measurements/europe/"
NOAA_HTTPS_DIR = "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/europe/"

def _download_via_ftp(cache_dir, standard_file_pattern):
    ftp = ftplib.FTP(NOAA_FTP_HOST, timeout=60); ftp.login(); ftp.cwd(NOAA_FTP_DIR)
    all_server_files = ftp.nlst()
    files_to_download, skipped = [], []
    for f in all_server_files:
        if standard_file_pattern.match(f): files_to_download.append(f)
        elif f.lower().endswith('.rwl'): skipped.append(f)
    print(f"Found {len(all_server_files)} files on the FTP server.")
    print(f"-> {len(files_to_download)} match the standard format and will be downloaded.")
    if skipped: print(f"-> {len(skipped)} non-standard .rwl files will be skipped.")
    for filename in tqdm(files_to_download, desc="Downloading (FTP)"):
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            try:
                with open(local_path, 'wb') as fh: ftp.retrbinary(f"RETR {filename}", fh.write)
            except Exception as e: print(f"Warning: Failed to download {filename}: {e}"); continue
    ftp.quit()
    return files_to_download

def _download_via_https(cache_dir, standard_file_pattern):
    print(f"Falling back to HTTPS: {NOAA_HTTPS_DIR}")
    req = urllib.request.Request(NOAA_HTTPS_DIR, headers={"User-Agent": "gogo-dendro/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", "ignore")
    # Directory listing hrefs are bare filenames like 'germ12.rwl'.
    all_server_files = sorted(set(re.findall(r'href="([^"/?]+\.rwl)"', html, flags=re.IGNORECASE)))
    files_to_download = [f for f in all_server_files if standard_file_pattern.match(f)]
    print(f"Found {len(all_server_files)} .rwl files listed; {len(files_to_download)} match the standard format.")
    if not files_to_download:
        raise ConnectionError("HTTPS listing returned no standard .rwl files (page format may have changed).")
    for filename in tqdm(files_to_download, desc="Downloading (HTTPS)"):
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            try:
                fr = urllib.request.Request(NOAA_HTTPS_DIR + filename, headers={"User-Agent": "gogo-dendro/1.0"})
                with urllib.request.urlopen(fr, timeout=60) as r, open(local_path, 'wb') as fh:
                    shutil.copyfileobj(r, fh)
            except Exception as e: print(f"Warning: Failed to download {filename}: {e}"); continue
    return files_to_download

def download_and_index_files(index_filename="noaa_europe_index.csv"):
    print("--- Stage 1: Downloading standard .rwl files and creating index ---")
    cache_dir = "full_rwl_cache"
    os.makedirs(cache_dir, exist_ok=True)
    standard_file_pattern = re.compile(r"^[a-zA-Z]+[0-9]+\.rwl$")
    try:
        files_to_download = _download_via_ftp(cache_dir, standard_file_pattern)
    except Exception as e:
        print(f"FTP download unavailable ({e}).")
        try:
            files_to_download = _download_via_https(cache_dir, standard_file_pattern)
        except Exception as e2:
            raise ConnectionError(f"Both FTP and HTTPS downloads failed. FTP: {e} | HTTPS: {e2}")
    print("Download complete.")
    index_cache(cache_dir=cache_dir, index_filename=index_filename, only_files=set(files_to_download))

def index_cache(cache_dir="full_rwl_cache", index_filename="noaa_europe_index.csv", only_files=None):
    """(Re)build the site index from files already in the cache. No downloading.

    Kept separate from downloading so the index can be rebuilt after a parser change
    without re-fetching 1900+ files from NOAA."""
    print("\n--- Indexing cached files ---")
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory '{cache_dir}' not found. Run 'gogo.py index' first.")
    names = sorted(f for f in os.listdir(cache_dir) if f.lower().endswith('.rwl'))
    if only_files is not None:
        names = [f for f in names if f in only_files]
    all_metadata, skipped = [], 0
    for filename in tqdm(names, desc="Validating and Indexing"):
        metadata = get_metadata_from_rwl(os.path.join(cache_dir, filename))
        if metadata:
            metadata['filename'] = filename
            all_metadata.append(metadata)
        else:
            skipped += 1
    if not all_metadata:
        raise ValueError("No valid, dated Tucson-format files could be indexed.")
    df = pd.DataFrame(all_metadata)
    df.to_csv(index_filename, index=False)
    print(f"\nSUCCESS: Index with {len(df)} valid entries written to '{index_filename}'"
          f"{f' ({skipped} unreadable files skipped)' if skipped else ''}.")
    cembra = int(df['is_cembra'].sum()) if 'is_cembra' in df.columns else 0
    print(f"    genus counts: {df['species'].value_counts().to_dict()}")
    print(f"    Pinus cembra sites identified: {cembra}")
    return index_filename

def build_master_from_index(category, min_end_year=None, index_filename="noaa_europe_index.csv",
                            min_depth=None, detrend_mode=None, detrend_wavelength=None,
                            spline_stiffness_pct=None, exclude_series=None, index=None):
    """Build one named master chronology from the site index.

    `category` is a key of CATEGORY_PARAMS, which is the only place a category's species,
    countries and length rules are defined. `exclude_series` rebuilds the master without
    the named series (see build_site_chronology)."""
    min_end_year = DEFAULTS['min_end_year'] if min_end_year is None else min_end_year
    min_depth = DEFAULTS['min_series_depth'] if min_depth is None else min_depth
    if category not in CATEGORY_PARAMS:
        raise ValueError(f"Unknown category '{category}'. Valid categories: {', '.join(CATEGORY_NAMES)}.")
    params = CATEGORY_PARAMS[category]
    label = params['label']
    print(f"\nBUILDING: '{label}'")
    if not os.path.exists(index_filename):
        raise FileNotFoundError("Index missing. Run 'python gogo.py index' first.")
    index_df = pd.read_csv(index_filename)
    df_filtered = filter_index(index_df, params, min_end_year)
    file_list = df_filtered['filename'].tolist()
    if not file_list:
        raise ValueError(f"No files in index matched criteria (including min_end_year={min_end_year}).")
    print(f"Found {len(file_list)} matching sites. Processing...")
    paths = [os.path.join("full_rwl_cache", f) for f in file_list]
    master, site_count, tree_count, kept, raw_mean, _depth = _combine_site_chronologies(
        paths, f"Building {label}", min_depth,
        detrend_mode, detrend_wavelength, spline_stiffness_pct,
        exclude_series=exclude_series, index=index)
    suffix = '' if resolve_index(index) == 'spline' else f"_{resolve_index(index)}"
    output_filename = f"master_{label.lower().replace(' ', '_')}{suffix}.csv"
    _save_master(master, site_count, tree_count, output_filename, label,
                 source_paths=paths, min_depth=min_depth, exclude_series=exclude_series,
                 detrend_mode=detrend_mode, detrend_wavelength=detrend_wavelength,
                 spline_stiffness_pct=spline_stiffness_pct, index=index, raw_mean=raw_mean)
    return output_filename

def run_create_master(input_folder, output_filename, min_depth=3, detrend_mode=None,
                      detrend_wavelength=None, spline_stiffness_pct=None,
                      exclude_series=None, index=None):
    print(f"\nCREATING CUSTOM MASTER: {input_folder}")
    if not os.path.isdir(input_folder): raise FileNotFoundError(f"Folder '{input_folder}' does not exist.")
    file_list = [f for f in os.listdir(input_folder) if f.lower().endswith('.rwl')]
    if not file_list: raise ValueError(f"No .rwl files found in '{input_folder}'.")
    print(f"Found {len(file_list)} .rwl files. Processing...")
    paths = [os.path.join(input_folder, f) for f in file_list]
    master, site_count, tree_count, kept, raw_mean, _depth = _combine_site_chronologies(
        paths, "Processing files", min_depth,
        detrend_mode, detrend_wavelength, spline_stiffness_pct,
        exclude_series=exclude_series, index=index)
    if not output_filename.lower().endswith('.csv'): output_filename += ".csv"
    _save_master(master, site_count, tree_count, output_filename, os.path.basename(output_filename),
                 source_paths=paths, min_depth=min_depth, exclude_series=exclude_series,
                 detrend_mode=detrend_mode, detrend_wavelength=detrend_wavelength,
                 spline_stiffness_pct=spline_stiffness_pct, index=index, raw_mean=raw_mean)
    return output_filename

def fetch_and_build_violin_master():
    """Gather the curated tonewood forest chronologies and build a master from them.

    These are ITRDB chronologies measured from standing trees in regions the
    instrument-dating literature identifies as tonewood sources. No instrument was
    measured to produce them."""
    source_dir = "full_rwl_cache"
    dest_dir = VIOLIN_REFERENCE_DIR
    master_filename = VIOLIN_MASTER_FILENAME

    print(f"\n--- Preparing {VIOLIN_REFERENCE_LABEL} ---")
    print(textwrap.fill(VIOLIN_REFERENCE_BLURB, 78))
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' not found. Please run 'Download and Create Index' first.")

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Destination folder '{dest_dir}' is ready.")

    copied_count = 0
    for filename in VIOLIN_FILES:
        source_path = os.path.join(source_dir, filename)
        if os.path.exists(source_path):
            shutil.copy2(source_path, os.path.join(dest_dir, filename))
            copied_count += 1
        else:
            print(f"Warning: '{filename}' not found in '{source_dir}'. Skipping.")
    
    print(f"\nGathering complete. Copied {copied_count} of {len(VIOLIN_FILES)} files to '{dest_dir}'.")
    
    if copied_count > 0:
        print(f"\nNow building the master file: '{master_filename}'")
        try:
            # Call the existing create master logic
            run_create_master(dest_dir, os.path.join(dest_dir, master_filename))
            print(f"\nSUCCESS: setup complete. Use '{os.path.join(dest_dir, master_filename)}' as a "
                  f"reference file, or select '{VIOLIN_REFERENCE_LABEL}' as the detective target.")
        except Exception as e:
            print(f"\nERROR building the tonewood reference master: {e}")
    else:
        print("\nNo files were copied, so no master chronology was built.")

# --- 4. MAIN DISPATCHER ---
def main():
    parser = argparse.ArgumentParser(description=f"Dendrochronology toolkit (v{__version__}).", formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
        WORKFLOW:
          1. python gogo.py index         (Downloads standard files and creates the index. Run once.)
          2. python gogo.py violin-setup  (Fetch curated tonewood forest chronologies + build their master.)
          3. python gogo.py build         (Build other master chronologies from the index)
          4. python gogo.py date ...      (Date a sample against a master)
        """))
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('index', help="Download standard-format NOAA files and create the data index.")
    subparsers.add_parser('violin-setup', help="Fetch curated tonewood forest chronologies and build their master.")
    subparsers.add_parser('reindex', help="Rebuild the index from the existing cache, without downloading.")
    
    # 'every' builds each category separately, one master file per category; they are
    # deliberately not merged, so an Alpine spruce result is never blended with a Baltic
    # pine or Alpine cembra one.
    build_targets = CATEGORY_NAMES + ['every']
    parser_build = subparsers.add_parser('build', help="Build master chronologies from the clean data index.")
    parser_build.add_argument('--target', choices=build_targets, default='every',
                              help=f"Which master to build: {', '.join(build_targets)}. (Default: every)")
    parser_build.add_argument('--min_end_year', type=int, default=DEFAULTS['min_end_year'],
                              help=f"Only include reference sites that end after this year. Default: {DEFAULTS['min_end_year']}")
    parser_build.add_argument('--min_depth', type=int, default=DEFAULTS['min_series_depth'],
                              help=f"Sites required at a year for it to appear in the master. Default: {DEFAULTS['min_series_depth']}")

    parser_create = subparsers.add_parser('create', help="Create a custom master from a local folder of .rwl files.")
    parser_create.add_argument('input_folder', help="Path to the folder containing your .rwl files.")
    parser_create.add_argument('output_filename', help="Name for the new master .csv file (e.g., 'my_master.csv').")
    parser_create.add_argument('--min_depth', type=int, default=3, help="Sites required at a year. Default: 3")

    # A master can be built without named series, so a series can be tested against a
    # reference that does not contain it. 'date' and 'detective' do this automatically.
    for p in (parser_build, parser_create):
        p.add_argument('--exclude_series', nargs='*', default=None, metavar='ID',
                       help="Series IDs to leave out. The biweight mean and the depth per year "
                            "are recomputed from the series that remain.")

    parser_date = subparsers.add_parser('date', help='Cross-date a sample against a master or another .rwl file.')
    parser_date.add_argument('sample_file', help="Path to your sample .rwl file.")
    parser_date.add_argument('master_file', help="Path to the reference .csv or .rwl file.")
    parser_date.add_argument('--min_overlap', type=int, default=DEFAULTS['min_overlap'],
                             help=f"Minimum overlap in years. (Default: {DEFAULTS['min_overlap']})")
    parser_date.add_argument('--stability', action='store_true',
                             help="Re-date under several detrending settings and report whether the end year holds.")

    parser_detective = subparsers.add_parser('detective', help="Run a sample against ALL individual files in a category or folder.")
    parser_detective.add_argument('sample_file', help="Path to your sample .rwl file.")
    parser_detective.add_argument('target', nargs='?', default='violin',
                                  help=f"Reference: a category ({', '.join(DETECTIVE_TARGETS)}) or a folder path. (Default: violin)")
    parser_detective.add_argument('--top_n', type=int, default=DEFAULTS['top_n'],
                                  help=f"Number of top results to display. (Default: {DEFAULTS['top_n']})")
    parser_detective.add_argument('--min_overlap', type=int, default=DEFAULTS['min_overlap'],
                                  help=f"Minimum overlap in years to consider a match. (Default: {DEFAULTS['min_overlap']})")
    parser_detective.add_argument('--min_end_year', type=int, default=DEFAULTS['min_end_year'],
                                  help=f"Only include reference sites that end after this year. Default: {DEFAULTS['min_end_year']}")
    parser_detective.add_argument('--min_country_series', type=int, default=DEFAULTS['min_country_series'],
                                  help=("Below this many measured series, a country prefix's best t is "
                                        f"flagged as too thin to mean anything. Default: {DEFAULTS['min_country_series']}"))

    parser_date.add_argument('--index', choices=list(INDEX_METHODS), default=None,
                             help=("Restrict the run to one index. Default: all three "
                                   f"({', '.join(INDEX_METHODS)}) are computed, so the index "
                                   "cannot be chosen after seeing which gives the better t."))
    parser_detective.add_argument('--search-index', dest='index', choices=list(INDEX_METHODS),
                                  default=None,
                                  help=("Restrict the sweep to one index, for speed. Default: the "
                                        "full search runs under all three and the report states "
                                        "whether the same reference wins under each. When "
                                        "restricted, the report says the comparison was not "
                                        "performed."))
    # A master must be built under the index it will be searched under: a log index cannot
    # be recovered from a finished spline index.
    for p in (parser_build, parser_create):
        p.add_argument('--index', choices=list(INDEX_METHODS), default=DEFAULTS['master_index'],
                       help=("Index for the master's 'value' column. The raw mean ring width per "
                             "year is stored too, so the other indices can be derived from it at "
                             f"comparison time. Default: {DEFAULTS['master_index']}"))

    for p in (parser_date, parser_detective):
        p.add_argument('--detrend_mode', choices=['fixed', 'percent'], default=DEFAULTS['detrend_mode'],
                       help=f"Spline cutoff: a fixed wavelength, or a %% of series length. (Default: {DEFAULTS['detrend_mode']})")
        p.add_argument('--detrend_wavelength', type=int, default=DEFAULTS['detrend_wavelength'],
                       help=f"Spline cutoff in years when --detrend_mode=fixed. (Default: {DEFAULTS['detrend_wavelength']})")
        p.add_argument('--stiffness', type=int, default=DEFAULTS['spline_stiffness_pct'],
                       help=f"Spline stiffness %% when --detrend_mode=percent. (Default: {DEFAULTS['spline_stiffness_pct']})")

    args = parser.parse_args()

    try:
        if args.command == 'index':
            download_and_index_files()
        elif args.command == 'reindex':
            index_cache()
        elif args.command == 'violin-setup':
            fetch_and_build_violin_master()
        elif args.command == 'build':
            wanted = CATEGORY_NAMES if args.target == 'every' else [args.target]
            for name in wanted:
                try:
                    build_master_from_index(name, min_end_year=args.min_end_year, min_depth=args.min_depth,
                                            exclude_series=args.exclude_series, index=args.index)
                except Exception as e:
                    print(f"  -> Could not build '{name}': {e}")
        elif args.command == 'create':
            run_create_master(args.input_folder, args.output_filename, min_depth=args.min_depth,
                              exclude_series=args.exclude_series, index=args.index)
        elif args.command == 'date':
            result = run_date_analysis(args.sample_file, args.master_file, args.min_overlap,
                                       spline_stiffness_pct=args.stiffness, detrend_mode=args.detrend_mode,
                                       detrend_wavelength=args.detrend_wavelength, index=args.index)
            if result:
                best = result.get('results', {}).get('best_match', {})
                print("\n" + format_classification(
                    _classify_dendro_match(best.get('t_value', 0.0), best.get('overlap_n', 0),
                                           best.get('glk', 0.0)),
                    index=result.get('index')))
                if result.get('reference_metadata_text'):
                    print("\n" + result['reference_metadata_text'])
                print("\n" + format_run_manifest(result.get('run_manifest', {})))
                if result.get('depth_at_match') is not None:
                    print(f"Reference depth at the reported end year: {result['depth_at_match']} site(s).")
                print("\n" + terminus_post_quem_note(best.get('end_year')))
                if args.stability:
                    table = stability_check(args.sample_file, args.master_file,
                                            min_overlap=args.min_overlap, index=args.index)
                    verdict = stability_verdict(table)
                    index_verdict = index_stability_verdict(table)
                    print("\n--- STABILITY UNDER DETRENDING SETTING AND INDEX ---")
                    print(table.to_string(index=False))
                    print(verdict['summary'])
                    print(f"end_year_stable_across_index: {index_stability_flag(index_verdict)}"
                          f" -- {index_verdict['summary']}")
                plot_results(result) # Pass the whole dict
        elif args.command == 'detective':
            result = run_detective_analysis(args.sample_file, args.target, args.top_n, args.min_overlap,
                                            min_end_year=args.min_end_year, spline_stiffness_pct=args.stiffness,
                                            detrend_mode=args.detrend_mode, detrend_wavelength=args.detrend_wavelength,
                                            min_country_series=args.min_country_series, index=args.index)
            if result:
                plot_results(result) # Pass the whole dict
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")

if __name__ == '__main__':
    main()
