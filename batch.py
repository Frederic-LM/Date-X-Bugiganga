#!/usr/bin/env python3
"""batch.py (Version 10.2) - batch-date and auto cross-match measured instrument series.

Reads measured series from a folder (Tucson ``*.rwl`` exports and/or Pennyscope
``*.dendro.json`` projects), then:

  1. dates each series against a reference master ("batch dating"), and
  2. cross-matches every series against every other ("auto cross-match"),

writing a summary table for each.

--------------------------------------------------------------------------------
READ THIS BEFORE INTERPRETING THE OUTPUT
--------------------------------------------------------------------------------
A strong cross-match means two pieces of wood share a common year-to-year growth
signal. That is evidence they *could* come from the same tree or the same region
and climate — it is NOT proof of identity or geographic origin.

Two well-known traps this tool deliberately refuses to fall into:

  * Reference-collection bias. If the reference set is (say) mostly Alpine, a
    violin will correlate "best" with something Alpine for statistical reasons
    that have nothing to do with its true origin — there are simply more chances
    to hit an Alpine match. So this tool never ranks or announces an "origin".

  * Circular provenancing. If expertise says "French", you compare against French
    references, and the dendro result then "confirms French", the confirmation is
    not independent — the attribution chose the comparison set. Keep the expert
    attribution (recorded in the Pennyscope metadata) separate from these numbers.

The tables below report *relationships and dates*, never identities or origins.
Use them as one input alongside independent expertise, not as a verdict.

Results from auto_crossmatch must never be folded back into a reference master.
Averaging series because they cross-matched builds a chronology out of the very
similarity you then go on to measure, and every later sample will match it.

Usage:
    python batch.py <folder> [--master master_tonewood_forest_references.csv]
                             [--min-overlap 60] [--detrend-mode fixed]
                             [--detrend-wavelength 32] [--stiffness 67]
                             [--index spline|bp|hollstein]
                             [--min-country-series 5]
                             [--xmatch-overlap 60] [--out-prefix batch]

Every series is dated under all three indices (spline, Baillie-Pilcher, Hollstein)
unless --index names one, and batch_dating.csv reports the end year under each plus
whether they agree. The log indices are computed on raw ring widths and lose rings at
the series ends, so each t carries its own overlap n and t-values from different
indices must not be compared with one another. A reference built under a different
index is refused rather than silently correlated.

Defaults come from gogo.DEFAULTS, so a batch run uses the same settings as the
CLI, the desktop app and Pennyscope. Series shorter than the overlap threshold are
listed as excluded rather than being quietly dated against a smaller overlap.
"""

import os, sys, glob, json, math, argparse
import pandas as pd

from gogo import (
    parse_as_floating_series, cross_date_indexed,
    index_series, index_edge_loss, resolve_index, describe_index,
    load_master_csv_series, index_agreement,
    shared_variation, classification_handle, convention_note, INDEX_METHODS,
    _classify_dendro_match,
    build_site_chronology, DEFAULTS, tonewood_master_path,
    run_manifest, write_manifest_sidecar, write_csv_with_manifest,
    describe_reference_set, format_reference_set, reference_depth_at,
    stability_check, stability_verdict, terminus_post_quem_note,
    country_coverage, format_country_coverage, COUNTRY_COVERAGE_COLUMNS,
)


# ── Loading measured series ───────────────────────────────────────────────────

def _series_from_dendro(js, piece_idx=0):
    """Reconstruct a floating ring-width series from a Pennyscope project piece."""
    pieces = js.get('pieces') or []
    if piece_idx >= len(pieces):
        return None
    pc = pieces[piece_idx]
    A, B = pc.get('axisA'), pc.get('axisB')
    rings = sorted(pc.get('rings', []), key=lambda r: r['t'])
    if not A or not B or len(rings) < 2:
        return None
    axlen = math.hypot(B['x'] - A['x'], B['y'] - A['y'])
    scale = js.get('scale')  # px per mm, or None -> widths stay in px (fine for dating)
    widths = []
    for k in range(1, len(rings)):
        px = abs(rings[k]['t'] - rings[k - 1]['t']) * axlen
        widths.append(px / scale if scale else px)
    if not widths:
        return None
    return pd.Series(widths, index=pd.RangeIndex(1, len(widths) + 1, name='ring_number'))


def load_folder(folder):
    """Return {name: (series, meta_dict)} for every measurable file in `folder`."""
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, '*'))):
        low = path.lower()
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            if low.endswith('.rwl'):
                s = parse_as_floating_series(path)
                if not s.empty:
                    out[name] = (s, {})
            elif low.endswith('.dendro.json') or low.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    js = json.load(f)
                meta = js.get('meta', {}) or {}
                if (js.get('frontType') == 'two'):
                    for pi, tag in ((0, 'bass'), (1, 'treble')):
                        s = _series_from_dendro(js, pi)
                        if s is not None and not s.empty:
                            out[f"{name}[{tag}]"] = (s, meta)
                else:
                    s = _series_from_dendro(js, 0)
                    if s is not None and not s.empty:
                        out[name] = (s, meta)
        except Exception as e:
            print(f"  ! skipped {os.path.basename(path)}: {e}")
    return out


# ── 1. Batch dating against a reference master ────────────────────────────────

def _load_master_under(master_path, index, stiffness, detrend_mode, detrend_wavelength):
    """The reference chronology under one index, or a stated reason it cannot be had."""
    try:
        if master_path.lower().endswith('.rwl'):
            master_det, _depth = build_site_chronology(
                master_path, mode=detrend_mode, wavelength=detrend_wavelength,
                spline_stiffness_pct=stiffness, index=index)
        else:
            _raw, master_det, _depth = load_master_csv_series(
                master_path, index=index, mode=detrend_mode, wavelength=detrend_wavelength,
                spline_stiffness_pct=stiffness)
        if master_det.empty:
            return None, f"reference produced no usable chronology under the {index} index"
        return master_det, None
    except Exception as e:
        return None, str(e)


def _date_series_under(s, master_det, index, min_overlap, stiffness, detrend_mode,
                       detrend_wavelength):
    """One series against one reference under one index: a run dict for index_runs_table."""
    lead_loss, trail_loss = index_edge_loss(index)
    run = {'index': index, 'end_year': None, 't_value': None, 'overlap_n': None,
           'glk': None, 'r2': None, 'criteria_met': None}
    det, _ = index_series(s, index=index, spline_stiffness_pct=stiffness,
                          mode=detrend_mode, wavelength=detrend_wavelength)
    if det.empty:
        run['error'] = (f'does not survive the {index} index, which needs at least 15 usable '
                        f'rings after losing {lead_loss} at the start and {trail_loss} at the end')
        return run, det
    if len(det) < min_overlap:
        run['error'] = (f'{len(det)} of {len(s)} rings survive the {index} index ({lead_loss} lost '
                        f'at the start, {trail_loss} at the end), below the {min_overlap}-year '
                        f'minimum overlap')
        return run, det
    res = cross_date_indexed(det, master_det, min_overlap=min_overlap, index=index)
    if 'error' in res:
        run['error'] = res['error']
        return run, det
    b = res['best_match']
    t, n = float(b['t_value']), int(b['overlap_n'])
    run.update({'end_year': int(b['end_year']), 't_value': round(t, 2), 'overlap_n': n,
                'glk': round(float(b.get('glk', 0)), 1), 'r2': shared_variation(t, n),
                'criteria_met': classification_handle(
                    _classify_dendro_match(t, n, b.get('glk', 0))),
                'stands_out_sd': round(float(b.get('t_zscore', 0)), 1),
                'second_best_t': round(float(b.get('second_best_t', 0)), 2)})
    return run, det


def batch_date(series_map, master_path, min_overlap, stiffness, detrend_mode=None,
               detrend_wavelength=None, check_stability=True, index=None):
    """Date every series against one reference, under every index unless one is named."""
    indices = [resolve_index(index)] if index else list(INDEX_METHODS)
    # The lead index fills the row's t_value / overlap_n columns; the 'index' column names it.
    lead = resolve_index(index) if index else DEFAULTS['lead_index']
    masters = {}
    for name in indices:
        masters[name] = _load_master_under(master_path, name, stiffness, detrend_mode,
                                           detrend_wavelength)
    if masters[lead][0] is None:
        raise ValueError(f"Reference master '{master_path}' could not be read under the {lead} "
                         f"index: {masters[lead][1]}")
    reference_info = describe_reference_set(master_path)

    rows = []
    for name, (s, meta) in series_map.items():
        # One overlap threshold for every series: a t over 20 years and a t over 60 do not
        # mean the same thing and must not sit in one column.
        if len(s) < min_overlap:
            rows.append({'series': name, 'rings': len(s), 'rings_after_index': None,
                         'result': f'excluded: {len(s)} rings is below the {min_overlap}-year minimum overlap',
                         'attributed_origin': meta.get('attributedOrigin', '')})
            continue
        runs = []
        for name_index in indices:
            master_det, master_error = masters[name_index]
            if master_det is None:
                runs.append({'index': name_index, 'end_year': None, 't_value': None,
                             'overlap_n': None, 'glk': None, 'r2': None, 'criteria_met': None,
                             'error': master_error})
                continue
            run, _det = _date_series_under(s, master_det, name_index, min_overlap, stiffness,
                                           detrend_mode, detrend_wavelength)
            runs.append(run)
        agreement = index_agreement(runs)
        ok = [r for r in runs if r['end_year'] is not None]
        if not ok:
            rows.append({'series': name, 'rings': len(s), 'rings_after_index': None,
                         'result': 'excluded: ' + '; '.join(
                             f"{r['index']}: {r.get('error', 'no result')}" for r in runs),
                         'attributed_origin': meta.get('attributedOrigin', '')})
            continue
        primary = next((r for r in ok if r['index'] == lead), ok[0])
        # The filter dimension only: the index dimension is already covered by the runs above.
        end_year_stable, stability_note = '', ''
        if check_stability:
            try:
                table = stability_check(s, master_path, min_overlap=min_overlap,
                                        index=primary['index'], index_list=[primary['index']])
                verdict = stability_verdict(table)
                end_year_stable = 'yes' if verdict['stable'] else 'no'
                stability_note = verdict['summary']
            except Exception as e:
                end_year_stable, stability_note = 'error', str(e)
        rows.append({
            'series': name, 'rings': len(s), 'rings_after_index': primary['overlap_n'],
            'index': primary['index'],
            'end_year': primary['end_year'],
            't_value': primary['t_value'],
            'glk': primary['glk'],
            'overlap_n': primary['overlap_n'],
            'r2': None if primary['r2'] is None else round(primary['r2'], 2),
            'criteria_met': primary['criteria_met'],
            'stands_out_sd': primary.get('stands_out_sd'),
            'second_best_t': primary.get('second_best_t'),
            'ref_depth_at_year': reference_depth_at(reference_info, primary['end_year']),
            'end_year_by_index': '; '.join(
                f"{r['index']} {r['end_year']} t={r['t_value']} n={r['overlap_n']}"
                if r['end_year'] is not None else f"{r['index']} none"
                for r in runs),
            'end_year_stable': end_year_stable,
            'end_year_stable_across_index': ('not tested' if agreement['agree'] is None
                                             else ('yes' if agreement['agree'] else 'no')),
            'stability_note': stability_note,
            'index_note': agreement['headline'],
            # Expert opinion, shown alongside but deliberately NOT fed into the dating:
            'attributed_origin': meta.get('attributedOrigin', ''),
        })
    table = pd.DataFrame(rows)
    table.attrs['master_errors'] = {name: error for name, (chrono, error) in masters.items()
                                    if chrono is None}
    table.attrs['indices'] = indices
    return table


# ── 2. Auto cross-match: every series against every other ─────────────────────

def auto_crossmatch(series_map, min_overlap, stiffness, detrend_mode=None, detrend_wavelength=None,
                    index=None):
    """Cross-match every series against every other.

    This is a best-of-many search: n(n-1)/2 pairs, each tested at every offset. The
    highest t-value in such a search is selected precisely because it is the highest, so
    it needs the same standing-out control that batch_date reports. t_zscore (how far the
    winning offset sits above the population of offsets tested for that pair) and
    second_best_t (the runner-up it beat) are carried through from cross_date, and the
    number of pairs and alignments tested is reported alongside."""
    index = resolve_index(index)
    _lead_loss, trail_loss = index_edge_loss(index)
    names = list(series_map.keys())
    det = {n: index_series(series_map[n][0], index=index, spline_stiffness_pct=stiffness,
                           mode=detrend_mode, wavelength=detrend_wavelength)[0] for n in names}
    excluded = {n: len(series_map[n][0]) for n in names if det[n].empty}
    edges = []
    n_pairs_tested = 0
    n_alignments_total = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            # One threshold for every pair, for the same reason as in batch_date. Lengths
            # here are surviving rings, not measured rings.
            if min(len(det[a]), len(det[b])) < min_overlap:
                continue
            res = cross_date_indexed(det[a], det[b], min_overlap=min_overlap, index=index)
            if 'error' in res:
                continue
            n_pairs_tested += 1
            n_offsets = len(res.get('all_correlations', []))
            n_alignments_total += n_offsets
            m = res['best_match']
            edges.append({
                'series_a': a, 'series_b': b,
                't_value': round(float(m['t_value']), 2),
                'glk': round(float(m.get('glk', 0)), 1),
                'overlap_n': int(m['overlap_n']),
                'r2': round(shared_variation(m['t_value'], int(m['overlap_n'])) or 0.0, 2),
                # How far a's last measured ring sits past b's: b's trailing loss is added
                # back, since det[b]'s last label is its last surviving ring.
                'offset_years': int(m['end_year'] - (det[b].index.max() + trail_loss)),
                'stands_out_sd': round(float(m.get('t_zscore', 0)), 1),
                'second_best_t': round(float(m.get('second_best_t', 0)), 2),
                'alignments_tested': int(n_offsets),
                # A relationship label — explicitly not an origin or identity claim.
                'relationship': _relationship_label(m['t_value'], int(m['overlap_n']), m.get('glk', 0)),
            })
    df = pd.DataFrame(edges)
    df.attrs['index'] = index
    df.attrs['excluded_by_index'] = excluded
    if df.empty:
        df.attrs['n_pairs_tested'] = 0
        df.attrs['n_alignments_total'] = 0
        return df
    # Ranked by how far the winner stands out as well as by raw t: a high t drawn from a
    # search that produced many similar values is a weaker finding than the same t that
    # towered over its alternatives.
    df = df.sort_values(['stands_out_sd', 't_value'], ascending=False).reset_index(drop=True)
    df.attrs['n_pairs_tested'] = n_pairs_tested
    df.attrs['n_alignments_total'] = n_alignments_total
    df.attrs['index'] = index
    df.attrs['excluded_by_index'] = excluded
    return df


def _relationship_label(t, overlap, glk):
    """Name the criteria met — never a quality adjective, an origin or an identity."""
    classification = _classify_dendro_match(t, overlap, glk)
    if not classification['meets_any']:
        return f"no tier met ({classification['criteria']} not met)"
    handle = classification['handle']
    if classification['tier_index'] == 0:
        return f"tier {handle} — consistent with a shared growth signal (corroborate independently)"
    return f"tier {handle} — possible shared growth signal (corroborate independently)"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Batch dating + auto cross-match for measured violin series.")
    ap.add_argument('folder', help="Folder of .rwl and/or .dendro.json measurements.")
    ap.add_argument('--master', default=tonewood_master_path(),
                    help="Reference master (.csv or .rwl) for batch dating.")
    ap.add_argument('--min-overlap', type=int, default=DEFAULTS['min_overlap'],
                    help=f"Minimum overlap in years. Default: {DEFAULTS['min_overlap']}")
    ap.add_argument('--stiffness', type=int, default=DEFAULTS['spline_stiffness_pct'],
                    help="Spline stiffness %% when --detrend-mode=percent.")
    ap.add_argument('--detrend-mode', choices=['fixed', 'percent'], default=DEFAULTS['detrend_mode'],
                    help=f"Spline cutoff: fixed wavelength or %% of length. Default: {DEFAULTS['detrend_mode']}")
    ap.add_argument('--detrend-wavelength', type=int, default=DEFAULTS['detrend_wavelength'],
                    help=f"Spline cutoff in years when --detrend-mode=fixed. Default: {DEFAULTS['detrend_wavelength']}")
    ap.add_argument('--xmatch-overlap', type=int, default=DEFAULTS['min_overlap_internal'],
                    help=f"Min overlap for cross-matching. Default: {DEFAULTS['min_overlap_internal']}")
    ap.add_argument('--index', choices=list(INDEX_METHODS), default=None,
                    help=("Restrict the run to one index. Default: all three "
                          f"({', '.join(INDEX_METHODS)}) are computed per series."))
    ap.add_argument('--min-country-series', type=int, default=DEFAULTS['min_country_series'],
                    help=("Below this many measured series, a country prefix's best t is flagged as "
                          f"too thin to mean anything. Default: {DEFAULTS['min_country_series']}"))
    ap.add_argument('--out-prefix', default='batch', help="Prefix for the two output CSVs.")
    ap.add_argument('--no-stability', action='store_true',
                    help="Skip the detrend-setting stability check (faster, less informative).")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print(f"No such folder: {args.folder}"); sys.exit(1)

    print(f"Loading measurements from '{args.folder}' ...")
    series_map = load_folder(args.folder)
    if not series_map:
        print("No usable .rwl / .dendro.json series found."); sys.exit(1)
    indices_used = [resolve_index(args.index)] if args.index else list(INDEX_METHODS)
    lead_index = resolve_index(args.index) if args.index else DEFAULTS['lead_index']
    print(f"Loaded {len(series_map)} series: {', '.join(series_map.keys())}")
    for name in indices_used:
        print(describe_index(name))
    print()

    # Conditions of this run, recorded once and attached to every artefact below.
    manifest = run_manifest(
        master_file=args.master if os.path.exists(args.master) else None,
        min_overlap=args.min_overlap, detrend_mode=args.detrend_mode, index=lead_index,
        indices_reported='+'.join(indices_used), index_restricted_to=args.index,
        detrend_wavelength=args.detrend_wavelength, spline_stiffness_pct=args.stiffness,
        n_series=len(series_map), input_folder=os.path.abspath(args.folder),
        xmatch_overlap=args.xmatch_overlap,
        min_country_series=args.min_country_series,
        stability_checked=not args.no_stability)

    # 1. Batch dating
    if os.path.exists(args.master):
        print(f"--- Batch dating against '{args.master}' ---")
        reference_info = describe_reference_set(args.master)
        print(format_reference_set(reference_info))
        manifest['reference_n_sites'] = reference_info.get('n_sites')
        manifest['reference_year_span'] = (
            f"{reference_info.get('year_min')}-{reference_info.get('year_max')}"
            if reference_info.get('year_min') is not None else None)
        print()
        dated = batch_date(series_map, args.master, args.min_overlap, args.stiffness,
                           args.detrend_mode, args.detrend_wavelength,
                           check_stability=not args.no_stability, index=args.index)
        print(dated.drop(columns=['stability_note', 'index_note'],
                         errors='ignore').to_string(index=False))
        out1 = f"{args.out_prefix}_dating.csv"
        write_csv_with_manifest(dated, out1, manifest)
        print(f"-> saved {out1}")
        print()
        for name, error in (dated.attrs.get('master_errors') or {}).items():
            print(f"{name}: not computed for any series -- {error}")
        # Disagreement is the finding worth reading, so only that is repeated per series.
        for row in dated.to_dict('records'):
            note = row.get('index_note')
            if row.get('end_year_stable_across_index') == 'no' and isinstance(note, str):
                print(f"  {row['series']}: {note}")
        print("Tier criteria are conventions from the literature. " + convention_note(lead_index))
        print("Each t above is stated with the n it was computed over. The indices are not on a "
              "common scale; t-values from different indices must not be compared with each other.")

        # What was searched, per country prefix of the reference set, with the same columns
        # the detective export uses. A merged master .csv carries no per-country breakdown,
        # so that is stated rather than guessed at.
        coverage_rows = [{'source_file': os.path.basename(args.master),
                          't_value': r.get('t_value'), 'overlap_n': r.get('overlap_n')}
                         for r in dated.to_dict('records') if r.get('t_value') is not None]
        coverage = country_coverage(reference_info, coverage_rows,
                                    min_country_series=args.min_country_series)
        print()
        if coverage.empty:
            print("WHAT WAS SEARCHED, PER ITRDB COUNTRY PREFIX\n"
                  f"  '{os.path.basename(args.master)}' is a merged master chronology: it carries no "
                  "per-country breakdown, so per-country coverage could not be reported for this run.")
        else:
            print(format_country_coverage(coverage, min_overlap=args.min_overlap))
            out_cov = f"{args.out_prefix}_country_coverage.csv"
            write_csv_with_manifest(coverage[list(COUNTRY_COVERAGE_COLUMNS)], out_cov, manifest)
            print(f"-> saved {out_cov}")
        print()
        print(terminus_post_quem_note())
        print()
    else:
        print(f"(Skipping batch dating — master not found: {args.master})\n")

    # 2. Auto cross-match
    print("--- Auto cross-match (every series vs every other) ---")
    xm = auto_crossmatch(series_map, args.xmatch_overlap, args.stiffness,
                         args.detrend_mode, args.detrend_wavelength, index=lead_index)
    for name, n_rings in (xm.attrs.get('excluded_by_index') or {}).items():
        print(f"  excluded from cross-match: {name} ({n_rings} rings does not survive the "
              f"{lead_index} index)")
    if xm.empty:
        print("No pairs shared enough overlap to compare.")
    else:
        # The size of the search is part of the result: the best pair was chosen from
        # this many, so the reader needs the denominator to judge the numerator.
        pairs = xm.attrs.get('n_pairs_tested', len(xm))
        aligns = xm.attrs.get('n_alignments_total', 0)
        print(f"{pairs} pairs compared, {aligns} alignments evaluated in total "
              f"({aligns // pairs if pairs else 0} per pair on average).")
        print(xm.to_string(index=False))
        out2 = f"{args.out_prefix}_crossmatch.csv"
        xm_manifest = dict(manifest, n_pairs_tested=pairs, n_alignments_total=aligns,
                           min_overlap=args.xmatch_overlap)
        write_csv_with_manifest(xm, out2, xm_manifest)
        print(f"-> saved {out2}")

    sidecar = write_manifest_sidecar(manifest, args.out_prefix)
    print(f"-> saved {sidecar}")
    print("\nReminder: cross-matches show a shared growth signal, not origin or identity."
          " Keep expert attribution independent of these numbers.")


if __name__ == '__main__':
    main()
