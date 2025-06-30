# gogo.py (Version 9.8 - Integrated Violin Setup)
# ==============================================================================
import os, ftplib, argparse, textwrap, multiprocessing, shutil, re
import warnings
from typing import Tuple, Dict
import pandas as pd, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings("ignore", message="The maximal number of iterations")

# This curated list is based on dendrochronological literature for violin wood (Cybis wiki).
VIOLIN_FILES = [
    "fran7.rwl","fran039.rwl","swit204.rwl","swit203.rwl","swit189.rwl","swit193.rwl",
    "swit177.rwl","swit169.rwl","swit215.rwl","swit184.rwl","swit173.rwl","swit181.rwl",
    "aust003.rwl","aust007.rwl","germ12.rwl","germ11.rwl","ital007.rwl","ital006.rwl",
    "ital025.rwl","germ036.rwl","germ4.rwl","germ5.rwl","germ14.rwl","germ040.rwl",
    "germ033.rwl","germ020.rwl","czec001.rwl","czec002.rwl","czec3.rwl","czec.rwl",
    "pola022.rwl","pola019.rwl","pola020.rwl","roma002.rwl","roma005.rwl","yugo001.rwl",
    "slov001.rwl","ital022.rwl","swed311.rwl","finl012.rwl","swed312.rwl","swed011.rwl"
]

def _classify_dendro_match(t_value, overlap_years, gleich_percent):
    """Classify match strength based on multi-dimensional thresholds. DUPLICATED LOGIC for backend independence."""
    if t_value >= 7.0 and overlap_years >= 80 and gleich_percent >= 70:
        return "Very Strong Match"
    elif t_value >= 6.0 and overlap_years >= 70 and gleich_percent >= 65:
        return "Strong Match"
    elif t_value >= 5.0 and overlap_years >= 50 and gleich_percent >= 60:
        return "Significant Match"
    else:
        return "Tentative/Insufficient Match"

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

    # --- The rest of the report ---
    best_match = res.get('results', {}).get('best_match', {})
    if best_match:
        t_value, overlap, glk = best_match.get('t_value', 0.0), best_match.get('overlap_n', 0), best_match.get('glk', 0.0)
        classification = _classify_dendro_match(t_value, overlap, glk)
        end_year = int(best_match.get('end_year', 0))

        if classification != "Tentative/Insufficient Match":
            dating_result = f"A felling date after {end_year} is proposed. This is supported by a '{classification}' classification based on the multi-dimensional criteria."
        else:
            dating_result = f"While a potential end year of {end_year} is suggested, it should be classified as 'Tentative/Insufficient' in absence of further evidence."
        report_lines.append(textwrap.fill(dating_result, width=width))
    else:
        report_lines.append("Dating analysis did not produce a statistically significant result.")
    
    if res.get('analysis_type') == 'detective':
        df = res.get('enriched_results_df')
        if df is not None and not df.empty:
            top_match = df.iloc[0].to_dict()
            t, o, g = top_match.get('t_value',0), top_match.get('overlap_n',0), top_match.get('glk',0)
            top_classification = _classify_dendro_match(t, o, g)
            top_location = top_match.get('location', 'N/A').strip()
            
            if top_location != 'N/A' and top_location and top_classification != "Tentative/Insufficient Match":
                geo_conclusion = f"\nThe strongest match is a '{top_classification}' with a chronology from {top_location}, suggesting a probable geographic origin."
                report_lines.append(textwrap.fill(geo_conclusion, width=width))

    return "\n".join(report_lines)

def _parse_rwl_header(file_path: str) -> dict:
    """Parses the first 3 lines of an RWL file to extract metadata."""
    header_info = {'site_name': 'N/A', 'location': 'N/A', 'pi': 'N/A', 'collector': 'N/A'}
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(3)]
        if len(lines) >= 3:
            header_info['site_name'] = lines[0][12:55].strip()
            header_info['location'] = lines[1][12:55].strip()
            pi_collector = lines[2][12:55].strip().split()
            # This is a heuristic guess as the format varies
            if len(pi_collector) > 1:
                header_info['pi'] = pi_collector[0]
                header_info['collector'] = ' '.join(pi_collector[1:])
            elif len(pi_collector) == 1:
                header_info['collector'] = pi_collector[0]
    except Exception:
        pass # Fail silently, return defaults
    return header_info

def parse_as_floating_series(file_path: str) -> pd.Series:
    all_widths = [];
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                try: int(parts[1]); value_parts = parts[2:]
                except (ValueError, IndexError): continue
                for val_str in value_parts:
                    try:
                        width = int(val_str)
                        if width in [-9999, 999]: continue
                        all_widths.append(width / 100.0)
                    except ValueError: continue
    except Exception as e: raise IOError(f"Error parsing {file_path}: {e}")
    if not all_widths: return pd.Series(dtype=np.float64)
    return pd.Series(all_widths, index=pd.RangeIndex(start=1, stop=len(all_widths) + 1, name='ring_number'))

def _build_master_from_rwl_file(file_path: str) -> pd.Series:
    all_rings = [];
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                try:
                    year_val = int(parts[1])
                    if not (100 < year_val < 3000): continue
                    start_year, value_parts = (year_val // 10) * 10, parts[2:]
                    for i, val_str in enumerate(value_parts):
                        try:
                            width = int(val_str)
                            if width in [-9999, 999]: continue
                            current_year = year_val if i == 0 else start_year + i
                            all_rings.append({'year': current_year, 'width': width / 100.0})
                        except ValueError: continue
                except (ValueError, IndexError): continue
    except Exception as e: print(f"Warning: Could not parse dated file {file_path}: {e}"); return pd.Series(dtype=np.float64)
    if not all_rings: return pd.Series(dtype=np.float64)
    df = pd.DataFrame(all_rings).drop_duplicates(subset='year').set_index('year')
    return df['width'].dropna().sort_index()

def detrend(series: pd.Series, spline_stiffness_pct: int = 67) -> Tuple[pd.Series, pd.Series]:
    series = series.dropna()
    if len(series) < 15: return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    x, y = series.index.values, series.values
    s = len(series) * (spline_stiffness_pct / 100)**3
    spline = UnivariateSpline(x, y, s=s)
    spline_fit = pd.Series(spline(x), index=x)
    detrended_series = series / (spline_fit + 1e-6)
    return detrended_series, spline_fit

def calculate_t_value(r: float, n: int) -> float:
    if n < 3 or abs(r) >= 1.0: return np.inf * np.sign(r) if r != 0 else 0
    return r * np.sqrt((n - 2) / (1 - r**2))

def calculate_glk(series1: pd.Series, series2: pd.Series) -> float:
    diff1 = series1.diff().dropna(); diff2 = series2.diff().dropna()
    common_index = diff1.index.intersection(diff2.index)
    if len(common_index) < 2: return 0.0
    agreements = np.sum(np.sign(diff1.loc[common_index]) == np.sign(diff2.loc[common_index]))
    return (agreements / len(common_index)) * 100

def cross_date(sample_series: pd.Series, master_series: pd.Series, min_overlap: int = 50) -> dict:
    if sample_series.empty or master_series.empty or len(sample_series) < min_overlap:
        return {"error": "Input series is empty or shorter than minimum overlap."}
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
    if not corrs: return {"error": f"No suitable overlap found (min_overlap = {min_overlap} years)."}
    rdf = pd.DataFrame(corrs)
    if rdf['t_value'].isnull().all(): return {"error": "Correlation calculation failed for all overlaps."}
    return {"best_match": rdf.loc[rdf['t_value'].idxmax()].to_dict(), "all_correlations": rdf.set_index('end_year')}

def plot_results(analysis_dict: Dict):
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

    if "error" in results:
        print(f"Cannot plot: {results['error']}")
        return
        
    best_match = results['best_match']
    all_correlations = results['all_correlations']
    best_end_year = int(best_match['end_year'])
    r_val, t_val, n_val, glk_val = best_match['correlation'], best_match['t_value'], int(best_match['overlap_n']), best_match.get('glk', 0.0)
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
    ax1.axhline(5.0, color='orange', linestyle='--', linewidth=1, label='t=5.0 (Significant)'); ax1.axhline(7.0, color='firebrick', linestyle='--', linewidth=1, label='t=7.0 (Very Strong)')
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
    summary_text_body = textwrap.dedent(f"""
    Sample: {sample_label}
    Reference: {master_label}

    Most Likely End Year: {best_end_year}
    (Start Year: {best_start_year})

    Best Match Statistics:
      T-Value: {t_val:.2f}
      Correlation (r): {r_val:.2f}
      GLK (%): {glk_val:.1f}
      Overlap (n): {n_val}
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
    plt.show()
    
def get_metadata_from_rwl(file_path):
    series = _build_master_from_rwl_file(file_path)
    if series.empty: return None
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: header_content = "".join([f.readline() for _ in range(5)]).lower()
    species_map = {"picea": "PICEA", "spruce": "PICEA", "pinus": "PINUS", "pine": "PINUS", "abies": "ABIES", "fir": "ABIES", "larix": "LARIX", "larch": "LARIX"}
    for key, val in species_map.items():
        if key in header_content: return {"species": val, "start_year": int(series.index.min()), "end_year": int(series.index.max()), "length": len(series)}
    return {"species": "UNKNOWN", "start_year": int(series.index.min()), "end_year": int(series.index.max()), "length": len(series)}

def run_date_analysis(sample_file, master_file, min_overlap=50, is_internal_test=False, reverse_sample=False, spline_stiffness_pct=67):
    if not is_internal_test: print(f"\n--- Running Analysis: {os.path.basename(sample_file)} vs {os.path.basename(master_file)} ---")
    sample_chrono = parse_as_floating_series(sample_file)
    if sample_chrono.empty: raise ValueError(f"Could not read data from sample file: {sample_file}")
    if reverse_sample:
        print(f"-> Reversing data for sample: {os.path.basename(sample_file)}")
        sample_chrono = sample_chrono.iloc[::-1].reset_index(drop=True)
        sample_chrono.index = pd.RangeIndex(start=1, stop=len(sample_chrono) + 1, name='ring_number')
    reference_is_rwl = master_file.lower().endswith('.rwl')
    if is_internal_test: master_chrono = parse_as_floating_series(master_file)
    elif reference_is_rwl: master_chrono = _build_master_from_rwl_file(master_file)
    else: master_chrono = pd.read_csv(master_file, index_col='year').squeeze("columns")
    if master_chrono.empty: raise ValueError(f"Could not read data from reference file: {master_file}")
    if min_overlap > len(sample_chrono): raise ValueError(f"CONFIG ERROR: min_overlap ({min_overlap}) > sample length ({len(sample_chrono)}).")
    
    sample_detrended, sample_spline_fit = detrend(sample_chrono, spline_stiffness_pct=spline_stiffness_pct)
    master_detrended, _ = detrend(master_chrono, spline_stiffness_pct=spline_stiffness_pct)
    
    analysis_results = cross_date(sample_detrended, master_detrended, min_overlap=min_overlap)
    if "error" in analysis_results: raise ValueError(analysis_results['error'])
    if not is_internal_test: print(f"\n--- Cross-Dating Complete ---\nMost Likely End Year: {int(analysis_results['best_match']['end_year'])}")
    
    # Pass back the reverse flag for the smart report
    return {'raw_sample': sample_chrono, 'master_detrended': master_detrended, 'detrended_sample': sample_detrended, 
            'results': analysis_results, 'sample_filename': sample_file, 'master_filename': master_file, 
            'reference_is_rwl': reference_is_rwl, 'raw_master': master_chrono if not is_internal_test else None,
            'sample_spline_fit': sample_spline_fit, 'reverse_sample': reverse_sample, 'analysis_mode': 'single'}

def process_single_file(args):
    filename, sample_detrended, sample_basename, base_path, min_overlap, spline_stiffness_pct = args
    if filename == sample_basename: return None
    master_path = os.path.join(base_path, filename)
    master_raw = _build_master_from_rwl_file(master_path)
    if master_raw.empty or len(master_raw) < min_overlap: return None
    master_detrended, _ = detrend(master_raw, spline_stiffness_pct=spline_stiffness_pct)
    analysis_results = cross_date(sample_detrended, master_detrended, min_overlap=min_overlap)
    if "error" in analysis_results: return None
    best_match = analysis_results['best_match']
    if best_match['correlation'] > 0.99: return None
    best_match['source_file'] = filename
    return best_match

def run_detective_analysis(sample_file, target, top_n, min_overlap=80, min_end_year=1500, reverse_sample=False, spline_stiffness_pct=67):
    print(f"\n--- Running Detective Analysis on {os.path.basename(sample_file)} ---")
    sample_chrono = parse_as_floating_series(sample_file)
    if sample_chrono.empty: raise ValueError("Could not process sample file.")
    if reverse_sample:
        print(f"-> Reversing data for sample: {os.path.basename(sample_file)}")
        sample_chrono = sample_chrono.iloc[::-1].reset_index(drop=True)
        sample_chrono.index = pd.RangeIndex(start=1, stop=len(sample_chrono) + 1, name='ring_number')
    if min_overlap > len(sample_chrono): raise ValueError(f"CONFIG ERROR: min_overlap ({min_overlap}) > sample length ({len(sample_chrono)}).")
    sample_detrended, _ = detrend(sample_chrono, spline_stiffness_pct=spline_stiffness_pct)
    file_list, base_path_for_masters = [], ""
    
    if target == 'violin':
        base_path_for_masters = "violin_references"
        if not os.path.exists(base_path_for_masters) or not os.listdir(base_path_for_masters):
             raise ValueError(f"The 'violin' category requires reference files in '{base_path_for_masters}'.\nPlease run the 'Fetch & Build Violin Reference' tool in the Setup tab first.")
        file_list = [f for f in os.listdir(base_path_for_masters) if f.lower().endswith('.rwl')]
    elif os.path.isdir(target):
        base_path_for_masters = target
        file_list = [f for f in os.listdir(target) if f.lower().endswith('.rwl')]
    else:
        base_path_for_masters = "full_rwl_cache"
        index_filename = "noaa_europe_index.csv"
        if not os.path.exists(index_filename): raise ValueError("Index file missing. Run 'gogo index' first.")
        category_params = {'alpine': {'species': ['PICEA', 'ABIES'], 'countries': ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 'min_len': 150, 'min_start': 1750}, 'baltic': {'species': ['PINUS', 'PICEA'], 'countries': ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 'min_len': 150, 'min_start': 1750}, 'all': {'species': ['PICEA', 'ABIES', 'PINUS', 'LARIX'], 'countries': ['aust', 'fran', 'germ', 'ital', 'swit', 'slov', 'finl', 'lith', 'norw', 'pola', 'swed'], 'min_len': 100, 'min_start': 1800}}
        if target not in category_params: raise ValueError(f"Invalid category '{target}'.")
        params = category_params[target]
        index_df = pd.read_csv(index_filename)
        df_filtered = index_df[(index_df['species'].isin(params['species'])) & (index_df['filename'].str.lower().str.startswith(tuple(params['countries']))) & (index_df['length'] >= params['min_len']) & (index_df['start_year'] < params['min_start']) & (index_df['end_year'] >= min_end_year)]
        file_list = df_filtered['filename'].tolist()

    if not file_list: raise ValueError(f"No reference files found for target '{target}' (including min_end_year={min_end_year}).")
    tasks = [(filename, sample_detrended, os.path.basename(sample_file), base_path_for_masters, min_overlap, spline_stiffness_pct) for filename in file_list]
    print(f"Testing against {len(file_list)} sites using {multiprocessing.cpu_count()} CPU cores...")
    with multiprocessing.Pool() as pool:
        all_best_results = [res for res in tqdm(pool.imap(process_single_file, tasks), total=len(tasks)) if res is not None]
    if not all_best_results: print("\nAnalysis complete, no significant correlations found."); return None
    results_df = pd.DataFrame(all_best_results).sort_values(by='t_value', ascending=False)
    top_results = results_df.head(top_n)
    
    header_infos = [_parse_rwl_header(os.path.join(base_path_for_masters, row.source_file)) for _, row in top_results.iterrows()]
    header_df = pd.DataFrame(header_infos, index=top_results.index)
    enriched_results_df = pd.concat([top_results, header_df], axis=1)

    print(f"\n--- Top {top_n} Matching Sites (Sorted by T-Value) ---")
    print(top_results[['end_year', 't_value', 'glk', 'correlation', 'overlap_n', 'source_file']].to_string(index=False))
    
    top_match_file_name = top_results.iloc[0]['source_file']
    top_match_full_path = os.path.join(base_path_for_masters, top_match_file_name)
    print(f"\nGenerating plot for top match: {top_match_file_name}")
    plot_data_dict = run_date_analysis(sample_file, top_match_full_path, min_overlap=min_overlap, reverse_sample=reverse_sample, spline_stiffness_pct=spline_stiffness_pct)
    if not plot_data_dict:
        print("Could not generate plot for the top match.")
        return {"analysis_mode": "single", "analysis_type": "detective", "sample_file": sample_file, "target": target, "min_overlap": min_overlap, "min_end_year": min_end_year, "enriched_results_df": enriched_results_df}
    
    # Update dictionary with all detective mode context
    plot_data_dict.update({"analysis_type": "detective", "enriched_results_df": enriched_results_df, "min_end_year": min_end_year, "target": target, "min_overlap": min_overlap, "top_n": top_n})
    return plot_data_dict

def run_two_piece_mean_analysis(bass_file, treble_file, final_analysis_func, final_analysis_args, reverse_bass=False, reverse_treble=False, spline_stiffness_pct=67):
    print("\n--- Starting Two-Piece Mean Analysis ---")
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
    print("--- Internal Cross-Match (Bass vs. Treble) ---")
    bass_detrended, _ = detrend(bass_chrono, spline_stiffness_pct=spline_stiffness_pct)
    treble_detrended, _ = detrend(treble_chrono, spline_stiffness_pct=spline_stiffness_pct)
    internal_results = cross_date(bass_detrended, treble_detrended, min_overlap=40)
    if "error" in internal_results: raise ValueError(f"Internal cross-dating failed: {internal_results['error']}")
    internal_best = internal_results['best_match']
    internal_t, internal_glk = internal_best['t_value'], internal_best.get('glk', 0.0)
    print(f"Internal Match Stats: t-value = {internal_t:.2f}, Glk = {internal_glk:.1f}%")
    if internal_t < 6.0: raise ValueError("Weak internal match (t < 6.0).")
    print("\nStrong internal match found. Creating mean chronology.")
    end_pos_of_bass_relative_to_treble = internal_best['end_year']
    offset = int(end_pos_of_bass_relative_to_treble - bass_chrono.index.max())
    bass_chrono.index += offset
    mean_chrono_series = pd.concat([bass_chrono, treble_chrono], axis=1).mean(axis=1).dropna()
    temp_mean_file = "_temp_mean_chrono.rwl"
    with open(temp_mean_file, 'w') as f:
        for i in range(0, len(mean_chrono_series), 10):
            decade_vals, start_marker = mean_chrono_series.iloc[i:i+10], int(mean_chrono_series.iloc[i:i+10].index[0])
            vals_str = " ".join([f"{int(v*100): >5}" for v in decade_vals])
            f.write(f"MEAN      {start_marker} {vals_str}\n")
    print(f"Temporary mean chronology created.\n--- Running Final Analysis on Mean Chronology ---")
    final_analysis_args[0] = temp_mean_file
    final_results = final_analysis_func(*final_analysis_args)
    if os.path.exists(temp_mean_file): os.remove(temp_mean_file)
    if not final_results: return None
    # Add all necessary info for the smart report
    final_results.update({'analysis_mode': 'two_piece', 'internal_stats': {'t_value': internal_t, 'glk': internal_glk}, 
                           'bass_file': bass_file, 'treble_file': treble_file, 'sample_filename': "Mean Chronology (Bass+Treble)",
                           'mean_series_length': len(mean_chrono_series), 'reverse_bass': reverse_bass, 'reverse_treble': reverse_treble})
    return final_results

# --- COMMAND LOGIC ---

def download_and_index_files(index_filename="noaa_europe_index.csv"):
    print("--- Stage 1: Downloading standard .rwl files and creating index ---")
    cache_dir = "full_rwl_cache"
    os.makedirs(cache_dir, exist_ok=True)
    standard_file_pattern = re.compile(r"^[a-zA-Z]+[0-9]+\.rwl$")
    try:
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov", timeout=60); ftp.login(); ftp.cwd("/pub/data/paleo/treering/measurements/europe/")
        all_server_files = ftp.nlst()
    except Exception as e: raise ConnectionError(f"FTP Error: {e}")
    files_to_download = []; skipped_files = []
    for f in all_server_files:
        if standard_file_pattern.match(f): files_to_download.append(f)
        elif f.lower().endswith('.rwl'): skipped_files.append(f)
    print(f"Found {len(all_server_files)} total files on server.")
    print(f"-> {len(files_to_download)} files match standard format and will be downloaded.")
    if skipped_files: print(f"-> {len(skipped_files)} non-standard .rwl files will be skipped (e.g., 'e', 'l', etc.).")
    for filename in tqdm(files_to_download, desc="Downloading Standard Files"):
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            try:
                with open(local_path, 'wb') as f: ftp.retrbinary(f"RETR {filename}", f.write)
            except Exception as e: print(f"Warning: Failed to download {filename}: {e}"); continue
    ftp.quit()
    print("Download complete.")
    print("\n--- Indexing downloaded files ---")
    all_metadata = []
    local_files_to_index = [f for f in os.listdir(cache_dir) if f in files_to_download]
    for filename in tqdm(local_files_to_index, desc="Validating and Indexing"):
        local_path = os.path.join(cache_dir, filename)
        metadata = get_metadata_from_rwl(local_path)
        if metadata:
            metadata['filename'] = filename
            all_metadata.append(metadata)
        else:
            tqdm.write(f"  -> WARNING: Downloaded file '{filename}' could not be parsed, excluded from index.")
    if not all_metadata: raise ValueError("No valid, dated Tucson-format files could be indexed from download.")
    pd.DataFrame(all_metadata).to_csv(index_filename, index=False)
    print(f"\nSUCCESS: Index with {len(all_metadata)} valid entries created: '{index_filename}'.")

def build_master_from_index(chronology_name, target_species, country_prefixes, min_series_length, min_start_year, min_end_year=1500, index_filename="noaa_europe_index.csv"):
    print(f"\nBUILDING: '{chronology_name}'")
    if not os.path.exists(index_filename): raise FileNotFoundError("Index missing. Run 'python gogo.py index' first.")
    index_df = pd.read_csv(index_filename)
    df_filtered = index_df[(index_df['species'].isin(target_species)) & (index_df['filename'].str.lower().str.startswith(tuple(country_prefixes))) & (index_df['length'] >= min_series_length) & (index_df['start_year'] < min_start_year) & (index_df['end_year'] >= min_end_year)]
    file_list = df_filtered['filename'].tolist()
    if not file_list: raise ValueError(f"No files in index matched criteria (including min_end_year={min_end_year}).")
    print(f"Found {len(file_list)} matching files. Processing...")
    all_series = [_build_master_from_rwl_file(os.path.join("full_rwl_cache", filename)) for filename in tqdm(file_list, desc=f"Building {chronology_name}")]
    all_series = [s for s in all_series if not s.empty]
    if not all_series: raise ValueError("Failed to process any files.")
    combined_df = pd.concat(all_series, axis=1)
    master_chronology = combined_df.mean(axis=1); series_count = combined_df.notna().sum(axis=1)
    master_chronology = master_chronology[series_count >= 5].dropna()
    output_filename = f"master_{chronology_name.lower().replace(' ', '_')}.csv"; master_chronology.to_csv(output_filename, header=['value'], index_label='year')
    print(f"--- SUCCESS! Saved to '{output_filename}' ---")

def run_create_master(input_folder, output_filename):
    print(f"\nCREATING CUSTOM MASTER: {input_folder}")
    if not os.path.isdir(input_folder): raise FileNotFoundError(f"Folder '{input_folder}' does not exist.")
    file_list = [f for f in os.listdir(input_folder) if f.lower().endswith('.rwl')]
    if not file_list: raise ValueError(f"No .rwl files found in '{input_folder}'.")
    print(f"Found {len(file_list)} .rwl files. Processing...")
    all_series = [_build_master_from_rwl_file(os.path.join(input_folder, filename)) for filename in tqdm(file_list, desc="Processing files")]
    all_series = [s for s in all_series if not s.empty]
    if not all_series: raise ValueError("Failed to process any .rwl files.")
    print(f"Combining {len(all_series)} series...")
    combined_df = pd.concat(all_series, axis=1)
    master_chronology = combined_df.mean(axis=1); series_count = combined_df.notna().sum(axis=1)
    master_chronology = master_chronology[series_count >= 2].dropna()
    if not master_chronology.empty:
        if not output_filename.lower().endswith('.csv'): output_filename += ".csv"
        master_chronology.to_csv(output_filename, header=['value'], index_label='year')
        print(f"--- SUCCESS! Saved to: '{output_filename}' ---")
    else: raise ValueError("Resulting chronology was empty.")

def fetch_and_build_violin_master():
    """
    Fetches the curated violin reference files from the main cache
    and builds a dedicated master chronology from them.
    """
    source_dir = "full_rwl_cache"
    dest_dir = "violin_references"
    master_filename = "master_violin_chronology.csv"
    
    print(f"\n--- Preparing Specialist Violin Reference ---")
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
            print(f"\nSUCCESS: Violin setup is complete. You can now use '{os.path.join(dest_dir, master_filename)}' as a reference or select the 'violin' category in the Detective tab.")
        except Exception as e:
            print(f"\nERROR building violin master: {e}")
    else:
        print("\nNo files were copied, so no master chronology was built.")

# --- 4. MAIN DISPATCHER ---
def main():
    parser = argparse.ArgumentParser(description="Dendrochronology toolkit (V9.8).", formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
        WORKFLOW:
          1. python gogo.py index         (Downloads standard files and creates the index. Run once.)
          2. python gogo.py violin-setup  (Fetch curated violin files and build the specialist master.)
          3. python gogo.py build         (Build other master chronologies from the index)
          4. python gogo.py date ...      (Date a sample against a master)
        """))
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('index', help="Download standard-format NOAA files and create the data index.")
    subparsers.add_parser('violin-setup', help="Fetch curated violin files and build the specialist master.")
    
    parser_build = subparsers.add_parser('build', help="Build master chronologies from the clean data index.")
    parser_build.add_argument('--target', choices=['alpine', 'baltic', 'all'], default='all', help="Which master to build. (Default: all)")
    parser_build.add_argument('--min_end_year', type=int, default=1500, help="Only include reference sites that end after this year. Default: 1500")

    parser_create = subparsers.add_parser('create', help="Create a custom master from a local folder of .rwl files.")
    parser_create.add_argument('input_folder', help="Path to the folder containing your .rwl files.")
    parser_create.add_argument('output_filename', help="Name for the new master .csv file (e.g., 'my_master.csv').")

    parser_date = subparsers.add_parser('date', help='Cross-date a sample against a master or another .rwl file.')
    parser_date.add_argument('sample_file', help="Path to your sample .rwl file.")
    parser_date.add_argument('master_file', help="Path to the reference .csv or .rwl file.")
    parser_date.add_argument('--min_overlap', type=int, default=50, help="Minimum overlap in years. (Default: 50)")

    parser_detective = subparsers.add_parser('detective', help="Run a sample against ALL individual files in a category or folder.")
    parser_detective.add_argument('sample_file', help="Path to your sample .rwl file.")
    parser_detective.add_argument('target', nargs='?', default='violin', help="Reference: a category ('alpine', 'baltic', 'violin', 'all') or a folder path. (Default: violin)")
    parser_detective.add_argument('--top_n', type=int, default=10, help="Number of top results to display. (Default: 10)")
    parser_detective.add_argument('--min_overlap', type=int, default=80, help="Minimum overlap in years to consider a match. (Default: 80)")
    parser_detective.add_argument('--min_end_year', type=int, default=1500, help="Only include reference sites that end after this year. Default: 1500")
    
    args = parser.parse_args()

    try:
        if args.command == 'index':
            download_and_index_files()
        elif args.command == 'violin-setup':
            fetch_and_build_violin_master()
        elif args.command == 'build':
            min_end_year = args.min_end_year
            if args.target in ['alpine', 'all']:
                build_master_from_index("Alpine Instrument Wood", ['PICEA', 'ABIES'], ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 150, 1750, min_end_year=min_end_year)
            if args.target in ['baltic', 'all']:
                build_master_from_index("Baltic Northern Timber", ['PINUS', 'PICEA'], ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 150, 1750, min_end_year=min_end_year)
        elif args.command == 'create':
            run_create_master(args.input_folder, args.output_filename)
        elif args.command == 'date':
            result = run_date_analysis(args.sample_file, args.master_file, args.min_overlap)
            if result:
                plot_results(result) # Pass the whole dict
        elif args.command == 'detective':
            result = run_detective_analysis(args.sample_file, args.target, args.top_n, args.min_overlap, min_end_year=args.min_end_year)
            if result:
                plot_results(result) # Pass the whole dict
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")

if __name__ == '__main__':
    main()
