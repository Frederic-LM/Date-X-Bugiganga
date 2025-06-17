# gogo.py (Version 8.0 - Two-Stage Validation Workflow)
# ==============================================================================
import os, ftplib, argparse, textwrap, multiprocessing, shutil
import pandas as pd, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import UnivariateSpline

# --- 2. CORE DENDROCHRONOLOGY FUNCTIONS ---

def parse_as_floating_series(file_path: str) -> pd.Series:
    """A robust parser that reads ANY rwl-like file and treats it as a floating sample."""
    all_widths = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                try:
                    int(parts[1]); value_parts = parts[2:]
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
    """Original, simple parser for DATED .rwl files ONLY."""
    all_rings = []
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

# ... All other core functions (detrend, calculate_t_value, etc.) are correct ...
def detrend(series: pd.Series, spline_stiffness_pct: int = 67) -> pd.Series:
    series = series.dropna()
    if len(series) < 15: return pd.Series(dtype=np.float64)
    x, y = series.index.values, series.values
    s = len(series) * (spline_stiffness_pct / 100)**3
    spline = UnivariateSpline(x, y, s=s)
    spline_fit = pd.Series(spline(x), index=x)
    return series / (spline_fit + 1e-6)

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

def plot_results(raw_sample, master_detrended, detrended_sample, results, sample_filename, master_filename, reference_is_rwl=False, raw_master=None, **kwargs):
    print("Generating enhanced diagnostic plot...")
    if "error" in results: print(f"Cannot plot: {results['error']}"); return
    best_match = results['best_match']; all_correlations = results['all_correlations']
    best_end_year = int(best_match['end_year'])
    r_val, t_val, n_val, glk_val = best_match['correlation'], best_match['t_value'], int(best_match['overlap_n']), best_match.get('glk', 0.0)
    best_start_year = best_end_year - (raw_sample.index.max() - raw_sample.index.min())
    sample_index_at_best = pd.RangeIndex(start=best_start_year, stop=best_end_year + 1)
    overlap_index = master_detrended.index.intersection(sample_index_at_best)
    overlap_start_year, overlap_end_year = (overlap_index.min(), overlap_index.max()) if not overlap_index.empty else (best_start_year, best_end_year)
    fig = plt.figure(figsize=(16, 12)); plt.style.use('seaborn-v0_8-whitegrid')
    sample_label, master_label = os.path.basename(sample_filename), os.path.basename(master_filename)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(all_correlations.index, all_correlations['t_value'], color='gray', zorder=1, label=f'All Offsets (Best t={t_val:.2f})')
    ax1.scatter(best_end_year, t_val, color='red', s=120, zorder=2, ec='black', label=f'Best Match Year: {best_end_year}')
    ax1.axhline(3.5, color='orange', linestyle='--', linewidth=1, label='t=3.5 (Significant)'); ax1.axhline(5.0, color='firebrick', linestyle='--', linewidth=1, label='t=5.0 (Very Strong)')
    ax1.set_xlabel("Potential End Year of Sample"); ax1.set_ylabel("T-Value"); ax1.set_title("1. Cross-Dating Significance Plot"); ax1.legend()
    ax2 = plt.subplot(2, 2, 2)
    aligned_sample_detrended = detrended_sample.copy(); aligned_sample_detrended.index = aligned_sample_detrended.index - aligned_sample_detrended.index.min() + best_start_year
    ax2.plot(master_detrended.index, master_detrended.values, label=f'Reference: {master_label}', color='blue', alpha=0.8)
    ax2.plot(aligned_sample_detrended.index, aligned_sample_detrended.values, label=f'Sample: {sample_label}', color='red', linestyle='--')
    ax2.axvspan(overlap_start_year, overlap_end_year, color='gray', alpha=0.2, label=f'Overlap Region (n={n_val})'); ax2.set_xlim(overlap_start_year - 20, overlap_end_year + 20)
    ax2.set_xlabel("Year"); ax2.set_ylabel("Detrended Index"); ax2.set_title(f"2. Aligned Detrended Series (r={r_val:.3f})"); ax2.legend()
    ax3 = plt.subplot(2, 2, 3)
    aligned_raw_sample = raw_sample.copy(); aligned_raw_sample.index = aligned_raw_sample.index - aligned_raw_sample.index.min() + best_start_year
    ax3.plot(aligned_raw_sample.index, aligned_raw_sample.values, label=f'Sample: {sample_label}', color='green')
    if reference_is_rwl and raw_master is not None: ax3.plot(raw_master.index, raw_master.values, label=f'Reference: {master_label}', color='black', alpha=0.7)
    else: rescaled_master_for_plot = master_detrended * raw_sample.mean(); ax3.plot(rescaled_master_for_plot.index, rescaled_master_for_plot.values, label=f'Ref (scaled): {master_label}', color='black', alpha=0.7)
    ax3.axvspan(overlap_start_year, overlap_end_year, color='gray', alpha=0.2, label=f'Overlap Region (n={n_val})'); ax3.set_xlim(overlap_start_year - 20, overlap_end_year + 20)
    ax3.set_xlabel("Year"); ax3.set_ylabel("Ring Width (mm)"); ax3.set_title("3. Raw Data Visual Match"); ax3.legend()
    ax4 = plt.subplot(2, 2, 4); ax4.axis('off')
    summary_text = textwrap.dedent(f"""Cross-Dating Report\n-----------------------------\nSample File: {sample_label}\nReference File: {master_label}\n\nMost Likely End Year: {best_end_year}\n(Sample Start Year: {best_start_year})\n\nStatistics for this Position:\n  Correlation (r): {r_val:.4f}\n  T-Value: {t_val:.4f}\n  Gleichläufigkeit (Glk): {glk_val:.1f}%\n  Overlap (n years): {n_val}\n    """)
    ax4.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=12, fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", fc='aliceblue', ec='lightsteelblue', lw=2))
    ax4.set_title("4. Summary Statistics")
    plt.suptitle(f"Cross-Dating Analysis: {sample_label} vs. {master_label}", fontsize=16, fontweight='bold'); plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

# --- COMMAND LOGIC FUNCTIONS ---
def get_metadata_from_rwl(file_path):
    series = _build_master_from_rwl_file(file_path)
    if series.empty: return None
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: header_content = "".join([f.readline() for _ in range(5)]).lower()
    species_map = {"picea": "PICEA", "spruce": "PICEA", "pinus": "PINUS", "pine": "PINUS", "abies": "ABIES", "fir": "ABIES", "larix": "LARIX", "larch": "LARIX"}
    for key, val in species_map.items():
        if key in header_content: return {"species": val, "start_year": int(series.index.min()), "end_year": int(series.index.max()), "length": len(series)}
    return {"species": "UNKNOWN", "start_year": int(series.index.min()), "end_year": int(series.index.max()), "length": len(series)}

def download_all_files():
    """STAGE 1: Downloads all .rwl files from NOAA FTP server without validation."""
    print("--- Stage 1: Downloading all .rwl files from NOAA ---")
    cache_dir = "full_rwl_cache"
    os.makedirs(cache_dir, exist_ok=True)
    try:
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov", timeout=60); ftp.login(); ftp.cwd("/pub/data/paleo/treering/measurements/europe/")
        file_list = [f for f in ftp.nlst() if f.lower().endswith('.rwl')]
        print(f"Found {len(file_list)} files on server.")
    except Exception as e: raise ConnectionError(f"FTP Error: {e}")

    for filename in tqdm(file_list, desc="Downloading Files"):
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            try:
                with open(local_path, 'wb') as f: ftp.retrbinary(f"RETR {filename}", f.write)
            except Exception: continue
    ftp.quit()
    print("Download complete. Run 'python gogo.py clean' to validate and create the index.")

def clean_and_index_cache(index_filename="noaa_europe_index.csv"):
    """STAGE 2: Validates all downloaded files, quarantines bad ones, and creates the index."""
    print("\n--- Stage 2: Cleaning cache and creating index ---")
    cache_dir = "full_rwl_cache"
    quarantine_dir = os.path.join(cache_dir, "unsupported_format")
    if not os.path.isdir(cache_dir): raise FileNotFoundError(f"Cache directory '{cache_dir}' not found. Run 'python gogo.py index' first.")
    os.makedirs(quarantine_dir, exist_ok=True)
    
    all_metadata = []
    file_list = [f for f in os.listdir(cache_dir) if f.lower().endswith('.rwl') and os.path.isfile(os.path.join(cache_dir, f))]
    
    for filename in tqdm(file_list, desc="Validating and Indexing"):
        local_path = os.path.join(cache_dir, filename)
        # We use the strict, dated-only parser for validation. If it fails, the file is bad.
        metadata = get_metadata_from_rwl(local_path)
        if metadata:
            metadata['filename'] = filename
            all_metadata.append(metadata)
        else:
            tqdm.write(f"  -> Quarantining non-standard file: {filename}")
            shutil.move(local_path, os.path.join(quarantine_dir, filename))

    if not all_metadata: raise ValueError("No valid, dated Tucson-format files could be indexed.")
    pd.DataFrame(all_metadata).to_csv(index_filename, index=False)
    print(f"\nSUCCESS: Index with {len(all_metadata)} valid entries created and saved to '{index_filename}'.")
    print(f"Unsupported files moved to '{quarantine_dir}'.")

def build_master_from_index(chronology_name, target_species, country_prefixes, min_series_length, min_start_year, index_filename="noaa_europe_index.csv"):
    # This logic is correct now that the index is clean.
    print(f"\nBUILDING: '{chronology_name}'")
    if not os.path.exists(index_filename): raise FileNotFoundError("Index file missing. Run 'python gogo.py clean' first.")
    index_df = pd.read_csv(index_filename)
    df_filtered = index_df[(index_df['species'].isin(target_species)) & (index_df['filename'].str.lower().str.startswith(tuple(country_prefixes))) & (index_df['length'] >= min_series_length) & (index_df['start_year'] < min_start_year)]
    file_list = df_filtered['filename'].tolist()
    if not file_list: raise ValueError("No files in index matched criteria.")
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
    # This logic is correct.
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

def run_date_analysis(sample_file, master_file, min_overlap=50, is_internal_test=False):
    # The analysis functions from the previous final version were correct,
    # it was the data feeding them that was wrong. This logic is restored.
    if not is_internal_test: print(f"\n--- Running Analysis: {os.path.basename(sample_file)} vs {os.path.basename(master_file)} ---")
    sample_chrono = parse_as_floating_series(sample_file)
    if sample_chrono.empty: raise ValueError(f"Could not read data from sample file: {sample_file}")

    reference_is_rwl = master_file.lower().endswith('.rwl')
    if is_internal_test:
        master_chrono = parse_as_floating_series(master_file)
    elif reference_is_rwl:
        master_chrono = _build_master_from_rwl_file(master_file)
    else:
        master_chrono = pd.read_csv(master_file, index_col='year').squeeze("columns")

    if master_chrono.empty: raise ValueError(f"Could not read data from reference file: {master_file}")
    if min_overlap > len(sample_chrono): raise ValueError(f"CONFIG ERROR: min_overlap > sample length.")

    sample_detrended = detrend(sample_chrono); master_detrended = detrend(master_chrono)
    analysis_results = cross_date(sample_detrended, master_detrended, min_overlap=min_overlap)
    if "error" in analysis_results: raise ValueError(analysis_results['error'])
    if not is_internal_test: print(f"\n--- Cross-Dating Complete ---\nMost Likely End Year: {int(analysis_results['best_match']['end_year'])}")
    return {'raw_sample': sample_chrono, 'master_detrended': master_detrended, 'detrended_sample': sample_detrended, 'results': analysis_results, 'sample_filename': sample_file, 'master_filename': master_file, 'reference_is_rwl': reference_is_rwl, 'raw_master': master_chrono if not is_internal_test else None}

def process_single_file(args):
    filename, sample_detrended, sample_basename, base_path, min_overlap = args
    if filename == sample_basename: return None
    master_path = os.path.join(base_path, filename)
    master_raw = _build_master_from_rwl_file(master_path)
    if master_raw.empty or len(master_raw) < min_overlap: return None
    master_detrended = detrend(master_raw)
    analysis_results = cross_date(sample_detrended, master_detrended, min_overlap=min_overlap)
    if "error" in analysis_results: return None
    best_match = analysis_results['best_match']
    if best_match['correlation'] > 0.99: return None
    best_match['source_file'] = filename
    return best_match

def run_detective_analysis(sample_file, target, top_n, min_overlap=80):
    print(f"\n--- Running Detective Analysis on {os.path.basename(sample_file)} ---")
    sample_chrono = parse_as_floating_series(sample_file)
    if sample_chrono.empty: raise ValueError("Could not process sample file.")
    if min_overlap > len(sample_chrono): raise ValueError(f"CONFIG ERROR: min_overlap > sample length.")
    sample_detrended = detrend(sample_chrono)
    file_list, base_path_for_masters = [], ""
    if os.path.isdir(target):
        base_path_for_masters = target
        file_list = [f for f in os.listdir(target) if f.lower().endswith('.rwl')]
    else:
        base_path_for_masters = "full_rwl_cache"
        index_filename = "noaa_europe_index.csv"
        if not os.path.exists(index_filename): raise ValueError("Index file missing. Run 'gogo clean' first.")
        category_params = {'alpine': {'species': ['PICEA', 'ABIES'], 'countries': ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 'min_len': 150, 'min_start': 1750}, 'baltic': {'species': ['PINUS', 'PICEA'], 'countries': ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 'min_len': 150, 'min_start': 1750}, 'all': {'species': ['PICEA', 'ABIES', 'PINUS', 'LARIX'], 'countries': ['aust', 'fran', 'germ', 'ital', 'swit', 'slov', 'finl', 'lith', 'norw', 'pola', 'swed'], 'min_len': 100, 'min_start': 1800}}
        if target not in category_params: raise ValueError(f"Invalid category '{target}'.")
        params = category_params[target]
        index_df = pd.read_csv(index_filename)
        df_filtered = index_df[(index_df['species'].isin(params['species'])) & (index_df['filename'].str.lower().str.startswith(tuple(params['countries']))) & (index_df['length'] >= params['min_len']) & (index_df['start_year'] < params['min_start'])]
        file_list = df_filtered['filename'].tolist()

    if not file_list: raise ValueError(f"No reference files found for target '{target}'.")
    tasks = [(filename, sample_detrended, os.path.basename(sample_file), base_path_for_masters, min_overlap) for filename in file_list]
    print(f"Testing against {len(file_list)} sites using {multiprocessing.cpu_count()} CPU cores...")
    with multiprocessing.Pool() as pool:
        all_best_results = [res for res in tqdm(pool.imap(process_single_file, tasks), total=len(tasks)) if res is not None]
    if not all_best_results: print("\nAnalysis complete, no significant correlations found."); return None
    results_df = pd.DataFrame(all_best_results).sort_values(by='t_value', ascending=False)
    top_results = results_df.head(top_n)
    print(f"\n--- Top {top_n} Matching Sites (Sorted by T-Value) ---")
    top_results_display = top_results.copy(); top_results_display['glk'] = top_results_display['glk'].round(1)
    print(top_results_display[['end_year', 't_value', 'glk', 'correlation', 'overlap_n', 'source_file']].to_string(index=False))
    top_match_file_name = top_results.iloc[0]['source_file']
    top_match_full_path = os.path.join(base_path_for_masters, top_match_file_name)
    print(f"\nGenerating plot for top match: {top_match_file_name}")
    plot_data_dict = run_date_analysis(sample_file, top_match_full_path, min_overlap=min_overlap)
    if not plot_data_dict:
        print("Could not generate plot for the top match.")
        return {"analysis_mode": "single", "analysis_type": "detective", "sample_file": sample_file, "target": target, "min_overlap": min_overlap, "results_df": top_results}
    plot_data_dict.update({"analysis_type": "detective", "results_df": top_results})
    return plot_data_dict

def run_two_piece_mean_analysis(bass_file, treble_file, final_analysis_func, final_analysis_args):
    print("\n--- Starting Two-Piece Mean Analysis ---")
    bass_chrono = parse_as_floating_series(bass_file)
    treble_chrono = parse_as_floating_series(treble_file)
    if bass_chrono.empty or treble_chrono.empty: raise ValueError("Could not process two-piece sample files.")
    print("--- Internal Cross-Match (Bass vs. Treble) ---")
    bass_detrended, treble_detrended = detrend(bass_chrono), detrend(treble_chrono)
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
    final_results.update({'analysis_mode': 'two_piece', 'internal_stats': {'t_value': internal_t, 'glk': internal_glk}, 'bass_file': bass_file, 'treble_file': treble_file, 'sample_filename': "Mean Chronology (Bass+Treble)"})
    return final_results
    
# --- 4. MAIN DISPATCHER ---
def main():
    parser = argparse.ArgumentParser(description="Dendrochronology toolkit (V8.0).", formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
        WORKFLOW:
          1. python gogo.py index     (Download all files from NOAA, run once)
          2. python gogo.py clean     (Validate files, create index, run after index or if cache changes)
          3. python gogo.py build     (Build master chronologies from the clean index)
          4. python gogo.py date ...  (Date a sample against a master)
        """))
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('index', help="STAGE 1: Download all .rwl files from the NOAA server.")
    subparsers.add_parser('clean', help="STAGE 2: Clean the downloaded cache and create the data index.")
    parser_build = subparsers.add_parser('build', help="Build master chronologies from the clean data index.")
    parser_build.add_argument('--target', choices=['alpine', 'baltic', 'all'], default='all', help="Which master to build. (Default: all)")
    parser_create = subparsers.add_parser('create', help="Create a custom master from a local folder of .rwl files.")
    parser_create.add_argument('input_folder', help="Path to the folder containing your .rwl files.")
    parser_create.add_argument('output_filename', help="Name for the new master .csv file (e.g., 'my_master.csv').")
    parser_date = subparsers.add_parser('date', help='Cross-date a sample against a master or another .rwl file.')
    parser_date.add_argument('sample_file', help="Path to your sample .rwl file.")
    parser_date.add_argument('master_file', help="Path to the reference .csv or .rwl file.")
    parser_date.add_argument('--min_overlap', type=int, default=50, help="Minimum overlap in years. (Default: 50)")
    parser_detective = subparsers.add_parser('detective', help="Run a sample against ALL individual files in a category or folder.")
    parser_detective.add_argument('sample_file', help="Path to your sample .rwl file.")
    parser_detective.add_argument('target', nargs='?', default='alpine', help="Reference: a category ('alpine', 'baltic', 'all') or a folder path. (Default: alpine)")
    parser_detective.add_argument('--top_n', type=int, default=10, help="Number of top results to display. (Default: 10)")
    parser_detective.add_argument('--min_overlap', type=int, default=80, help="Minimum overlap in years to consider a match. (Default: 80)")
    args = parser.parse_args()

    try:
        if args.command == 'index': download_all_files()
        elif args.command == 'clean': clean_and_index_cache()
        elif args.command == 'build':
            if args.target in ['alpine', 'all']: build_master_from_index("Alpine Instrument Wood", ['PICEA', 'ABIES'], ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 150, 1750)
            if args.target in ['baltic', 'all']: build_master_from_index("Baltic Northern Timber", ['PINUS', 'PICEA'], ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 150, 1750)
        elif args.command == 'create': run_create_master(args.input_folder, args.output_filename)
        elif args.command == 'date': 
            result = run_date_analysis(args.sample_file, args.master_file, args.min_overlap)
            if result: plot_results(**result)
        elif args.command == 'detective':
            result = run_detective_analysis(args.sample_file, args.target, args.top_n, args.min_overlap)
            if result: plot_results(**result)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")

if __name__ == '__main__':
    main()
