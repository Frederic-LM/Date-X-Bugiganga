# ==============================================================================
#
#  The Cross-Dating Tool for Historical Instrument Analysis
#
#  Version: 3.5 (Final - All Functions Included and Fully Functional)
#
#  OVERVIEW:
#  This is the complete, working version of the script. It includes a robust
#  parser and indexer, and the cross-dating and plotting functions are now
#  fully implemented and called correctly from a clean command-line interface.
#
#  WORKFLOW:
#  STEP 1: Create a Local Index (Run ONCE, takes 15-30 mins)
#     python dd.py index
#
#  STEP 2: Build Master Chronologies from the Local Index (Fast)
#     python dd.py build
#
#  STEP 3: Date Your Sample (Two powerful methods)
#     # A) Broad search against all sites in a category
#     python dd.py detective your_sample.rwl --category alpine
#
#     # B) Direct 1-on-1 comparison against a specific master OR another .rwl file
#     python dd.py date your_sample.rwl master_alpine_instrument_wood.csv
#     python dd.py date your_sample.rwl swed023e.rwl
#
# ==============================================================================

# --- 1. IMPORTS ---
import os
import ftplib
import argparse
import textwrap
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import UnivariateSpline

# --- 2. CORE DENDROCHRONOLOGY FUNCTIONS ---

def parse_rwl_robust(file_path: str) -> pd.Series:
    """A highly robust parser for messy, real-world RWL files."""
    all_rings = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 2: continue
                try: year_val = int(parts[1])
                except (ValueError, IndexError): continue
                if not (100 < year_val < 2100): continue
                start_year = (year_val // 10) * 10
                for i, val_str in enumerate(parts[2:]):
                    try:
                        width = int(val_str)
                        if width in [-9999, 999]: continue
                        current_year = start_year + i
                        if i == 0 and current_year != year_val: start_year = year_val; current_year = year_val
                        all_rings.append({'year': current_year, 'width': width / 100.0})
                    except ValueError: continue
    except Exception: return pd.Series(dtype=np.float64)
    if not all_rings: return pd.Series(dtype=np.float64)
    df = pd.DataFrame(all_rings).drop_duplicates(subset='year').set_index('year')
    return df['width'].sort_index()

def detrend(series: pd.Series, spline_stiffness_pct: int = 67) -> pd.Series:
    """Detrends a time series using a standard smoothing spline."""
    if len(series) < 15: return pd.Series(dtype=np.float64)
    x, y = series.index.values, series.values
    s = len(series) * (spline_stiffness_pct / 100)**3
    spline = UnivariateSpline(x, y, s=s)
    spline_fit = pd.Series(spline(x), index=x)
    return series / (spline_fit + 1e-6)

def calculate_t_value(r: float, n: int) -> float:
    """Calculates the Student's t-statistic for a Pearson correlation."""
    if n < 3 or abs(r) >= 1.0: return np.inf * np.sign(r) if r != 0 else 0
    return r * np.sqrt((n - 2) / (1 - r**2))

def cross_date(sample_series: pd.Series, master_series: pd.Series, min_overlap: int = 50) -> dict:
    """Performs cross-dating using a sliding window algorithm."""
    if sample_series.empty or master_series.empty: return {"error": "Input series empty."}
    s_len, s_first_year = len(sample_series), sample_series.index.min()
    m_start, m_end = master_series.index.min(), master_series.index.max()
    start_rng = range(int(m_start - s_len + min_overlap), int(m_end - min_overlap + 2))
    corrs = []
    for s_start in start_rng:
        s_idx_shifted = pd.RangeIndex(start=s_start, stop=s_start + s_len)
        o_idx = master_series.index.intersection(s_idx_shifted)
        if len(o_idx) >= min_overlap:
            s_idx_original = o_idx - s_start + s_first_year
            sample_segment = sample_series.loc[s_idx_original]
            master_segment = master_series.loc[o_idx]
            r, _ = pearsonr(sample_segment, master_segment)
            corrs.append({"end_year": s_start + s_len - 1, "correlation": r, "t_value": calculate_t_value(r, len(o_idx)), "overlap_n": len(o_idx)})
    if not corrs: return {"error": f"No suitable overlap found (min_overlap = {min_overlap} years)."}
    rdf = pd.DataFrame(corrs)
    return {"best_match": rdf.loc[rdf['t_value'].idxmax()].to_dict(), "all_correlations": rdf.set_index('end_year')}

# --- Replace your entire plot_results function with this one ---
def plot_results(
    raw_sample,
    master_detrended,
    detrended_sample,
    results,
    sample_filename,
    master_filename,
    reference_is_rwl=False,
    raw_master=None
):
    """
    Generates an improved 4-panel plot with a highlighted overlap region
    and dynamic legends using the source filenames.
    """
    print("Generating enhanced diagnostic plot...")
    if "error" in results:
        print(f"Cannot plot: {results['error']}")
        return

    # --- Extract data from results ---
    best_match = results['best_match']
    all_correlations = results['all_correlations']

    best_end_year = int(best_match['end_year'])
    r_val, t_val, n_val = best_match['correlation'], best_match['t_value'], int(best_match['overlap_n'])
    best_start_year = best_end_year - len(raw_sample) + 1
    
    # --- Determine the precise overlap start and end for shading ---
    # We need to find the intersection of the two series' indices at the best alignment
    sample_index_at_best = pd.RangeIndex(start=best_start_year, stop=best_end_year + 1)
    overlap_index = master_detrended.index.intersection(sample_index_at_best)
    
    # Handle cases where there might not be an overlap found
    if overlap_index.empty:
        overlap_start_year = best_start_year
        overlap_end_year = best_end_year
    else:
        overlap_start_year = overlap_index.min()
        overlap_end_year = overlap_index.max()


    # --- Setup the plot ---
    fig = plt.figure(figsize=(16, 12))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Use just the filename part, not the full path, for cleaner labels
    sample_label = os.path.basename(sample_filename)
    master_label = os.path.basename(master_filename)

    # --- Plot 1: Correlogram (t-value vs. year) ---
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(all_correlations.index, all_correlations['t_value'], color='gray', zorder=1, label=f'All Offsets (Best t={t_val:.2f})')
    ax1.scatter(best_end_year, t_val, color='red', s=120, zorder=2, ec='black', label=f'Best Match Year: {best_end_year}')
    ax1.set_xlabel("Potential End Year of Sample"); ax1.set_ylabel("T-Value")
    ax1.set_title("1. Cross-Dating Significance Plot"); ax1.legend()
    ax1.axhline(3.5, color='orange', linestyle='--', linewidth=1, label='t=3.5 (Significant)')
    ax1.axhline(5.0, color='firebrick', linestyle='--', linewidth=1, label='t=5.0 (Very Strong)')
    ax1.legend()

    # --- Plot 2: Detrended Series Alignment ---
    ax2 = plt.subplot(2, 2, 2)
    aligned_sample = detrended_sample.copy()
    aligned_sample.index = pd.RangeIndex(start=best_start_year, stop=best_end_year + 1)
    # NEW: Add dynamic filenames to legends
    ax2.plot(master_detrended.index, master_detrended.values, label=f'Reference: {master_label}', color='blue', alpha=0.8)
    ax2.plot(aligned_sample.index, aligned_sample.values, label=f'Sample: {sample_label}', color='red', linestyle='--')
    # NEW: Highlight the overlap region
    ax2.axvspan(overlap_start_year, overlap_end_year, color='gray', alpha=0.2, label=f'Overlap Region (n={n_val})')
    ax2.set_xlim(overlap_start_year - 10, overlap_end_year + 10) # Zoom in on the overlap
    ax2.set_xlabel("Year"); ax2.set_ylabel("Detrended Index")
    ax2.set_title(f"2. Aligned Detrended Series (r={r_val:.3f})"); ax2.legend()

    # --- Plot 3: Raw Data Comparison ---
    ax3 = plt.subplot(2, 2, 3)
    aligned_raw_sample = raw_sample.copy()
    aligned_raw_sample.index = pd.RangeIndex(start=best_start_year, stop=best_end_year + 1)
    # NEW: Use dynamic filenames in legends
    ax3.plot(aligned_raw_sample.index, aligned_raw_sample.values, label=f'Sample: {sample_label}', color='green')
    if reference_is_rwl and raw_master is not None:
        ax3.plot(raw_master.index, raw_master.values, label=f'Reference: {master_label}', color='black', alpha=0.7)
    else:
        rescaled_master_for_plot = master_detrended * raw_sample.mean()
        ax3.plot(rescaled_master_for_plot.index, rescaled_master_for_plot.values, label=f'Ref (scaled): {master_label}', color='black', alpha=0.7)
    # NEW: Highlight the overlap region
    ax3.axvspan(overlap_start_year, overlap_end_year, color='gray', alpha=0.2, label=f'Overlap Region (n={n_val})')
    ax3.set_xlim(overlap_start_year - 10, overlap_end_year + 10) # Zoom in
    ax3.set_xlabel("Year"); ax3.set_ylabel("Ring Width (mm)"); ax3.set_title("3. Raw Data Visual Match"); ax3.legend()

    # --- Plot 4: Summary Report (Unchanged) ---
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    summary_text = textwrap.dedent(f"""Cross-Dating Report
-----------------------------
Sample File: {sample_label}
Reference File: {master_label}

Most Likely End Year: {best_end_year}
(Sample Start Year: {best_start_year})

Statistics for this Position:
  Correlation (r): {r_val:.4f}
  T-Value: {t_val:.4f}
  Overlap (n years): {n_val}
    """)
    ax4.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=12, fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", fc='aliceblue', ec='lightsteelblue', lw=2))
    ax4.set_title("4. Summary Statistics")

    plt.suptitle(f"Cross-Dating Analysis: {sample_label} vs. {master_label}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- 3. INDEXING AND BUILDING FUNCTIONS ---

def get_metadata_from_rwl(file_path):
    """Extracts metadata using the robust parser."""
    try:
        series = parse_rwl_robust(file_path)
        if series.empty: return None
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            header_content = "".join([f.readline() for _ in range(5)]).lower()
        species_name = "UNKNOWN"
        if "picea" in header_content or "spruce" in header_content: species_name = "PICEA"
        elif "pinus" in header_content or "pine" in header_content: species_name = "PINUS"
        elif "abies" in header_content or "fir" in header_content: species_name = "ABIES"
        elif "larix" in header_content or "larch" in header_content: species_name = "LARIX"
        return {"species": species_name, "start_year": int(series.index.min()), "end_year": int(series.index.max()), "length": len(series)}
    except Exception: return None

def create_ftp_index(index_filename="noaa_europe_index.csv"):
    """The one-time task. Scans FTP and builds the local index."""
    print("Starting the one-time indexing of the NOAA FTP server.")
    cache_dir = "full_rwl_cache"; os.makedirs(cache_dir, exist_ok=True)
    try:
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov", timeout=60); ftp.login()
        ftp.cwd("/pub/data/paleo/treering/measurements/europe/")
        file_list = [f for f in ftp.nlst() if f.lower().endswith('.rwl')]
        print(f"Found {len(file_list)} .rwl files to index.")
    except Exception as e: print(f"FATAL: FTP Error. {e}"); return
    all_metadata = []
    for filename in tqdm(file_list, desc="Indexing NOAA Files"):
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            try:
                with open(local_path, 'wb') as f: ftp.retrbinary(f"RETR {filename}", f.write)
            except Exception: continue
        metadata = get_metadata_from_rwl(local_path)
        if metadata: metadata['filename'] = filename; all_metadata.append(metadata)
    ftp.quit()
    if not all_metadata: print("\nERROR: No metadata could be extracted."); return
    pd.DataFrame(all_metadata).to_csv(index_filename, index=False)
    print(f"\nSUCCESS: Index with {len(all_metadata)} entries created and saved to '{index_filename}'.")

def build_master_from_index(chronology_name, target_species, country_prefixes, min_series_length, min_start_year, index_filename="noaa_europe_index.csv"):
    """Builds a master chronology using the local index."""
    print("\n" + "="*60); print(f"BUILDING: '{chronology_name}'")
    if not os.path.exists(index_filename): print(f"ERROR: Index file missing."); return
    index_df = pd.read_csv(index_filename)
    df_filtered = index_df[(index_df['species'].isin(target_species)) & (index_df['filename'].str.lower().str.startswith(tuple(country_prefixes))) & (index_df['length'] >= min_series_length) & (index_df['start_year'] < min_start_year)]
    file_list = df_filtered['filename'].tolist()
    if not file_list: print("ERROR: No files in index matched criteria."); return
    print(f"Found {len(file_list)} matching files. Processing...")
    detrended_series_list = []
    for filename in tqdm(file_list, desc=f"Building {chronology_name}"):
        series = parse_rwl_robust(os.path.join("full_rwl_cache", filename))
        if not series.empty: detrended_series_list.append(detrend(series))
    if not detrended_series_list: print("ERROR: Failed to process files."); return
    combined_df = pd.concat(detrended_series_list, axis=1)
    master_chronology = combined_df.mean(axis=1)
    series_count = combined_df.notna().sum(axis=1)
    master_chronology = master_chronology[series_count >= 5].dropna()
    output_filename = f"master_{chronology_name.lower().replace(' ', '_')}.csv"
    master_chronology.to_csv(output_filename, header=['value'], index_label='year')
    print(f"\n--- SUCCESS! Saved to '{output_filename}' ---")

# --- 4. COMMAND LOGIC FUNCTIONS ---

def run_date_analysis(sample_file, master_file):
    """Performs cross-dating of a sample against a reference file (.csv or .rwl)."""
    if not os.path.exists(sample_file) or not os.path.exists(master_file):
        print("Error: One or both specified files do not exist."); return
    print(f"\n--- Running Cross-Dating Analysis ---\nSample: '{sample_file}'\nReference: '{master_file}'")
    sample_raw = parse_rwl_robust(sample_file)
    if sample_raw.empty: print("Could not parse sample file."); return
    sample_detrended = detrend(sample_raw)

    master_raw = None
    reference_is_rwl = False
    if master_file.lower().endswith('.csv'):
        print("Reference is a .csv master chronology. Loading...")
        master_detrended = pd.read_csv(master_file, index_col='year').squeeze("columns")
    elif master_file.lower().endswith('.rwl'):
        print("Reference is an .rwl file. Parsing and detrending on the fly...")
        master_raw = parse_rwl_robust(master_file)
        if master_raw.empty: print("Could not parse reference .rwl file."); return
        master_detrended = detrend(master_raw)
        reference_is_rwl = True
    else: print(f"Error: Unknown reference file type '{master_file}'."); return

    print("Performing sliding window correlation...")
    analysis_results = cross_date(sample_detrended, master_detrended)

    if "error" not in analysis_results:
        best = analysis_results['best_match']
        print("\n--- Cross-Dating Complete ---")
        print(f"Most Likely End Year: {int(best['end_year'])}")
        print(f"Correlation (r):      {best['correlation']:.4f}")
        print(f"T-Value:              {best['t_value']:.4f} (A value > 4.0 is a strong indicator)")
        print(f"Overlap (n):          {int(best['overlap_n'])} years")
        plot_results(sample_raw, master_detrended, sample_detrended, analysis_results, sample_file, master_file, reference_is_rwl, master_raw)
    else: print(f"\nAnalysis failed: {analysis_results['error']}")

def run_detective_analysis(sample_file, category, top_n):
    """Performs site-by-site analysis against a category of files from the index."""
    print(f"\n--- Running Detective Analysis ---\nSample: '{sample_file}' | Category: '{category}'")
    index_filename = "noaa_europe_index.csv"
    if not os.path.exists(index_filename): print(f"ERROR: Index file missing."); return
    sample_raw = parse_rwl_robust(sample_file)
    if sample_raw.empty: print("Could not parse sample file."); return
    sample_detrended = detrend(sample_raw)

    category_params = {
        'alpine': {'species': ['PICEA', 'ABIES'], 'countries': ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 'min_len': 150, 'min_start': 1750},
        'baltic': {'species': ['PINUS', 'PICEA'], 'countries': ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 'min_len': 150, 'min_start': 1750},
        'all': {'species': ['PICEA', 'ABIES', 'PINUS', 'LARIX'], 'countries': ['aust', 'fran', 'germ', 'ital', 'swit', 'slov', 'finl', 'lith', 'norw', 'pola', 'swed'], 'min_len': 100, 'min_start': 1800}
    }
    params = category_params[category]
    index_df = pd.read_csv(index_filename)
    df_filtered = index_df[(index_df['species'].isin(params['species'])) & (index_df['filename'].str.lower().str.startswith(tuple(params['countries']))) & (index_df['length'] >= params['min_len']) & (index_df['start_year'] < params['min_start'])]
    file_list = df_filtered['filename'].tolist()
    if not file_list: print(f"ERROR: No files in index matched criteria for '{category}'."); return

    all_best_results = []
    for filename in tqdm(file_list, desc=f"Testing {len(file_list)} sites in '{category}'"):
        master_raw = parse_rwl_robust(os.path.join("full_rwl_cache", filename))
        if master_raw.empty: continue
        analysis_results = cross_date(sample_detrended, detrend(master_raw), min_overlap=40)
        if "error" not in analysis_results:
            best_match = analysis_results['best_match']; best_match['source_file'] = filename
            all_best_results.append(best_match)

    if not all_best_results: print("\nAnalysis complete, no correlations found."); return
    results_df = pd.DataFrame(all_best_results)
    top_results = results_df.sort_values(by='t_value', ascending=False).head(top_n)
    print(f"\n--- Top {top_n} Matching Sites (Sorted by T-Value) ---")
    top_results['end_year'] = top_results['end_year'].astype(int)
    top_results['overlap_n'] = top_results['overlap_n'].astype(int)
    print(top_results[['end_year', 't_value', 'correlation', 'overlap_n', 'source_file']].to_string(index=False))
    year_counts = top_results['end_year'].value_counts()
    if not year_counts.empty:
        most_likely_year, count = year_counts.index[0], year_counts.iloc[0]
        print("\n--- Conclusion ---")
        if count > 1 and top_results['t_value'].iloc[0] > 3.5:
            print(f"A consensus is forming: the year {most_likely_year} appeared {count} times.")
        else:
            print("No clear consensus found. The chronology is likely 'floating'.")

# --- 5. MAIN DISPATCHER ---
def main():
    parser = argparse.ArgumentParser(description="Dendrochronology toolkit for instrument analysis (V3.5).", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('index', help="Create a local index of NOAA data (RUN THIS FIRST).")
    parser_build = subparsers.add_parser('build', help="Build master chronologies from the local index (fast).")
    parser_build.add_argument('--target', choices=['alpine', 'baltic', 'all'], default='all', help="Which master to build. (Default: all)")
    parser_date = subparsers.add_parser('date', help='Cross-date a sample against a master or another .rwl file.')
    parser_date.add_argument('sample_file', help="Path to your sample .rwl file.")
    parser_date.add_argument('master_file', help="Path to the reference .csv or .rwl file.")
    parser_detective = subparsers.add_parser('detective', help="Run a sample against ALL individual files in a category.")
    parser_detective.add_argument('sample_file', help="Path to your sample .rwl file.")
    parser_detective.add_argument('--category', choices=['alpine', 'baltic', 'all'], default='alpine', help="Which category to test against. (Default: alpine)")
    parser_detective.add_argument('--top_n', type=int, default=10, help="Number of top results to display. (Default: 10)")
    args = parser.parse_args()

    if args.command == 'index': create_ftp_index()
    elif args.command == 'build':
        if args.target in ['alpine', 'all']: build_master_from_index("Alpine Instrument Wood", ['PICEA', 'ABIES'], ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 150, 1750)
        if args.target in ['baltic', 'all']: build_master_from_index("Baltic Northern Timber", ['PINUS', 'PICEA'], ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 150, 1750)
    elif args.command == 'date': run_date_analysis(args.sample_file, args.master_file)
    elif args.command == 'detective': run_detective_analysis(args.sample_file, args.category, args.top_n)

if __name__ == '__main__':
    main()
