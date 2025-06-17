# ==============================================================================
#
#  The Cross-Dating Tool for Historical Instrument Analysis
#
#  Version: 5.2 
#
#  OVERVIEW:
#  This version significantly enhances the 'detective' mode. It can now run
#  analysis against both the large, predefined categories from the NOAA index
#  AND a local folder containing your own curated set of .rwl files. This makes
#  targeted analysis (e.g., against known violin references) fast and easy.
#
#  WORKFLOW:
#  STEP 1: Create a Local Index (Run ONCE, takes 15-30 mins)
#     python gogo.py index
#
#  STEP 2: Build or Create Reference Chronologies
#     # A) Build master chronologies from the full local index (Fast)
#     python gogo.py build
#
#     # B) Create a custom master from your own .rwl files (e.g., violin references)
#     python gogo.py create violin_references master_violin_chronology.csv
#
#  STEP 3: Date Your "Floating" Sample (Three powerful methods)
#     Note: The script automatically ignores existing dates in your sample for the
#     analysis, finds the best match, and then reports if its finding
#     agrees with the original file's dating.
#
#     # A) Detective Mode: Broad search against many individual files
#     #    Use a predefined category...
#     python gogo.py detective your_sample.rwl alpine
#
#     #    ...OR use your own local folder of references! (NEW)
#     python gogo.py detective your_sample.rwl violin_references
#
#     # B) Date Mode: Direct 1-on-1 comparison for final verification
#     #    Against a master .csv file...
#     python gogo.py date your_sample.rwl master_violin_chronology.csv
#
#     #    ...OR against another single .rwl file
#     python gogo.py date your_sample.rwl swed023e.rwl
#
# =============================================================================

# --- 1. IMPORTS ---
import os
import ftplib
import argparse
import textwrap
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import UnivariateSpline

# --- 2. CORE DENDROCHRONOLOGY FUNCTIONS ---

def parse_rwl_robust(file_path: str, is_floating: bool = False) -> pd.Series:
    """A highly robust parser for messy, real-world RWL files."""
    all_rings = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 2: continue
                if not is_floating:
                    try: year_val = int(parts[1])
                    except (ValueError, IndexError): continue
                    if not (100 < year_val < 2100): continue
                    start_year = (year_val // 10) * 10
                    value_parts = parts[2:]
                    for i, val_str in enumerate(value_parts):
                        try:
                            width = int(val_str)
                            if width in [-9999, 999]: continue
                            current_year = start_year + i
                            if i == 0 and current_year != year_val: start_year = year_val; current_year = year_val
                            all_rings.append({'year': current_year, 'width': width / 100.0})
                        except ValueError: continue
                else:
                    value_parts = parts[1:]
                    for val_str in value_parts:
                        try:
                            width = int(val_str)
                            if width in [-9999, 999]: continue
                            all_rings.append({'year': len(all_rings) + 1, 'width': width / 100.0})
                        except ValueError: continue
    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {file_path}")
        return pd.Series(dtype=np.float64)
    except Exception as e:
        print(f"An unexpected error occurred while parsing {file_path}: {e}")
        return pd.Series(dtype=np.float64)

    if not all_rings: return pd.Series(dtype=np.float64)
    df = pd.DataFrame(all_rings).drop_duplicates(subset='year').set_index('year')
    clean_series = df['width'].dropna().sort_index()

    if is_floating:
        clean_series.index = pd.RangeIndex(start=1, stop=len(clean_series) + 1, name='ring_number')
    return clean_series

def calculate_glk(series1: pd.Series, series2: pd.Series) -> float:
    """Calculates the Gleichläufigkeit (percentage of sign agreement)."""
    # Get the year-over-year differences for both series
    diff1 = series1.diff().dropna()
    diff2 = series2.diff().dropna()
    
    # Find the common index between the two difference series
    common_index = diff1.index.intersection(diff2.index)
    if len(common_index) < 2:
        return 0.0

    # Get the aligned differences
    aligned_diff1 = diff1.loc[common_index]
    aligned_diff2 = diff2.loc[common_index]

    # Count the number of times the sign of the difference is the same
    agreements = np.sum(np.sign(aligned_diff1) == np.sign(aligned_diff2))
    
    # Return the percentage
    return (agreements / len(common_index)) * 100

def _build_master_from_rwl_file(file_path: str) -> pd.Series:
    """Builds a proper master chronology from a single RWL file by averaging all cores."""
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
    df = pd.DataFrame(all_rings)
    master_series = df.groupby('year')['width'].mean().dropna().sort_index()
    return master_series

def detrend(series: pd.Series, spline_stiffness_pct: int = 67) -> pd.Series:
    """Detrends a time series using a standard smoothing spline."""
    series = series.dropna()
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
    """Performs cross-dating using a sliding window that correctly handles gaps in data."""
    if sample_series.empty or master_series.empty:
        return {"error": "Input series is empty."}
    if len(sample_series) < min_overlap:
        return {"error": f"Sample length ({len(sample_series)}) is less than min_overlap ({min_overlap})."}

    s_first_year, s_last_year = sample_series.index.min(), sample_series.index.max()
    s_span = s_last_year - s_first_year + 1
    m_start, m_end = master_series.index.min(), master_series.index.max()
    start_offset_year = int(m_start - s_last_year + min_overlap)
    end_offset_year = int(m_end - s_first_year - min_overlap + 2)
    corrs = []

    for end_year_for_sample in range(start_offset_year + s_span, end_offset_year + s_span):
        offset = end_year_for_sample - s_last_year
        sample_index_shifted = sample_series.index + offset
        overlap_index = master_series.index.intersection(sample_index_shifted)
        
        if len(overlap_index) >= min_overlap:
            master_segment = master_series.loc[overlap_index]
            original_sample_years = overlap_index - offset
            sample_segment = sample_series.loc[original_sample_years]
            
            # Create temporary series with a matching index for GLK calculation
            # This is crucial because GLK compares year-over-year.
            glk_sample = pd.Series(sample_segment.values, index=overlap_index)
            glk_master = master_segment

            # Calculate r, t, and now GLK
            r, _ = pearsonr(sample_segment, master_segment)
            t = calculate_t_value(r, len(overlap_index))
            glk = calculate_glk(glk_sample, glk_master)

            corrs.append({
                "end_year": end_year_for_sample,
                "correlation": r,
                "t_value": t,
                "glk": glk,  # <-- ADDED GLK
                "overlap_n": len(overlap_index)
            })

    if not corrs: return {"error": f"No suitable overlap found (min_overlap = {min_overlap} years)."}
    rdf = pd.DataFrame(corrs)
    if rdf['t_value'].isnull().all(): return {"error": "Correlation calculation failed for all overlaps."}
    return {"best_match": rdf.loc[rdf['t_value'].idxmax()].to_dict(), "all_correlations": rdf.set_index('end_year')}

def plot_results(raw_sample, master_detrended, detrended_sample, results, sample_filename, master_filename, reference_is_rwl=False, raw_master=None):
    """Generates an improved 4-panel plot with a highlighted overlap region."""
    print("Generating enhanced diagnostic plot...")
    if "error" in results: print(f"Cannot plot: {results['error']}"); return

    best_match, all_correlations = results['best_match'], results['all_correlations']
    best_end_year = int(best_match['end_year'])
    r_val, t_val, n_val = best_match['correlation'], best_match['t_value'], int(best_match['overlap_n'])
    glk_val = best_match.get('glk', 0.0) # <-- Get the new GLK value
    best_start_year = best_end_year - (detrended_sample.index.max() - detrended_sample.index.min())

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
    aligned_sample = detrended_sample.copy(); aligned_sample.index += (best_end_year - aligned_sample.index.max())
    ax2.plot(master_detrended.index, master_detrended.values, label=f'Reference: {master_label}', color='blue', alpha=0.8)
    ax2.plot(aligned_sample.index, aligned_sample.values, label=f'Sample: {sample_label}', color='red', linestyle='--')
    ax2.axvspan(overlap_start_year, overlap_end_year, color='gray', alpha=0.2, label=f'Overlap Region (n={n_val})')
    ax2.set_xlim(overlap_start_year - 10, overlap_end_year + 10)
    ax2.set_xlabel("Year"); ax2.set_ylabel("Detrended Index"); ax2.set_title(f"2. Aligned Detrended Series (r={r_val:.3f})"); ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    aligned_raw_sample = raw_sample.copy(); aligned_raw_sample.index += (best_end_year - aligned_raw_sample.index.max())
    ax3.plot(aligned_raw_sample.index, aligned_raw_sample.values, label=f'Sample: {sample_label}', color='green')
    if reference_is_rwl and raw_master is not None:
        ax3.plot(raw_master.index, raw_master.values, label=f'Reference: {master_label}', color='black', alpha=0.7)
    else:
        rescaled_master_for_plot = master_detrended * raw_sample.mean(); ax3.plot(rescaled_master_for_plot.index, rescaled_master_for_plot.values, label=f'Ref (scaled): {master_label}', color='black', alpha=0.7)
    ax3.axvspan(overlap_start_year, overlap_end_year, color='gray', alpha=0.2, label=f'Overlap Region (n={n_val})')
    ax3.set_xlim(overlap_start_year - 10, overlap_end_year + 10)
    ax3.set_xlabel("Year"); ax3.set_ylabel("Ring Width (mm)"); ax3.set_title("3. Raw Data Visual Match"); ax3.legend()

    ax4 = plt.subplot(2, 2, 4); ax4.axis('off')
    summary_text = textwrap.dedent(f"""Cross-Dating Report
-----------------------------
Sample File: {sample_label}
Reference File: {master_label}

Most Likely End Year: {best_end_year}
(Sample Start Year: {best_start_year})

Statistics for this Position:
  Correlation (r): {r_val:.4f}
  T-Value: {t_val:.4f}
  Gleichläufigkeit (Glk): {glk_val:.1f}%
  Overlap (n years): {n_val}
    """)
    ax4.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=12, fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", fc='aliceblue', ec='lightsteelblue', lw=2))
    ax4.set_title("4. Summary Statistics")

    plt.suptitle(f"Cross-Dating Analysis: {sample_label} vs. {master_label}", fontsize=16, fontweight='bold'); plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

# --- 3. COMMAND LOGIC FUNCTIONS ---

def get_metadata_from_rwl(file_path):
    """Extracts metadata using the robust parser for the indexer."""
    try:
        series = _build_master_from_rwl_file(file_path) # Use the averaging builder
        if series.empty: return None
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: header_content = "".join([f.readline() for _ in range(5)]).lower()
        species_name = "UNKNOWN"
        if "picea" in header_content or "spruce" in header_content: species_name = "PICEA"
        elif "pinus" in header_content or "pine" in header_content: species_name = "PINUS"
        elif "abies" in header_content or "fir" in header_content: species_name = "ABIES"
        elif "larix" in header_content or "larch" in header_content: species_name = "LARIX"
        return {"species": species_name, "start_year": int(series.index.min()), "end_year": int(series.index.max()), "length": len(series)}
    except Exception: return None

def create_ftp_index(index_filename="noaa_europe_index.csv"):
    """The one-time task. Scans FTP and builds the local index."""
    print("Starting the one-time indexing of the NOAA FTP server."); cache_dir = "full_rwl_cache"; os.makedirs(cache_dir, exist_ok=True)
    try:
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov", timeout=60); ftp.login(); ftp.cwd("/pub/data/paleo/treering/measurements/europe/")
        file_list = [f for f in ftp.nlst() if f.lower().endswith('.rwl')]; print(f"Found {len(file_list)} .rwl files to index.")
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
    pd.DataFrame(all_metadata).to_csv(index_filename, index=False); print(f"\nSUCCESS: Index with {len(all_metadata)} entries created and saved to '{index_filename}'.")

def build_master_from_index(chronology_name, target_species, country_prefixes, min_series_length, min_start_year, index_filename="noaa_europe_index.csv"):
    """Builds a master chronology using the local index."""
    print("\n" + "="*60); print(f"BUILDING: '{chronology_name}'")
    if not os.path.exists(index_filename): print(f"ERROR: Index file missing."); return
    index_df = pd.read_csv(index_filename)
    df_filtered = index_df[(index_df['species'].isin(target_species)) & (index_df['filename'].str.lower().str.startswith(tuple(country_prefixes))) & (index_df['length'] >= min_series_length) & (index_df['start_year'] < min_start_year)]
    file_list = df_filtered['filename'].tolist()
    if not file_list: print("ERROR: No files in index matched criteria."); return
    print(f"Found {len(file_list)} matching files. Processing...")
    all_series = [_build_master_from_rwl_file(os.path.join("full_rwl_cache", filename)) for filename in tqdm(file_list, desc=f"Building {chronology_name}")]
    all_series = [s for s in all_series if not s.empty]
    if not all_series: print("ERROR: Failed to process files."); return
    combined_df = pd.concat(all_series, axis=1)
    master_chronology = combined_df.mean(axis=1); series_count = combined_df.notna().sum(axis=1)
    master_chronology = master_chronology[series_count >= 5].dropna()
    output_filename = f"master_{chronology_name.lower().replace(' ', '_')}.csv"; master_chronology.to_csv(output_filename, header=['value'], index_label='year')
    print(f"\n--- SUCCESS! Saved to '{output_filename}' ---")

def run_create_master(input_folder, output_filename):
    """Creates a new master chronology from a local folder of .rwl files."""
    print("\n" + "="*60); print(f"CREATING CUSTOM MASTER CHRONOLOGY\nSource Folder: '{input_folder}'"); print("="*60)
    if not os.path.isdir(input_folder): print(f"ERROR: The specified folder '{input_folder}' does not exist."); return
    file_list = [f for f in os.listdir(input_folder) if f.lower().endswith('.rwl')]
    if not file_list: print(f"ERROR: No .rwl files were found in '{input_folder}'."); return
    
    print(f"Found {len(file_list)} .rwl files. Processing...")
    
    # Use the robust _build_master_from_rwl_file to correctly handle each file
    all_series = [_build_master_from_rwl_file(os.path.join(input_folder, filename)) for filename in tqdm(file_list, desc="Processing custom files")]
    all_series = [s for s in all_series if not s.empty]
    
    if not all_series: print("ERROR: Failed to process any of the .rwl files in the folder."); return
    
    print(f"\nCombining {len(all_series)} series into a new master...")
    combined_df = pd.concat(all_series, axis=1)
    
    # Calculate the mean and replication count
    master_chronology = combined_df.mean(axis=1)
    series_count = combined_df.notna().sum(axis=1)
    
    # Apply the standard replication filter
    master_chronology = master_chronology[series_count >= 2].dropna()

    # --- NEW FILTERING STEP ADDED HERE ---
    start_year_filter = 1450
    original_start = int(master_chronology.index.min())
    
    # Apply the filter to keep only data from the specified year onwards
    master_chronology = master_chronology[master_chronology.index >= start_year_filter]
    
    print(f"Applying start year filter: Keeping data from {start_year_filter} onwards.")
    if original_start < start_year_filter:
        print(f"(Removed data from {original_start} to {start_year_filter - 1})")
    # --- END OF NEW FILTERING STEP ---

    if not master_chronology.empty:
        if not output_filename.lower().endswith('.csv'): output_filename += ".csv"
        master_chronology.to_csv(output_filename, header=['value'], index_label='year')
        
        print("\n--- SUCCESS! ---")
        print(f"Custom master chronology created.")
        print(f"Final Time Span: {int(master_chronology.index.min())} to {int(master_chronology.index.max())}")
        print(f"Saved to: '{output_filename}'")
        print("="*60)
    else:
        print("\n--- ERROR ---")
        print(f"The chronology was empty after applying the start year filter ({start_year_filter}).")
        print("No .csv file was created.")
def run_date_analysis(sample_file, master_file, min_overlap=50):
    """Performs cross-dating and returns a dictionary of arguments for plotting."""
    if not os.path.exists(sample_file) or not os.path.exists(master_file):
        print("Error: One or both specified files do not exist.")
        return None
    print(f"\n--- Running Cross-Dating Analysis ---\nSample: '{sample_file}'\nReference: '{master_file}'")
    reference_is_rwl = master_file.lower().endswith('.rwl')
    if reference_is_rwl:
        master_chronology = _build_master_from_rwl_file(master_file)
    else:
        master_chronology = pd.read_csv(master_file, index_col='year').squeeze("columns")
    if master_chronology.empty: print("Could not process reference file."); return None
    sample_chronology = _build_master_from_rwl_file(sample_file)
    if sample_chronology.empty: print("Could not process sample .rwl file."); return None
    if min_overlap > len(sample_chronology):
        print(f"\n--- CONFIGURATION ERROR ---\nThe minimum overlap ({min_overlap}) cannot be greater than the sample length ({len(sample_chronology)}).\nPlease reduce the 'Minimum Overlap' setting.")
        return None
    sample_for_detrend = sample_chronology.copy(); sample_for_detrend.index = pd.RangeIndex(start=1, stop=len(sample_for_detrend) + 1)
    master_for_detrend = master_chronology.copy(); master_for_detrend.index = pd.RangeIndex(start=1, stop=len(master_for_detrend) + 1)
    sample_detrended = detrend(sample_for_detrend)
    master_detrended = pd.Series(detrend(master_for_detrend).values, index=master_chronology.index)
    print("Performing sliding window correlation..."); analysis_results = cross_date(sample_detrended, master_detrended, min_overlap=min_overlap)
    if "error" not in analysis_results:
        best = analysis_results['best_match']
        print("\n--- Cross-Dating Complete ---"); print(f"Most Likely End Year: {int(best['end_year'])}")
        print(f"Correlation (r):      {best['correlation']:.4f}"); print(f"T-Value:              {best['t_value']:.4f} (A value > 4.0 is a strong indicator)"); print(f"Overlap (n):          {int(best['overlap_n'])} years")
        sample_raw_for_plot = sample_chronology.copy(); sample_raw_for_plot.index = pd.RangeIndex(start=1, stop=len(sample_raw_for_plot) + 1)
        master_raw_for_plot = master_chronology if reference_is_rwl else None
        sample_with_original_dates = _build_master_from_rwl_file(sample_file)
        if not sample_with_original_dates.empty:
            original_end_year, script_found_year = int(sample_with_original_dates.index.max()), int(best['end_year'])
            print("\n--- Original Date Verification ---"); print(f"The original file had a last ring year of: {original_end_year}")
            if original_end_year == script_found_year: print(f"✅ SUCCESS: The script's finding ({script_found_year}) MATCHES the date in the file!")
            else: print(f"⚠️ NOTE: The script's finding ({script_found_year}) DOES NOT MATCH the file's date.")
        plot_args = {'raw_sample': sample_raw_for_plot, 'master_detrended': master_detrended, 'detrended_sample': sample_detrended, 'results': analysis_results, 'sample_filename': sample_file, 'master_filename': master_file, 'reference_is_rwl': reference_is_rwl, 'raw_master': master_raw_for_plot}
        return plot_args
    else:
        print(f"\nAnalysis failed: {analysis_results['error']}")
        return None

def process_single_file(args):
    """Helper function for multiprocessing. Now correctly unpacks and uses min_overlap."""
    filename, sample_detrended, sample_basename, base_path, min_overlap = args
    if filename == sample_basename: return None
    master_path = os.path.join(base_path, filename)
    master_raw = _build_master_from_rwl_file(master_path)
    if master_raw.empty: return None
    
    # Detrend the reference master with its true year index for this mode
    master_detrended = detrend(master_raw)
    
    analysis_results = cross_date(sample_detrended, master_detrended, min_overlap=min_overlap)
    if "error" in analysis_results: return None
    best_match = analysis_results['best_match']
    if best_match['correlation'] > 0.90: return None
    best_match['source_file'] = filename
    return best_match

def run_detective_analysis(sample_file, target, top_n, min_overlap=80):
    """Performs site-by-site analysis using a predefined category OR a local folder."""
    print(f"\n--- Running Detective Analysis (in parallel) ---\nSample: '{sample_file}' | Target: '{target}'")
    sample_chronology = _build_master_from_rwl_file(sample_file)
    if sample_chronology.empty: print("Could not process sample .rwl file."); return
    if min_overlap > len(sample_chronology):
        print(f"\n--- CONFIGURATION ERROR ---\nThe minimum overlap ({min_overlap}) cannot be greater than the sample length ({len(sample_chronology)}).\nPlease reduce the 'Minimum Overlap' setting in the GUI.")
        return
    sample_for_detrend = sample_chronology.copy(); sample_for_detrend.index = pd.RangeIndex(start=1, stop=len(sample_for_detrend) + 1)
    sample_detrended = detrend(sample_for_detrend)
    file_list, base_path_for_masters = [], ""
    if os.path.isdir(target):
        print(f"INFO: Detected '{target}' as a local folder. Using files from this directory."); base_path_for_masters = target
        file_list = [f for f in os.listdir(target) if f.lower().endswith('.rwl')]
        if not file_list: print(f"ERROR: No .rwl files were found in the folder '{target}'."); return
    else:
        print(f"INFO: Using predefined category '{target}' from the NOAA index."); base_path_for_masters = "full_rwl_cache"
        index_filename = "noaa_europe_index.csv"
        if not os.path.exists(index_filename): print(f"ERROR: Index file '{index_filename}' missing."); return
        category_params = {'alpine': {'species': ['PICEA', 'ABIES'], 'countries': ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 'min_len': 150, 'min_start': 1750}, 'baltic': {'species': ['PINUS', 'PICEA'], 'countries': ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 'min_len': 150, 'min_start': 1750}, 'all': {'species': ['PICEA', 'ABIES', 'PINUS', 'LARIX'], 'countries': ['aust', 'fran', 'germ', 'ital', 'swit', 'slov', 'finl', 'lith', 'norw', 'pola', 'swed'], 'min_len': 100, 'min_start': 1800}}
        if target not in category_params: print(f"ERROR: Target '{target}' is not a valid predefined category or a local folder."); return
        params = category_params[target]; index_df = pd.read_csv(index_filename)
        df_filtered = index_df[(index_df['species'].isin(params['species'])) & (index_df['filename'].str.lower().str.startswith(tuple(params['countries']))) & (index_df['length'] >= params['min_len']) & (index_df['start_year'] < params['min_start'])]
        file_list = df_filtered['filename'].tolist()
        if not file_list: print(f"ERROR: No files in index matched criteria for '{target}'."); return
    sample_basename = os.path.basename(sample_file); tasks = [(filename, sample_detrended, sample_basename, base_path_for_masters, min_overlap) for filename in file_list]
    print(f"Testing against {len(file_list)} sites using {multiprocessing.cpu_count()} CPU cores...")
    with multiprocessing.Pool() as pool:
        results_iterator = pool.imap(process_single_file, tasks)
        all_best_results = [res for res in tqdm(results_iterator, total=len(tasks)) if res is not None]
    if not all_best_results: print("\nAnalysis complete, no correlations found."); return
    results_df = pd.DataFrame(all_best_results); top_results = results_df.sort_values(by='t_value', ascending=False).head(top_n)
    print(f"\n--- Top {top_n} Matching Sites (Sorted by T-Value) ---"); top_results['end_year'] = top_results['end_year'].astype(int); top_results['overlap_n'] = top_results['overlap_n'].astype(int)
    top_results['glk'] = top_results['glk'].round(1) # Format the GLK column
     # Add 'glk' to the list of columns to print
    print(top_results[['end_year', 't_value', 'glk', 'correlation', 'overlap_n', 'source_file']].to_string(index=False))
    year_counts = top_results['end_year'].value_counts()
    if not year_counts.empty:
        most_likely_year, count = year_counts.index[0], year_counts.iloc[0]
        print("\n--- Conclusion ---")
        if count > 1 and top_results['t_value'].iloc[0] > 3.5:
            print(f"A consensus is forming: the year {most_likely_year} appeared {count} times.")
            sample_with_original_dates = _build_master_from_rwl_file(sample_file)
            if not sample_with_original_dates.empty:
                original_end_year, script_found_year = int(sample_with_original_dates.index.max()), int(most_likely_year)
                print(f"The original file had a last ring year of: {original_end_year}")
                if original_end_year == script_found_year: print(f"✅ SUCCESS: The consensus finding ({script_found_year}) MATCHES the date in the file!")
                else: print(f"⚠️ NOTE: The consensus finding ({script_found_year}) DOES NOT MATCH the file's date.")
        else: print("No clear consensus found. The chronology is likely 'floating'.")

# --- 4. MAIN DISPATCHER ---
def main():
    parser = argparse.ArgumentParser(description="Dendrochronology toolkit for instrument analysis (V5.2).", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('index', help="Create a local index of NOAA data (RUN THIS FIRST).")
    parser_build = subparsers.add_parser('build', help="Build master chronologies from the online data index."); parser_build.add_argument('--target', choices=['alpine', 'baltic', 'all'], default='all', help="Which master to build. (Default: all)")
    parser_create = subparsers.add_parser('create', help="Create a custom master from a local folder of .rwl files."); parser_create.add_argument('input_folder', help="Path to the folder containing your .rwl files."); parser_create.add_argument('output_filename', help="Name for the new master .csv file (e.g., 'my_master.csv').")
    parser_date = subparsers.add_parser('date', help='Cross-date a sample against a master or another .rwl file.'); parser_date.add_argument('sample_file', help="Path to your sample .rwl file."); parser_date.add_argument('master_file', help="Path to the reference .csv or .rwl file."); parser_date.add_argument('--min_overlap', type=int, default=50, help="Minimum overlap in years. (Default: 50)")
    parser_detective = subparsers.add_parser('detective', help="Run a sample against ALL individual files in a category or folder."); parser_detective.add_argument('sample_file', help="Path to your sample .rwl file."); parser_detective.add_argument('target', nargs='?', default='alpine', help="Reference: a category ('alpine', 'baltic', 'all') or a folder path. (Default: alpine)"); parser_detective.add_argument('--top_n', type=int, default=10, help="Number of top results to display. (Default: 10)"); parser_detective.add_argument('--min_overlap', type=int, default=80, help="Minimum overlap in years to consider a match. (Default: 80)")
    args = parser.parse_args()

    if args.command == 'index': create_ftp_index()
    elif args.command == 'build':
        if args.target in ['alpine', 'all']: build_master_from_index("Alpine Instrument Wood", ['PICEA', 'ABIES'], ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 150, 1750)
        if args.target in ['baltic', 'all']: build_master_from_index("Baltic Northern Timber", ['PINUS', 'PICEA'], ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 150, 1750)
    elif args.command == 'create': run_create_master(args.input_folder, args.output_filename)
    elif args.command == 'date': run_date_analysis(args.sample_file, args.master_file, args.min_overlap)
    elif args.command == 'detective': run_detective_analysis(args.sample_file, args.target, args.top_n, args.min_overlap)

if __name__ == '__main__':
    main()
