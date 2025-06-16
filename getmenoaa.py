# ==============================================================================
#
#  The Cross-Dating Tool for Historical Instrument Analysis
#
#  Version: 3.1 (Final - Robust Parser and Indexer Method)
#
#  OVERVIEW:
#  This script uses a professional, two-stage approach. It corrects the previous
#  parsing errors with a new, highly robust function for reading real-world,
#  messy .rwl files.
#
#  WORKFLOW:
#  STEP 1: Create a Local Index (Run ONCE, takes 15-30 mins)
#     python your_script_name.py index
#
#  STEP 2: Build Master Chronologies from the Local Index (Fast)
#     python your_script_name.py build
#
#  STEP 3: Cross-Date Your Sample (Fast)
#     python your_script_name.py date your_sample.rwl master_alpine_instrument_wood.csv
#
# ==============================================================================

# --- 1. IMPORTS ---
import os
import ftplib
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm # A progress bar library, run: pip install tqdm
# Import other libraries as needed (matplotlib, etc.) when you add back the plotting function

# --- 2. CORE DENDROCHRONOLOGY FUNCTIONS (WITH CORRECTIONS) ---

def parse_rwl_robust(file_path: str) -> pd.Series:
    """
    A new, highly robust parser designed to handle messy, real-world RWL files.
    This is the key fix to the script.
    """
    all_rings = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    # The year is usually the second element
                    year_val = int(parts[1])
                except (ValueError, IndexError):
                    continue

                # Check if it's a valid year or just a random number
                if not (100 < year_val < 2100):
                    continue

                # This is likely a data line. The first value that looks like a year
                # is the start of a decade.
                start_year = (year_val // 10) * 10
                
                # The rest of the parts are potential ring widths
                measurements = parts[2:]
                
                for i, val_str in enumerate(measurements):
                    try:
                        width = int(val_str)
                        if width in [-9999, 999]: # Stop markers
                            continue
                        
                        # Calculate the actual year for this measurement
                        current_year = start_year + i
                        
                        # A sanity check for the year calculation
                        if i == 0 and current_year != year_val:
                            # If the first value doesn't align, there's a format issue.
                            # We'll trust the explicit year and adjust.
                            start_year = year_val
                            current_year = year_val

                        all_rings.append({'year': current_year, 'width': width / 100.0})
                    except ValueError:
                        continue # Skip non-integer values

    except Exception:
        return pd.Series(dtype=np.float64) # Return empty on any major read error

    if not all_rings:
        return pd.Series(dtype=np.float64)

    df = pd.DataFrame(all_rings).drop_duplicates(subset='year').set_index('year')
    return df['width'].sort_index()


def detrend(series: pd.Series, spline_stiffness_pct: int = 67) -> pd.Series:
    if len(series) < 15: return pd.Series(dtype=np.float64)
    x, y = series.index.values, series.values
    s = len(series) * (spline_stiffness_pct / 100)**3
    from scipy.interpolate import UnivariateSpline
    spline = UnivariateSpline(x, y, s=s)
    spline_fit = pd.Series(spline(x), index=x)
    return series / (spline_fit + 1e-6)

# --- 3. INDEXING AND BUILDING LOGIC (Using the new robust parser) ---

def get_metadata_from_rwl(file_path):
    """Quickly extracts metadata using the new ROBUST parser."""
    try:
        # ** THE CRITICAL CHANGE IS HERE **
        series = parse_rwl_robust(file_path)

        if series.empty:
            return None

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            header_content = "".join([f.readline() for _ in range(5)]).lower()
        
        species_name = "UNKNOWN"
        if "picea" in header_content or "spruce" in header_content: species_name = "PICEA"
        elif "pinus" in header_content or "pine" in header_content: species_name = "PINUS"
        elif "abies" in header_content or "fir" in header_content: species_name = "ABIES"
        elif "larix" in header_content or "larch" in header_content: species_name = "LARIX"

        return {
            "species": species_name,
            "start_year": int(series.index.min()),
            "end_year": int(series.index.max()),
            "length": len(series)
        }
    except Exception:
        return None

def create_ftp_index(index_filename="noaa_europe_index.csv"):
    """The one-time task. Scans FTP and builds the local index."""
    print("Starting the one-time indexing of the NOAA FTP server.")
    print("This will take 15-30 minutes but only needs to be run once.")
    
    cache_dir = "full_rwl_cache"
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)

    try:
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov", timeout=60)
        ftp.login()
        ftp.cwd("/pub/data/paleo/treering/measurements/europe/")
        file_list = [f for f in ftp.nlst() if f.lower().endswith('.rwl')]
        print(f"Found {len(file_list)} .rwl files to index.")
    except Exception as e:
        print(f"FATAL: Could not connect to FTP server. {e}"); return

    all_metadata = []
    for filename in tqdm(file_list, desc="Indexing NOAA Files"):
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            try:
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)
            except Exception: continue
        
        metadata = get_metadata_from_rwl(local_path)
        if metadata:
            metadata['filename'] = filename
            all_metadata.append(metadata)
    
    ftp.quit()
    
    if not all_metadata:
        print("\nERROR: No metadata could be extracted. The index could not be built.")
        print("This may be due to a network issue or a fundamental change in the FTP server data.")
        return

    index_df = pd.DataFrame(all_metadata)
    index_df.to_csv(index_filename, index=False)
    print(f"\nSUCCESS: Index with {len(index_df)} entries created and saved to '{index_filename}'.")
    print("You can now run the 'build' command.")


def build_master_from_index(
    chronology_name: str,
    target_species: list,
    country_prefixes: list,
    min_series_length: int,
    min_start_year: int,
    index_filename="noaa_europe_index.csv"
):
    """Builds a master chronology using the pre-built local index."""
    print("\n" + "="*60)
    print(f"BUILDING MASTER CHRONOLOGY: '{chronology_name}'")

    if not os.path.exists(index_filename):
        print(f"ERROR: Index file '{index_filename}' not found."); print("Please run 'python your_script_name.py index' first."); return

    index_df = pd.read_csv(index_filename)
    print("Loaded local index. Filtering for target files...")

    df_filtered = index_df[
        (index_df['species'].isin(target_species)) &
        (index_df['filename'].str.lower().str.startswith(tuple(country_prefixes))) &
        (index_df['length'] >= min_series_length) &
        (index_df['start_year'] < min_start_year)
    ]

    file_list = df_filtered['filename'].tolist()

    if not file_list: print("ERROR: No files in the index matched your criteria."); return
        
    print(f"Found {len(file_list)} matching files in the index. Processing from cache...")
    
    detrended_series_list = []
    cache_dir = "full_rwl_cache"
    for filename in tqdm(file_list, desc=f"Building {chronology_name}"):
        local_path = os.path.join(cache_dir, filename)
        if os.path.exists(local_path):
            series = parse_rwl_robust(local_path) # Use the robust parser
            if not series.empty:
                detrended_series_list.append(detrend(series))

    if not detrended_series_list: print("ERROR: Failed to process any of the filtered files."); return
    
    combined_df = pd.concat(detrended_series_list, axis=1)
    master_chronology = combined_df.mean(axis=1)
    series_count = combined_df.notna().sum(axis=1)
    master_chronology = master_chronology[series_count >= 5].dropna()
    
    output_filename = f"master_{chronology_name.lower().replace(' ', '_')}.csv"
    master_chronology.to_csv(output_filename)
    print(f"\n--- SUCCESS! Saved to '{output_filename}' ---")

# --- 4. COMMAND-LINE INTERFACE AND EXECUTION ---
def main():
    parser = argparse.ArgumentParser(description="Dendrochronology toolkit for instrument analysis (V3.1).")
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('index', help="Create a local index of the NOAA FTP server (RUN THIS FIRST).")
    parser_build = subparsers.add_parser('build', help="Build master chronologies using the local index (fast).")
    parser_build.add_argument('--target', choices=['alpine', 'baltic', 'all'], default='all', help="Which master to build.")
    parser_date = subparsers.add_parser('date', help='Cross-date a sample against a master.')
    parser_date.add_argument('sample_file'); parser_date.add_argument('master_file')
    
    args = parser.parse_args()

    if args.command == 'index':
        create_ftp_index()
    elif args.command == 'build':
        if args.target in ['alpine', 'all']:
            build_master_from_index("Alpine Instrument Wood", ['PICEA', 'ABIES'], ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 250, 1700)
        if args.target in ['baltic', 'all']:
             build_master_from_index("Baltic Northern Timber", ['PINUS', 'PICEA'], ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 200, 1750)
    elif args.command == 'date':
        # This part requires the full code for cross_date and plot_results
        print("Date command placeholder. Add full functions to use.")

if __name__ == '__main__':
    main()
