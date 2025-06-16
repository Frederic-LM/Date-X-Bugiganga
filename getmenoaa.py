# ==============================================================================
#
#  The Cross-Dating Tool for Historical Instrument Analysis
#
#  Version: 3.0 (The Indexer Method - Correct and Robust)
#
#  OVERVIEW:
#  This script uses a professional, two-stage approach to solve the problem of
#  finding the correct reference data on the messy NOAA FTP server.
#
#
#  STEP 1: CREATE A LOCAL INDEX (Run this only ONCE)
#  This new 'index' command does the slow, hard work of scanning the entire
#  NOAA Europe directory. It downloads each file, extracts its metadata (species,
#  years), and saves it to a local file: 'noaa_europe_index.csv'.
#
#  --> THIS COMMAND WILL TAKE 15-30 MINUTES TO RUN. <--
#
#  COMMAND:
#     python your_script_name.py index
#
#  STEP 2: BUILD MASTER CHRONOLOGIES (Now very fast)
#  The 'build' command now uses your local index file. It no longer connects
#  to the internet to search. It reads 'noaa_europe_index.csv', instantly
#  finds the exact files it needs, and builds the masters. This is fast and accurate.
#
#  COMMANDS:
#     python your_script_name.py build
#     python your_script_name.py build --target alpine
#
#  STEP 3: CROSS-DATE YOUR SAMPLE (Unchanged)
#  COMMAND:
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
# (Other imports are the same: textwrap, time, matplotlib, scipy)

# --- 2. CORE DENDROCHRONOLOGY FUNCTIONS ---
# These are correct. For brevity, their code is represented by their signatures.
# In the final file, you will have their full code.
def parse_rwl(file_path: str) -> pd.Series:
    # ... Full function code here ...
    pass
def detrend(series: pd.Series) -> pd.Series:
    # ... Full function code here ...
    pass
def cross_date(sample, master) -> dict:
    # ... Full function code here ...
    pass
def plot_results(raw_sample, master, detrended_sample, results):
    # ... Full function code here ...
    pass

# --- 3. NEW INDEXING AND BUILDING LOGIC (THE CORRECTED PART) ---

def get_metadata_from_rwl(file_path):
    """Quickly extracts species, start/end years from a single RWL file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            header_lines = [f.readline() for _ in range(5)] # Read first 5 lines for species
            f.seek(0) # Rewind to read the whole file for years
            series = parse_rwl(file_path)

        if series.empty:
            return None

        # Find species name - it's usually in the first few lines
        species_name = "UNKNOWN"
        for line in header_lines:
            if "picea" in line.lower() or "spruce" in line.lower():
                species_name = "PICEA"
                break
            if "pinus" in line.lower() or "pine" in line.lower():
                species_name = "PINUS"
                break
            if "abies" in line.lower() or "fir" in line.lower():
                species_name = "ABIES"
                break
            if "larix" in line.lower() or "larch" in line.lower():
                species_name = "LARIX"
                break
        
        return {
            "species": species_name,
            "start_year": series.index.min(),
            "end_year": series.index.max(),
            "length": len(series)
        }
    except Exception:
        return None

def create_ftp_index(index_filename="noaa_europe_index.csv"):
    """
    The slow, one-time task. Scans the entire NOAA Europe FTP directory and
    builds a local CSV index of all .rwl files and their metadata.
    """
    print("Starting the one-time indexing process of the NOAA FTP server.")
    print("This will take a significant amount of time (15-30 minutes). Please be patient.")
    print("A local cache of all files will be created in 'full_rwl_cache'.")

    cache_dir = "full_rwl_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    try:
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov", timeout=60)
        ftp.login()
        ftp.cwd("/pub/data/paleo/treering/measurements/europe/")
        file_list = [f for f in ftp.nlst() if f.lower().endswith('.rwl')]
        print(f"Found {len(file_list)} .rwl files to index.")
    except Exception as e:
        print(f"FATAL: Could not connect or list files on FTP server. {e}")
        return

    all_metadata = []
    for filename in tqdm(file_list, desc="Indexing NOAA Files"):
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            try:
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)
            except Exception:
                continue
        
        metadata = get_metadata_from_rwl(local_path)
        if metadata:
            metadata['filename'] = filename
            all_metadata.append(metadata)
    
    ftp.quit()
    
    if not all_metadata:
        print("ERROR: No metadata could be extracted. The index could not be built.")
        return

    index_df = pd.DataFrame(all_metadata)
    index_df.to_csv(index_filename, index=False)
    print(f"\nSUCCESS: Index created and saved to '{index_filename}'.")
    print("You can now run the 'build' command, which will be very fast.")

def build_master_from_index(
    chronology_name: str,
    target_species: list,
    country_prefixes: list,
    min_series_length: int,
    min_start_year: int,
    index_filename="noaa_europe_index.csv"
):
    """
    Builds a master chronology using the pre-built local index. This is FAST.
    """
    print("\n" + "="*60)
    print(f"BUILDING MASTER CHRONOLOGY: '{chronology_name}'")

    if not os.path.exists(index_filename):
        print(f"ERROR: Index file '{index_filename}' not found.")
        print("Please run the 'index' command first: python your_script_name.py index")
        return

    index_df = pd.read_csv(index_filename)
    print("Loaded local index. Filtering for target files...")

    # Filter the DataFrame to get the list of files we need
    df_filtered = index_df[
        (index_df['species'].isin(target_species)) &
        (index_df['filename'].str.lower().str.startswith(tuple(country_prefixes))) &
        (index_df['length'] >= min_series_length) &
        (index_df['start_year'] < min_start_year)
    ]

    file_list = df_filtered['filename'].tolist()

    if not file_list:
        print("ERROR: No files in the index matched your criteria.")
        return
        
    print(f"Found {len(file_list)} matching files in the index. Processing...")
    
    detrended_series_list = []
    cache_dir = "full_rwl_cache" # The same cache used by the indexer
    for filename in file_list:
        local_path = os.path.join(cache_dir, filename)
        if os.path.exists(local_path):
            series = parse_rwl(local_path)
            if not series.empty:
                detrended_series_list.append(detrend(series))
                print(f"  + Processed '{filename}'")

    if not detrended_series_list:
        print("ERROR: Failed to process any of the filtered files.")
        return
    
    # ... Combination logic is the same ...
    combined_df = pd.concat(detrended_series_list, axis=1)
    master_chronology = combined_df.mean(axis=1)
    series_count = combined_df.notna().sum(axis=1)
    master_chronology = master_chronology[series_count >= 5].dropna()
    
    output_filename = f"master_{chronology_name.lower().replace(' ', '_')}.csv"
    master_chronology.to_csv(output_filename)
    print("\n--- SUCCESS! ---")
    print(f"Saved to: '{output_filename}'")
    print("="*60)


# --- 4. COMMAND-LINE INTERFACE AND EXECUTION ---
def main():
    # ... The argparse setup is the same, but with the new 'index' command ...
    parser = argparse.ArgumentParser(description="Dendrochronology toolkit for instrument analysis (V3).")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # New 'index' command
    subparsers.add_parser('index', help="Create a local index of the NOAA FTP server (RUN THIS FIRST, TAKES A LONG TIME).")

    # Updated 'build' command
    parser_build = subparsers.add_parser('build', help="Build master chronologies using the local index (fast).")
    parser_build.add_argument('--target', choices=['alpine', 'baltic', 'all'], default='all', help="Which master to build.")

    # 'date' command is unchanged
    parser_date = subparsers.add_parser('date', help='Cross-date a sample against a master.')
    parser_date.add_argument('sample_file')
    parser_date.add_argument('master_file')
    
    args = parser.parse_args()

    if args.command == 'index':
        create_ftp_index()
    
    elif args.command == 'build':
        if args.target in ['alpine', 'all']:
            build_master_from_index(
                chronology_name="Alpine Instrument Wood",
                target_species=['PICEA', 'ABIES'], # Use full names
                country_prefixes=['aust', 'fran', 'germ', 'ital', 'swit', 'slov'],
                min_series_length=250,
                min_start_year=1700
            )
        if args.target in ['baltic', 'all']:
             build_master_from_index(
                chronology_name="Baltic Northern Timber",
                target_species=['PINUS', 'PICEA'], # Use full names
                country_prefixes=['finl', 'germ', 'lith', 'norw', 'pola', 'swed'],
                min_series_length=200,
                min_start_year=1750
            )
    
    elif args.command == 'date':
        # ... Dating logic is unchanged ...
        pass

if __name__ == '__main__':
    # Make sure you have the full code for the helper functions
    # before running the main function.
    main()
