# fetch_files.py
import os
import shutil

# --- Configuration ---
SOURCE_DIR = "full_rwl_cache"
DEST_DIR = "violin_references"
MASTER_FILENAME = "master_violin_chronology.csv"

# List of files from the Cybis wiki
VIOLIN_FILES = [
"fran7.rwl","fran039.rwl","swit204.rwl","swit203.rwl","swit189.rwl","swit193.rwl","swit177.rwl","swit169.rwl","swit215.rwl","swit184.rwl","swit173.rwl","swit181.rwl","aust003.rwl","aust007.rwl","germ12.rwl","germ11.rwl","ital007.rwl","ital006.rwl","ital025.rwl","germ036.rwl","germ4.rwl","germ5.rwl","germ14.rwl","germ040.rwl","germ033.rwl","germ020.rwl","czec001.rwl","czec002.rwl","czec3.rwl","czec.rwl","pola022.rwl","pola019.rwl","pola020.rwl","roma002.rwl","roma005.rwl","yugo001.rwl","slov001.rwl","ital022.rwl","swed311.rwl","finl012.rwl","swed312.rwl","swed011.rwl"

]

# --- Main Logic ---
def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        print("Please run 'python gogo.py index' first to download the files.")
        return

    # Create destination directory if it doesn't exist
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Destination folder '{DEST_DIR}' is ready.")

    copied_count = 0
    for filename in VIOLIN_FILES:
        source_path = os.path.join(SOURCE_DIR, filename)
        dest_path = os.path.join(DEST_DIR, filename)

        if os.path.exists(source_path):
            print(f"Copying '{filename}'...")
            shutil.copy2(source_path, dest_path)
            copied_count += 1
        else:
            print(f"Warning: '{filename}' not found in '{SOURCE_DIR}'. Skipping.")
    
    print(f"\n--- Gathering Complete ---")
    print(f"Copied {copied_count} of {len(VIOLIN_FILES)} files to '{DEST_DIR}'.")
    
    if copied_count > 0:
        print("\nNow, to create the master file, run this command:")
        print(f"python gogo.py create {DEST_DIR} {MASTER_FILENAME}")

if __name__ == "__main__":
    main()