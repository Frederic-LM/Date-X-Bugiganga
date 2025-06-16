# GoGo-Bugiganga: A Cross-Dating Tools Set

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

*GoGo-Bugiganga to jump into X-Dating.*

---


While powerful enough for any dendrochronological task, it is by default tailors to work on wood usualy used in music instruments.

## Features

*   **FTP Indexer (`index`):** Automatically connects to the International Tree-Ring Data Bank (ITRDB) and builds a comprehensive local "map" of all European tree-ring data.
*   **Chrono-Builder (`build`):** Assembles broad, regional master chronologies (like "Alpine" or "Baltic") from the data index.
*   **The Detective (`detective`):** The ultimate search tool. Compares your mystery sample against hundreds of individual site records to find the strongest possible leads.
*   **The Curator (`create`):** Allows you to become the chief inspector. Hand-pick a team of "elite" reference files and forge them into a new, hyper-specific custom master chronology.
*   **The Finalizer (`date`):** Performs the definitive one-to-one cross-dating analysis, providing a statistical verdict and a detailed visual report.

##  A Step-by-Step Workflow

### Step 1: Installation

Before you begin, You'll need to Intall the full Gang: Pandas, Numpy, Scipy, etc.

```bash
pip install pandas numpy scipy matplotlib tqdm
```

### Step 2: Building your DB

The `index` command is your brilliant assistant working overnight to build this indispensable resource. It scans the entire public database and creates a local map.

**This mission, will take 15-30 minutes. It only needs to be run ONCE.**

```bash
>_ python gogo.py index
```
This will create `noaa_europe_index.csv`.  But you won't need to run this command again unless the main database is updated years from now.

### Step 3: The "X-Detective" Work 

Use the `detective`  to compare your sample against hundreds of individual sample to find the best matches.

```bash
# Search for matches within the most likely category (Alpine)
>_ python gogo.py detective your_sample.rwl --category alpine

# Widen the search to the Baltic region
>_ python gogo.py detective your_sample.rwl --category baltic

# Or activate Everything for the widest possible search
>_ python gogo.py detective your_sample.rwl --category all --top_n 20
```
#### Fine-Tuning Your  Search Parameters

Inside the `gogo.py` script, within the `run_detective_analysis` and `build_master_from_index` functions, you will find parameters that act as the "dials" on your equipment. The two most important are:

*   `'min_len'`: **Minimum Sample Length.** This is your "quality control".  It tells the script to ignore any reference files that don't have enough data to be reliable. `min_len: 150` means "Don't bother with any clue that has fewer than 150 rings." A longer sample provides a more unique fingerprint and avoids false leads.

*   `'min_start'`: **The Time-Travel Cut-Off.** This is your historical focus. It tells the script to only consider reference trees that were already alive during a specific era. `min_start: 1750` means "Only show me clues from trees that were already growing *before* the year 1750."

**The Golden Rule:** To find a date for a very old object (e.g., from the 1600s), you must use a **low `min_start` value** (like `1700` or `1650`). This makes your search highly selective, filtering out all the "noise" from younger, irrelevant trees and focusing only on the high-value, old-growth witnesses from the correct time period.

Examine the output table. Look for a **consensus**—do multiple top-ranking files point to the same end year? These are your prime suspects.

### Step 4: The "Curator" Work

You've found a "family" of highly correlated reference sample! Now, activate the "Custom Lib-Maker."

1.  Create a new, empty folder (e.g., `all_that_strad/`).
2.  Go into your `full_rwl_cache/` directory.
3.  Copy the best `.rwl` files you identified in the detective step into `all_that_strad/`.

Now, use the `create` command to forge your own high-precision tool from this elite team.

```bash
>_ python gogo.py create all_that_strad/ all_that_strad.csv
```

### Step 5: The Final Verdict (The "Date" Command)

Use the `date` argument to compare your sample against your custom-built `all_that_strad.csv`. This will provide the most definitive result.

```bash
>_ python gogo.py date your_sample.rwl all_that_strad.csv
```
This command can also be used for a quick one-on-one comparison between any two `.rwl` files.

```bash
>_ python gogo.py date sample_A.rwl sample_B.rwl
```

## Interpreting

The script will report several statistics, but the **T-Value** is usualy the value people look for.

**As a Rule of Thumb:**
*   **t < 3.5:** inconclusive. Keep searching.
*   **3.5 < t < 5.0:** Not so great. Keep searching.
*   **t > 5.0:** Not so bad, if it's the best you find 
*   **t > 7.0:** Meaningfull

The script will also generate a 4-panel diagnostic plot for visual confirmation. 

---
