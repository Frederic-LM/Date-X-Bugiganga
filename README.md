# 🌳 Gogo & Date-X Bugiganga: A Dendro-X-Dating Tool

<div align="center">

*GoGo-Bugiganga to jump into X-Dating.*

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/status-active-brightgreen)

**An open-source dendrochronology toolkit for cross-dating tree-ring measurement series**

[Quick Start](#-quick-start) • [Features](#-core-features) • [GUI Guide](#-gui-guide) • [CLI Reference](#-cli-reference) • [Examples](#-examples)

</div>

---

## 🎯 What is Date-X Bugiganga?

**Date-X Bugiganga** cross-dates tree-ring series against reference chronologies and reports the result with the information needed to judge it. It features:

- 🔬 **Stated statistics**: T-Value always reported with its overlap, plus Gleichläufigkeit
- 🚀 **Detrending**: Cubic smoothing spline, fixed 32-year cutoff by default, parameters stated in every report
- 📐 **Three indices, all computed by default**: spline, Baillie–Pilcher and Hollstein. The
  end year is reported under each, so the index cannot be shopped for a flattering t
- 🎨 **Dual Interface**: Choose between intuitive GUI or powerful CLI
- 📊 **Rich Visualizations**: Comprehensive 2x2 plots with narrative interpretations
- 🌍 **Public reference data**: Integrated NOAA/ITRDB access — forest chronologies, not a private corpus
- ⚡ **Batch Processing**: Handle hundreds of samples efficiently

---

## 🚀 Quick Start

### Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/Frederic-LM/Date-XBugiganga.git
cd Date-XBugiganga

# Install dependencies
pip install pandas numpy matplotlib scipy tqdm

# Launch the GUI
python date-x.py
```

### Your First Analysis (6 minutes)

1. **Setup Database**: Click `Setup` tab → `Download NOAA Database` → `Build Masters`
2. **Load Sample**: `Date` tab → Browse for your `.rwl` file
3. **Select Master**: Choose `master_alpine_instrument_wood.csv`
4. **Run Analysis**: Click `Date Sample` and view results!

---

## ✨ Core Features

### 🔬 Advanced Science
- **Detrending**: Cubic smoothing spline removes the biological age trend; the cutoff is a fixed 32 years by default, configurable, and recorded in every artefact
- **Overlap reported throughout**: no T-Value is printed without the *n* it was computed over
- **Search context**: reports state how many alignments were tested and how many survived each threshold
- **Two-Piece Analysis**: Specialized mode for matching and merging separate measurement series

### 🎯 Powerful Analysis Tools
- **Interactive GUI**: Point-and-click interface for single sample analysis
- **Scriptable CLI**: Batch processing and automation capabilities
- **Survey Mode**: rank which references a series correlates with, across a whole collection
- **Custom Masters**: Build chronologies from your own reference collections

### 📊 Rich Output
- **Comprehensive Plots**: 2x2 visual summaries with statistical overlays
- **Narrative Reports**: measured values, the tier criteria they meet, r², and who says what that means
- **Export Options**: Save results as images, CSV, or formatted text reports

---

## 🖥️ System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.9+ |
| **OS** | Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+) |
| **RAM** | 4GB minimum, 8GB recommended |
| **CPU** | Multicore for // Processing |
| **Storage** | 100MB free space (for NOAA database) |
| **Display** | 1920x1080 recommended for GUI |

### Supported File Formats
- **Input**: `.rwl` (Tucson format), `.csv` (custom format)
- **Output**: `.png`, `.pdf`, `.csv`, `.txt`

---

## 📋 Table of Contents

### Getting Started
- [Quick Start](#-quick-start)
- [Core Features](#-core-features)
- [System Requirements](#️-system-requirements)

### User Guides
- [🖼️ GUI Guide (`date-x.py`)](#-gui-guide)
- [⌨️ CLI Reference (`gogo.py`)](#-cli-reference)
- [📖 Examples & Use Cases](#-examples)

### Advanced Topics
- [🔬 Scientific Workflow](#-the-scientific-workflow)
- [🏗️ Building Executables](#-building-executables)
- [🔧 Troubleshooting](#-troubleshooting)

### Reference
- [📚 Methods & References](#-methods--references)


---

## 🔬 The Scientific Workflow

```
┌────────────────────────────┐
│ 📥 Download NOAA Database  │
└─────────────┬──────────────┘
              │
┌─────────────▼────────────────┐
│ 🏗️ Build Master Chronologies │
└─────────────┬────────────────┘
              │
         ┌────▼────┐
         │Reference│
         │ chosen  │
         │   ?     │
         └────┬────┘
              │
    ┌─────────┼─────────┐
    │ Yes     │      No │
    ▼         ▼         ▼
┌───────┐ ┌───────────────────┐
│  Date │ │ 🕵️ Survey Mode    │
│Against│ │ Rank All Refs     │
│Master │ └─────────┬─────────┘
└───┬───┘           │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │ 📈 Analyze    │
    │   Results     │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │ 📄 Generate   │
    │   Report      │
    └───────────────┘
```
### Step-by-Step Process

1. **🗄️ Build Reference Database** *(One-time setup)*
   - Download standard `.rwl` files from NOAA server
   - Create local cache and index (`noaa_europe_index.csv`)

2. **🏗️ Create Master Chronologies**
   - Build regional masters (e.g., 'Alpine Instrument Wood')
   - Average multiple series to amplify climate signals

3. **🎯 Analyze Your Sample**
   - **Reference chosen**: test against a specific master chronology
   - **No reference chosen**: use Survey mode to rank correlations across a collection

4. **📊 Interpret Results**
   - Read whether the end year held under all three indices — that comes first
   - Review comprehensive visual plots
   - Read each t-value against the overlap *n* it was computed over, and the tier criteria it meets
   - Read the alignment count before reading the top result
   - Generate shareable reports

---

## 🖼️ GUI Guide

### Launch the GUI
```bash
python date-x.py
```

<!-- Screenshot placeholder -->
> 📸 *[Screenshot of main GUI interface would go here]*

### Tab 1: 📊 Date 
**Purpose**: Date samples against known reference chronologies

| Field | Description | Recommendation |
|-------|-------------|----------------|
| **Analysis Type** | Single Sample or Two-Piece Mean | Use Two-Piece for instrument analysis |
| **Sample File** | Your `.rwl` measurement file | Ensure proper Tucson format |
| **Master File** | Reference chronology | Use relevant regional master |
| **Reverse** | Sample measured center→edge | Check for radial measurements |
| **Min Overlap** | Required overlap years | 60+ for reliable results |
| **Detrending** | Spline cutoff | Fixed 32-year (standard) |

### Tab 2: 🕵️ Survey
**Purpose**: Rank a series against every reference in a collection

- **Predefined Categories**: `violin` (tonewood forest references), `alpine` (Picea/Abies),
  `alpine_pine` (Pinus cembra), `baltic`, `all`
- **Custom Folders**: Use your own reference collections
- **Top N Results**: Display best matches (recommended: 5-10)
- **High Overlap**: Use 80+ years — a large search needs a stronger floor

### Tab 3: 🏗️ Create Master
**Purpose**: Build custom chronologies from local collections

1. Select folder containing `.rwl` files
2. Choose output filename
3. Software automatically processes and averages series

### Tab 4: ⚙️ Setup
**Purpose**: Database management

- **Step 1**: Download NOAA database *(run once)*
- **Step 2**: Build predefined masters
- Monitor progress with built-in progress bars

### Tab 5: 📚 Methods & References
**Purpose**: Scientific methodology and citation information

---

## ⌨️ CLI Reference

### Global Options
```bash
python gogo.py [command] -h  # Help for any command
```

### Database Management

#### `index` - Download NOAA Database
```bash
python gogo.py index
```
*Downloads and indexes the complete NOAA Europe database*

#### `reindex` - Rebuild the index without downloading
```bash
python gogo.py reindex
```
*Re-scans the existing `full_rwl_cache/` and rewrites `noaa_europe_index.csv`. Use this
after upgrading, or whenever the index and the cache may have fallen out of step — it is
minutes rather than the ~15 of a full download.*

#### `violin-setup` - Fetch the tonewood forest references
```bash
python gogo.py violin-setup
```
*Copies the curated ITRDB sites that the instrument-dating literature cites into
`tonewood_references/` and builds `master_tonewood_forest_references.csv` from them.
These are chronologies from **standing trees** — nothing here was measured from an
instrument.*

#### `build` - Create Master Chronologies
```bash
# Build every predefined master (one file each; they are never merged)
python gogo.py build --target every

# Build a specific target
python gogo.py build --target alpine        # Picea/Abies, Alpine countries
python gogo.py build --target alpine_pine   # Pinus cembra, Alpine countries
python gogo.py build --target baltic        # Pinus/Picea, Baltic countries
python gogo.py build --target all           # all four genera, all countries

# Options:
--min_end_year 1500     # Only include reference sites ending after this year
--min_depth 5           # Sites required at a year for it to appear in the master
```

`--min_depth` controls how many contributing sites a year needs before the master
reports an index for it. Raising it shortens the master but strengthens every year it
keeps; lowering it extends the early tail at the cost of depth. The value used is
recorded in the run manifest.

```bash
--exclude_series VIG405 VIG406   # build the master without these series
```

Every master this program builds is written with a `<master>.sources.json` sidecar
listing the files and series IDs it was built from. That record is what makes the
holdout below possible; without it a master `.csv` is an opaque average.

### Holding a series out of its own master

COFECHA (Holmes 1983) removes each series from the master before testing that series
against it, because a series scored against a mean that contains it is scored partly
against itself and cannot fail. `gogo.py create` makes that easy to do by accident —
build a master from a folder, then date one of the folder's own series against it — so
`date` and `detective` now do the holdout themselves:

- The sample's series ID is read from its `.rwl` (the series `parse_as_floating_series`
  would use), or supplied by Pennyscope as `series_id`.
- If that ID is one of the reference's series, the reference is **rebuilt without it**.
  The biweight mean and the depth per year are both recomputed from the series that
  remain — neither can be obtained by subtracting the held-out series from the finished
  master, because a biweight mean is not linear.
- `run_manifest` records `held_out_series`, `holdout_performed`, the year the depths refer
  to, and `depth_before_holdout` / `depth_after_holdout` (counted in sites for a master
  `.csv`, in measured series for a single `.rwl` site — `holdout_depth_basis` says which).
- **Every report says what happened, in one sentence, before the result.** Where the
  reference is a prebuilt `.csv` with no source record, the sentence states that the check
  **could not be performed** and that the sample being inside its own reference cannot be
  ruled out. There is no silent case.
- If the holdout drops the depth at the reported year below the minimum depth floor, the
  result is reported as **unusable** rather than as a slightly weaker date. In a detective
  search, a reference that falls below the floor after the holdout is named and dropped
  from the ranking rather than quietly ranked.

Holding a series out lowers the t-value — that is the point, and a test asserts it. A
t-value obtained without the holdout is not comparable to one obtained with it.

### Analysis Commands

#### `date` - Single Master Analysis
```bash
python gogo.py date "sample.rwl" "master_alpine.csv" [options]

# Options:
--min_overlap 60            # Minimum overlap in years (default: 60)
--stability                 # Re-date under several filter lengths AND all three indices
--index bp                  # restrict to one index; default computes all three
--detrend_mode fixed        # 'fixed' (default) or 'percent'
--detrend_wavelength 32     # Spline cutoff in years when --detrend_mode=fixed
--stiffness 67              # Spline cutoff as % of length, only when --detrend_mode=percent
```

#### `detective` - Multi-Master Search *(Survey mode)*
```bash
# Search predefined category
python gogo.py detective "sample.rwl" alpine --top_n 5

# Search local folder
python gogo.py detective "sample.rwl" "/path/to/references/"

# Options:
--min_country_series 5      # below this, a country's best t is flagged too thin (default: 5)
```

##### What was searched, per country prefix

After the ranked table, detective mode prints one row per ITRDB country prefix in the
reference set:

| Column | Meaning |
|--------|---------|
| `country` | ITRDB country prefix of the reference filenames |
| `n_sites` | sites of that prefix in the set |
| `n_series` | measured series of that prefix in the set |
| `best_t` | highest t obtained anywhere in that prefix |
| `overlap_n` | the overlap **that** t was computed over |
| `best_site` | the site file that produced it |
| `too_thin` | `yes` when `n_series` is below `--min_country_series` |

Sorted by `best_t` descending. **Every country in the set gets a row, including those
where nothing matched** — a `-` means no alignment there met the overlap floor. This is
what makes an absence readable: "no Austrian match" means one thing when the set holds
112 Austrian series and something else entirely when it holds four, and until the counts
are printed beside the statistic the reader cannot tell which.

A row flagged `too_thin` supports no reading in **either** direction: not a match, and not
the absence of one.

The block describes the reference set that was searched. It is not a ranking of candidate
origins, and it names none — see the note on origin under Methods below. The same columns
are written to `<report>_country_coverage.csv` from the GUI, to
`<prefix>_country_coverage.csv` by `batch.py`, and returned as `country_coverage` in
`finotserv.py`'s JSON. A merged master `.csv` carries no per-country breakdown; batch
mode states that rather than printing a table it cannot fill.

#### `create` - Custom Master
```bash
python gogo.py create "/path/to/rwl_folder/" "custom_master.csv" --min_depth 3
```

---

## 🧰 The Other Tools

### `brain.py` — the engine

Reading Tucson files, the three indices, the biweight site chronologies, cross-dating, the
holdout, the tier criteria and the reporting helpers all live in `brain.py`. `gogo.py` keeps
the searches, the plots and the command line, and re-exports everything from `brain`, so
`from gogo import ...` keeps working and `gogo.__version__` is still the single version.
`brain.py` imports nothing from `gogo` and can be used on its own.

### `batch.py` — batch dating and auto cross-match

Dates a whole folder of measurements against one reference, then cross-matches every
series against every other.

```bash
python batch.py ./measurements/ \
    --master tonewood_references/master_tonewood_forest_references.csv \
    --min-overlap 60 --detrend-mode fixed --detrend-wavelength 32 \
    --xmatch-overlap 60 --out-prefix batch
```

Reads Tucson `.rwl` exports and Pennyscope `.dendro.json` projects. Writes
`batch_dating.csv`, `batch_crossmatch.csv` and `batch_run.json`. Series shorter than the
overlap floor are listed as **excluded**, not quietly dated against a smaller overlap —
t-values computed over different overlaps are not comparable to each other.

`--no-stability` skips the per-series stability check if you want speed over information.

> Cross-match results must **never** be folded back into a reference master. Averaging
> series *because* they cross-matched builds a chronology out of the similarity you then
> go on to measure, and everything you test against it afterwards will match.

### `finotserv.py` — local server for Pennyscope

```bash
python finotserv.py            # serves http://localhost:5174
PENNYSCOPE_PORT=5199 python finotserv.py   # or pick another port
```

Bridges the `ring-measurer.html` browser measuring tool (Pennyscope) to the engine.
Serves the measuring UI, runs analyses on the measurements it sends, and returns results
as JSON including the run manifest. It reports `gogo.__version__`, so the browser can
never advertise a version the engine has moved past.

### `fetch.py`

Small helper that gathers the curated tonewood site files. `gogo.py violin-setup` is the
supported route; this exists for scripted use.

---

## 📖 Examples

### Example 1: Dating a Violin Top Plate
```bash
# Two-piece analysis for violin belly (bass and treble sides)
python date-x.py
# Select "Two-Piece Mean", load both measurement files
# Choose alpine master, check "Reverse" if needed
```

### Example 2: Surveying a Whole Collection
```bash
# CLI approach when no single reference has been chosen
python gogo.py detective "mystery_sample.rwl" alpine --top_n 10 --min_overlap 80

# Output: ranked correlations with overlap and alignment count.
# A ranking is not an origin — see Methods & References below.
```

### Example 3: Building Custom Regional Master
```bash
# Create master from local collection
python gogo.py create "./my_oak_collection/" "regional_oak_master.csv"
```

### Example 4: Batch Processing Multiple Samples
```bash
# Process multiple files (requires simple script)
for file in *.rwl; do
    python gogo.py date "$file" "master_alpine.csv" --min_overlap 60
done
```

---

## 🏗️ Building Executables

Create standalone `.exe` files for easy distribution without Python dependencies.

### Prerequisites
```bash
pip install pyinstaller
```

### Build Command
```bash
pyinstaller --name "Date-X" --onefile --windowed --icon="icon.ico" date-x.py
```

### Options Explained
- `--onefile`: Single executable file
- `--windowed`: No console window (GUI only)
- `--icon`: Custom application icon
- `--name`: Output executable name

### Distribution
- Executable located in `dist/` folder
- Fully portable - no installation required
- ~50-100MB file size (includes Python runtime)

---

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems
**Issue**: `ModuleNotFoundError: No module named 'pandas'`
```bash
# Solution: Install dependencies
pip install pandas numpy matplotlib scipy tqdm
```

#### File Format Errors
**Issue**: "Cannot parse .rwl file"
```bash
# Check file format:
# - Must be standard Tucson format
# - Headers should contain site codes
# - Measurements in 0.01mm units
```

#### Memory Issues
**Issue**: "Memory error during large detective search"
```bash
# Solutions:
# 1. Increase minimum overlap (reduces comparisons)
# 2. Use smaller reference categories
# 3. Process in smaller batches
```

#### GUI Won't Start
**Issue**: Tkinter-related errors
```bash
# Linux users may need:
sudo apt-get install python3-tk

# macOS users with Homebrew:
brew install python-tk
```

### Performance Notes
- **Database download**: ~10-15 minutes (one-time)
- **Master building**: ~2-5 minutes per category
- **Single analysis**: ~1-5 seconds
- **Detective mode**: ~30 seconds to 5 minutes (depends on database size)

### Getting Help
- 📖 Check the [Methods & References](#-methods--references) tab in GUI
- 🐛 Report bugs via GitHub Issues
- 💡 Feature requests welcome
- 📧 Contact: via GitHub Issues

---

## 📚 Methods & References

### Statistical Methods

### The three indices — all computed by default

A t-value is a property of the index it was computed on. This tool computes **all three by
default** and reports the end year under each, so the index cannot be chosen after seeing
which one gives the better t. `--index bp` (or `hollstein`, or `spline`) restricts a run to
one, for speed and for scripting; the setting is recorded in the run manifest either way.

`--index` on `date` restricts the run; the lead index — the one that fills the single
`t_value` / `overlap_n` pair a CSV row or a headline can hold — is **Baillie–Pilcher**, because
that is the index the laboratory reports a reader is holding were computed on. Every row and
every manifest names the index its numbers came from, and `build` / `create` still standardise
a master's `value` column with the spline.

| index | Definition | Rings lost (start + end) |
|-------|------------|--------------------------|
| `spline` | ratio to a cubic smoothing spline fit | 0 + 0 |
| `hollstein` | `d(y) = ln( w(y) / w(y-1) )` | 1 + 0 |
| `bp` | `d(y) = ln( 5·w(y) / (w(y-2)+w(y-1)+w(y)+w(y+1)+w(y+2)) )` | 2 + 2 |

Both log indices are computed on **raw ring widths**, not on spline indices: they replace
the spline, they do not follow it. Under `bp` and `hollstein` the spline cutoff setting is
not used at all, and the report says so rather than quoting a cutoff that was not applied.

**Agreement leads every report.** All three indices giving the same end year *is* the
finding, and it is the first sentence: the year survived the parameter choice. Disagreement
leads harder, in plain words at the top — a date that moves with the index was produced
partly by the index. Where only one index could be computed (a prebuilt master under another
index), the report says the year has **not** been tested, rather than calling that agreement.

**The three t-values are not on a common scale.** Every t prints with its own *n*, and every
report carries one line saying so: t-values from different indices must not be compared with
one another, and no threshold carries from one index to another.

In `detective` mode the **full search runs under all three**. Re-dating a single winner under
the other two would test whether the *year* holds; it cannot test whether the *winner* holds,
and if spline ranks one chronology first while Hollstein ranks another, that is the finding —
searching under one index guarantees it never surfaces. So the report carries:

- the top candidate list **per index**;
- whether the same reference ranks first under all three, stated plainly at the top;
- where the winner differs by index, that leads the report alongside the end-year case: the
  ranking depends on the index, so no single best-matching reference can be reported.

`--search-index {bp,hollstein,spline}` restricts the sweep to one for speed. It is not the
default, and when used the report states that the cross-index comparison was not performed.

**Rankings from different indices are not on a common scale.** Compare which reference each
index puts first, not the t-values against each other.

- **The same transform is applied to sample and reference, always.** Two series carrying
  different index settings are **refused**, not warned about: `cross_date()` raises.
- **One master serves all three indices.** A master `.csv` stores the mean **raw** ring width
  per year (`raw_mean`) beside its standardised `value`, so the log indices are derived from
  the raw widths at comparison time rather than being unavailable. `value` holds the index
  named in the file's `# key: value` header — built by standardising each series and taking a
  biweight mean, which is a step better than standardising a mean — so a run under that index
  uses `value` and a run under another derives it from `raw_mean`; `reference_metadata` says
  which. Only a prebuilt file with **no** `raw_mean` column is limited to one index, and such a
  file is refused for the others with a message naming the index it was built under.
- **A small constant is added before the log** (`index_log_epsilon`, default 0.001 mm) so a
  locally absent ring of width 0 gives a finite number instead of `-inf`. The constant is
  recorded in the run manifest.
- **Edge losses are real rings, and `overlap_n` counts survivors.** `bp` loses two rings at
  each end, `hollstein` one at the start. `overlap_n` is the number of *compared* rings, not
  the number measured, because `r² = t²/(t²+n−2)` and the shared-variance figures in the
  table below are wrong by up to four years' worth of *n* otherwise. Reports print the
  measured count and the surviving count side by side. The reported end year is always the
  date of the **last measured ring**, under every index.
- **A series too short to survive the transform is excluded with a stated reason**, exactly
  as a sub-overlap series is — never truncated quietly into a smaller comparison.

> ⚠️ **The published violin-report thresholds — 3.5 as a floor, 6 as significant, 8–10 for
> same-tree — were calibrated on the Baillie–Pilcher and Hollstein indices.** A t produced
> here on the `spline` index is therefore **not directly comparable** to a t on a lab report;
> under `bp` or `hollstein` the convention applies directly. Even then, reference set, overlap
> and detrending differ between laboratories, so compare t-values only alongside the index,
> the overlap and the reference they were computed against. See *Tier criteria* below.

**Detrending**: Cubic smoothing spline (Cook & Peters, 1981), implemented as an order-2 Whittaker-Henderson smoother. **The default is a fixed 32-year cutoff**, applied identically to sample and reference so both are filtered in the same frequency band. A percentage-of-length cutoff is available as `--detrend_mode percent` (default 67%), but is not the default: sample and reference differ in length, so a proportional cutoff filters them in *different* bands — an 80-ring plate at 67% gets a 54-year cutoff while a 450-year reference gets a 300-year one. **Autocorrelation is not removed** — series are standard-index, not residual.

**Cross-Dating Validation**:
- **T-Value**: Student's *t* from the Pearson correlation of the two detrended series, df = overlap − 2
- **Overlap (n)**: Number of overlapping years — required to interpret the T-Value
- **Gleichläufigkeit (GLK)**: Sign-test for year-to-year changes (Eckstein & Bauch, 1969).
  **Ties are counted strictly**: a year where both series are flat counts as agreement, a year
  where only one is flat counts as disagreement. Part of the literature scores a single flat
  step as a half-agreement instead (*semi-Gleichläufigkeit*, after Huber 1943); this tool does
  not, because the tiers below were set against the strict rule and switching would move every
  reported GLK. A test pins the convention.

> ⚠️ **The T-Value is not a probability.** T-values of this family do not follow Student's *t*-distribution, because consecutive rings are autocorrelated and degrees of freedom are overstated (Baillie, 1982). Significance for oak had to be established empirically rather than theoretically (Fowler & Bridge, 2017). Read it as an index of similarity, not as odds.

**Converting to shared variation**: `r² = t² / (t² + n − 2)` — exact, and dependent on overlap as well as T. The same T-Value describes a weaker relationship on a longer series.

### Tier criteria

Tiers are named by the criteria they apply, not by a quality adjective. An adjective is an
unattributed judgement wearing the authority of the measurement, and it is quotable out of
context — "tier t7/n80/g70" is not. `_classify_dendro_match()` returns the tier's criteria and
the observed values; nothing in this tool applies a word like "strong" to a result.

All three conditions must hold for a tier. The tiers are **conventions from the literature,
not derived constants**, and published thresholds vary between laboratories.

| Tier | T-Value | Overlap | GLK | Shared var.* | Conventionally read as |
|------|---------|---------|-----|--------------|------------------------|
| `t7/n80/g70` | ≥7.0 | ≥80 | ≥70% | ~39% | a strong dating candidate, to be checked against competing alignments |
| `t6/n70/g65` | ≥6.0 | ≥70 | ≥65% | ~35% | plausible, but in need of independent corroboration |
| `t5/n50/g60` | ≥5.0 | ≥50 | ≥60% | ~34% | weak, and meaningful only with independent support |
| no tier met | – | – | – | – | not conventionally treated as evidential on its own |

<sub>* r² = t²/(t²+n−2), evaluated at each tier's minimum T and minimum overlap</sub>

> A test in `tests/test_reporting.py` parses this table's tier definitions and compares them
> against `gogo.CLASSIFICATION_TIERS`, and fails if the two disagree. If you change one,
> change both.

Every place a result is printed states: the measured values with their overlap, which tier's
criteria were met (as the criteria: `t>=7.0, n>=80, GLK>=70`), the shared year-to-year
variation `r² = t²/(t²+n−2)`, and a sentence attributing the reading to the literature rather
than asserting it. The bottom tier gets the same treatment with no verdict noun: the
alignment is not carrying weight by itself, which is not the same as the instrument failing.

**The published violin-report thresholds are not these tiers.** The values commonly quoted on
instrument reports — 3.5 as a floor, 6 as significant, 8–10 for same-tree — were calibrated on
the **Baillie–Pilcher and Hollstein** indices. A t computed here on the `spline` index is not
directly comparable to a t on such a report; under `bp` or `hollstein` the convention applies
directly, and the reports say which case applies.

**Reading the tiers:**
- These describe correspondence between **ring sequences**. They say nothing about maker, workshop or region.
- Short overlaps flatter the statistic — below ~50 rings, treat every tier as one step weaker.
- No tier is interpretable without the alignment count. See the `SEARCH CONTEXT` block in generated reports.
- Visual comparison of growth curves remains necessary. This software does not perform it.

> ⚠️ **On origin.** A high correlation with a reference from a given region does not place the wood there. The signal that makes cross-dating work is the shared regional climate signal, which is by construction similar across a whole region; the site-specific component that would carry origin is small and confounded with individual-tree variation. Survey mode ranks *correlations with references*, which is a different thing from provenance. Establishing origin requires purpose-built local or elevation-specific chronologies and usually a second line of evidence.

### What the output records

A date is only a result if someone else can arrive at it again. Every artefact — CSV,
text report, JSON response — carries the following.

**1. Run provenance.** Tool version, UTC timestamp, reference file name **and the first
12 characters of its SHA-256**, the index and its log constant, detrend mode, wavelength,
stiffness, minimum overlap, minimum depth, the series held out of the reference (with the
depth before and after), and the number of series in the run. The hash matters because masters get
rebuilt: a filename alone does not identify what you actually measured against. CSVs get
these as leading `# key: value` lines *and* a `<prefix>_run.json` sidecar.

**2. Reference-set composition.** Number of sites and series, counts per ITRDB country
prefix, year span, and depth at each year — printed **before** any ranked table. A "best
match" is uninterpretable without knowing what it could have matched: a reference set
that is 80% Alpine returns an Alpine best match for almost any Alpine-climate sample,
because that is simply where the chances are. If one country prefix exceeds half the
set, the summary says so in as many words. Where a result is reported for a given end
year, the master's **depth at that year** is reported beside it.

**3. Best-of-many control.** `batch_dating.csv` reports `stands_out_sd` (how many
standard deviations the winning alignment sits above the population of alignments tested)
and `second_best_t`. `batch_crossmatch.csv` now carries the same, plus
`alignments_tested`, and reports how many pairs were compared — a crossmatch over *n*
series is `n(n−1)/2` pairs × every offset, and the highest t in such a search was chosen
*because* it was highest. Cross-match rows are ranked by `stands_out_sd` as well as by t.

**4. Stability under the detrending setting and the index.** `--stability` on `gogo.py date`
(always on in the GUI report and in `batch_dating.csv` as `end_year_stable`) re-dates the
winning candidate under several filter lengths — fixed 20, 32, 64 years and percent-67 — and
reports whether the end year holds. Filter length has documented effects on cross-dating
statistics (Holmes 1983) and no cutoff is canonical, so exposing the setting without
testing sensitivity to it invites parameter shopping: try cutoffs until the t-value looks
good, publish that one. A date that holds its year under every filter is stable; one that
moves was produced by a parameter choice and is reported as such.

`end_year_stable_across_index` (`yes` / `no` / `not tested`) reports the index dimension, in
`batch_dating.csv` and in the GUI report. Since every run already dates under all three
indices, that column comes from those runs; `stability_check()` can also test the index
dimension itself (`index_list`), and does so by default when called on its own.

`not tested` is a real answer, not a failure: a prebuilt master under one index cannot be
re-expressed under another, so only one index could be computed and the year was never
compared. Build the reference under each index, or date against a `.rwl` reference, which can
be standardised any way. Under `bp` and `hollstein` the *filter* rows are identical by
construction — a log index has no filter parameter — and the summary says so instead of
counting it as four passed checks.

Every row of the stability table carries its own `overlap_n`, because the indices lose
different numbers of rings at the series ends and the *n* is not the same from row to row.


### Key References

**Methods implemented**
1. **Cook, E.R. & Peters, K. (1981)** — The smoothing spline: a new approach to standardizing forest interior tree-ring width series. *Tree-Ring Bulletin* 41, 45–53.
2. **Baillie, M.G.L. & Pilcher, J.R. (1973)** — A simple cross-dating program for tree-ring research. *Tree-Ring Bulletin* 33, 7–14.
3. **Eckstein, D. & Bauch, J. (1969)** — Beitrag zur Rationalisierung eines dendrochronologischen Verfahrens. *Forstwissenschaftliches Centralblatt* 88, 230–250.



---

## 🧪 Tests

```bash
python tests/test_dating_pipeline.py
python tests/test_location_parsing.py
python tests/test_reporting.py
```

`test_reporting.py` fails if this README and the code disagree: the tier table is parsed and
compared against `brain.CLASSIFICATION_TIERS`, and the documented indices, edge losses and
flags are checked against the module.

---

## 🤝 Contributing

Contributions are welcome. Here's how you can help:

### 🐛 Bug Reports
- Use GitHub Issues with detailed descriptions
- Include sample files if possible
- Specify OS and Python version

### 💡 Feature Requests  
- Describe use case and expected behavior
- Consider scientific validity and user needs
- Check existing issues first

### 🔧 Code Contributions
- Fork the repository
- Create feature branch (`git checkout -b feature/amazing-feature`)
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Submit pull request with clear description

### 📖 Documentation
- Improve README clarity
- Add examples and use cases
- Translate to other languages
- Create video tutorials

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### What this means:
- ✅ Commercial use allowed
- ✅ Modification and distribution permitted
- ✅ Patent use granted
- ❗ Must disclose source code
- ❗ Must include license and copyright notice
- ❗ Derivative works must use same license

---

## 🙏 Acknowledgments

- **NOAA Paleoclimatology Database** for providing open access to tree-ring data
- **International Tree-Ring Data Bank (ITRDB)** for standardized data formats
- **Scientific Community** for decades of dendrochronological research


---

<div align="center">

**Made with 🌳 for the community**

⭐ Star this repository if you find it useful!

[🏠 Home](#-gogo--date-x-bugiganga-a-dendro-x-dating-tool) • [📚 Documentation](#-table-of-contents) • [🐛 Issues](https://github.com/Frederic-LM/Date-XBugiganga/issues) 

</div>
