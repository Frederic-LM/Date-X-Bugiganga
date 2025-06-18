# Gogo & Date-X Bugiganga: A Dendro-X-Dating Tool
*GoGo-Bugiganga to jump into X-Dating.*

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Date-X Bugiganga** is a professional-grade dendrochronology toolkit designed for the scientific cross-dating of tree-ring measurement series. It combines a powerful command-line interface (CLI) for batch processing and a user-friendly graphical user interface (GUI) for interactive analysis.

The software is built on established scientific principles, using modern detrending methods (Cubic Smoothing Spline) and a robust, multi-dimensional statistical validation approach (T-Value, Overlap, and Gleichläufigkeit) to ensure reliable and defensible results.

## Table of Contents

1.  [Core Features](#core-features)
2.  [The Scientific Workflow](#the-scientific-workflow)
3.  [Software Components](#software-components)
4.  [GUI vs. CLI: Which to Use?](#gui-vs-cli-which-to-use)
5.  [Installation and Setup](#installation-and-setup)
6.  [Usage Guide: The `Date-X.py` GUI](#usage-guide-the-date-xpy-gui)
    *   [Tab 1: Date](#tab-1-date)
    *   [Tab 2: Detective](#tab-2-detective)
    *   [Tab 3: Create Master](#tab-3-create-master)
    *   [Tab 4: Setup](#tab-4-setup)
    *   [Tab 5: Methods & References](#tab-5-methods--references)
7.  [Usage Guide: The `gogo.py` CLI](#usage-guide-the-gogopy-cli)
    *   [`index`](#index)
    *   [`build`](#build)
    *   [`create`](#create)
    *   [`date`](#date-1)
    *   [`detective`](#detective-1)
8.  [Building a Standalone Executable (`.exe`)](#building-a-standalone-executable-exe)

## Core Features

*   **Modern Detrending:** Uses a Cubic Smoothing Spline to remove biological age trends, superior to older methods for its flexibility and accuracy.
*   **Multi-Dimensional Validation:** Classifies match strength using a combination of **T-Value**, **Overlap (n)**, and **Gleichläufigkeit (GLK)**, preventing false positives from statistically-plausible but biologically-unlikely matches.
*   **Interactive GUI (`Date-X.py`):** An intuitive interface for running analyses, visualizing results, and generating reports.
*   **Powerful CLI (`gogo.py`):** A scriptable backend for batch processing, building master chronologies, and managing the reference database.
*   **Comprehensive Plotting:** Generates a detailed 2x2 plot including graphs, key statistics, and a full narrative interpretation for a complete, shareable analysis summary.
*   **Two-Piece Analysis:** Specialized mode to cross-match, validate, and merge two separate measurement series (e.g., the bass and treble sides of a violin belly) into a single, more robust mean chronology before dating.
*   **Reference Database Management:** Tools to download, index, and build regional master chronologies from the NOAA public database.

## The Scientific Workflow

The recommended workflow ensures that your analysis is built upon a solid foundation of validated reference data.

1.  **Step 1: Build the Reference Database (Once)**
    *   Use the `Setup` tab in the GUI or the `gogo.py index` command to download and index standard-format `.rwl` files from the NOAA server. This creates a local cache and an index file (`noaa_europe_index.csv`).
    *   This step is only required once or to update the database.

2.  **Step 2: Create Master Chronologies**
    *   Use the `Setup` tab in the GUI or the `gogo.py build` command to create regional master chronologies (e.g., 'Alpine Instrument Wood') from the indexed files. These masters average many tree-ring series, amplifying the common climate signal and are excellent for initial dating.

3.  **Step 3: Analyze Your Sample**
    *   **For a known origin:** Use the `Date` tab in the GUI or the `gogo.py date` command to run your sample against a specific, relevant master chronology.
    *   **For an unknown origin:** Use the `Detective` tab in the GUI or the `gogo.py detective` command. This runs your sample against *every individual site chronology* in a specified category, helping to pinpoint the most likely geographic origin with high precision.

4.  **Step 4: Interpret Results**
    *   Review the generated plot, which provides a complete visual and narrative summary. The classification ("Very Strong Match", "Significant Match", etc.) is your primary guide to the reliability of the result.
    *   Use the "Save Text Report" button in the GUI to generate a clean, shareable text file of the full analysis.

## Software Components

*   `gogo.py`: The core scientific backend and command-line interface. It contains all the logic for parsing files, performing calculations, and generating plots.
*   `Date-X.py`: The graphical user interface (GUI). It provides a user-friendly front-end for the functions within `gogo.py`.

## GUI vs. CLI: Which to Use?

| Feature / Use Case                  | `Date-X.py` (GUI)                               | `gogo.py` (CLI)                                    |
| ----------------------------------- | ----------------------------------------------- | -------------------------------------------------- |
| **Primary Use**                     | Interactive, visual analysis of single samples. | Batch processing, scripting, and automation.       |
| **Ease of Use**                     | **High.** Point-and-click interface.            | **Medium.** Requires comfort with the command line. |
| **Two-Piece Mean Analysis**         | **Fully supported** with simple radio buttons.  | Not directly exposed; requires manual scripting.   |
| **Advanced Options**                | Easy access to `Reverse` and `Stiffness` flags. | Flags are available but must be typed.             |
| **Building Chronologies**           | Simple buttons in the `Setup` tab.              | Flexible commands for building specific targets.   |
| **Output**                          | Interactive plot window, auto-logged report.    | Plot window, console output.                       |
| **Best For...**                     | Analyzing a new instrument; generating reports. | Re-analyzing 100s of files with new parameters.    |

## Installation and Setup

**Prerequisites:** Python 3.7+

1.  **Clone or download** the repository.
2.  **Install dependencies** from the command line:
    ```bash
    pip install pandas numpy matplotlib scipy tqdm
    ```
3.  **Place the files** `gogo.py` and `Date-X.py` in the same directory.

## Usage Guide: The `Date-X.py` GUI

Run the GUI from your terminal:
```bash
python Date-X.py
```

### Tab 1: Date

*Purpose:* To date a sample against a single, known reference file (e.g., `master_alpine_instrument_wood.csv`).
*   **Analysis Type:** Choose "Single Sample" or "Two-Piece Mean".
*   **File Inputs:** Browse for your sample file(s) and the reference master file.
*   **Options:**
    *   `Reverse`: Check this if your sample was measured from the center joint outwards.
    *   `Minimum Overlap`: Sets the minimum number of years the sample and master must overlap to be considered.
    *   `Detrending Stiffness`: 'Standard (67%)' is suitable for most cases. 'Stiff (80%)' is better for sensitive series with a weak age trend.

### Tab 2: Detective

*Purpose:* To date a sample of unknown origin against a large database of individual site chronologies to find the best match and likely origin.
*   **File Inputs:** Select your sample file(s).
*   **Reference Target:**
    *   `Predefined Category`: Use a category (`alpine`, `baltic`, etc.) built from the NOAA database.
    *   `Local Folder`: Use a local folder containing your own collection of `.rwl` files.
*   **Options:**
    *   `Top N Results`: How many top matches to display in the console log.
    *   `Minimum Overlap`: Higher values (e.g., 80) are recommended for detective work to ensure reliable matches.

### Tab 3: Create Master

*Purpose:* To create your own custom master chronology from a local folder of `.rwl` files.
*   **Input Folder:** The folder containing your measurement files.
*   **Output Filename:** The name for the resulting `.csv` master file.

### Tab 4: Setup

*Purpose:* To manage the reference database.
*   **Step 1:** Download and index the NOAA database. **Run this once.**
*   **Step 2:** Build the predefined master chronologies from the indexed data.

### Tab 5: Methods & References

Provides a detailed explanation of the scientific methods and statistical thresholds used by the software, along with key scientific references.

## Usage Guide: The `gogo.py` CLI

All commands are run from the terminal. Use `-h` for help on any command (e.g., `python gogo.py date -h`).

### `index`
Downloads and indexes the NOAA Europe database.
```bash
python gogo.py index
```

### `build`
Builds master chronologies from the indexed data.
```bash
# Build both alpine and baltic chronologies
python gogo.py build

# Build only the alpine chronology
python gogo.py build --target alpine
```

### `create`
Creates a custom master from a local folder.
```bash
python gogo.py create "path/to/my_rwl_folder" "my_custom_master.csv"
```

### `date`
Dates a sample against a single master.
```bash
python gogo.py date "path/to/my_sample.rwl" "master_alpine_instrument_wood.csv" --min_overlap 60
```

### `detective`
Runs a sample against a category or folder.
```bash
# Against a predefined category
python gogo.py detective "my_sample.rwl" alpine --top_n 5

# Against a local folder
python gogo.py detective "my_sample.rwl" "path/to/my_rwl_folder"
```

## Building a Standalone Executable (`.exe`)

You can package the GUI application into a single `.exe` file for easy distribution on Windows, so users don't need to install Python or any dependencies. The recommended tool is **PyInstaller**.

1.  **Install PyInstaller:**
    ```bash
    pip install pyinstaller
    ```

2.  **Run the PyInstaller command:**
    Open a terminal in the directory containing `Date-X.py` and `gogo.py`. Run the following command:

    ```bash
    pyinstaller --name "Date-X" --onefile --windowed --icon="path/to/your_icon.ico" Date-X.py
    ```

    *   `--name "Date-X"`: Sets the name of the output executable.
    *   `--onefile`: Packages everything into a single `.exe` file.
    *   `--windowed`: Prevents a console window from appearing in the background when you run the GUI.
    *   `--icon="..."`: (Optional) Associates a custom icon with your executable. The icon file must be in `.ico` format.

3.  **Find your executable:**
    PyInstaller will create a few folders. Your final application, `Date-X.exe`, will be inside the `dist` folder. You can copy this file to any other Windows machine and run it directly.
