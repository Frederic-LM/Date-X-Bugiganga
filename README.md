# 🌳 Gogo & Date-X Bugiganga: A Dendro-X-Dating Tool

<div align="center">

*GoGo-Bugiganga to jump into X-Dating.*

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/status-active-brightgreen)

**Professional-grade dendrochronology toolkit for scientific cross-dating of tree-ring measurement series**

[Quick Start](#-quick-start) • [Features](#-core-features) • [GUI Guide](#-gui-guide) • [CLI Reference](#-cli-reference) • [Examples](#-examples)

</div>

---

## 🎯 What is Date-X Bugiganga?

**Date-X Bugiganga** is a comprehensive dendrochronology toolkit that combines cutting-edge statistical methods with user-friendly interfaces to provide accurate tree-ring dating. Whether you're analyzing musical instrument wood, archaeological samples, or building materials, this tool delivers reliable, scientifically-defensible results.

### Why Choose Date-X Bugiganga?

- 🔬 **Scientific Rigor**: Multi-dimensional validation using T-Value, Overlap, and Gleichläufigkeit
- 🚀 **Modern Methods**: Cubic Smoothing Spline detrending superior to traditional approaches  
- 🎨 **Dual Interface**: Choose between intuitive GUI or powerful CLI
- 📊 **Rich Visualizations**: Comprehensive 2x2 plots with narrative interpretations
- 🌍 **Global Database**: Integrated NOAA reference database access
- ⚡ **Batch Processing**: Handle hundreds of samples efficiently

---

## 🚀 Quick Start

### Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/date-x-bugiganga.git
cd date-x-bugiganga

# Install dependencies
pip install pandas numpy matplotlib scipy tqdm

# Launch the GUI
python Date-X.py
```

### Your First Analysis (6 minutes)

1. **Setup Database**: Click `Setup` tab → `Download NOAA Database` → `Build Masters`
2. **Load Sample**: `Date` tab → Browse for your `.rwl` file
3. **Select Master**: Choose `master_alpine_instrument_wood.csv`
4. **Run Analysis**: Click `Date Sample` and view results!

---

## ✨ Core Features

### 🔬 Advanced Science
- **Modern Detrending**: Cubic Smoothing Spline removes biological age trends with superior flexibility
- **Multi-Dimensional Validation**: Prevents false positives through comprehensive statistical analysis
- **Two-Piece Analysis**: Specialized mode for matching and merging separate measurement series

### 🎯 Powerful Analysis Tools
- **Interactive GUI**: Point-and-click interface for single sample analysis
- **Scriptable CLI**: Batch processing and automation capabilities
- **Detective Mode**: Unknown origin samples tested against entire database
- **Custom Masters**: Build chronologies from your own reference collections

### 📊 Rich Output
- **Comprehensive Plots**: 2x2 visual summaries with statistical overlays
- **Narrative Reports**: Full interpretation with match strength classification
- **Export Options**: Save results as images, CSV, or formatted text reports

---

## 🖥️ System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.7+ (3.9+ recommended) |
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
- [Installation & Setup](#installation-and-setup)
- [Quick Start Guide](#quick-start-guide)
- [System Requirements](#system-requirements)

### User Guides
- [🖼️ GUI Guide (`Date-X.py`)](#-gui-guide)
- [⌨️ CLI Reference (`gogo.py`)](#-cli-reference)
- [📖 Examples & Use Cases](#-examples)

### Advanced Topics
- [🔬 Scientific Workflow](#-the-scientific-workflow)
- [🏗️ Building Executables](#-building-executables)
- [🔧 Troubleshooting](#-troubleshooting)

### Reference
- [📚 Methods & References](#-methods--references)
- [🤝 Contributing](#-contributing)
- [📝 Changelog](#-changelog)

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
         │ Sample  │
         │ Origin  │
         │ Known?  │
         └────┬────┘
              │
    ┌─────────┼─────────┐
    │ Yes     │      No │
    ▼         ▼         ▼
┌───────┐ ┌───────────────────┐
│  Date │ │ 🕵️ Detective Mode │
│Against│ │ Test All Sites    │
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
   - **Known origin**: Test against specific master chronology
   - **Unknown origin**: Use Detective mode for geographic identification

4. **📊 Interpret Results**
   - Review comprehensive visual plots
   - Check statistical classification (Very Strong, Significant, etc.)
   - Generate shareable reports

---

## 🖼️ GUI Guide

### Launch the GUI
```bash
python Date-X.py
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
| **Stiffness** | Detrending sensitivity | Standard (67%) for most cases |

### Tab 2: 🕵️ Detective
**Purpose**: Identify unknown sample origins

- **Predefined Categories**: `alpine`, `baltic`
- **Custom Folders**: Use your own reference collections
- **Top N Results**: Display best matches (recommended: 5-10)
- **High Overlap**: Use 80+ years for detective work

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

#### `build` - Create Master Chronologies
```bash
# Build all predefined masters
python gogo.py build

# Build specific target
python gogo.py build --target alpine
python gogo.py build --target baltic
```

### Analysis Commands

#### `date` - Single Master Analysis
```bash
python gogo.py date "sample.rwl" "master_alpine.csv" [options]

# Options:
--min_overlap 60        # Minimum overlap years
--reverse              # Reverse measurement direction  
--stiffness 0.67       # Detrending stiffness (0.5-0.8)
```

#### `detective` - Multi-Master Search
```bash
# Search predefined category
python gogo.py detective "sample.rwl" alpine --top_n 5

# Search local folder
python gogo.py detective "sample.rwl" "/path/to/references/"
```

#### `create` - Custom Master
```bash
python gogo.py create "/path/to/rwl_folder/" "custom_master.csv"
```

---

## 📖 Examples

### Example 1: Dating a Violin Top Plate
```bash
# Two-piece analysis for violin belly (bass and treble sides)
python Date-X.py
# Select "Two-Piece Mean", load both measurement files
# Choose alpine master, check "Reverse" if needed
```

### Example 2: Unknown Origin Investigation
```bash
# CLI approach for unknown sample
python gogo.py detective "mystery_sample.rwl" alpine --top_n 10 --min_overlap 80

# Expected output: Ranked list of best matches with statistics
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
pyinstaller --name "Date-X" --onefile --windowed --icon="icon.ico" Date-X.py
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
- 📧 Contact: your.email@domain.com

---

## 📚 Methods & References

### Statistical Methods

**Detrending**: Cubic Smoothing Spline with 67% cutoff frequency (Cook & Peters, 1981)

**Cross-Dating Validation**:
- **T-Value**: Student's t-test for correlation significance
- **Overlap (n)**: Number of overlapping years
- **Gleichläufigkeit (GLK)**: Sign-test for year-to-year changes

### Classification Thresholds

| Classification | T-Value | Overlap | GLK | Interpretation |
|----------------|---------|---------|-----|----------------|
| **Very Strong** | >6.0 | >80 | >65% | Highly reliable match |
| **Strong** | >4.0 | >60 | >60% | Reliable match |
| **Significant** | >3.0 | >40 | >55% | Likely correct match |
| **Weak** | >2.0 | >30 | >50% | Possible match, verify |
| **No Match** | <2.0 | - | - | No significant correlation |

### Key References

1. **Cook, E.R. & Peters, K. (1981)** *The smoothing spline: a new approach to standardizing forest interior tree-ring width*
2. **Baillie, M.G.L. & Pilcher, J.R. (1973)** *A simple crossdating program for tree-ring research*
3. **Eckstein, D. & Bauch, J. (1969)** *Beitrag zur Rationalisierung eines dendrochronologischen Verfahrens*

---

## 🤝 Contributing

We welcome contributions to improve Date-X Bugiganga! Here's how you can help:

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

[🏠 Home](#-gogo--date-x-bugiganga-a-dendro-x-dating-tool) • [📚 Documentation](#-table-of-contents) • [🐛 Issues](https://github.com/Frederic-LM/Date-XBugiganga/issues) • [💬 Discussions](https://github.com/Frederic-LM/Date-XBugiganga/discussions)

</div>
