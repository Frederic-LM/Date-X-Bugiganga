# buildme.py (Version 10.2)
# build_exe.py
import os
import subprocess
import sys
import shutil

# --- Configuration ---
# Name of your virtual environment directory
VENV_DIR = "date_x_venv"

# Main script for the GUI application
MAIN_SCRIPT = "date-x.py"

# Name of the final executable
EXE_NAME = "Date-X_DendroTool"

# Icon file (optional, path to a .ico file)
# If you don't have one, PyInstaller will use a default Python icon.
# You can generate .ico files from .png using online converters.
ICON_FILE = "app_icon.ico" # e.g., create an 'app_icon.ico' in the same directory

# List of Python packages required by your application
# Ensure these are the exact names used when installing with pip
REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "tqdm",
    # tkinter is usually built-in with Python, so no need to list it here
    # PyInstaller usually handles standard library modules automatically
]

# --- Build Process ---
def run_command(command, cwd=None):
    """Executes a shell command and prints its output."""
    print(f"\n--- Running: {' '.join(command) if isinstance(command, list) else command} ---")
    process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
    print(process.stdout)
    if process.stderr:
        print("--- STDERR ---")
        print(process.stderr)
    return process

def create_venv():
    """Creates a virtual environment if it doesn't exist."""
    if os.path.exists(VENV_DIR):
        print(f"Virtual environment '{VENV_DIR}' already exists.")
    else:
        print(f"Creating virtual environment '{VENV_DIR}'...")
        run_command([sys.executable, "-m", "venv", VENV_DIR])
        print("Virtual environment created.")

def get_venv_python_path():
    """Returns the path to the Python executable within the venv."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")

def install_dependencies():
    """Installs required packages into the virtual environment."""
    venv_python = get_venv_python_path()
    
    # Ensure pip is up-to-date in the venv
    run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"])
    
    print(f"Installing dependencies into '{VENV_DIR}'...")
    for package in REQUIRED_PACKAGES:
        try:
            run_command([venv_python, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"ERROR installing {package}: {e.stderr}")
            sys.exit(1)
    
    # Install PyInstaller itself
    try:
        run_command([venv_python, "-m", "pip", "install", "pyinstaller"])
    except subprocess.CalledProcessError as e:
        print(f"ERROR installing PyInstaller: {e.stderr}")
        sys.exit(1)
    print("All dependencies and PyInstaller installed.")

def clean_build_folders():
    """Removes previous build and dist folders for a clean build."""
    print("\n--- Cleaning previous build/dist folders ---")
    if os.path.exists("build"):
        shutil.rmtree("build")
        print("Removed 'build/' directory.")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
        print("Removed 'dist/' directory.")
    if os.path.exists(f"{EXE_NAME}.spec"):
        os.remove(f"{EXE_NAME}.spec")
        print(f"Removed '{EXE_NAME}.spec' file.")

def build_executable():
    """Builds the single-file executable using PyInstaller."""
    venv_python = get_venv_python_path()
    
    print(f"\n--- Building '{EXE_NAME}.exe' ---")
    pyinstaller_command = [
        venv_python,
        "-m",
        "PyInstaller",
        "--noconsole",  # Suppress console window for GUI app
        "--onefile",    # Create a single executable file
        f"--name={EXE_NAME}", # Name of the executable
        MAIN_SCRIPT,
        # Add any other necessary files here.
        # For your application, rwl_cache, tonewood_references, and csvs are runtime data,
        # so they should NOT be bundled into the EXE. The user will manage them externally.
    ]

    if os.path.exists(ICON_FILE):
        pyinstaller_command.append(f"--icon={ICON_FILE}")
        print(f"Using custom icon: {ICON_FILE}")
    else:
        print("No custom icon found. Using PyInstaller default icon.")

    try:
        run_command(pyinstaller_command)
        print(f"\nSUCCESS: Executable created at 'dist/{EXE_NAME}.exe'")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: PyInstaller failed with exit code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        sys.exit(1)

def main():
    print("--- Starting Date-X Executable Build Process ---")
    
    # 1. Create (or ensure) the virtual environment
    create_venv()
    
    # 2. Install dependencies and PyInstaller into the venv
    install_dependencies()
    
    # 3. Clean up previous build artifacts
    clean_build_folders()
    
    # 4. Build the executable
    build_executable()
    
    print("\n--- Build process finished ---")
    print("\nImportant Notes:")
    print(f"1. Your executable is located in the '{os.path.join('dist', EXE_NAME)}.exe' folder.")
    print(f"2. You will need to run the 'Download and Create Index' step from within the GUI (Setup tab)")
    print(f"   to populate the 'full_rwl_cache' and 'noaa_europe_index.csv' files.")
    print(f"3. If you use the 'Tonewood Forest References' feature, run that step from the GUI too,")
    print(f"   which will create the 'tonewood_references' folder and its master CSV.")
    print("4. The executable bundles only the application code. External data files are handled at runtime.")

if __name__ == "__main__":
    main()