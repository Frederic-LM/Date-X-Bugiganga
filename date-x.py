date-x.py
# ==============================================================================
#
#  GUI for the Cross-Dating Tool for Historical Instrument Analysis
#
#  Version: 5.3 (UX & Persistence Update)
#
#  This version adds significant user experience improvements:
#  1. Settings Persistence: Last used paths are saved on exit and reloaded on start.
#  2. Shared Sample Path: Selecting a sample on one tab updates it on others.
#
# ==============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import queue
import json

# --- Import all the core functions from your original script ---
try:
    from gogo import (
        create_ftp_index, build_master_from_index, run_create_master,
        run_date_analysis, run_detective_analysis, plot_results
    )
except ImportError:
    messagebox.showerror("Error", "Could not find gogo.py. Make sure it is in the same directory.")
    sys.exit(1)


class TextRedirector(object):
    """A class to redirect stdout to a Tkinter Text widget."""
    def __init__(self, widget): self.widget = widget
    def write(self, str_val):
        self.widget.config(state=tk.NORMAL)
        self.widget.insert(tk.END, str_val)
        self.widget.see(tk.END)
        self.widget.config(state=tk.DISABLED)
        self.widget.update_idletasks()
    def flush(self): pass

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GoGo Dendro-Dating Tool v5.3")
        self.geometry("800x700")
        self.settings_file = "gogo_settings.json"

        self.plot_queue = queue.Queue()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")

        self._create_tabs()
        self._create_log_widget()

        # --- NEW: Load settings on startup and save on exit ---
        self.load_settings()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.check_plot_queue()
        print("Welcome to the GoGo Dendro-Dating Tool GUI!")

    def on_closing(self):
        """Handle window closing: save settings then destroy."""
        print("Saving settings...")
        self.save_settings()
        self.destroy()

    def save_settings(self):
        """Saves current paths from entry widgets to a JSON file."""
        settings = {
            'date_sample': self.date_sample_entry.get(),
            'date_master': self.date_master_entry.get(),
            'detective_sample': self.detective_sample_entry.get(),
            'detective_folder': self.detective_folder_entry.get(),
            'create_folder': self.create_folder_entry.get(),
            'create_output': self.create_output_entry.get()
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=4)

    def load_settings(self):
        """Loads paths from JSON file and populates entry widgets."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.date_sample_entry.insert(0, settings.get('date_sample', ''))
                    self.date_master_entry.insert(0, settings.get('date_master', ''))
                    self.detective_sample_entry.insert(0, settings.get('detective_sample', ''))
                    self.detective_folder_entry.insert(0, settings.get('detective_folder', ''))
                    self.create_folder_entry.insert(0, settings.get('create_folder', ''))
                    self.create_output_entry.insert(0, settings.get('create_output', ''))
        except (json.JSONDecodeError, KeyError):
            print(f"Warning: Could not read '{self.settings_file}'. Using defaults.")

    def check_plot_queue(self):
        try:
            plot_args = self.plot_queue.get_nowait()
            if plot_args: plot_results(**plot_args)
        except queue.Empty: pass
        finally: self.after(100, self.check_plot_queue)

    def _run_in_thread(self, target_func, args, button_to_disable):
        def thread_target():
            if button_to_disable: self.after(0, lambda: button_to_disable.config(state=tk.DISABLED))
            try: target_func(*args)
            except Exception as e:
                print(f"\n--- AN ERROR OCCURRED ---\nError: {e}")
                self.after(0, lambda: messagebox.showerror("Thread Error", f"An error occurred:\n\n{e}"))
            finally:
                if button_to_disable: self.after(0, lambda: button_to_disable.config(state=tk.NORMAL))
        thread = threading.Thread(target=thread_target); thread.daemon = True; thread.start()

    def _create_tabs(self):
        self._create_date_tab()
        self._create_detective_tab()
        self._create_master_tab()
        self._create_index_build_tab()

    def _create_log_widget(self):
        log_frame = ttk.LabelFrame(self, text="Output Log")
        log_frame.pack(pady=10, padx=10, expand=True, fill="both")
        self.log_widget = tk.Text(log_frame, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.log_widget.pack(expand=True, fill="both", padx=5, pady=5)
        sys.stdout = TextRedirector(self.log_widget)

    def _create_date_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="1. Date (1-vs-1)")
        frame = ttk.LabelFrame(tab, text="Direct 1-vs-1 Comparison"); frame.pack(padx=20, pady=20, fill="x")
        ttk.Label(frame, text="Sample File (.rwl):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.date_sample_entry = ttk.Entry(frame, width=60); self.date_sample_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse...", command=self._browse_for_sample_file).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(frame, text="Reference File (.csv/.rwl):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.date_master_entry = ttk.Entry(frame, width=60); self.date_master_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.date_master_entry, is_master=True)).grid(row=1, column=2, padx=5, pady=5)
        ttk.Label(frame, text="Minimum Overlap (years):").grid(row=2, column=0, padx=5, pady=10, sticky="w")
        self.date_min_overlap_spinbox = ttk.Spinbox(frame, from_=30, to=500, increment=10, width=5); self.date_min_overlap_spinbox.set(50)
        self.date_min_overlap_spinbox.grid(row=2, column=1, padx=5, pady=10, sticky="w")
        run_button = ttk.Button(tab, text="Run Date Analysis & Plot", command=self._run_date); run_button.pack(pady=20)
    
    def _create_detective_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="2. Detective (1-vs-Many)")
        frame = ttk.LabelFrame(tab, text="Detective Mode: Compare a sample against many files"); frame.pack(padx=20, pady=20, fill="x")
        ttk.Label(frame, text="Sample File (.rwl):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.detective_sample_entry = ttk.Entry(frame, width=60); self.detective_sample_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse...", command=self._browse_for_sample_file).grid(row=0, column=2, padx=5, pady=5)
        self.detective_target_var = tk.StringVar(value="category")
        ttk.Radiobutton(frame, text="Predefined Category:", variable=self.detective_target_var, value="category").grid(row=1, column=0, padx=5, pady=10, sticky="w")
        self.detective_category_combo = ttk.Combobox(frame, values=['alpine', 'baltic', 'all']); self.detective_category_combo.set('alpine')
        self.detective_category_combo.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        ttk.Radiobutton(frame, text="Local Folder:", variable=self.detective_target_var, value="folder").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.detective_folder_entry = ttk.Entry(frame, width=60); self.detective_folder_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_folder(self.detective_folder_entry)).grid(row=2, column=2, padx=5, pady=5)
        ttk.Label(frame, text="Show Top N Results:").grid(row=3, column=0, padx=5, pady=10, sticky="w")
        self.detective_top_n_spinbox = ttk.Spinbox(frame, from_=1, to=100, width=5); self.detective_top_n_spinbox.set(10)
        self.detective_top_n_spinbox.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        ttk.Label(frame, text="Minimum Overlap (years):").grid(row=4, column=0, padx=5, pady=10, sticky="w")
        self.detective_min_overlap_spinbox = ttk.Spinbox(frame, from_=30, to=200, increment=10, width=5); self.detective_min_overlap_spinbox.set(80)
        self.detective_min_overlap_spinbox.grid(row=4, column=1, padx=5, pady=10, sticky="w")
        run_button = ttk.Button(tab, text="Run Detective Analysis", command=self._run_detective); run_button.pack(pady=20)
    
    def _create_master_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="3. Create Master")
        frame = ttk.LabelFrame(tab, text="Create a Custom Master Chronology from a Folder"); frame.pack(padx=20, pady=20, fill="x")
        ttk.Label(frame, text="Input Folder (.rwl files):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.create_folder_entry = ttk.Entry(frame, width=60); self.create_folder_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_folder(self.create_folder_entry)).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(frame, text="Output Filename (.csv):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.create_output_entry = ttk.Entry(frame, width=60); self.create_output_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Save As...", command=self._save_file_as).grid(row=1, column=2, padx=5, pady=5)
        run_button = ttk.Button(tab, text="Create Master File", command=self._run_create); run_button.pack(pady=20)
        
    def _create_index_build_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="4. Setup (Index/Build)")
        index_frame = ttk.LabelFrame(tab, text="Step 1: Create Local Index (Run once, takes 15-30 mins)"); index_frame.pack(padx=20, pady=20, fill="x")
        ttk.Label(index_frame, text="Downloads all reference files from the NOAA server.").pack(pady=5)
        self.index_button = ttk.Button(index_frame, text="Create / Update Index", command=self._run_index); self.index_button.pack(pady=10)
        build_frame = ttk.LabelFrame(tab, text="Step 2: Build Master Chronologies from Index"); build_frame.pack(padx=20, pady=20, fill="x")
        ttk.Label(build_frame, text="Select a predefined master to build:").grid(row=0, column=0, padx=5, pady=5)
        self.build_target_combo = ttk.Combobox(build_frame, values=['alpine', 'baltic', 'all']); self.build_target_combo.set('all')
        self.build_target_combo.grid(row=0, column=1, padx=5, pady=5)
        self.build_button = ttk.Button(build_frame, text="Build Selected", command=self._run_build); self.build_button.grid(row=0, column=2, padx=10, pady=5)

    def _browse_file(self, entry_widget, is_master=False):
        types = (("RWL files", "*.rwl"), ("All files", "*.*")) if not is_master else (("All files", "*.*"),("CSV files", "*.csv"),("RWL files", "*.rwl"))
        filename = filedialog.askopenfilename(title="Select a file", filetypes=types)
        if filename: entry_widget.delete(0, tk.END); entry_widget.insert(0, filename)

    def _browse_for_sample_file(self):
        """NEW: Special browser that updates all sample fields."""
        filename = filedialog.askopenfilename(title="Select a Sample File", filetypes=(("RWL files", "*.rwl"), ("All files", "*.*")))
        if filename:
            # Update both tabs
            self.date_sample_entry.delete(0, tk.END); self.date_sample_entry.insert(0, filename)
            self.detective_sample_entry.delete(0, tk.END); self.detective_sample_entry.insert(0, filename)

    def _browse_folder(self, entry_widget):
        foldername = filedialog.askdirectory(title="Select a folder")
        if foldername: entry_widget.delete(0, tk.END); entry_widget.insert(0, foldername)

    def _save_file_as(self):
        filename = filedialog.asksaveasfilename(title="Save Master As", defaultextension=".csv", filetypes=(("CSV files", "*.csv"),))
        if filename: self.create_output_entry.delete(0, tk.END); self.create_output_entry.insert(0, filename)

    def _run_date(self):
        sample = self.date_sample_entry.get(); master = self.date_master_entry.get()
        if not sample or not master: messagebox.showerror("Error", "Please select both a sample and a reference file."); return
        min_overlap = int(self.date_min_overlap_spinbox.get())
        run_button = self.notebook.nametowidget(self.notebook.select()).winfo_children()[-1]
        def thread_target():
            self.after(0, lambda: run_button.config(state=tk.DISABLED))
            try:
                plot_args_dict = run_date_analysis(sample, master, min_overlap)
                if plot_args_dict: self.plot_queue.put(plot_args_dict)
            except Exception as e:
                print(f"\n--- AN ERROR OCCURRED ---\nError: {e}")
                self.after(0, lambda: messagebox.showerror("Thread Error", f"An error occurred:\n\n{e}"))
            finally: self.after(0, lambda: run_button.config(state=tk.NORMAL))
        thread = threading.Thread(target=thread_target); thread.daemon = True; thread.start()

    def _run_detective(self):
        sample = self.detective_sample_entry.get()
        if not sample: messagebox.showerror("Error", "Please select a sample file."); return
        if self.detective_target_var.get() == "category": target = self.detective_category_combo.get()
        else: target = self.detective_folder_entry.get()
        if not target: messagebox.showerror("Error", "Please select a target category or folder."); return
        top_n = int(self.detective_top_n_spinbox.get()); min_overlap = int(self.detective_min_overlap_spinbox.get())
        button = self.notebook.nametowidget(self.notebook.select()).winfo_children()[-1]
        self._run_in_thread(run_detective_analysis, (sample, target, top_n, min_overlap), button)

    def _run_create(self):
        folder = self.create_folder_entry.get(); output = self.create_output_entry.get()
        if not folder or not output: messagebox.showerror("Error", "Please select an input folder and an output file."); return
        button = self.notebook.nametowidget(self.notebook.select()).winfo_children()[-1]
        self._run_in_thread(run_create_master, (folder, output), button)

    def _run_index(self):
        self._run_in_thread(create_ftp_index, (), self.index_button)

    def _run_build(self):
        target = self.build_target_combo.get()
        if target in ['alpine', 'all']:
            self._run_in_thread(build_master_from_index, ("Alpine Instrument Wood", ['PICEA', 'ABIES'], ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 150, 1750), self.build_button)
        if target in ['baltic', 'all']:
            self._run_in_thread(build_master_from_index, ("Baltic Northern Timber", ['PINUS', 'PICEA'], ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 150, 1750), self.build_button)

if __name__ == "__main__":
    app = App()
    app.mainloop()
