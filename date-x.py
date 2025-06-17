# gogo_gui.py (Version 8.4 - Sample Reversal Feature)
# ==============================================================================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, sys, threading, queue, json, textwrap
from datetime import datetime

try:
    from gogo import (
        download_and_index_files, build_master_from_index, run_create_master,
        run_date_analysis, run_detective_analysis, plot_results,
        run_two_piece_mean_analysis, parse_as_floating_series
    )
except ImportError:
    messagebox.showerror("Error", "Could not find gogo.py. Please ensure it's in the same directory.")
    sys.exit(1)

class TextRedirector:
    def __init__(self, widget): self.widget = widget
    def write(self, str_val): self.widget.config(state=tk.NORMAL); self.widget.insert(tk.END, str_val); self.widget.see(tk.END); self.widget.config(state=tk.DISABLED); self.widget.update_idletasks()
    def flush(self): pass

class App(tk.Tk):
    DEFAULT_OVERLAP_PERCENTAGE = 0.8

    def __init__(self):
        super().__init__()
        self.title("GoGo Dendro-Dating Tool v8.4")
        self.geometry("850x800") # Slightly wider for new checkboxes
        self.settings_file = "gogo_settings.json"
        
        self.last_analysis_results = None
        self.plot_queue = queue.Queue()
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=5, padx=10, expand=True, fill="both")
        self._create_tabs()
        self._create_report_widget()
        self._create_log_widget()

        self.load_settings()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_plot_queue()
        print("Welcome! Ready for analysis.")
    # ... (on_closing, save_settings, load_settings, check_plot_queue, _run_in_thread, etc. are unchanged)...
    def on_closing(self):
        print("Saving settings..."); self.save_settings(); self.destroy()

    def save_settings(self):
        settings = {'date_sample': self.date_sample_entry.get(), 'date_master': self.date_master_entry.get(), 'treble_file': self.date_treble_entry.get(), 'detective_sample': self.detective_sample_entry.get(), 'detective_treble': self.detective_treble_entry.get(), 'detective_folder': self.detective_folder_entry.get(), 'create_folder': self.create_folder_entry.get(), 'create_output': self.create_output_entry.get()}
        with open(self.settings_file, 'w') as f: json.dump(settings, f, indent=4)

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.date_sample_entry.insert(0, settings.get('date_sample', ''))
                    self.date_master_entry.insert(0, settings.get('date_master', ''))
                    self.date_treble_entry.insert(0, settings.get('treble_file', ''))
                    self.detective_sample_entry.insert(0, settings.get('detective_sample', ''))
                    self.detective_treble_entry.insert(0, settings.get('detective_treble', ''))
                    self.detective_folder_entry.insert(0, settings.get('detective_folder', ''))
                    self.create_folder_entry.insert(0, settings.get('create_folder', ''))
                    self.create_output_entry.insert(0, settings.get('create_output', ''))
        except (json.JSONDecodeError, KeyError): print(f"Warning: Could not read '{self.settings_file}'.")

    def check_plot_queue(self):
        try:
            plot_args = self.plot_queue.get_nowait()
            if plot_args: plot_results(**plot_args)
        except queue.Empty: pass
        finally: self.after(100, self.check_plot_queue)

    def _run_in_thread(self, target_func, args, button_to_disable, is_analysis=False):
        def thread_target():
            if button_to_disable: self.after(0, lambda: button_to_disable.config(state=tk.DISABLED))
            if is_analysis: self.after(0, lambda: self.generate_report_button.config(state=tk.DISABLED))
            try:
                result = target_func(*args)
                if is_analysis and result:
                    self.last_analysis_results = result
                    if 'raw_sample' in result:
                        plot_args = {k: v for k, v in result.items() if k in plot_results.__code__.co_varnames}
                        self.plot_queue.put(plot_args)
                    self.after(0, lambda: self.generate_report_button.config(state=tk.NORMAL))
            except Exception as e:
                error_message = f"An error occurred:\n\n{e}"
                print(f"\n--- ERROR ---\n{error_message}")
                self.after(0, lambda: messagebox.showerror("Operation Error", error_message))
            finally:
                if button_to_disable: self.after(0, lambda: button_to_disable.config(state=tk.NORMAL))
        thread = threading.Thread(target=thread_target); thread.daemon = True; thread.start()

    def _create_tabs(self):
        self._create_date_tab()
        self._create_detective_tab()
        self._create_master_tab()
        self._create_index_build_tab()

    def _create_report_widget(self):
        report_frame = ttk.LabelFrame(self, text="Report Generation")
        report_frame.pack(pady=5, padx=10, fill="x")
        self.generate_report_button = ttk.Button(report_frame, text="Generate Text Report...", command=self._generate_report, state=tk.DISABLED)
        self.generate_report_button.pack(pady=5)

    def _create_log_widget(self):
        log_frame = ttk.LabelFrame(self, text="Output Log")
        log_frame.pack(pady=5, padx=10, expand=True, fill="both")
        self.log_widget = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_widget.pack(expand=True, fill="both", padx=5, pady=5)
        sys.stdout = TextRedirector(self.log_widget)
        
    def _create_date_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="1. Date")
        # --- NEW: BooleanVars for checkboxes ---
        self.date_reverse_sample_var = tk.BooleanVar()
        self.date_reverse_treble_var = tk.BooleanVar()

        type_frame = ttk.LabelFrame(tab, text="Analysis Type"); type_frame.pack(padx=20, pady=5, fill="x")
        self.date_type_var = tk.StringVar(value="single")
        def toggle():
            is_two_piece = self.date_type_var.get() == "two_piece"
            self.date_treble_label.grid_forget(); self.date_treble_entry.grid_forget(); self.date_treble_browse.grid_forget(); self.date_treble_reverse_check.grid_forget()
            if is_two_piece:
                self.date_treble_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
                self.date_treble_entry.grid(row=1, column=1, padx=5, pady=5)
                self.date_treble_browse.grid(row=1, column=2, padx=5, pady=5)
                self.date_treble_reverse_check.grid(row=1, column=3, padx=5, pady=5) # Show treble reverse
                self.date_sample_label.config(text="Bass Side File (.rwl):")
            else: self.date_sample_label.config(text="Sample File (.rwl):")
            self._update_default_overlap()
        ttk.Radiobutton(type_frame, text="Single Sample", variable=self.date_type_var, value="single", command=toggle).pack(side="left", padx=10)
        ttk.Radiobutton(type_frame, text="Two-Piece Mean", variable=self.date_type_var, value="two_piece", command=toggle).pack(side="left", padx=10)
        
        frame = ttk.LabelFrame(tab, text="File Inputs & Options"); frame.pack(padx=20, pady=5, fill="x")
        self.date_sample_label = ttk.Label(frame, text="Sample File (.rwl):"); self.date_sample_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.date_sample_entry = ttk.Entry(frame, width=60); self.date_sample_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse...", command=self._browse_for_sample_file).grid(row=0, column=2, padx=5, pady=5)
        ttk.Checkbutton(frame, text="Reverse", variable=self.date_reverse_sample_var).grid(row=0, column=3, padx=5, pady=5)

        self.date_treble_label = ttk.Label(frame, text="Treble Side File (.rwl):"); self.date_treble_entry = ttk.Entry(frame, width=60)
        self.date_treble_browse = ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.date_treble_entry, callback=self._update_default_overlap))
        self.date_treble_reverse_check = ttk.Checkbutton(frame, text="Reverse", variable=self.date_reverse_treble_var)
        
        ttk.Label(frame, text="Reference File (.csv/.rwl):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.date_master_entry = ttk.Entry(frame, width=60); self.date_master_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.date_master_entry, is_master=True)).grid(row=2, column=2, padx=5, pady=5)
        ttk.Label(frame, text="Minimum Overlap (years):").grid(row=3, column=0, padx=5, pady=10, sticky="w")
        self.date_min_overlap_spinbox = ttk.Spinbox(frame, from_=30, to=500, increment=10, width=5); self.date_min_overlap_spinbox.set(50)
        self.date_min_overlap_spinbox.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        run_button = ttk.Button(tab, text="Run Date Analysis", command=self._run_date); run_button.pack(pady=10)
        toggle()

    def _create_detective_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="2. Detective")
        # --- NEW: BooleanVars for checkboxes ---
        self.detective_reverse_sample_var = tk.BooleanVar()
        self.detective_reverse_treble_var = tk.BooleanVar()

        type_frame = ttk.LabelFrame(tab, text="Analysis Type"); type_frame.pack(padx=20, pady=5, fill="x")
        self.detective_type_var = tk.StringVar(value="single")
        def toggle():
            is_two_piece = self.detective_type_var.get() == "two_piece"
            self.detective_treble_label.grid_forget(); self.detective_treble_entry.grid_forget(); self.detective_treble_browse.grid_forget(); self.detective_treble_reverse_check.grid_forget()
            if is_two_piece:
                self.detective_treble_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
                self.detective_treble_entry.grid(row=1, column=1, padx=5, pady=5)
                self.detective_treble_browse.grid(row=1, column=2, padx=5, pady=5)
                self.detective_treble_reverse_check.grid(row=1, column=3, padx=5, pady=5)
                self.detective_sample_label.config(text="Bass Side File (.rwl):")
            else: self.detective_sample_label.config(text="Sample File (.rwl):")
            self._update_default_overlap()
        ttk.Radiobutton(type_frame, text="Single Sample", variable=self.detective_type_var, value="single", command=toggle).pack(side="left", padx=10)
        ttk.Radiobutton(type_frame, text="Two-Piece Mean", variable=self.detective_type_var, value="two_piece", command=toggle).pack(side="left", padx=10)
        
        frame = ttk.LabelFrame(tab, text="File Inputs"); frame.pack(padx=20, pady=5, fill="x")
        self.detective_sample_label = ttk.Label(frame, text="Sample File (.rwl):"); self.detective_sample_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.detective_sample_entry = ttk.Entry(frame, width=60); self.detective_sample_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse...", command=self._browse_for_sample_file).grid(row=0, column=2, padx=5, pady=5)
        ttk.Checkbutton(frame, text="Reverse", variable=self.detective_reverse_sample_var).grid(row=0, column=3, padx=5, pady=5)

        self.detective_treble_label = ttk.Label(frame, text="Treble Side File (.rwl):"); self.detective_treble_entry = ttk.Entry(frame, width=60)
        self.detective_treble_browse = ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.detective_treble_entry, callback=self._update_default_overlap))
        self.detective_treble_reverse_check = ttk.Checkbutton(frame, text="Reverse", variable=self.detective_reverse_treble_var)
        
        target_frame = ttk.LabelFrame(tab, text="Reference Target"); target_frame.pack(padx=20, pady=5, fill="x")
        self.detective_target_var = tk.StringVar(value="category")
        ttk.Radiobutton(target_frame, text="Predefined Category:", variable=self.detective_target_var, value="category").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.detective_category_combo = ttk.Combobox(target_frame, values=['alpine', 'baltic', 'all']); self.detective_category_combo.set('alpine')
        self.detective_category_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(target_frame, text="Local Folder:", variable=self.detective_target_var, value="folder").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.detective_folder_entry = ttk.Entry(target_frame, width=60); self.detective_folder_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(target_frame, text="Browse...", command=lambda: self._browse_folder(self.detective_folder_entry)).grid(row=1, column=2, padx=5, pady=5)
        options_frame = ttk.LabelFrame(tab, text="Options"); options_frame.pack(padx=20, pady=5, fill="x")
        ttk.Label(options_frame, text="Show Top N Results:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.detective_top_n_spinbox = ttk.Spinbox(options_frame, from_=1, to=100, width=5); self.detective_top_n_spinbox.set(10)
        self.detective_top_n_spinbox.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(options_frame, text="Minimum Overlap (years):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.detective_min_overlap_spinbox = ttk.Spinbox(options_frame, from_=30, to=500, increment=10, width=5); self.detective_min_overlap_spinbox.set(80)
        self.detective_min_overlap_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(options_frame, text="Only Include Sites Ending After:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.detective_min_end_year_spinbox = ttk.Spinbox(options_frame, from_=0, to=2100, increment=50, width=5); self.detective_min_end_year_spinbox.set(1500)
        self.detective_min_end_year_spinbox.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        run_button = ttk.Button(tab, text="Run Detective Analysis", command=self._run_detective); run_button.pack(pady=10)
        toggle()

    # ... (_create_master_tab, _create_index_build_tab, and helper functions are unchanged) ...
    def _create_master_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="3. Create Master")
        frame = ttk.LabelFrame(tab, text="Create a Custom Master Chronology"); frame.pack(padx=20, pady=20, fill="x")
        ttk.Label(frame, text="Input Folder (.rwl files):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.create_folder_entry = ttk.Entry(frame, width=60); self.create_folder_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_folder(self.create_folder_entry)).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(frame, text="Output Filename (.csv):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.create_output_entry = ttk.Entry(frame, width=60); self.create_output_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Save As...", command=self._save_file_as).grid(row=1, column=2, padx=5, pady=5)
        run_button = ttk.Button(tab, text="Create Master File", command=self._run_create); run_button.pack(pady=20)
        
    def _create_index_build_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="4. Setup")
        index_frame = ttk.LabelFrame(tab, text="Step 1: Download & Index NOAA Files (Run once)")
        index_frame.pack(padx=20, pady=10, fill="x")
        self.index_button = ttk.Button(index_frame, text="Download and Create Index", command=self._run_download)
        self.index_button.pack(pady=5)
        build_frame = ttk.LabelFrame(tab, text="Step 2: Build Master Chronologies from Index")
        build_frame.pack(padx=20, pady=10, fill="x")
        build_options_grid = ttk.Frame(build_frame)
        build_options_grid.pack(pady=5)
        ttk.Label(build_options_grid, text="Select a predefined master to build:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.build_target_combo = ttk.Combobox(build_options_grid, values=['alpine', 'baltic', 'all']); self.build_target_combo.set('all')
        self.build_target_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(build_options_grid, text="Only Include Sites Ending After:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.build_min_end_year_spinbox = ttk.Spinbox(build_options_grid, from_=0, to=2100, increment=50, width=5); self.build_min_end_year_spinbox.set(1500)
        self.build_min_end_year_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.build_button = ttk.Button(build_frame, text="Build Selected", command=self._run_build); self.build_button.pack(pady=10)

    def _get_rwl_length(self, file_path):
        if not file_path or not os.path.exists(file_path): return 0
        try: return len(parse_as_floating_series(file_path))
        except Exception: return 0

    def _update_default_overlap(self):
        new_overlap = 0
        active_tab_index = self.notebook.index(self.notebook.select())
        is_date_tab = active_tab_index == 0
        mode_var = self.date_type_var if is_date_tab else self.detective_type_var
        sample_entry = self.date_sample_entry if is_date_tab else self.detective_sample_entry
        treble_entry = self.date_treble_entry if is_date_tab else self.detective_treble_entry
        if mode_var.get() == "single":
            sample_len = self._get_rwl_length(sample_entry.get())
            if sample_len > 0:
                new_overlap = int(sample_len * self.DEFAULT_OVERLAP_PERCENTAGE)
                print(f"Sample length is {sample_len} rings. Default overlap set to ~80%: {new_overlap} years.")
        else:
            bass_len = self._get_rwl_length(sample_entry.get()); treble_len = self._get_rwl_length(treble_entry.get())
            if bass_len > 0 and treble_len > 0:
                shorter_len = min(bass_len, treble_len)
                new_overlap = int(shorter_len * self.DEFAULT_OVERLAP_PERCENTAGE)
                print(f"Bass/Treble lengths: {bass_len}/{treble_len}. Default overlap set to ~80% of shorter sample: {new_overlap} years.")
            elif bass_len > 0:
                 new_overlap = int(bass_len * self.DEFAULT_OVERLAP_PERCENTAGE)
                 print(f"Bass sample length is {bass_len} rings. Default overlap set to ~80%: {new_overlap} years.")
        if new_overlap >= 30:
            self.date_min_overlap_spinbox.set(new_overlap)
            self.detective_min_overlap_spinbox.set(new_overlap)

    def _browse_file(self, entry_widget, is_master=False, callback=None):
        types = (("RWL files", "*.rwl"), ("All files", "*.*")) if not is_master else (("All files", "*.*"),("CSV files", "*.csv"),("RWL files", "*.rwl"))
        filename = filedialog.askopenfilename(title="Select a file", filetypes=types)
        if filename:
            entry_widget.delete(0, tk.END); entry_widget.insert(0, filename)
            if callback: callback()

    def _browse_for_sample_file(self):
        filename = filedialog.askopenfilename(title="Select a Sample File", filetypes=(("RWL files", "*.rwl"), ("All files", "*.*")))
        if filename:
            self.date_sample_entry.delete(0, tk.END); self.date_sample_entry.insert(0, filename)
            self.detective_sample_entry.delete(0, tk.END); self.detective_sample_entry.insert(0, filename)
            self._update_default_overlap()

    def _browse_folder(self, entry_widget):
        foldername = filedialog.askdirectory(title="Select a folder")
        if foldername: entry_widget.delete(0, tk.END); entry_widget.insert(0, foldername)

    def _save_file_as(self):
        filename = filedialog.asksaveasfilename(title="Save Master As", defaultextension=".csv", filetypes=(("CSV files", "*.csv"),))
        if filename: self.create_output_entry.delete(0, tk.END); self.create_output_entry.insert(0, filename)

    def _run_date(self):
        run_button = self.notebook.nametowidget(self.notebook.select()).winfo_children()[-1]
        analysis_type = self.date_type_var.get()
        min_overlap = int(self.date_min_overlap_spinbox.get())
        master = self.date_master_entry.get()
        if not master: messagebox.showerror("Error", "Please select a reference file."); return
        if analysis_type == "single":
            sample = self.date_sample_entry.get()
            if not sample: messagebox.showerror("Error", "Please select a sample file."); return
            reverse_sample = self.date_reverse_sample_var.get()
            self._run_in_thread(run_date_analysis, (sample, master, min_overlap, False, reverse_sample), run_button, is_analysis=True)
        else:
            bass = self.date_sample_entry.get(); treble = self.date_treble_entry.get()
            if not bass or not treble: messagebox.showerror("Error", "Please select both bass and treble files."); return
            reverse_bass = self.date_reverse_sample_var.get()
            reverse_treble = self.date_reverse_treble_var.get()
            final_args = ["placeholder", master, min_overlap] # Note: final_analysis_func doesn't have reverse flag, handled in two_piece func
            self._run_in_thread(run_two_piece_mean_analysis, (bass, treble, run_date_analysis, final_args, reverse_bass, reverse_treble), run_button, is_analysis=True)

    def _run_detective(self):
        run_button = self.notebook.nametowidget(self.notebook.select()).winfo_children()[-1]
        analysis_type = self.detective_type_var.get()
        if self.detective_target_var.get() == "category": target = self.detective_category_combo.get()
        else: target = self.detective_folder_entry.get()
        if not target: messagebox.showerror("Error", "Please select a target category or folder."); return
        top_n = int(self.detective_top_n_spinbox.get()); min_overlap = int(self.detective_min_overlap_spinbox.get()); min_end_year = int(self.detective_min_end_year_spinbox.get())
        if analysis_type == "single":
            sample = self.detective_sample_entry.get()
            if not sample: messagebox.showerror("Error", "Please select a sample file."); return
            reverse_sample = self.detective_reverse_sample_var.get()
            self._run_in_thread(run_detective_analysis, (sample, target, top_n, min_overlap, min_end_year, reverse_sample), run_button, is_analysis=True)
        else:
            bass = self.detective_sample_entry.get(); treble = self.detective_treble_entry.get()
            if not bass or not treble: messagebox.showerror("Error", "Please select both bass and treble files."); return
            reverse_bass = self.detective_reverse_sample_var.get()
            reverse_treble = self.detective_reverse_treble_var.get()
            final_args = ["placeholder", target, top_n, min_overlap, min_end_year, False] # Reversal is handled in two_piece func
            self._run_in_thread(run_two_piece_mean_analysis, (bass, treble, run_detective_analysis, final_args, reverse_bass, reverse_treble), run_button, is_analysis=True)
    # ... (other run methods unchanged) ...
    def _run_create(self):
        folder = self.create_folder_entry.get(); output = self.create_output_entry.get()
        if not folder or not output: messagebox.showerror("Error", "Please select an input folder and an output file."); return
        button = self.notebook.nametowidget(self.notebook.select()).winfo_children()[-1]
        self._run_in_thread(run_create_master, (folder, output), button)
        
    def _run_download(self):
        self._run_in_thread(download_and_index_files, (), self.index_button)
        
    def _run_build(self):
        target = self.build_target_combo.get()
        min_end_year = int(self.build_min_end_year_spinbox.get())
        if target in ['alpine', 'all']: 
            self._run_in_thread(build_master_from_index, ("Alpine Instrument Wood", ['PICEA', 'ABIES'], ['aust', 'fran', 'germ', 'ital', 'swit', 'slov'], 150, 1750, min_end_year), self.build_button)
        if target in ['baltic', 'all']: 
            self._run_in_thread(build_master_from_index, ("Baltic Northern Timber", ['PINUS', 'PICEA'], ['finl', 'germ', 'lith', 'norw', 'pola', 'swed'], 150, 1750, min_end_year), self.build_button)
    # ... (_generate_report unchanged) ...
    def _generate_report(self):
        if not self.last_analysis_results: messagebox.showerror("Error", "No analysis data found."); return
        res = self.last_analysis_results
        report_lines = ["="*70, "         DENDROCHRONOLOGICAL ANALYSIS REPORT", "="*70, f"ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
        if res.get('analysis_mode') == 'two_piece':
            report_lines.append("--- 1. SAMPLE INFORMATION ---")
            report_lines.append(f"Analysis Mode: Two-Piece Mean")
            report_lines.append(f"  Bass Side:   {os.path.basename(res['bass_file'])}")
            report_lines.append(f"  Treble Side: {os.path.basename(res['treble_file'])}\n")
            report_lines.append("--- 2. INTERNAL CROSS-MATCHING OF SAMPLES ---")
            stats = res['internal_stats']
            report_lines.append(f"  Result: Strong Match Found (t > 6.0)")
            report_lines.append(f"  T-Value:              {stats['t_value']:.2f}")
            report_lines.append(f"  Gleichläufigkeit (Glk): {stats.get('glk', 0.0):.1f}%\n")
            report_lines.append("--- 3. FINAL ANALYSIS OF MEAN CHRONOLOGY ---")
        else: # Single Mode
            report_lines.append("--- 1. SAMPLE INFORMATION ---")
            report_lines.append(f"Analysis Mode: Single Sample")
            report_lines.append(f"  Sample File: {os.path.basename(res.get('sample_file') or res.get('sample_filename'))}\n")
            report_lines.append("--- 2. FINAL ANALYSIS RESULTS ---")
            
        if res.get('analysis_type') == 'detective':
            report_lines.append(f"Reference Target: {res['target']}")
            report_lines.append(f"Minimum Overlap: {res['min_overlap']} years")
            report_lines.append(f"Sites Ending Before {res.get('min_end_year', 1500)} Were Excluded.\n")
            df = res['results_df'][['end_year', 't_value', 'glk', 'correlation', 'overlap_n', 'source_file']]
            report_lines.append(df.to_string(index=False))
        else: # Date mode
            best = res['results']["best_match"]
            report_lines.append(f"Reference: {os.path.basename(res['master_filename'])}\n")
            report_lines.append(f"Most Likely End Year: {int(best['end_year'])}")
            report_lines.append(f"  T-Value:              {best['t_value']:.2f}")
            report_lines.append(f"  Gleichläufigkeit (Glk): {best.get('glk', 0.0):.1f}%")
            report_lines.append(f"  Correlation (r):      {best['correlation']:.4f}")
            report_lines.append(f"  Overlap (n):          {int(best['overlap_n'])} years")
        
        report_lines.extend(["\n\n--- NOTES & INTERPRETATION (User editable) ---", "\n[Enter qualitative analysis and conclusions here...]\n"])
        report_content = "\n".join(report_lines)
        report_filename = filedialog.asksaveasfilename(initialfile="dendro_report.txt", defaultextension=".txt", filetypes=(("Text Files", "*.txt"), ("All files", "*.*")))
        if report_filename:
            with open(report_filename, 'w') as f: f.write(report_content)
            print(f"Report saved to {report_filename}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
