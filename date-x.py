# gogo_gui.py (Version 9.6 - Streamlined Reporting Workflow)
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
except ImportError as e:
    messagebox.showerror("Import Error", f"Could not import from gogo.py. Please ensure it is in the same directory and contains no syntax errors.\n\nDetails: {e}")
    sys.exit(1)

class TextRedirector:
    def __init__(self, widget): self.widget = widget
    def write(self, str_val): self.widget.config(state=tk.NORMAL); self.widget.insert(tk.END, str_val); self.widget.see(tk.END); self.widget.config(state=tk.DISABLED); self.widget.update_idletasks()
    def flush(self): pass

class App(tk.Tk):
    DEFAULT_OVERLAP_PERCENTAGE = 0.8
    def _create_main_layout(self):
        """Creates the main left/right pane structure for the GUI."""
        main_container = ttk.Frame(self)
        main_container.pack(expand=True, fill="both", padx=10, pady=5)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=3)
        main_container.grid_columnconfigure(1, weight=2)
        self.left_pane = ttk.Frame(main_container)
        self.right_pane = ttk.Frame(main_container)
        self.left_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.right_pane.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

    def __init__(self):
        super().__init__()
        self.title("GoGo Dendro-Dating Tool v9.6") # Version bump
        self.geometry("1450x550")
        self.settings_file = "gogo_settings.json"
        self.last_analysis_results = None
        self.plot_queue = queue.Queue()
        self._create_main_layout()
        self.notebook = ttk.Notebook(self.left_pane)
        self.notebook.pack(pady=5, padx=0, expand=True, fill="both")
        self._create_tabs()
        self._create_report_widget(parent=self.left_pane)
        self._create_log_widget(parent=self.right_pane)
        self.load_settings()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_plot_queue()
        print("Welcome! Ready for analysis.")

    def _get_stiffness_from_string(self, value_str):
        try: return int(value_str.split('(')[1].split('%')[0])
        except (IndexError, ValueError): return 67

    def _run_date(self):
        analysis_type = self.date_type_var.get()
        if analysis_type == "single":
            sample = self.date_sample_entry.get()
            if not sample or not os.path.exists(sample): messagebox.showerror("Error", "Please select a valid sample file."); return
            sample_len = self._get_rwl_length(sample)
            if 0 < sample_len <= 60:
                messagebox.showwarning("Short Sample Warning", f"The sample '{os.path.basename(sample)}' has {sample_len} rings (60 or fewer).\n\nDating may be statistically unreliable due to the short length.")
        else:
            bass, treble = self.date_sample_entry.get(), self.date_treble_entry.get()
            if not bass or not os.path.exists(bass) or not treble or not os.path.exists(treble): messagebox.showerror("Error", "Please select valid files for both Bass and Treble sides."); return
            bass_len, treble_len = self._get_rwl_length(bass), self._get_rwl_length(treble)
            if (0 < bass_len <= 60) or (0 < treble_len <= 60):
                messagebox.showwarning("Short Sample Warning", f"One or both samples have 60 or fewer rings (Bass: {bass_len}, Treble: {treble_len}).\n\nDating may be statistically unreliable.")
        
        run_button = self.notebook.nametowidget(self.notebook.select()).winfo_children()[-1]
        min_overlap = int(self.date_min_overlap_spinbox.get())
        master = self.date_master_entry.get()
        stiffness_pct = self._get_stiffness_from_string(self.date_stiffness_combo.get())
        if not master: messagebox.showerror("Error", "Please select a reference file."); return
        if analysis_type == "single":
            reverse_sample = self.date_reverse_sample_var.get()
            self._run_in_thread(run_date_analysis, (sample, master, min_overlap, False, reverse_sample, stiffness_pct), run_button, is_analysis=True)
        else:
            reverse_bass = self.date_reverse_sample_var.get(); reverse_treble = self.date_reverse_treble_var.get()
            final_args = ["placeholder", master, min_overlap, False, False, stiffness_pct]
            self._run_in_thread(run_two_piece_mean_analysis, (bass, treble, run_date_analysis, final_args, reverse_bass, reverse_treble, stiffness_pct), run_button, is_analysis=True)

    def _run_detective(self):
        analysis_type = self.detective_type_var.get()
        if analysis_type == "single":
            sample = self.detective_sample_entry.get()
            if not sample or not os.path.exists(sample): messagebox.showerror("Error", "Please select a valid sample file."); return
            sample_len = self._get_rwl_length(sample)
            if 0 < sample_len <= 60:
                messagebox.showwarning("Short Sample Warning", f"The sample '{os.path.basename(sample)}' has {sample_len} rings (60 or fewer).\n\nDating may be statistically unreliable due to the short length.")
        else:
            bass, treble = self.detective_sample_entry.get(), self.detective_treble_entry.get()
            if not bass or not os.path.exists(bass) or not treble or not os.path.exists(treble): messagebox.showerror("Error", "Please select valid files for both Bass and Treble sides."); return
            bass_len, treble_len = self._get_rwl_length(bass), self._get_rwl_length(treble)
            if (0 < bass_len <= 60) or (0 < treble_len <= 60):
                messagebox.showwarning("Short Sample Warning", f"One or both samples have 60 or fewer rings (Bass: {bass_len}, Treble: {treble_len}).\n\nDating may be statistically unreliable.")

        run_button = self.notebook.nametowidget(self.notebook.select()).winfo_children()[-1]
        if self.detective_target_var.get() == "category": target = self.detective_category_combo.get()
        else: target = self.detective_folder_entry.get()
        if not target: messagebox.showerror("Error", "Please select a target category or folder."); return
        top_n = int(self.detective_top_n_spinbox.get()); min_overlap = int(self.detective_min_overlap_spinbox.get()); min_end_year = int(self.detective_min_end_year_spinbox.get())
        stiffness_pct = self._get_stiffness_from_string(self.detective_stiffness_combo.get())
        if analysis_type == "single":
            reverse_sample = self.detective_reverse_sample_var.get()
            self._run_in_thread(run_detective_analysis, (sample, target, top_n, min_overlap, min_end_year, reverse_sample, stiffness_pct), run_button, is_analysis=True)
        else:
            reverse_bass = self.detective_reverse_sample_var.get(); reverse_treble = self.detective_reverse_treble_var.get()
            final_args = ["placeholder", target, top_n, min_overlap, min_end_year, False, stiffness_pct]
            self._run_in_thread(run_two_piece_mean_analysis, (bass, treble, run_detective_analysis, final_args, reverse_bass, reverse_treble, stiffness_pct), run_button, is_analysis=True)
            
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
            if is_analysis: self.after(0, lambda: self.save_report_button.config(state=tk.DISABLED))
            try:
                result = target_func(*args)
                if is_analysis and result:
                    self.last_analysis_results = result
                    if 'raw_sample' in result:
                        plot_args = {k: v for k, v in result.items() if k in plot_results.__code__.co_varnames}
                        self.plot_queue.put(plot_args)
                    self.after(0, lambda: self.save_report_button.config(state=tk.NORMAL))
                    
                    # --- NEW: Auto-print report to log ---
                    report_content = self._create_report_content()
                    if report_content:
                        self.after(0, lambda: print("\n\n" + "="*70 + "\n           AUTOMATICALLY GENERATED REPORT\n" + "="*70 + "\n" + report_content))
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
        self._create_methodology_tab()
    def _create_report_widget(self, parent):
        report_frame = ttk.LabelFrame(parent, text="Report Generation")
        report_frame.pack(pady=5, padx=0, fill="x")
        self.save_report_button = ttk.Button(report_frame, text="Save Text Report...", command=self._save_report, state=tk.DISABLED)
        self.save_report_button.pack(pady=5)
    def _create_log_widget(self, parent):
        log_frame = ttk.LabelFrame(parent, text="Output Log")
        log_frame.pack(pady=5, padx=0, expand=True, fill="both")
        self.log_widget = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_widget.pack(expand=True, fill="both", padx=5, pady=5)
        sys.stdout = TextRedirector(self.log_widget)
    def _create_date_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="1. Date")
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
                self.date_treble_reverse_check.grid(row=1, column=3, padx=5, pady=5)
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
        ttk.Label(frame, text="Detrending Stiffness:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.date_stiffness_combo = ttk.Combobox(frame, values=['Standard (67%)', 'Stiff (80%)'], width=15, state="readonly"); self.date_stiffness_combo.set('Standard (67%)')
        self.date_stiffness_combo.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        run_button = ttk.Button(tab, text="Run Date Analysis", command=self._run_date); run_button.pack(pady=10)
        toggle()
    def _create_detective_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="2. Detective")
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
        self.detective_category_combo = ttk.Combobox(target_frame, values=['alpine', 'baltic', 'all'], state="readonly"); self.detective_category_combo.set('alpine')
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
        ttk.Label(options_frame, text="Detrending Stiffness:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.detective_stiffness_combo = ttk.Combobox(options_frame, values=['Standard (67%)', 'Stiff (80%)'], width=15, state="readonly"); self.detective_stiffness_combo.set('Standard (67%)')
        self.detective_stiffness_combo.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        run_button = ttk.Button(tab, text="Run Detective Analysis", command=self._run_detective); run_button.pack(pady=10)
        toggle()
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
        self.build_target_combo = ttk.Combobox(build_options_grid, values=['alpine', 'baltic', 'all'], state="readonly"); self.build_target_combo.set('all')
        self.build_target_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(build_options_grid, text="Only Include Sites Ending After:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.build_min_end_year_spinbox = ttk.Spinbox(build_options_grid, from_=0, to=2100, increment=50, width=5); self.build_min_end_year_spinbox.set(1500)
        self.build_min_end_year_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.build_button = ttk.Button(build_frame, text="Build Selected", command=self._run_build); self.build_button.pack(pady=10)
    def _create_methodology_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="5. Methods & References")
        text_frame = ttk.Frame(tab); text_frame.pack(padx=10, pady=10, expand=True, fill="both")
        methodology_text = tk.Text(text_frame, wrap=tk.WORD, padx=5, pady=5, font=("Helvetica", 10), background="#f0f0f0")
        scrollbar = ttk.Scrollbar(text_frame, command=methodology_text.yview); methodology_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y); methodology_text.pack(side=tk.LEFT, expand=True, fill="both")
        content = textwrap.dedent("""
            This document explains the scientific choices and methods used by this software to ensure accurate and reliable dendrochronological analysis, grounded in established scientific literature.

            --- MEASUREMENT DIRECTION (CRITICAL!) ---
            For two-piece, book-matched instrument tops and backs, the wood is processed in a way that places the YOUNGEST wood at the CENTER JOINT and the OLDEST wood at the OUTER EDGES. The correct measurement direction is therefore:
            FROM THE OUTER EDGE (Ring 1) INWARDS TO THE CENTER JOINT (Final Ring).
            If you have measured in the opposite direction, use the "Reverse" checkbox.

            --- DETRENDING METHOD ---
            This software exclusively uses the Cubic Smoothing Spline for detrending. This is the modern standard for its flexibility in modeling real-world growth patterns without the endpoint instability issues found in older methods like polynomials. The stiffness of the spline can be adjusted:
            • Standard (67%): The scientific default, best for general-purpose dating (Cook & Peters, 1981).
            • Stiff (80%): A less flexible spline, better for "sensitive" trees that have a weak age trend but a strong climate signal, preventing the removal of the signal itself.

            --- STATISTICAL VALIDATION: A MULTI-DIMENSIONAL APPROACH ---
            To determine the strength of a tree-ring match, this method uses three key metrics simultaneously, as a simple t-value can be misleading.

            1. t-value (Baillie-Pilcher): Measures statistical similarity.
            2. Overlap (Years): The number of shared rings; longer overlaps provide more reliable matches.
            3. Gleichläufigkeit (%): A classical German statistic measuring agreement in year-to-year growth direction (Eckstein & Bauch, 1969).

            This approach ensures matches are statistically significant, biologically meaningful, and visually consistent—crucial for high-stakes applications like dating antique instruments.

            --- CLASSIFICATION THRESHOLDS ---

            - Very Strong Match:
                T ≥ 7.0, Overlap ≥ 80 years, and Gleichläufigkeit ≥ 70%

            - Strong Match:
                T ≥ 6.0, Overlap ≥ 70 years, and Gleichläufigkeit ≥ 65%

            - Significant Match:
                T ≥ 5.0, Overlap ≥ 50 years, and Gleichläufigkeit ≥ 60%

            - Tentative/Insufficient Match:
                Any case failing to meet all three criteria for a "Significant Match". A high T-value with a low overlap or low Gleichläufigkeit is not considered a reliable match.

            --- KEY SCIENTIFIC REFERENCES ---
            • Baillie, M.G.L. & Pilcher, J.R. (1973). "A simple cross-dating program for tree-ring research." Tree-Ring Bulletin 33, 7-14.
            • Cook, E.R. & Peters, K. (1981). "The smoothing spline: a new approach to standardizing tree-ring width series for dendroclimatic studies." Tree-Ring Bulletin 41, 45-53.
            • Eckstein, D. & Bauch, J. (1969). "Beitrag zur Rationalisierung eines dendrochronologischen Verfahrens..." Forstwiss. Centralbl. 88, 230-250.
            • Fritts, H.C. (1976). Tree Rings and Climate. Academic Press, New York.
        """)
        methodology_text.config(state=tk.NORMAL)
        methodology_text.insert(tk.END, content)
        methodology_text.config(state=tk.DISABLED)
    def _get_rwl_length(self, file_path):
        if not file_path or not os.path.exists(file_path): return 0
        try: return len(parse_as_floating_series(file_path))
        except Exception: return 0
    def _update_default_overlap(self):
        new_overlap = 0; active_tab_index = self.notebook.index(self.notebook.select())
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
    
    def _classify_dendro_match(self, t_value, overlap_years, gleich_percent):
        """Classify match strength based on multi-dimensional thresholds."""
        if t_value >= 7.0 and overlap_years >= 80 and gleich_percent >= 70:
            return "Very Strong Match"
        elif t_value >= 6.0 and overlap_years >= 70 and gleich_percent >= 65:
            return "Strong Match"
        elif t_value >= 5.0 and overlap_years >= 50 and gleich_percent >= 60:
            return "Significant Match"
        else:
            return "Tentative/Insufficient Match"

    def _create_report_content(self):
        """Generates the full report string based on the last analysis."""
        if not self.last_analysis_results:
            return ""

        res = self.last_analysis_results
        paragraphs = []

        # Paragraph 1: Physical Description
        is_two_piece = res.get('analysis_mode') == 'two_piece'
        physical_desc = f"The belly appears to be constructed from {'two sections' if is_two_piece else 'one section'}."

        if is_two_piece:
            bass_rev, treble_rev = res.get('reverse_bass', False), res.get('reverse_treble', False)
            if bass_rev and treble_rev: orientation_desc = "Both halves were measured from the centre joint outwards."
            elif bass_rev: orientation_desc = "The bass side was measured from the centre joint outwards, and the treble side from the outer edge inwards."
            elif treble_rev: orientation_desc = "The treble side was measured from the centre joint outwards, and the bass side from the outer edge inwards."
            else: orientation_desc = "Both halves were measured from the outer edge inwards, which is the standard orientation."
        else:
            orientation_desc = "The sample was measured from the centre joint outwards." if res.get('reverse_sample', False) else "The sample was measured from the outer edge inwards, which is the standard orientation."
        
        physical_desc += " " + orientation_desc
        ring_count = res.get('mean_series_length') if is_two_piece else len(res.get('raw_sample', []))
        physical_desc += f" The {'final mean chronology' if is_two_piece else 'sample'} contains {ring_count} rings."
        paragraphs.append(physical_desc)

        # Paragraph 2: Dating Result
        best_match = res.get('results', {}).get('best_match', {})
        if best_match:
            t_value, overlap, glk = best_match.get('t_value', 0.0), best_match.get('overlap_n', 0), best_match.get('glk', 0.0)
            classification = self._classify_dendro_match(t_value, overlap, glk)
            end_year = int(best_match.get('end_year', 0))

            if classification != "Tentative/Insufficient Match":
                dating_result = f"Dendrochronological analysis indicates a felling date for the tree after the growing season of {end_year}. This conclusion is supported by a '{classification}' classification based on the multi-dimensional criteria (T-value: {t_value:.2f}, Overlap: {overlap} yrs, GLK: {glk:.1f}%)."
            else:
                dating_result = f"Analysis suggests a potential end year of {end_year} (T-value: {t_value:.2f}, Overlap: {overlap} yrs, GLK: {glk:.1f}%). However, this is classified as a 'Tentative/Insufficient Match' because it fails to meet the required thresholds for a conclusive scientific date. This result should be considered a proposal requiring further evidence."
            paragraphs.append(dating_result)
        else:
            paragraphs.append("Dating analysis did not produce a statistically significant result.")

        # Paragraph 3: Two-Piece Confirmation
        if is_two_piece:
            internal_t = res.get('internal_stats', {}).get('t_value', 0.0)
            if internal_t >= 6.0:
                paragraphs.append(f"The two halves of the belly show a very strong internal cross-match (T-value = {internal_t:.2f}), confirming they likely originate from the same tree. They were subsequently combined into a single mean series for the final dating analysis.")

        # Paragraphs 4 & 5: Reference Context and Geographic Conclusion
        if res.get('analysis_type') == 'detective':
            df = res.get('enriched_results_df')
            if df is not None and not df.empty:
                top_match = df.iloc[0].to_dict()
                context_lines = ["The analysis was performed against a database of regional chronologies. The top matching reference sites are:"]
                for i in range(min(5, len(df))):
                    row = df.iloc[i]
                    site_info = f"{row.get('location', 'Unknown Location').strip()} ({row.get('site_name', 'Unknown Site').strip()})"
                    context_lines.append(f"  - {site_info} (T={row.get('t_value', 0.0):.2f}, O={row.get('overlap_n', 0)}, G={row.get('glk', 0.0):.1f}%)")
                paragraphs.append("\n".join(context_lines))

                t, o, g = top_match.get('t_value',0), top_match.get('overlap_n',0), top_match.get('glk',0)
                top_classification = self._classify_dendro_match(t, o, g)
                top_location = top_match.get('location', 'N/A').strip()
                
                if top_location != 'N/A' and top_location:
                    if top_classification != "Tentative/Insufficient Match":
                        geo_conclusion = f"The strongest alignment is classified as a '{top_classification}' with a chronology from {top_location}. This strongly suggests a probable geographic origin for the instrument's wood in or around this region."
                    else:
                        geo_conclusion = f"The top alignment in the database is with a site from {top_location}. However, this is classified as a 'Tentative/Insufficient Match' and cannot be used to confidently assign a geographic origin."
                    paragraphs.append(geo_conclusion)
            else:
                 paragraphs.append("Detective analysis was run, but no significantly matching reference sites were found in the target database.")
        else:
            paragraphs.append(f"The sample was dated against the single reference chronology: {os.path.basename(res.get('master_filename', 'N/A'))}.")

        # Final Assembly
        title = "DENDROCHRONOLOGICAL ANALYSIS REPORT"
        header = f"{'='*70}\n{title:^70}\n{'='*70}\nANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_body = "\n\n".join(paragraphs)
        return header + report_body

    def _save_report(self):
        """Saves the generated text report to a file."""
        if not self.last_analysis_results:
            messagebox.showerror("Error", "No analysis data to save. Please run an analysis first.")
            return

        res = self.last_analysis_results
        
        # --- Determine default save path and filename ---
        if res.get('analysis_mode') == 'two_piece':
            sample_path = res.get('bass_file')
        else:
            sample_path = res.get('sample_filename')

        if sample_path and os.path.exists(sample_path):
            default_dir = os.path.dirname(sample_path)
            base_name = os.path.basename(sample_path)
            filename_sans_ext, _ = os.path.splitext(base_name)
            default_filename = f"dendro_report_{filename_sans_ext}.txt"
        else:
            default_dir = os.getcwd()
            default_filename = "dendro_report.txt"

        report_content = self._create_report_content()
        if not report_content:
            messagebox.showerror("Error", "Could not generate report content.")
            return

        report_filename = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=default_filename,
            defaultextension=".txt",
            filetypes=(("Text Files", "*.txt"), ("All files", "*.*"))
        )
        if report_filename:
            try:
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"Report saved to {report_filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save report file:\n{e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
