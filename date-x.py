# date-x.py (Version 10.2) - desktop front end; version comes from gogo.__version__
# ==============================================================================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, sys, threading, queue, json, textwrap
from datetime import datetime
from functools import partial

try:
    from gogo import (
        download_and_index_files, build_master_from_index, run_create_master,
        run_date_analysis, run_detective_analysis, plot_results,
        run_two_piece_mean_analysis, parse_as_floating_series,
        fetch_and_build_violin_master, _classify_dendro_match as _gogo_classify,
        DEFAULTS, CATEGORY_NAMES, CATEGORY_PARAMS, DETECTIVE_TARGETS, index_cache,
        __version__ as GOGO_VERSION, APP_NAME, TARGET_LABELS,
        VIOLIN_REFERENCE_LABEL, VIOLIN_REFERENCE_BLURB, VIOLIN_MASTER_FILENAME,
        VIOLIN_REFERENCE_DIR, run_manifest, format_run_manifest,
        terminus_post_quem_note, format_reference_set,
        stability_check, stability_verdict,
        format_country_coverage, COUNTRY_COVERAGE_COLUMNS,
        describe_index, write_csv_with_manifest,
        format_classification, classification_handle, convention_note, shared_variation
    )
except ImportError as e:
    messagebox.showerror("Import Error", f"Could not import from gogo.py. Please ensure it is in the same directory and contains no syntax errors.\n\nDetails: {e}")
    sys.exit(1)

class TextRedirector:
    """stdout -> a queue. Touches no widget, so it is safe to write from a worker thread.

    Analyses run in background threads and print as they go. Writing into the Text widget
    from there -- and worse, calling update_idletasks(), which pumps the event loop -- is a
    Tk call from a non-Tk thread: freezes and crashes that are hard to reproduce. The main
    thread drains this queue on its existing timer (App._drain_log)."""
    def __init__(self, log_queue): self.queue = log_queue
    def write(self, str_val):
        if str_val:
            self.queue.put(str_val)
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
        self.title(f"{APP_NAME} Bugiganga: A Dendro-Dating Tool v{GOGO_VERSION}")
        self.geometry("1450x550")
        self.settings_file = "date-x_settings.json"
        self.last_analysis_results = None
        self.plot_queue = queue.Queue()
        self.log_queue = queue.Queue()
        self.ui_queue = queue.Queue()
        self._stdout_before = sys.stdout
        # Shared, report-affecting settings (Task 1 & Task 3 disclosures). Defaults are
        # conservative / "not recorded" so an old settings file without these keys still
        # loads safely and the report says so rather than silently omitting the caveat.
        self.same_tree_mode_var = tk.StringVar(value="conservative")
        self.image_resolution_var = tk.StringVar(value="")
        self.image_scale_calibrated_var = tk.BooleanVar(value=False)
        self.image_source_note_var = tk.StringVar(value="")
        self._create_main_layout()
        self.notebook = ttk.Notebook(self.left_pane)
        self.notebook.pack(pady=5, padx=0, expand=True, fill="both")
        self._create_tabs()
        self._create_report_widget(parent=self.left_pane)
        self._create_log_widget(parent=self.right_pane)
        self.load_settings()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_plot_queue()
        self._check_setup_requirements()
        print("Welcome! Ready for analysis.")

    # Detrending presets. A fixed cutoff is the default because sample and reference are
    # of very different lengths, and a %-of-length cutoff therefore filters them in
    # different frequency bands (see gogo.detrend). The percent options are kept so old
    # runs can be reproduced.
    DETREND_PRESETS = {
        'Fixed 32-year (standard)': ('fixed', 32, 67),
        'Fixed 20-year (flexible)': ('fixed', 20, 67),
        'Fixed 50-year (stiff)':    ('fixed', 50, 67),
        'Percent 67% (legacy)':     ('percent', 32, 67),
        'Percent 80% (legacy stiff)': ('percent', 32, 80),
    }
    DETREND_DEFAULT = 'Fixed 32-year (standard)'

    # All three by default; the single-index options are for speed and for reproducing a run.
    INDEX_PRESETS = {
        'All three (spline + bp + hollstein)': None,
        'Spline only': 'spline',
        'Baillie-Pilcher only': 'bp',
        'Hollstein only': 'hollstein',
    }
    INDEX_DEFAULT = 'All three (spline + bp + hollstein)'

    def _get_detrend_from_string(self, value_str):
        """(mode, wavelength, stiffness_pct) for a preset label."""
        return self.DETREND_PRESETS.get(value_str, self.DETREND_PRESETS[self.DETREND_DEFAULT])

    def _get_index_from_string(self, value_str):
        """The index key for a preset label; None means all three."""
        return self.INDEX_PRESETS.get(value_str, None)

    # A master is built under exactly one index, so there is no "all" to choose here.
    MASTER_INDEX_PRESETS = {'Spline (standard)': 'spline',
                            'Baillie-Pilcher (bp)': 'bp',
                            'Hollstein': 'hollstein'}
    MASTER_INDEX_DEFAULT = 'Spline (standard)'

    def _path_row(self, parent, row, label_text, browse_command, button_text="Browse...",
                  reverse_var=None, grid_now=True):
        """One label / entry / browse-button row, optionally with a Reverse checkbox.

        Returns (label, entry, button, checkbox) so a caller that needs to show and hide the
        row later (the two-piece treble side) keeps the handles it needs. With grid_now=False
        the widgets are built but not placed, which is what hiding one amounts to here."""
        label = ttk.Label(parent, text=label_text)
        entry = ttk.Entry(parent, width=60)
        button = ttk.Button(parent, text=button_text, command=browse_command)
        check = (ttk.Checkbutton(parent, text="Reverse", variable=reverse_var)
                 if reverse_var is not None else None)
        if grid_now:
            self._grid_path_row(row, label, entry, button, check)
        return label, entry, button, check

    def _show_treble_row(self, is_two_piece, sample_label, treble_widgets):
        """Show or hide the treble row, and rename the first row to match the mode."""
        for widget in treble_widgets:
            widget.grid_forget()
        if is_two_piece:
            self._grid_path_row(1, *treble_widgets)
        sample_label.config(text="Bass Side File (.rwl):" if is_two_piece
                            else "Sample File (.rwl):")
        self._update_default_overlap()

    @staticmethod
    def _grid_path_row(row, label, entry, button, check=None):
        """Place a path row's widgets on one grid row."""
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        entry.grid(row=row, column=1, padx=5, pady=5)
        button.grid(row=row, column=2, padx=5, pady=5)
        if check is not None:
            check.grid(row=row, column=3, padx=5, pady=5)

    def _add_index_combo(self, parent, row, column=0, single=False):
        """The index selector, laid out as a label/combobox pair in a grid."""
        presets = self.MASTER_INDEX_PRESETS if single else self.INDEX_PRESETS
        default = self.MASTER_INDEX_DEFAULT if single else self.INDEX_DEFAULT
        ttk.Label(parent, text="Index:").grid(row=row, column=column, padx=5, pady=5, sticky="w")
        combo = ttk.Combobox(parent, values=list(presets), width=42, state="readonly")
        combo.set(default)
        combo.grid(row=row, column=column + 1, padx=5, pady=5, sticky="w")
        return combo

    def _get_stiffness_from_string(self, value_str):
        return self._get_detrend_from_string(value_str)[2]

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
        mode, wavelength, stiffness_pct = self._get_detrend_from_string(self.date_detrend_combo.get())
        index = self._get_index_from_string(self.date_index_combo.get())
        if not master: messagebox.showerror("Error", "Please select a reference file."); return
        # Every setting is passed by name: these argument lists grow, and a positional list
        # silently means something different the next time one is added in the middle.
        date_one = partial(run_date_analysis, master_file=master, min_overlap=min_overlap,
                           spline_stiffness_pct=stiffness_pct, detrend_mode=mode,
                           detrend_wavelength=wavelength, index=index)
        if analysis_type == "single":
            reverse_sample = self.date_reverse_sample_var.get()
            self._run_in_thread(partial(date_one, sample, reverse_sample=reverse_sample),
                                (), run_button, is_analysis=True)
        else:
            reverse_bass = self.date_reverse_sample_var.get(); reverse_treble = self.date_reverse_treble_var.get()
            self._run_in_thread(
                partial(run_two_piece_mean_analysis, bass, treble, date_one,
                        reverse_bass=reverse_bass, reverse_treble=reverse_treble,
                        spline_stiffness_pct=stiffness_pct, detrend_mode=mode,
                        detrend_wavelength=wavelength, index=index),
                (), run_button, is_analysis=True)

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
        if self.detective_target_var.get() == "category":
            label = self.detective_category_combo.get()
            target = self._target_by_label.get(label, label)
        else: target = self.detective_folder_entry.get()
        if not target: messagebox.showerror("Error", "Please select a target category or folder."); return
        top_n = int(self.detective_top_n_spinbox.get()); min_overlap = int(self.detective_min_overlap_spinbox.get()); min_end_year = int(self.detective_min_end_year_spinbox.get())
        mode, wavelength, stiffness_pct = self._get_detrend_from_string(self.detective_detrend_combo.get())
        index = self._get_index_from_string(self.detective_index_combo.get())
        search_one = partial(run_detective_analysis, target=target, top_n=top_n,
                             min_overlap=min_overlap, min_end_year=min_end_year,
                             spline_stiffness_pct=stiffness_pct, detrend_mode=mode,
                             detrend_wavelength=wavelength, index=index)
        if analysis_type == "single":
            reverse_sample = self.detective_reverse_sample_var.get()
            self._run_in_thread(partial(search_one, sample, reverse_sample=reverse_sample),
                                (), run_button, is_analysis=True)
        else:
            reverse_bass = self.detective_reverse_sample_var.get(); reverse_treble = self.detective_reverse_treble_var.get()
            self._run_in_thread(
                partial(run_two_piece_mean_analysis, bass, treble, search_one,
                        reverse_bass=reverse_bass, reverse_treble=reverse_treble,
                        spline_stiffness_pct=stiffness_pct, detrend_mode=mode,
                        detrend_wavelength=wavelength, index=index),
                (), run_button, is_analysis=True)
            
    def _check_setup_requirements(self):
        if not os.path.exists("full_rwl_cache"):
            self.notebook.select(3)
            print("\n⚠️  SETUP REQUIRED: The RWL cache hasn't been downloaded yet.")
            print("Please click 'Download and Create Index' in the Setup tab to begin.")
            print("This step may take 30-45 minutes on first run.")
            messagebox.showinfo("Initial Setup", "Welcome! Before you can run analysis, you need to download the RWL reference files.\n\nClick 'Download and Create Index' in the Setup tab.\nThis may take 30-45 minutes on first run.")

    def on_closing(self):
        print("Saving settings..."); self.save_settings()
        # The widget is about to go; anything printed after this belongs on the console.
        sys.stdout = self._stdout_before
        self.destroy()
    def save_settings(self):
        settings = {'date_sample': self.date_sample_entry.get(), 'date_master': self.date_master_entry.get(), 'treble_file': self.date_treble_entry.get(), 'detective_sample': self.detective_sample_entry.get(), 'detective_treble': self.detective_treble_entry.get(), 'detective_folder': self.detective_folder_entry.get(), 'create_folder': self.create_folder_entry.get(), 'create_output': self.create_output_entry.get(),
                    'date_index': self.date_index_combo.get(), 'detective_index': self.detective_index_combo.get(),
                    'same_tree_mode': self.same_tree_mode_var.get(), 'image_resolution_px_per_mm': self.image_resolution_var.get(),
                    'image_scale_calibrated': self.image_scale_calibrated_var.get(), 'image_source_note': self.image_source_note_var.get()}
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
                    # An unknown or missing index label falls back to the spline default
                    # rather than silently selecting whatever happens to be first.
                    for combo, key in ((self.date_index_combo, 'date_index'),
                                       (self.detective_index_combo, 'detective_index')):
                        label = settings.get(key, self.INDEX_DEFAULT)
                        combo.set(label if label in self.INDEX_PRESETS else self.INDEX_DEFAULT)
                    self.same_tree_mode_var.set(settings.get('same_tree_mode', 'conservative'))
                    self.image_resolution_var.set(settings.get('image_resolution_px_per_mm', ''))
                    self.image_scale_calibrated_var.set(settings.get('image_scale_calibrated', False))
                    self.image_source_note_var.set(settings.get('image_source_note', ''))
        except (json.JSONDecodeError, KeyError): print(f"Warning: Could not read '{self.settings_file}'.")
    def _drain_log(self):
        """Move whatever the worker threads printed into the log widget, on this thread."""
        chunks = []
        try:
            while True:
                chunks.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        if chunks:
            self.log_widget.config(state=tk.NORMAL)
            self.log_widget.insert(tk.END, ''.join(chunks))
            self.log_widget.see(tk.END)
            self.log_widget.config(state=tk.DISABLED)

    def _on_main(self, fn):
        """Queue a callable for the main thread. The only way a worker touches the GUI.

        `after()` is itself a Tk call, so calling it from a worker thread is the same
        violation in a quieter form; a plain queue drained by the timer below is not."""
        self.ui_queue.put(fn)

    def _drain_ui(self):
        while True:
            try:
                fn = self.ui_queue.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception as e:
                print(f"(GUI update failed: {e})")

    def check_plot_queue(self):
        try:
            self._drain_log()
            self._drain_ui()
            plot_args = self.plot_queue.get_nowait()
            if plot_args: plot_results(plot_args) # Pass the whole dict
        except queue.Empty: pass
        finally: self.after(100, self.check_plot_queue)
    def _attach_stability(self, result):
        """Re-date the winning candidate under several detrending settings.

        Runs on every analysis rather than behind a switch: a date that only survives one
        filter length is not a finding, and the user should not have to opt in to learning
        that. Four extra cross-dates against a single reference is negligible next to the
        search that produced the candidate."""
        try:
            sample = result.get('sample_filename')
            master = result.get('master_filename')
            if not sample or not master or not os.path.exists(str(master)):
                return
            # Filter lengths only; the index dimension is covered by index_agreement.
            lead = result.get('index')
            table = stability_check(sample, master,
                                    min_overlap=result.get('min_overlap'),
                                    index=lead, index_list=[lead])
            result['stability_table'] = table
            result['stability_verdict'] = stability_verdict(table)
            agreement = result.get('index_agreement') or {}
            result['end_year_stable_across_index'] = (
                'not tested' if agreement.get('agree') is None
                else ('yes' if agreement['agree'] else 'no'))
        except Exception as e:
            print(f"(Stability check could not be run: {e})")

    def _run_in_thread(self, target_func, args, button_to_disable, is_analysis=False):
        def thread_target():
            if button_to_disable: self._on_main(lambda: button_to_disable.config(state=tk.DISABLED))
            if is_analysis: self._on_main(lambda: self.save_report_button.config(state=tk.DISABLED))
            try:
                result = target_func(*args)
                if is_analysis and result:
                    self._attach_stability(result)
                    self.last_analysis_results = result
                    if 'raw_sample' in result:
                        # Pass the ENTIRE results dictionary to the plot queue
                        self.plot_queue.put(result)
                    self._on_main(lambda: self.save_report_button.config(state=tk.NORMAL))

                    report_content = self._create_report_content()
                    if report_content:
                        print("\n\n" + "="*70 + "\n           AUTOMATICALLY GENERATED REPORT\n"
                              + "="*70 + "\n" + report_content)
            except Exception as e:
                error_message = f"An error occurred:\n\n{e}"
                print(f"\n--- ERROR ---\n{error_message}")
                self._on_main(lambda: messagebox.showerror("Operation Error", error_message))
            finally:
                if button_to_disable: self._on_main(lambda: button_to_disable.config(state=tk.NORMAL))
        thread = threading.Thread(target=thread_target); thread.daemon = True; thread.start()
    def _create_tabs(self):
        self._create_date_tab()
        self._create_detective_tab()
        self._create_master_tab()
        self._create_index_build_tab()
        self._create_methodology_tab()
    def _create_report_widget(self, parent):
        basis_frame = ttk.LabelFrame(parent, text="Measurement Basis (Image Capture)")
        basis_frame.pack(pady=5, padx=0, fill="x")
        ttk.Label(basis_frame, text="Image Resolution (px/mm):").grid(row=0, column=0, padx=5, pady=3, sticky="w")
        ttk.Entry(basis_frame, textvariable=self.image_resolution_var, width=10).grid(row=0, column=1, padx=5, pady=3, sticky="w")
        ttk.Checkbutton(basis_frame, text="Scale/ruler calibrated in frame", variable=self.image_scale_calibrated_var).grid(row=0, column=2, padx=15, pady=3, sticky="w")
        ttk.Label(basis_frame, text="Image Source Note:").grid(row=1, column=0, padx=5, pady=3, sticky="w")
        ttk.Entry(basis_frame, textvariable=self.image_source_note_var, width=50).grid(row=1, column=1, columnspan=2, padx=5, pady=3, sticky="w")

        report_frame = ttk.LabelFrame(parent, text="Report Generation")
        report_frame.pack(pady=5, padx=0, fill="x")
        self.save_report_button = ttk.Button(report_frame, text="Save Text Report...", command=self._save_report, state=tk.DISABLED)
        self.save_report_button.pack(pady=5)
    def _create_log_widget(self, parent):
        log_frame = ttk.LabelFrame(parent, text="Output Log")
        log_frame.pack(pady=5, padx=0, expand=True, fill="both")
        self.log_widget = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_widget.pack(expand=True, fill="both", padx=5, pady=5)
        sys.stdout = TextRedirector(self.log_queue)
    def _create_date_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="1. Date")
        self.date_reverse_sample_var = tk.BooleanVar()
        self.date_reverse_treble_var = tk.BooleanVar()
        type_frame = ttk.LabelFrame(tab, text="Analysis Type"); type_frame.pack(padx=20, pady=5, fill="x")
        self.date_type_var = tk.StringVar(value="single")
        def toggle():
            is_two_piece = self.date_type_var.get() == "two_piece"
            self._show_treble_row(
                is_two_piece, self.date_sample_label,
                (self.date_treble_label, self.date_treble_entry, self.date_treble_browse,
                 self.date_treble_reverse_check))
        ttk.Radiobutton(type_frame, text="Single Sample", variable=self.date_type_var, value="single", command=toggle).pack(side="left", padx=10)
        ttk.Radiobutton(type_frame, text="Two-Piece Mean", variable=self.date_type_var, value="two_piece", command=toggle).pack(side="left", padx=10)
        ttk.Label(type_frame, text="   Same-tree threshold:").pack(side="left", padx=(20, 2))
        ttk.Radiobutton(type_frame, text="Permissive (T≥6)", variable=self.same_tree_mode_var, value="permissive").pack(side="left", padx=5)
        ttk.Radiobutton(type_frame, text="Conservative (T≥10)", variable=self.same_tree_mode_var, value="conservative").pack(side="left", padx=5)
        frame = ttk.LabelFrame(tab, text="File Inputs & Options"); frame.pack(padx=20, pady=5, fill="x")
        self.date_sample_label, self.date_sample_entry, _, _ = self._path_row(
            frame, 0, "Sample File (.rwl):",
            lambda: self._browse_file(self.date_sample_entry, title="Select a Sample File",
                                      callback=self._update_default_overlap),
            reverse_var=self.date_reverse_sample_var)
        # The treble row is built now and placed only in two-piece mode (see toggle).
        (self.date_treble_label, self.date_treble_entry, self.date_treble_browse,
         self.date_treble_reverse_check) = self._path_row(
            frame, 1, "Treble Side File (.rwl):",
            lambda: self._browse_file(self.date_treble_entry, callback=self._update_default_overlap),
            reverse_var=self.date_reverse_treble_var, grid_now=False)
        _, self.date_master_entry, _, _ = self._path_row(
            frame, 2, "Reference File (.csv/.rwl):",
            lambda: self._browse_file(self.date_master_entry, is_master=True))
        ttk.Label(frame, text="Minimum Overlap (years):").grid(row=3, column=0, padx=5, pady=10, sticky="w")
        self.date_min_overlap_spinbox = ttk.Spinbox(frame, from_=30, to=500, increment=10, width=5); self.date_min_overlap_spinbox.set(DEFAULTS['min_overlap'])
        self.date_min_overlap_spinbox.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        ttk.Label(frame, text="Detrending:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.date_detrend_combo = ttk.Combobox(frame, values=list(self.DETREND_PRESETS), width=26, state="readonly"); self.date_detrend_combo.set(self.DETREND_DEFAULT)
        self.date_detrend_combo.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.date_index_combo = self._add_index_combo(frame, row=5)
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
            self._show_treble_row(
                is_two_piece, self.detective_sample_label,
                (self.detective_treble_label, self.detective_treble_entry,
                 self.detective_treble_browse, self.detective_treble_reverse_check))
        ttk.Radiobutton(type_frame, text="Single Sample", variable=self.detective_type_var, value="single", command=toggle).pack(side="left", padx=10)
        ttk.Radiobutton(type_frame, text="Two-Piece Mean", variable=self.detective_type_var, value="two_piece", command=toggle).pack(side="left", padx=10)
        ttk.Label(type_frame, text="   Same-tree threshold:").pack(side="left", padx=(20, 2))
        ttk.Radiobutton(type_frame, text="Permissive (T≥6)", variable=self.same_tree_mode_var, value="permissive").pack(side="left", padx=5)
        ttk.Radiobutton(type_frame, text="Conservative (T≥10)", variable=self.same_tree_mode_var, value="conservative").pack(side="left", padx=5)
        frame = ttk.LabelFrame(tab, text="File Inputs"); frame.pack(padx=20, pady=5, fill="x")
        self.detective_sample_label, self.detective_sample_entry, _, _ = self._path_row(
            frame, 0, "Sample File (.rwl):",
            lambda: self._browse_file(self.detective_sample_entry, title="Select a Sample File",
                                      callback=self._update_default_overlap),
            reverse_var=self.detective_reverse_sample_var)
        (self.detective_treble_label, self.detective_treble_entry, self.detective_treble_browse,
         self.detective_treble_reverse_check) = self._path_row(
            frame, 1, "Treble Side File (.rwl):",
            lambda: self._browse_file(self.detective_treble_entry, callback=self._update_default_overlap),
            reverse_var=self.detective_reverse_treble_var, grid_now=False)
        target_frame = ttk.LabelFrame(tab, text="Reference Target"); target_frame.pack(padx=20, pady=5, fill="x")
        self.detective_target_var = tk.StringVar(value="category")
        ttk.Radiobutton(target_frame, text="Predefined Category:", variable=self.detective_target_var, value="category").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        # Show the readable label but keep the key the engine expects. Raw keys like
        # 'violin' and 'all' do not say what the reference set actually contains.
        self._target_by_label = {TARGET_LABELS.get(k, k): k for k in DETECTIVE_TARGETS}
        self.detective_category_combo = ttk.Combobox(target_frame, values=list(self._target_by_label), width=28, state="readonly")
        self.detective_category_combo.set(VIOLIN_REFERENCE_LABEL)
        self.detective_category_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        # The label here is a radio button rather than a plain label, so this row is placed
        # by hand; the entry and browse button still come from the shared helper.
        ttk.Radiobutton(target_frame, text="Local Folder:", variable=self.detective_target_var, value="folder").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        _, self.detective_folder_entry, folder_browse, _ = self._path_row(
            target_frame, 1, "", lambda: self._browse_folder(self.detective_folder_entry),
            grid_now=False)
        self.detective_folder_entry.grid(row=1, column=1, padx=5, pady=5)
        folder_browse.grid(row=1, column=2, padx=5, pady=5)
        options_frame = ttk.LabelFrame(tab, text="Options"); options_frame.pack(padx=20, pady=5, fill="x")
        ttk.Label(options_frame, text="Show Top N Results:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.detective_top_n_spinbox = ttk.Spinbox(options_frame, from_=1, to=100, width=5); self.detective_top_n_spinbox.set(DEFAULTS['top_n'])
        self.detective_top_n_spinbox.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(options_frame, text="Minimum Overlap (years):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.detective_min_overlap_spinbox = ttk.Spinbox(options_frame, from_=30, to=500, increment=10, width=5); self.detective_min_overlap_spinbox.set(DEFAULTS['min_overlap'])
        self.detective_min_overlap_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(options_frame, text="Only Include Sites Ending After:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.detective_min_end_year_spinbox = ttk.Spinbox(options_frame, from_=0, to=2100, increment=50, width=5); self.detective_min_end_year_spinbox.set(DEFAULTS['min_end_year'])
        self.detective_min_end_year_spinbox.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(options_frame, text="Detrending:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.detective_detrend_combo = ttk.Combobox(options_frame, values=list(self.DETREND_PRESETS), width=26, state="readonly"); self.detective_detrend_combo.set(self.DETREND_DEFAULT)
        self.detective_detrend_combo.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.detective_index_combo = self._add_index_combo(options_frame, row=4)
        run_button = ttk.Button(tab, text="Run Detective Analysis", command=self._run_detective); run_button.pack(pady=10)
        toggle()
    def _create_master_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="3. Create Master")
        frame = ttk.LabelFrame(tab, text="Create a Custom Master Chronology"); frame.pack(padx=20, pady=20, fill="x")
        _, self.create_folder_entry, _, _ = self._path_row(
            frame, 0, "Input Folder (.rwl files):",
            lambda: self._browse_folder(self.create_folder_entry))
        _, self.create_output_entry, _, _ = self._path_row(
            frame, 1, "Output Filename (.csv):", self._save_file_as, button_text="Save As...")
        # A master must be built under the index it will be searched under: a log index
        # cannot be recovered from a finished spline index.
        self.create_index_combo = self._add_index_combo(frame, row=2, single=True)
        run_button = ttk.Button(tab, text="Create Master File", command=self._run_create); run_button.pack(pady=20)
    def _create_index_build_tab(self):
        tab = ttk.Frame(self.notebook); self.notebook.add(tab, text="4. Setup")
        index_frame = ttk.LabelFrame(tab, text="Step 1: Download & Index NOAA Files (Run once)")
        index_frame.pack(padx=20, pady=10, fill="x")
        self.index_button = ttk.Button(index_frame, text="Download and Create Index", command=self._run_download)
        self.index_button.pack(pady=5)
        
        build_frame = ttk.LabelFrame(tab, text="Step 2: Build General Master Chronologies")
        build_frame.pack(padx=20, pady=10, fill="x")
        build_options_grid = ttk.Frame(build_frame)
        build_options_grid.pack(pady=5)
        ttk.Label(build_options_grid, text="Select a predefined master to build:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self._build_by_label = {TARGET_LABELS.get(k, k): k for k in CATEGORY_NAMES}
        self._build_by_label["Every master (all four)"] = 'every'
        self.build_target_combo = ttk.Combobox(build_options_grid, values=list(self._build_by_label), width=26, state="readonly")
        self.build_target_combo.set("Every master (all four)")
        self.build_target_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(build_options_grid, text="Only Include Sites Ending After:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.build_min_end_year_spinbox = ttk.Spinbox(build_options_grid, from_=0, to=2100, increment=50, width=5); self.build_min_end_year_spinbox.set(DEFAULTS['min_end_year'])
        self.build_min_end_year_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(build_options_grid, text="Minimum Sites Per Year:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.build_min_depth_spinbox = ttk.Spinbox(build_options_grid, from_=1, to=50, width=5); self.build_min_depth_spinbox.set(DEFAULTS['min_series_depth'])
        self.build_min_depth_spinbox.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.build_index_combo = self._add_index_combo(build_options_grid, row=3, single=True)
        self.build_button = ttk.Button(build_frame, text="Build Selected", command=self._run_build); self.build_button.pack(pady=10)
        ttk.Button(build_frame, text="Rebuild Index Only (no download)", command=self._run_reindex).pack(pady=(0, 8))

        violin_frame = ttk.LabelFrame(tab, text=f"Step 3: Build {VIOLIN_REFERENCE_LABEL} (Recommended)")
        violin_frame.pack(padx=20, pady=10, fill="x")
        violin_label = ttk.Label(violin_frame, wraplength=550,
                                 text=f"{VIOLIN_REFERENCE_BLURB}\n\nGathers them from the downloaded cache and "
                                      f"builds '{VIOLIN_MASTER_FILENAME}' in the '{VIOLIN_REFERENCE_DIR}' folder.")
        violin_label.pack(pady=(5,10), padx=5)
        self.violin_button = ttk.Button(violin_frame, text="Fetch Tonewood Forest References", command=self._run_violin_setup)
        self.violin_button.pack(pady=5)
    
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

            --- INDICES ---
            Three indices are computed by default, and the end year is reported under each,
            so the index cannot be chosen after seeing which one gives the better t-value.

            • spline: ratio to a cubic smoothing spline (Cook & Peters, 1981). Fixed 32-year
              cutoff by default, applied identically to sample and reference. Loses no rings.
            • hollstein: d(y) = ln( w(y) / w(y-1) ) on raw ring widths. Loses 1 ring at the start.
            • bp: d(y) = ln( 5*w(y) / (w(y-2)+w(y-1)+w(y)+w(y+1)+w(y+2)) ) on raw ring widths.
              Loses 2 rings at each end.

            Rings lost to a transform are not compared, so each t-value is reported with the
            overlap n it was computed over. The three are not on a common scale: t-values from
            different indices must not be compared with one another.

            --- STATISTICAL VALIDATION: A MULTI-DIMENSIONAL APPROACH ---
            To determine the strength of a tree-ring match, this method uses three key metrics simultaneously, as a simple t-value can be misleading.

            1. t-value (Baillie-Pilcher): Measures statistical similarity.
            2. Overlap (Years): The number of shared rings; longer overlaps provide more reliable matches.
            3. Gleichläufigkeit (%): A classical German statistic measuring agreement in year-to-year growth direction (Eckstein & Bauch, 1969).

            This approach ensures matches are statistically significant, biologically meaningful, and visually consistent—crucial for high-stakes applications like dating antique instruments.

            --- TIER CRITERIA ---
            Tiers are named by the criteria they apply. All three conditions must hold.

            - tier t7/n80/g70:  t ≥ 7.0, n ≥ 80, GLK ≥ 70
            - tier t6/n70/g65:  t ≥ 6.0, n ≥ 70, GLK ≥ 65
            - tier t5/n50/g60:  t ≥ 5.0, n ≥ 50, GLK ≥ 60
            - no tier met:      anything that fails all three. A high t over a short overlap,
                                or with a low GLK, meets no tier.

            These are conventions from the dendrochronological literature, not derived
            constants, and published thresholds vary between laboratories. The values commonly
            quoted on instrument reports (3.5 floor, 6 significant, 8-10 same-tree) were
            calibrated on the Baillie-Pilcher and Hollstein indices, so a t computed here on
            the spline index is not directly comparable to a t on such a report.

            Reports state the measured values, which tier's criteria they meet, and the shared
            year-to-year variation r² = t²/(t²+n-2). No adjective is applied to a result.

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
        # Only the tab being worked in is adjusted, and only while it still holds the
        # default. Previously this wrote to both tabs on every file selection, so picking
        # a sample in one tab silently changed the other tab's setting, and a value the
        # user had deliberately typed was overwritten without notice.
        target_spinbox = self.date_min_overlap_spinbox if is_date_tab else self.detective_min_overlap_spinbox
        if new_overlap < 30:
            return
        try:
            current = int(target_spinbox.get())
        except (TypeError, ValueError):
            current = DEFAULTS['min_overlap']
        if current != DEFAULTS['min_overlap']:
            print(f"Suggested overlap for this sample is {new_overlap} years; keeping your setting of {current}.")
            return
        target_spinbox.set(new_overlap)
    def _browse_file(self, entry_widget, is_master=False, callback=None, title="Select a file"):
        """Fill one entry from a file dialog. Only the entry it was given, never another.

        The Date and Survey tabs keep their own sample paths -- save_settings stores them
        under separate keys -- so browsing on one tab must not overwrite the other's."""
        types = (("RWL files", "*.rwl"), ("All files", "*.*")) if not is_master else (("All files", "*.*"),("CSV files", "*.csv"),("RWL files", "*.rwl"))
        filename = filedialog.askopenfilename(title=title, filetypes=types)
        if filename:
            entry_widget.delete(0, tk.END); entry_widget.insert(0, filename)
            if callback: callback()
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
        index = self.MASTER_INDEX_PRESETS.get(self.create_index_combo.get(), 'spline')
        self._run_in_thread(partial(run_create_master, folder, output, index=index), (), button)
    def _run_download(self):
        self._run_in_thread(download_and_index_files, (), self.index_button)
    def _run_build(self):
        label = self.build_target_combo.get()
        target = self._build_by_label.get(label, label)
        min_end_year = int(self.build_min_end_year_spinbox.get())
        min_depth = int(self.build_min_depth_spinbox.get())
        index = self.MASTER_INDEX_PRESETS.get(self.build_index_combo.get(), 'spline')
        # Each category is built into its own master file; they are never merged, so an
        # Alpine spruce reference stays distinct from Alpine stone pine and Baltic pine.
        wanted = list(CATEGORY_NAMES) if target == 'every' else [target]

        def build_all():
            for name in wanted:
                try:
                    build_master_from_index(name, min_end_year=min_end_year, min_depth=min_depth,
                                            index=index)
                except Exception as e:
                    print(f"  -> Could not build '{name}': {e}")
        self._run_in_thread(build_all, (), self.build_button)

    def _run_reindex(self):
        self._run_in_thread(index_cache, (), self.build_button)
    def _run_violin_setup(self):
        self._run_in_thread(fetch_and_build_violin_master, (), self.violin_button)
    
    def _classify_dendro_match(self, t_value, overlap_years, gleich_percent):
        """Classify match strength. Single source of truth lives in gogo.py."""
        return _gogo_classify(t_value, overlap_years, gleich_percent)

    def _processing_block(self, res):
        """Task 4: disclose the detrending/statistics pipeline actually used for this run."""
        stiffness = res.get('spline_stiffness_pct') or DEFAULTS['spline_stiffness_pct']
        min_overlap = res.get('min_overlap', 'not recorded')
        mode = res.get('detrend_mode', DEFAULTS['detrend_mode'])
        wavelength = res.get('detrend_wavelength', DEFAULTS['detrend_wavelength'])
        index = res.get('index', DEFAULTS['lead_index'])
        if index != 'spline':
            # There is no spline in a log index, so quoting a cutoff here would describe a
            # step that was not performed.
            detrend_line = ("Detrending: none -- the index above is computed directly on raw ring "
                            "widths.\n            The spline cutoff setting does not apply and was "
                            "not used.\n")
        elif mode == 'percent':
            detrend_line = (f"Detrending: Cook & Peters (1981) cubic smoothing spline, cutoff {stiffness}% of series length.\n"
                            "            NOTE: sample and reference differ in length, so this filters them in\n"
                            "            different frequency bands. A fixed cutoff is preferred.\n")
        else:
            detrend_line = (f"Detrending: Cook & Peters (1981) cubic smoothing spline, fixed {wavelength}-year cutoff,\n"
                            "            applied identically to sample and reference so both are filtered in the\n"
                            "            same frequency band.\n")
        rings_in, rings_out = res.get('rings_measured'), res.get('rings_after_index')
        reported = res.get('indices_reported') or [index]
        index_lines = [f"Indices computed: {', '.join(reported)} (statistics below are the "
                       f"{index} index)", "            " + describe_index(index)]
        if rings_in is not None and rings_out is not None:
            index_lines.append(f"            {rings_out} of {rings_in} measured rings survive this "
                               f"transform; every overlap n counts surviving rings.")
        index_lines.append("            " + convention_note(index))
        return (
            "PROCESSING\n"
            + '\n'.join(index_lines) + '\n'
            + detrend_line +
            "Reference chronologies: each tree standardised separately, then combined with a\n"
            "            biweight robust mean (sample depth reported alongside the index).\n"
            "Autocorrelation: NOT removed. Series are standard-index, not residual.\n"
            f"Minimum overlap required: {min_overlap} years.\n"
            "Statistics reported: Student's t, overlap (n years), Gleichläufigkeit (%)."
        )

    def _plate_relationship_block(self, res):
        """Report whether the two halves of a two-piece plate came from one wedge."""
        rel = res.get('plate_relationship')
        if not rel:
            return ""
        headline = {'same_wedge': 'ONE WEDGE (book-matched from a single tree)',
                    'different_logs': 'DIFFERENT LOGS (two separate trees)',
                    'inconclusive': 'INCONCLUSIVE'}.get(rel['verdict'], rel['verdict'])
        lines = ["PLATE RELATIONSHIP", f"Verdict: {headline}",
                 f"Bass vs. treble: t = {rel['t_value']:.2f}, Glk = {rel['glk']:.1f}%, overlap = {rel['overlap_n']} years.",
                 rel['note']]
        if not rel['same_wedge']:
            lines.append("Each half was therefore dated on its own; no averaged chronology was formed.")
        bass, treble = res.get('bass_result'), res.get('treble_result')

        def _summary(label, r):
            if not r:
                return f"{label}: no result."
            bm = r.get('results', {}).get('best_match', {})
            if not bm:
                return f"{label}: no result."
            return (f"{label}: end year {int(bm.get('end_year', 0))}, t = {bm.get('t_value', 0):.2f}, "
                    f"Glk = {bm.get('glk', 0):.1f}%, overlap = {int(bm.get('overlap_n', 0))}.")
        lines.append(_summary("Bass side alone", bass))
        lines.append(_summary("Treble side alone", treble))
        if bass and treble:
            b = bass.get('results', {}).get('best_match', {}).get('end_year')
            t = treble.get('results', {}).get('best_match', {}).get('end_year')
            if b is not None and t is not None:
                agree = "agree" if int(b) == int(t) else f"DISAGREE by {abs(int(b) - int(t))} years"
                lines.append(f"The two halves {agree} on the end year.")
        return "\n".join(lines)

    def _terminus_block(self, res):
        """State plainly what a last-ring date does and does not establish.

        The wording lives in gogo so the desktop report, the CLI and the server cannot
        drift into describing the same result differently."""
        bm = res.get('results', {}).get('best_match', {})
        end_year = bm.get('end_year')
        if end_year is None:
            return ""
        return terminus_post_quem_note(end_year)

    def _provenance_block(self, res):
        """The conditions this result was produced under, so it can be reproduced."""
        manifest = res.get('run_manifest')
        if not manifest:
            return ""
        block = format_run_manifest(manifest)
        depth = res.get('depth_at_match')
        if depth is not None:
            block += (f"\n  reference depth at the reported end year: {depth} site(s)")
        return block

    def _holdout_block(self, res):
        """Whether the sample's own series was inside the reference it was tested against.

        Always says something: 'the check could not be performed' is itself a finding, and
        a reader told nothing cannot distinguish it from a check that passed."""
        note = res.get('holdout_note')
        if not note:
            return ""
        lines = ["SERIES HOLDOUT", note]
        hold = res.get('holdout') or {}
        if hold.get('performed') and hold.get('depth_year') is not None:
            before, after = hold.get('depth_before_at_year'), hold.get('depth_after_at_year')
            label = hold.get('depth_label', 'series')
            lines.append(f"Reference depth at {hold['depth_year']}: "
                         f"{'-' if before is None else before} {label}(s) before the holdout, "
                         f"{'-' if after is None else after} after "
                         f"(minimum required: {hold.get('depth_floor', '-')}).")
        for info in (res.get('holdout_unusable_references') or []):
            lines.append(info.get('unusable_reason', ''))
        if res.get('unusable_reason'):
            lines.append(res['unusable_reason'])
        return "\n".join(l for l in lines if l)

    def _reference_set_block(self, res):
        """What the search could have matched against."""
        text = res.get('reference_composition_text')
        if text:
            return text
        composition = res.get('reference_composition')
        return format_reference_set(composition) if composition else ""

    def _stability_block(self, res):
        """Whether the reported end year survives a change of detrending setting."""
        table = res.get('stability_table')
        if table is None or getattr(table, 'empty', True):
            return ""
        verdict = res.get('stability_verdict') or {}
        agreement = res.get('index_agreement') or {}
        lines = ["STABILITY UNDER THE DETRENDING SETTING",
                 table.to_string(index=False),
                 verdict.get('summary', '')]
        if not verdict.get('stable', False):
            lines.append("Treat a date that moves with the filter length as unresolved, "
                         "not as the best of the alternatives.")
        lines.append(f"end_year_stable_across_index: "
                     f"{res.get('end_year_stable_across_index', 'not tested')} -- "
                     f"{agreement.get('headline', '')}")
        lines.append("Every t is read against its own overlap n: the indices lose different "
                     "numbers of rings at the series ends.")
        return "\n".join(l for l in lines if l)

    def _measurement_basis_block(self):
        """Task 3: disclose image resolution/calibration. Absence is itself the finding, so it is never omitted."""
        res_str = self.image_resolution_var.get().strip()
        calibrated = self.image_scale_calibrated_var.get()
        source_note = self.image_source_note_var.get().strip()
        try:
            px_per_mm = float(res_str) if res_str else None
        except ValueError:
            px_per_mm = None

        lines = ["MEASUREMENT BASIS", "Measurements were taken from digital images."]
        if px_per_mm and px_per_mm > 0:
            lines.append(f"Image resolution: {px_per_mm:g} px/mm (nominal ring-boundary resolution ≈ {1.0 / px_per_mm:.3f} mm)")
        else:
            lines.append("Image resolution not recorded. Measurement precision cannot be assessed from this report.")
        lines.append(f"Scale calibration: {'present in frame' if calibrated else 'NOT CALIBRATED — resolution figure is nominal only'}")
        lines.append(f"Source: {source_note if source_note else 'not recorded'}")
        if not calibrated:
            lines.append(
                "Without an in-frame scale, absolute ring widths may carry a systematic scaling error. This affects "
                "mean ring width and any comparison of absolute widths; it has limited effect on crossdating, which "
                "depends on relative year-to-year variation."
            )
        return "\n".join(lines)

    def _same_tree_paragraph(self, res):
        """Task 1: same-tree threshold is a separate, stronger claim than the dating (crossdating) threshold."""
        mode = self.same_tree_mode_var.get()
        if mode not in ("permissive", "conservative"):
            mode = "conservative"
        threshold = 6.0 if mode == "permissive" else 10.0
        wording = ("consistent with a common origin, possibly the same tree" if mode == "permissive"
                   else "consistent with an origin in the same tree")
        internal = res.get('internal_stats', {})
        internal_t = internal.get('t_value', 0.0)
        internal_overlap = internal.get('overlap_n', 0)
        caveat = (
            f"Threshold used: T ≥ {threshold:.1f} ({mode} mode). The reliability of T-value thresholds for "
            f"same-tree identification is contested in the literature; visual comparison of the growth curves is "
            f"required to support this reading and has not been performed by this software."
        )
        if internal_t >= threshold:
            return (
                f"The two halves of the belly show an internal cross-match (T-value = {internal_t:.2f}, "
                f"overlap = {internal_overlap} years) {wording}. They were subsequently combined into a single mean "
                f"series for the final dating analysis. {caveat}"
            )
        return (
            f"The two halves of the belly did not meet the same-tree criterion (T-value = {internal_t:.2f}, "
            f"overlap = {internal_overlap} years; threshold T ≥ {threshold:.1f}, {mode} mode). They were "
            f"nonetheless combined into a mean chronology for dating purposes: combining requires only that the two "
            f"halves be shown to be contemporaneous (T ≥ 6.0 in the internal cross-match), which is a weaker "
            f"claim than common origin. {caveat}"
        )

    def _search_context_block(self, res):
        """Brief 2, Task 1: where the winning alignment sits among every alignment tested,
        not only those that passed the classification threshold. Pure summarisation of data
        already produced by the search — no new search is performed here."""
        n = res.get('search_n_alignments')
        if n is None:
            return None  # older/single-mode result with no search-context data; skip silently
        if not res.get('search_stats_reliable', n >= 100):
            return (
                "SEARCH CONTEXT\n"
                f"Only {n} alignment positions were evaluated — too few to characterise the background "
                "distribution. The reported result cannot be placed in search context from this run."
            )

        ref_set_n = res.get('ref_set_n', 'not recorded')
        best_t = res.get('results', {}).get('best_match', {}).get('t_value', 0.0)
        rank = res.get('best_t_rank', 'n/a')
        pct = res.get('best_t_percentile', 0.0)
        t_p50, t_p95, t_p99 = res.get('t_p50', 0.0), res.get('t_p95', 0.0), res.get('t_p99', 0.0)
        n_t5, n_t6, n_t8 = res.get('n_above_t5', 0), res.get('n_above_t6', 0), res.get('n_above_t8', 0)

        lines = [
            "SEARCH CONTEXT",
            f"Alignment positions evaluated: {n:,} across {ref_set_n} reference chronologies.",
            "Distribution of all t-values produced by this search:",
            f"    median {t_p50:.2f}   95th percentile {t_p95:.2f}   99th percentile {t_p99:.2f}",
            f"The reported alignment (t = {best_t:.2f}) ranks {rank} of {n:,}",
            f"    — higher than {pct:.2%} of all positions tested.",
            f"Alignments exceeding t = 5.0: {n_t5}   t = 6.0: {n_t6}   t = 8.0: {n_t8}",
            "",
            "How to read this: a crossdating search evaluates many thousands of possible",
            "alignments and reports the highest-scoring one. The highest of many thousands of",
            "values is elevated by the size of the search alone, even when no genuine",
            "relationship is present. These figures show the background the reported result",
            "was drawn from. A result far out in the tail of this distribution carries weight;",
            "a result close to the levels many other alignments reached is largely a",
            "reflection of how many positions were tried.",
        ]
        if res.get('search_stats_sampled'):
            sample_n = len(res.get('search_t_values', []))
            lines.append("")
            lines.append(
                f"(Percentiles above are estimated from a random sample of {sample_n:,} of the {n:,} "
                "alignments evaluated, retained for memory efficiency.)"
            )
        return "\n".join(lines)

    def _reference_context_paragraphs(self, res, candidates_csv_name=None):
        """Task 5 (reference set searched) and Task 2 (every candidate alignment, not only the winner)."""
        paragraphs = []
        if res.get('analysis_type') == 'detective':
            df = res.get('enriched_results_df')
            if df is not None and not df.empty:
                top_match = df.iloc[0].to_dict()
                context_lines = ["The analysis was performed against a database of regional "
                                 "chronologies. The highest-scoring reference sites are:"]
                for i in range(min(5, len(df))):
                    row = df.iloc[i]
                    site_info = f"{row.get('location', 'Unknown Location').strip()} ({row.get('site_name', 'Unknown Site').strip()})"
                    t, o, g = row.get('t_value', 0.0), int(row.get('overlap_n', 0)), row.get('glk', 0.0)
                    r2 = shared_variation(t, o)
                    context_lines.append(
                        f"  - {site_info} (t={t:.2f} over n={o}, GLK={g:.1f}%, "
                        f"r2={'-' if r2 is None else f'{r2:.2f}'}, "
                        f"{classification_handle(self._classify_dendro_match(t, o, g))})")
                paragraphs.append("\n".join(context_lines))

                top_location = top_match.get('location', 'N/A').strip()
                if top_location != 'N/A' and top_location:
                    paragraphs.append(
                        f"The strongest alignment in the tested set is with the reference site "
                        f"'{top_location}' (location read from the file header, often approximate). "
                        f"That is a statement about which chronology in this set scored highest, "
                        f"not about where the wood grew.")
            else:
                paragraphs.append("Detective analysis was run, but no significantly matching reference sites were found in the target database.")

            # Task 5: describe the composition of the set that was searched.
            # Task 2 (hardening): header locations are frequently truncated or junk; if more
            # than 30% of chronologies yielded nothing usable, a location list would mislead
            # more than it informs, so state that instead of printing a partial list.
            n = res.get('ref_set_n', 'not recorded')
            folder = res.get('ref_set_dir', 'not recorded')
            locations = res.get('ref_set_locations') or []
            n_ok = res.get('ref_set_location_n_ok', len(locations))
            n_total = res.get('ref_set_location_n_total', n if isinstance(n, int) else 0)
            if n_total > 0 and (n_ok / n_total) <= 0.7:
                geo = f"not reliably derivable from file headers ({n_ok} of {n_total} parsed)."
            elif locations:
                geo = ", ".join(locations[:15])
                if len(locations) > 15:
                    geo += f", …and {len(locations) - 15} more"
            else:
                geo = "not derivable from file headers"
            start, end = res.get('ref_set_start'), res.get('ref_set_end')
            span = f"{start}–{end}" if start is not None and end is not None else "not recorded"
            paragraphs.append(
                "REFERENCE SET\n"
                f"Chronologies searched: {n}\n"
                f"Source directory: {folder}\n"
                f"Geographic coverage: {geo}\n"
                f"Date span covered: {span}\n\n"
                "A best match is the best chronology in this set. If the true source region is absent or thinly "
                "represented, the best available match may not be the correct one. Location strings are read from "
                "file headers and are often approximate."
            )

            # Task 2: every alignment meeting threshold, not only the winner.
            n_positions, n_passing = res.get('n_positions_total', 0), res.get('n_passing', 0)
            cand_lines = [
                "ALL CANDIDATE ALIGNMENTS MEETING THRESHOLD",
                f"{n_positions} alignment positions were evaluated across {n} reference chronologies. "
                f"{n_passing} met the stated threshold."
            ]
            cand_df = res.get('candidate_alignments_df')
            if cand_df is not None and not cand_df.empty:
                cand_lines.append("")
                cand_lines.append(f"{'End Year':>8}  {'T':>6}  {'Overlap':>7}  {'GLK%':>6}  Reference")
                shown = cand_df.head(20)
                for _, row in shown.iterrows():
                    ref_label = f"{row.get('location', 'N/A')} ({row.get('site_name', 'N/A')}, {row.get('source_file', 'N/A')})"
                    cand_lines.append(f"{int(row['end_year']):>8}  {row['t_value']:>6.2f}  {int(row['overlap_n']):>7}  {row['glk']:>6.1f}  {ref_label}")
                remaining = len(cand_df) - len(shown)
                if remaining > 0:
                    if candidates_csv_name:
                        cand_lines.append(f"...and {remaining} further candidates. Full uncapped list: {candidates_csv_name}")
                    else:
                        cand_lines.append(f"...and {remaining} more (full list written to the companion CSV alongside the saved report).")
                if len(cand_df) > 1:
                    cand_lines.append("")
                    cand_lines.append(
                        "More than one alignment met the threshold. The strongest is reported above; the presence "
                        "of competing candidates should be weighed when interpreting it."
                    )
            paragraphs.append("\n".join(cand_lines))
        else:
            master_name = os.path.basename(res.get('master_filename', 'N/A'))
            paragraphs.append(
                "REFERENCE SET\n"
                f"One reference chronology was tested: {master_name}. No alternative reference chronologies were "
                "considered; this result reflects agreement with that single series only."
            )
            # Task 2 (general case): disclose alternate offsets against this same reference.
            all_corr = res.get('results', {}).get('all_correlations')
            best_end_year = int(res.get('results', {}).get('best_match', {}).get('end_year', -1))
            if all_corr is not None and not all_corr.empty:
                rows = all_corr.reset_index()
                passing_mask = rows.apply(
                    lambda r: self._classify_dendro_match(r['t_value'], r['overlap_n'], r['glk'])['meets_any'],
                    axis=1)
                passing = rows[passing_mask]
                alt = passing[passing['end_year'] != best_end_year].sort_values(by='t_value', ascending=False)
                lines = [
                    "ALL CANDIDATE END YEARS MEETING THRESHOLD (this reference)",
                    f"{len(rows)} alignment positions were evaluated against this single reference. "
                    f"{len(passing)} met the stated threshold."
                ]
                if not alt.empty:
                    lines.append("")
                    for _, row in alt.head(20).iterrows():
                        lines.append(f"  - End Year {int(row['end_year'])}: T={row['t_value']:.2f}, Overlap={int(row['overlap_n'])} yrs, GLK={row['glk']:.1f}%")
                    remaining = len(alt) - min(20, len(alt))
                    if remaining > 0:
                        lines.append(f"  ...and {remaining} more.")
                    lines.append("")
                    lines.append(
                        "More than one alignment against this reference met the threshold. The strongest is "
                        "reported above; competing candidates should be weighed when interpreting it."
                    )
                paragraphs.append("\n".join(lines))
        return paragraphs

    def _format_ring_series(self, series):
        """Render a ring-width series (mm) as decade-grouped rows for Task 6 (raw measurements)."""
        if series is None or len(series) == 0:
            return "  (no data)"
        idx, vals = list(series.index), list(series.values)
        lines = []
        for i in range(0, len(vals), 10):
            chunk_idx, chunk_vals = idx[i:i + 10], vals[i:i + 10]
            vals_str = " ".join(f"{v:6.2f}" for v in chunk_vals)
            lines.append(f"  [{chunk_idx[0]:>5}]{vals_str}")
        return "\n".join(lines)

    def _raw_measurements_block(self, res):
        """Task 6: append raw ring widths so a reader can re-run the analysis independently."""
        lines = [
            "RAW MEASUREMENTS",
            "(Ring width in mm, grouped in rows of 10; the bracketed number is the ring position from the measured starting edge, not a calendar year.)",
        ]
        if res.get('analysis_mode') == 'two_piece':
            bass_rev, treble_rev = res.get('reverse_bass', False), res.get('reverse_treble', False)
            bass_series, treble_series = res.get('raw_bass_series'), res.get('raw_treble_series')
            mean_series = res.get('raw_sample')
            bass_orient = "measured centre joint outwards" if bass_rev else "measured outer edge inwards"
            treble_orient = "measured centre joint outwards" if treble_rev else "measured outer edge inwards"
            lines += ["", f"Bass side ({bass_orient}), {len(bass_series) if bass_series is not None else 0} rings:", self._format_ring_series(bass_series)]
            lines += ["", f"Treble side ({treble_orient}), {len(treble_series) if treble_series is not None else 0} rings:", self._format_ring_series(treble_series)]
            lines += ["", f"Mean chronology used for dating, {len(mean_series) if mean_series is not None else 0} rings:", self._format_ring_series(mean_series)]
        else:
            sample_series = res.get('raw_sample')
            orient = "measured centre joint outwards" if res.get('reverse_sample', False) else "measured outer edge inwards"
            lines += ["", f"Sample ({orient}), {len(sample_series) if sample_series is not None else 0} rings:", self._format_ring_series(sample_series)]
        return "\n".join(lines)

    def _create_report_content(self, candidates_csv_name=None):
        """Generates the full report string based on the last analysis.

        `candidates_csv_name` names the companion candidates CSV once the save filename is
        known (Task 4, Brief 2); it is None for the auto-printed log preview generated right
        after an analysis runs, before the user has chosen where to save."""
        if not self.last_analysis_results:
            return ""

        res = self.last_analysis_results
        paragraphs = []

        # Unusable first, then whether the year survived the choice of index.
        if res.get('unusable_reason'):
            paragraphs.append(res['unusable_reason'])
        # Whether the same reference wins under every index comes first in a survey: a year
        # that survives the choice of index is one finding, a winner that survives it another.
        if res.get('candidate_agreement_text'):
            paragraphs.append(res['candidate_agreement_text'])
        if res.get('index_comparison_text'):
            paragraphs.append(res['index_comparison_text'])
        for block in (self._provenance_block(res), res.get('reference_metadata_text', ''),
                      self._holdout_block(res), self._reference_set_block(res)):
            if block:
                paragraphs.append(block)
        paragraphs.append(self._processing_block(res))
        paragraphs.append(self._measurement_basis_block())
        stability_block = self._stability_block(res)
        if stability_block:
            paragraphs.append(stability_block)
        plate_block = self._plate_relationship_block(res)
        if plate_block:
            paragraphs.append(plate_block)
        terminus_block = self._terminus_block(res)
        if terminus_block:
            paragraphs.append(terminus_block)

        # Paragraph: Physical Description
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

        # Paragraph: the measured values, the criteria they meet, and whose reading that is.
        best_match = res.get('results', {}).get('best_match', {})
        if best_match:
            t_value, overlap, glk = best_match.get('t_value', 0.0), best_match.get('overlap_n', 0), best_match.get('glk', 0.0)
            end_year = int(best_match.get('end_year', 0))
            paragraphs.append(
                f"The youngest measured ring dates to {end_year} (see WHAT THIS DATE "
                f"ESTABLISHES above).\n\n"
                + format_classification(self._classify_dendro_match(t_value, overlap, glk),
                                        index=res.get('index')))
        else:
            paragraphs.append("No alignment met the overlap floor, so no year is reported.")

        # Paragraph: Two-Piece same-tree assessment (Task 1 — separate, contested threshold; never "confirms")
        if is_two_piece:
            paragraphs.append(self._same_tree_paragraph(res))

        # Search context (Brief 2, Task 1): where the winning alignment sits among everything
        # tested, inserted after the dating result and before the candidate alignment list.
        if res.get('analysis_type') == 'detective':
            search_block = self._search_context_block(res)
            if search_block:
                paragraphs.append(search_block)

        # Reference context: what was searched (Task 5) and every candidate that met threshold (Task 2)
        paragraphs.extend(self._reference_context_paragraphs(res, candidates_csv_name))

        # What was searched per country prefix, after the ranked list: the only way a reader
        # can tell an absent match from an absent reference. Describes the search, not the wood.
        coverage_text = res.get('country_coverage_text')
        if not coverage_text and res.get('country_coverage_df') is not None:
            coverage_text = format_country_coverage(res['country_coverage_df'])
        if coverage_text:
            paragraphs.append(coverage_text)

        # Raw measurements (Task 6), appended at the end so the analysis can be re-run independently.
        paragraphs.append(self._raw_measurements_block(res))

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
        
        # Determine default save path and filename
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

        report_filename = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=default_filename,
            defaultextension=".txt",
            filetypes=(("Text Files", "*.txt"), ("All files", "*.*"))
        )
        if not report_filename:
            return

        # The candidates CSV's name is resolved before the report is rendered, so the report
        # text can name it exactly (Task 4, Brief 2) instead of pointing vaguely at "the
        # companion CSV alongside the saved report".
        base, _ = os.path.splitext(report_filename)
        cand_df = res.get('candidate_alignments_df')
        has_candidates = cand_df is not None and not cand_df.empty
        candidates_filename = f"{base}_candidates.csv"
        candidates_basename = os.path.basename(candidates_filename)

        report_content = self._create_report_content(candidates_csv_name=candidates_basename if has_candidates else None)
        if not report_content:
            messagebox.showerror("Error", "Could not generate report content.")
            return

        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Report saved to {report_filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save report file:\n{e}")
            return

        # Task 2 (Brief 1) / Task 4 (Brief 2): write the full set of candidate alignments
        # alongside the report, since the report text itself only prints the first ~20 rows.
        # Row count must match n_passing exactly or the printed and CSV lists can't be
        # reconciled from the report text alone.
        if has_candidates:
            try:
                cand_df.to_csv(candidates_filename, index=False)
                n_passing = res.get('n_passing', len(cand_df))
                assert len(cand_df) == n_passing, "candidate CSV row count does not match n_passing"
                print(f"Full candidate alignment list saved to {candidates_filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save candidate alignments CSV:\n{e}")

        # Per-country coverage, with the same columns the printed block uses.
        coverage_df = res.get('country_coverage_df')
        if coverage_df is not None and not coverage_df.empty:
            coverage_filename = f"{base}_country_coverage.csv"
            try:
                write_csv_with_manifest(coverage_df[list(COUNTRY_COVERAGE_COLUMNS)],
                                        coverage_filename, res.get('run_manifest', {}))
                print(f"Per-country coverage of the reference set saved to {coverage_filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save country coverage CSV:\n{e}")

        # Task 1 (Brief 2): companion t-value distribution behind the SEARCH CONTEXT block, one value per row.
        t_values = res.get('search_t_values')
        if t_values is not None and len(t_values) > 0:
            t_dist_filename = f"{base}_t_distribution.csv"
            try:
                with open(t_dist_filename, 'w', encoding='utf-8') as f:
                    f.write("t_value\n")
                    for v in t_values:
                        f.write(f"{float(v):.6f}\n")
                print(f"T-value distribution saved to {t_dist_filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save t-value distribution CSV:\n{e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
