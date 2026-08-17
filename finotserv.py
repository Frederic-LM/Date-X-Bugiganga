#!/usr/bin/env python3
"""
finotserv.py (Version 10.2)  --  Local bridge between Pennyscope and gogo.py
Run:   python finotserv.py
Opens: http://localhost:5174   (override with PENNYSCOPE_PORT)

The version reported to the browser comes from gogo.__version__, not from a literal here.
"""

# ── Headless matplotlib BEFORE any other import ───────────────────────────────
import matplotlib
matplotlib.use('Agg')           # no GUI window; plot goes to PNG
import matplotlib.pyplot as plt
import io as _io, base64 as _b64
import threading

# pyplot's figure state is process-global and not thread-safe, and this server is threaded.
_PLOT_LOCK = threading.Lock()

def _render_plot(result):
    """Draw one result and return its PNG.

    plot_results(show=False) hands back the figure instead of trying to display it, so
    nothing here has to intercept plt.show() and no drawing state is shared between
    requests beyond the lock."""
    with _PLOT_LOCK:
        figure = None
        try:
            figure = plot_results(result, show=False)
            if figure is None:
                return ''
            buf = _io.BytesIO()
            figure.savefig(buf, format='png', dpi=110, bbox_inches='tight')
            return _b64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print(f"(Plot could not be generated: {e})")
            return ''
        finally:
            if figure is not None:
                plt.close(figure)

# ── Standard imports ──────────────────────────────────────────────────────────
import os, sys, json, re, tempfile, traceback, webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gogo import (
    run_date_analysis, run_detective_analysis,
    run_two_piece_mean_analysis,
    plot_results, _classify_dendro_match,
    download_and_index_files, fetch_and_build_violin_master,
    build_master_from_index, index_cache,
    DEFAULTS, CATEGORY_NAMES, DETECTIVE_TARGETS,
    __version__ as GOGO_VERSION, APP_NAME, TARGET_LABELS,
    VIOLIN_REFERENCE_LABEL, VIOLIN_REFERENCE_BLURB, VIOLIN_REFERENCE_DIR,
    run_manifest, terminus_post_quem_note, reference_depth_at,
    describe_reference_set, stability_check, stability_verdict,
    INDEX_METHODS, resolve_index, describe_index, index_edge_loss,
    format_classification, classification_handle, convention_note, shared_variation,
    CLASSIFICATION_TIERS,
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = os.path.join(BASE_DIR, 'ring-measurer.html')
PORT      = int(os.environ.get("PENNYSCOPE_PORT", 5174))


class _ThreadLocalStdout:
    """One process-wide stdout that sends each thread's writes to its own buffer.

    The engine prints as it works and the browser wants that log back with its own result.
    Assigning sys.stdout per request cannot do that: sys.stdout is global and this server is
    threaded, so a second request would silently take over the first request's output."""

    def __init__(self, fallback):
        self._fallback = fallback
        self._local = threading.local()

    def _target(self):
        return getattr(self._local, 'buffer', None) or self._fallback

    def capture(self):
        """Send this thread's writes to a fresh buffer, and return it."""
        buffer = _io.StringIO()
        self._local.buffer = buffer
        return buffer

    def release(self):
        self._local.buffer = None

    def write(self, text):
        return self._target().write(text)

    def flush(self):
        try:
            self._target().flush()
        except (ValueError, AttributeError):
            pass

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self._fallback, 'encoding', 'utf-8')

    @property
    def raw_stream(self):
        """The real console stream, for callers that need to configure it."""
        return self._fallback


_STDOUT = _ThreadLocalStdout(sys.stdout)
sys.stdout = _STDOUT


def _requested_index(req):
    """One index name, or None for the default: all three."""
    raw = str(req.get('index') or '').strip().lower()
    return resolve_index(raw) if raw and raw != 'all' else None


# ── HTTP request handler ──────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # silence per-request console noise

    # ── helpers ───────────────────────────────────────────────────────────────

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type',   'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(n)) if n else {}

    # ── routes ────────────────────────────────────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin',  '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = self.path.split('?')[0]

        if path == '/':
            with open(HTML_FILE, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type',   'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == '/api/status':
            # Reported from gogo.__version__, never a literal, so the browser cannot
            # advertise a version the engine has moved past.
            self._json({'ok': True, 'version': GOGO_VERSION, 'app': APP_NAME})
            return

        if path == '/api/masters':
            vdir  = os.path.join(BASE_DIR, VIOLIN_REFERENCE_DIR)
            count = 0
            if os.path.isdir(vdir):
                count = len([f for f in os.listdir(vdir)
                              if f.lower().endswith('.rwl')])
            cache_exists = os.path.isdir(os.path.join(BASE_DIR, 'full_rwl_cache'))
            setup_complete = cache_exists and count > 0
            self._json({
                'cache_ready': cache_exists,
                'violin_ready': count > 0,
                'violin_count': count,
                'setup_complete': setup_complete,
                # Served from gogo so the web UI cannot drift out of step with the
                # categories the analysis engine actually accepts.
                'categories':   list(DETECTIVE_TARGETS),
                'category_labels': dict(TARGET_LABELS),
                'violin_label': VIOLIN_REFERENCE_LABEL,
                'violin_blurb': VIOLIN_REFERENCE_BLURB,
                'defaults':     dict(DEFAULTS),
                # Served from gogo so the web UI offers exactly the indices the engine
                # implements, and no more.
                'indices':      list(INDEX_METHODS),
                'tiers':        [{'handle': t['handle'], 't': t['t'], 'n': t['n'], 'glk': t['glk']}
                                 for t in CLASSIFICATION_TIERS],
                'version':      GOGO_VERSION,
            })
            return

        if path == '/api/setup/status':
            cache_exists = os.path.isdir(os.path.join(BASE_DIR, 'full_rwl_cache'))
            self._json({'cache_ready': cache_exists})
            return

        if path.startswith('/api/image/'):
            from urllib.parse import unquote
            img_path = os.path.realpath(unquote(path[11:]))  # Remove '/api/image/'
            # Confine reads to the project tree so a crafted path can't exfiltrate
            # arbitrary files (e.g. /api/image/../../etc/passwd, or another drive on Windows).
            allowed_root = os.path.realpath(BASE_DIR)
            try:
                within = os.path.commonpath([img_path, allowed_root]) == allowed_root
            except ValueError:
                within = False  # different drive / invalid path
            if not within:
                self._json({'error': 'Forbidden'}, 403)
                return
            ext = os.path.splitext(img_path)[1].lower()
            mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif', '.bmp': 'image/bmp', '.tif': 'image/tiff',
                    '.tiff': 'image/tiff', '.webp': 'image/webp'}.get(ext, 'application/octet-stream')
            try:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(img_data)))
                self.end_headers()
                self.wfile.write(img_data)
            except (FileNotFoundError, IsADirectoryError, ValueError):
                self._json({'error': 'Image not found'}, 404)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path not in ('/api/analyze', '/api/crossmatch', '/api/setup/download', '/api/setup/violin'):
            self.send_response(404)
            self.end_headers()
            return

        req = self._body()

        if self.path == '/api/setup/download':
            try:
                print("\n--- Starting RWL Download & Index ---")
                download_and_index_files()
                self._json({'success': True, 'message': 'Download complete. Cache ready for analysis.'})
            except Exception as e:
                print(f"Error: {e}")
                self._json({'success': False, 'error': str(e)})
            return

        if self.path == '/api/setup/violin':
            try:
                print("\n--- Starting Violin Reference Setup ---")
                fetch_and_build_violin_master()
                self._json({'success': True, 'message': f'{VIOLIN_REFERENCE_LABEL} ready.'})
            except Exception as e:
                print(f"Error: {e}")
                self._json({'success': False, 'error': str(e)})
            return

        if self.path == '/api/crossmatch':
            self._json(self._handle_crossmatch(req))
            return

        # ── /api/analyze ──────────────────────────────────────────────────────
        # Defaults come from gogo.DEFAULTS so a run launched from Pennyscope uses the
        # same settings as the same run launched from the CLI or the desktop app.
        rwl     = req.get('rwl',         '')
        mode    = req.get('mode',         'violin')
        rev     = bool(req.get('reverse',  False))
        ovlp    = int(req.get('min_overlap', DEFAULTS['min_overlap']))
        spln    = int(req.get('spline',      DEFAULTS['spline_stiffness_pct']))
        top_n   = int(req.get('top_n',       DEFAULTS['top_n']))
        dmode   = req.get('detrend_mode',    DEFAULTS['detrend_mode'])
        dwave   = int(req.get('detrend_wavelength', DEFAULTS['detrend_wavelength']))
        mey     = int(req.get('min_end_year', DEFAULTS['min_end_year']))
        # The sample is written to a temp file, so its filename says nothing about which
        # series it is. Pennyscope sends the project's series ID so the engine can still
        # hold that series out of a reference that contains it.
        sid     = (req.get('series_id') or req.get('project_name') or '').strip() or None
        # Which index the t-value is computed on. Refused loudly (by the engine) if the
        # chosen reference was built under a different one.
        idx     = _requested_index(req)
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.rwl', delete=False,
            dir=BASE_DIR, encoding='utf-8', prefix='_tmp_'
        )
        tmp.write(rwl)
        tmp.close()
        sample_path = tmp.name

        ref_tmp = None
        # This thread's prints only, so another request's log cannot land in this one.
        log_buffer = _STDOUT.capture()
        try:
            mode, ref_tmp = self._resolve_reference(req, mode)
            detective = (
                mode in DETECTIVE_TARGETS
                or os.path.isdir(mode)
            )

            if detective:
                rd = run_detective_analysis(
                    sample_path, mode,
                    top_n=top_n, min_overlap=ovlp, min_end_year=mey,
                    reverse_sample=rev, spline_stiffness_pct=spln,
                    detrend_mode=dmode, detrend_wavelength=dwave,
                    sample_series_id_override=sid, index=idx,
                )
            else:
                rd = run_date_analysis(
                    sample_path, mode,
                    min_overlap=ovlp, reverse_sample=rev,
                    spline_stiffness_pct=spln,
                    detrend_mode=dmode, detrend_wavelength=dwave,
                    sample_series_id_override=sid, index=idx,
                )

            plot_png = _render_plot(rd) if rd else ''
            log = log_buffer.getvalue()   # after the plot, so plotting trouble is in the log

            best = rd['results']['best_match'] if rd else None

            if not best:
                self._json({'ok': False,
                            'error': 'Analysis returned no result.',
                            'log': log})
                return

            index_used = rd.get('index', idx or DEFAULTS['lead_index'])
            classification = _classify_dendro_match(
                best['t_value'], int(best['overlap_n']), best.get('glk', 0))
            src = best.get(
                'source_file',
                os.path.basename(rd.get('master_filename', '')),
            )

            composition = rd.get('reference_composition') or {}
            resp = {
                'ok':           True,
                'best_year':    int(best['end_year']),
                't_value':      round(float(best['t_value']),     2),
                'glk':          round(float(best.get('glk', 0)),  1),
                'correlation':  round(float(best['correlation']),  3),
                'overlap_n':    int(best['overlap_n']),
                'r2':           round(shared_variation(best['t_value'], int(best['overlap_n'])) or 0.0, 3),
                'stands_out_sd': round(float(best.get('t_zscore', 0)), 1),
                'second_best_t': round(float(best.get('second_best_t', 0)), 2),
                # Values and the criteria they meet -- never an adjective.
                'criteria_met': classification_handle(classification),
                'classification': classification,
                'classification_text': format_classification(classification, index=index_used),
                'convention_note': convention_note(index_used),
                'source_file':  src,
                'index':        index_used,
                'index_note':   describe_index(index_used),
                'index_edge_loss': list(rd.get('index_edge_loss') or index_edge_loss(index_used)),
                'rings_measured': rd.get('rings_measured'),
                'rings_after_index': rd.get('rings_after_index'),
                # Whether the end year survived the choice of index leads the report.
                'index_runs': rd.get('index_runs', []),
                'index_agreement': rd.get('index_agreement', {}),
                'index_comparison_text': rd.get('index_comparison_text', ''),
                # Whether the same reference ranks first under every index.
                'candidate_agreement': rd.get('candidate_agreement', {}),
                'candidate_agreement_text': rd.get('candidate_agreement_text', ''),
                'search_indices': rd.get('search_indices', []),
                'plot_png':     plot_png,
                'log':          log[-3000:],
                # Conditions of the run, so a JSON response is as reproducible as a CSV.
                'run_manifest': rd.get('run_manifest', {}),
                'terminus_note': terminus_post_quem_note(best['end_year']),
                # Whether the sample's own series was held out of the reference it was
                # scored against -- including when that could not be checked at all.
                'holdout_note': rd.get('holdout_note', ''),
                'holdout': {k: v for k, v in (rd.get('holdout') or {}).items()
                            if not k.startswith('_')},
                'unusable_reason': rd.get('unusable_reason'),
                # What the search could have matched, and how deep the reference is at
                # the year actually reported.
                'reference_set': {
                    'n_sites': composition.get('n_sites'),
                    'n_series': composition.get('n_series'),
                    'countries': composition.get('countries'),
                    'year_min': composition.get('year_min'),
                    'year_max': composition.get('year_max'),
                } if composition else None,
                'reference_set_text': rd.get('reference_composition_text', ''),
                # What the matched reference is, not just what it is called.
                'reference_metadata': rd.get('reference_metadata', {}),
                'reference_metadata_text': rd.get('reference_metadata_text', ''),
                'ref_depth_at_year': (reference_depth_at(composition, best['end_year'])
                                      if composition else rd.get('depth_at_match')),
            }

            # What was searched, per country prefix -- the only way a reader can tell an
            # absent match from an absent reference. Same columns as the CSV exports.
            coverage = rd.get('country_coverage_df') if rd else None
            if coverage is not None and not coverage.empty:
                resp['country_coverage'] = coverage.where(coverage.notna(), None).to_dict('records')
                resp['country_coverage_text'] = rd.get('country_coverage_text', '')

            # Top-N list (detective mode only)
            if rd and 'enriched_results_df' in rd:
                rows = []
                for _, row in rd['enriched_results_df'].iterrows():
                    rows.append({
                        'end_year':    int(row['end_year']),
                        't_value':     round(float(row['t_value']),         2),
                        'glk':         round(float(row.get('glk', 0)),      1),
                        'source_file': str(row.get('source_file', '')),
                    })
                resp['top_matches'] = rows

            self._json(resp)

        except Exception as e:
            self._json({
                'ok':    False,
                'error': str(e),
                'log':   log_buffer.getvalue() + '\n' + traceback.format_exc(),
            })
        finally:
            _STDOUT.release()
            for path in (sample_path, ref_tmp):
                if not path:
                    continue
                try:
                    os.unlink(path)
                except OSError:
                    pass

    # ── /api/crossmatch ───────────────────────────────────────────────────────

    @staticmethod
    def _write_tmp(rwl_text, suffix='.rwl'):
        """Write RWL content to a temp file, return its path."""
        t = tempfile.NamedTemporaryFile(
            mode='w', suffix=suffix, delete=False,
            dir=BASE_DIR, encoding='utf-8', prefix='_tmp_'
        )
        t.write(rwl_text); t.close()
        return t.name

    # 'custom' lets the user date against a reference file of their own -- a master .csv
    # or a .rwl chronology -- instead of one of the built-in categories. Only the file's
    # CONTENT is accepted, never a path: this server binds every interface, so honouring
    # a caller-supplied path would let anything on the network read arbitrary files.
    CUSTOM_MODE = 'custom'

    @classmethod
    def _resolve_reference(cls, req, mode):
        """(reference_path_or_mode, temp_path_to_delete) for this request."""
        if mode != cls.CUSTOM_MODE:
            return mode, None
        content = req.get('reference_content') or ''
        name = (req.get('reference_name') or 'reference.csv').strip()
        if not content.strip():
            raise ValueError("Custom reference selected but no file contents were sent.")
        suffix = '.rwl' if name.lower().endswith('.rwl') else '.csv'
        path = cls._write_tmp(content, suffix=suffix)
        return path, path

    @staticmethod
    def _extract_top(rd):
        """Pull top-N list from a detective result dict."""
        if not rd or 'enriched_results_df' not in rd:
            return []
        rows = []
        for _, row in rd['enriched_results_df'].iterrows():
            rows.append({
                'end_year':    int(row['end_year']),
                't_value':     round(float(row['t_value']),         2),
                'glk':         round(float(row.get('glk', 0)),      1),
                'source_file': str(row.get('source_file', '')),
            })
        return rows

    def _handle_crossmatch(self, req):
        bass_rwl    = req.get('bass_rwl',    '')
        treble_rwl  = req.get('treble_rwl',  '')
        mode        = req.get('mode',        'violin')
        rev_bass    = bool(req.get('reverse_bass',   False))
        rev_treble  = bool(req.get('reverse_treble', False))
        ovlp        = int(req.get('min_overlap', DEFAULTS['min_overlap']))
        spln        = int(req.get('spline',      DEFAULTS['spline_stiffness_pct']))
        top_n       = int(req.get('top_n',       DEFAULTS['top_n']))
        dmode       = req.get('detrend_mode',    DEFAULTS['detrend_mode'])
        dwave       = int(req.get('detrend_wavelength', DEFAULTS['detrend_wavelength']))
        mey         = int(req.get('min_end_year', DEFAULTS['min_end_year']))
        idx         = _requested_index(req)
        sid         = (req.get('series_id') or req.get('project_name') or '').strip() or None

        bass_path   = self._write_tmp(bass_rwl)
        treble_path = self._write_tmp(treble_rwl)

        log_buffer = _STDOUT.capture()
        ref_tmp = None

        try:
            mode, ref_tmp = self._resolve_reference(req, mode)
            # A single reference file is dated directly; a category is searched. Either way
            # the halves are dated by a one-argument callable, so nothing counts positions.
            if mode in DETECTIVE_TARGETS or os.path.isdir(mode):
                date_one = partial(run_detective_analysis, target=mode, top_n=top_n,
                                   min_overlap=ovlp, min_end_year=mey, spline_stiffness_pct=spln,
                                   detrend_mode=dmode, detrend_wavelength=dwave,
                                   sample_series_id_override=sid, index=idx)
            else:
                date_one = partial(run_date_analysis, master_file=mode, min_overlap=ovlp,
                                   spline_stiffness_pct=spln, detrend_mode=dmode,
                                   detrend_wavelength=dwave, sample_series_id_override=sid,
                                   index=idx)
            rd = run_two_piece_mean_analysis(
                bass_path, treble_path, date_one,
                reverse_bass=rev_bass,
                reverse_treble=rev_treble,
                spline_stiffness_pct=spln,
                detrend_mode=dmode, detrend_wavelength=dwave,
                index=idx,
            )
            log = log_buffer.getvalue()

            # Internal stats are stored in result dict
            int_stats  = (rd or {}).get('internal_stats', {})
            int_t      = round(float(int_stats.get('t_value', 0)), 2)
            int_glk    = round(float(int_stats.get('glk',     0)), 1)

            # Also try parsing from log as fallback
            if int_t == 0:
                m = re.search(r't-value\s*=\s*([\d.]+)', log)
                if m: int_t = round(float(m.group(1)), 2)
                m = re.search(r'Glk\s*=\s*([\d.]+)', log)
                if m: int_glk = round(float(m.group(1)), 1)

            plot_png = _render_plot(rd) if rd else ''
            log = log_buffer.getvalue()   # refreshed so plotting trouble is in the log too

            best = rd['results']['best_match'] if rd else None
            if not best:
                return {'ok': False, 'error': 'Mean chronology dating returned no result.',
                        'internal_t': int_t, 'internal_glk': int_glk,
                        'pieces_match': int_t >= 6.0, 'log': log}

            index_used = rd.get('index', idx or DEFAULTS['lead_index'])
            classification = _classify_dendro_match(
                best['t_value'], int(best['overlap_n']), best.get('glk', 0))
            src = best.get('source_file',
                           os.path.basename(rd.get('master_filename', '')))

            rel = (rd or {}).get('plate_relationship', {})

            def _half(r):
                if not r:
                    return None
                bm = (r.get('results') or {}).get('best_match') or {}
                if not bm:
                    return None
                return {'best_year': int(bm.get('end_year', 0)),
                        't_value': round(float(bm.get('t_value', 0)), 2),
                        'glk': round(float(bm.get('glk', 0)), 1),
                        'overlap_n': int(bm.get('overlap_n', 0)),
                        'source_file': bm.get('source_file', '')}

            return {
                'ok':           True,
                'internal_t':   int_t,
                'internal_glk': int_glk,
                'pieces_match': bool(rel.get('same_wedge', int_t >= 6.0)),
                # Both halves are always dated on their own, so a plate whose halves come
                # from different logs still returns two usable results instead of failing.
                'run_manifest': (rd or {}).get('run_manifest', {}),
                'terminus_note': terminus_post_quem_note(best['end_year']),
                'plate_verdict': rel.get('verdict', ''),
                'plate_note':    rel.get('note', ''),
                'bass_result':   _half((rd or {}).get('bass_result')),
                'treble_result': _half((rd or {}).get('treble_result')),
                'mean_used':     (rd or {}).get('mean_result') is not None,
                'best_year':    int(best['end_year']),
                't_value':      round(float(best['t_value']),    2),
                'glk':          round(float(best.get('glk', 0)), 1),
                'correlation':  round(float(best['correlation']), 3),
                'overlap_n':    int(best['overlap_n']),
                'r2':           round(shared_variation(best['t_value'], int(best['overlap_n'])) or 0.0, 3),
                'criteria_met': classification_handle(classification),
                'classification': classification,
                'classification_text': format_classification(classification, index=index_used),
                'convention_note': convention_note(index_used),
                'source_file':  src,
                'index':        index_used,
                'index_note':   describe_index(index_used),
                'rings_after_index': rd.get('rings_after_index'),
                'index_runs': rd.get('index_runs', []),
                'index_agreement': rd.get('index_agreement', {}),
                'index_comparison_text': rd.get('index_comparison_text', ''),
                'holdout_note': rd.get('holdout_note', ''),
                'unusable_reason': rd.get('unusable_reason'),
                'plot_png':     plot_png,
                'top_matches':  self._extract_top(rd),
                'log':          log[-3000:],
            }

        except ValueError as e:
            log = log_buffer.getvalue()
            # Parse internal stats from log even on failure
            int_t, int_glk = 0.0, 0.0
            m = re.search(r't-value\s*=\s*([\d.]+)', log)
            if m: int_t = round(float(m.group(1)), 2)
            m = re.search(r'Glk\s*=\s*([\d.]+)', log)
            if m: int_glk = round(float(m.group(1)), 1)
            return {'ok': False, 'error': str(e),
                    'internal_t': int_t, 'internal_glk': int_glk,
                    'pieces_match': False, 'log': log}

        except Exception as e:
            return {'ok': False, 'error': str(e),
                    'log': log_buffer.getvalue() + '\n' + traceback.format_exc()}
        finally:
            _STDOUT.release()
            for f in (bass_path, treble_path, ref_tmp):
                if not f:
                    continue
                try: os.unlink(f)
                except OSError: pass


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # A Windows console defaults to cp1252, which cannot encode the characters used in
    # this program's output; printing one raised UnicodeEncodeError and killed the server
    # before it began serving. Ask for UTF-8 and fall back to replacing what will not fit.
    # sys.stdout is the thread-local proxy by now, so the real console stream behind it is
    # what needs reconfiguring.
    for stream in (_STDOUT.raw_stream, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    # PORT may be taken by an already-running instance; say so plainly instead of
    # failing with a bare OSError traceback.
    ThreadingTCPServer.allow_reuse_address = True
    try:
        srv_ctx = ThreadingTCPServer(('', PORT), Handler)
    except OSError as e:
        print(f"\nCould not start on port {PORT}: {e}")
        print("Another Pennyscope server is probably already running. Stop it first, "
              "or open the existing one in your browser.")
        sys.exit(1)
    with srv_ctx as srv:
        url = f'http://localhost:{PORT}'
        print(f'\n  Pennyscope server  ->  {url}')
        print(f'  Press Ctrl+C to stop\n')
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print('\nServer stopped.')
