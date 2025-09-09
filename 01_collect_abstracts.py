# -*- coding: utf-8 -*-
"""
PubMed Fetcher GUI
------------------
A small GUI wrapper around Biopython's Entrez utilities to:
- Collect user settings (email, API key, query, batching, wait, output path)
- Run a PubMed search → fetch → save pipeline
- Show progress and logs, and preview a compact table snapshot before saving

Enhancements
------------
- Automatically saves CSV and a sidecar .query.txt on completion (no overwrite via timestamp).
- Abort button to stop a running job cleanly.
- Consistent fixed-width font for Logs/Preview via Tk's TkFixedFont.
- Default output folder set to ./compiled_articles/ (created automatically).

Dependencies
------------
- biopython  (pip install biopython)
- pandas     (pip install pandas)
- customtkinter (pip install customtkinter)

Usage
-----
python 01_collect_abstracts.py

Notes
-----
- Long-running network calls happen in a background thread to keep the UI responsive.
- The app can save/load a simple settings.json next to the script.
- Output is written to the CSV path (with a timestamp suffix) inside ./compiled_articles/.
- Both email and API key are required: register at NCBI to obtain them.
"""

# =============================================================================
#                                 Imports
# =============================================================================
import os
import json
import time
import threading
import traceback
from datetime import datetime
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox
from typing import Callable, List, Dict, Any

import customtkinter as ctk
import pandas as pd
from Bio import Entrez
# =============================================================================


# ------------------ Core fetch logic ------------------ #
def xml_to_df(article_details: Dict[str, Any]) -> pd.DataFrame:
    """Convert PubMed EFETCH XML response into a pandas DataFrame.

    Parameters
    ----------
    article_details : dict
        Parsed PubMed article details from Entrez.read() for an EFETCH call.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: PMID, Title, Abstract, Authors, PublicationDate, Journal, DOI.
    """
    articles_data: List[Dict[str, Any]] = []
    for article in article_details.get("PubmedArticle", []):
        pmid = article["MedlineCitation"]["PMID"]
        title = article["MedlineCitation"]["Article"].get("ArticleTitle", "")

        abstract_parts = (
            article["MedlineCitation"]["Article"]
            .get("Abstract", {})
            .get("AbstractText", [])
        )
        if abstract_parts:
            abstract = " ".join(
                part if isinstance(part, str) else part.get("#text", "")
                for part in abstract_parts
            )
        else:
            abstract = "No Abstract"

        authors_list = article["MedlineCitation"]["Article"].get("AuthorList", [])
        authors = "; ".join(
            f"{author.get('LastName', '')}, {author.get('ForeName', '')}"
            for author in authors_list
        )

        publication_date_info = article["MedlineCitation"]["Article"].get(
            "ArticleDate", []
        )
        publication_date = (
            publication_date_info[0]["Year"] if publication_date_info else "No Date"
        )
        journal = (
            article["MedlineCitation"]["Article"].get("Journal", {}).get("Title", "")
        )

        doi = "No DOI"
        for article_id in article["PubmedData"].get("ArticleIdList", []):
            if getattr(article_id, "attributes", {}).get("IdType") == "doi":
                doi = str(article_id)
                break

        articles_data.append(
            {
                "PMID": pmid,
                "Title": title,
                "Abstract": abstract,
                "Authors": authors,
                "PublicationDate": publication_date,
                "Journal": journal,
                "DOI": doi,
            }
        )

    return pd.DataFrame(articles_data)


def fetch_details_in_batches(
    id_list: List[str],
    batch_size: int = 100,
    wait_time: float = 0.2,
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Fetch PubMed records in batches and return the combined DataFrame.

    Parameters
    ----------
    id_list : list of str
        PubMed IDs to fetch.
    batch_size : int, optional
        Number of records per EFETCH request, by default 100.
    wait_time : float, optional
        Delay in seconds between batch requests, by default 0.2.
    on_progress : callable, optional
        Callback to update progress bar; called with (done, total).
    on_log : callable, optional
        Callback to log messages.
    should_stop : callable, optional
        Function returning True if the operation should abort.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame of all fetched records (empty if none or aborted early).
    """
    all_dfs: List[pd.DataFrame] = []
    total = len(id_list)

    for start in range(0, total, batch_size):
        if should_stop and should_stop():
            if on_log:
                on_log("Abort requested. Stopping before next batch …")
            break

        end = min(start + batch_size, total)
        if on_log:
            on_log(f"Fetching records {start + 1}–{end} of {total} …")
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=",".join(id_list[start:end]),
                retmode="xml",
                rettype="abstract",
            )
            batch_articles = Entrez.read(fetch_handle)
            fetch_handle.close()

            batch_df = xml_to_df(batch_articles)
            all_dfs.append(batch_df)
        except Exception as e:
            if on_log:
                on_log(f"Error during fetch {start + 1}–{end}: {e}")
            # continue to next batch to salvage remaining work
        finally:
            time.sleep(wait_time)

        if on_progress:
            on_progress(end, total)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ------------------ GUI ------------------ #
DEFAULTS = {
    "email": "",
    "api_key": "",
    "query": '"functional connectivity" AND (postpartum OR PPD OR "Postnatal depression" OR "Maternal depression")',
    "batch_size": 100,
    "wait_time": 0.2,
    # Default to ./compiled_articles/compiled_articles.csv (folder auto-created)
    "output_csv": os.path.abspath(os.path.join("compiled_articles", "compiled_articles.csv")),
}

SETTINGS_FILE = "settings.json"


class PubMedApp(ctk.CTk):
    """customtkinter-based GUI to search, fetch, preview, and export PubMed data."""

    def __init__(self, initial: Dict[str, Any] | None = None):
        """Initialize the GUI, fonts, tabs, and layout."""
        super().__init__()
        self.title("PubMed Fetcher")
        self.geometry("1000x760")
        self.minsize(900, 640)

        # Appearance & theme
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Normalize font sizing across envs: use TkFixedFont and bump size.
        try:
            self.tk.call("tk", "scaling", 1.3)
        except Exception:
            pass
        fixed = tkfont.nametofont("TkFixedFont")
        fixed.configure(size=16)
        self._mono_font = fixed

        self.cfg = json.loads(json.dumps(initial or DEFAULTS))
        self.result_df: pd.DataFrame | None = None
        self._worker: threading.Thread | None = None
        self._stop_flag = False

        # Root grid: tabs stretch; footer sticks
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Order: Setup → Logs → Preview (preview last)
        self.tabs.add("Setup")
        self.tabs.add("Logs")
        self.tabs.add("Preview")

        self._build_setup_tab()
        self._build_logs_tab()
        self._build_preview_tab()

        # Footer (Run / Abort / Save)
        footer = ctk.CTkFrame(self)
        footer.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        footer.grid_columnconfigure(0, weight=1)

        self.run_btn = ctk.CTkButton(footer, text="Run", command=self.on_run, height=38)
        self.run_btn.grid(row=0, column=1, padx=(0, 8))

        self.abort_btn = ctk.CTkButton(
            footer, text="Abort", command=self.on_abort, height=38, fg_color="#b4232c"
        )
        self.abort_btn.grid(row=0, column=2, padx=(0, 8))

        self.save_btn = ctk.CTkButton(
            footer, text="Save CSV", command=self.on_save_csv, height=38, state="disabled"
        )
        self.save_btn.grid(row=0, column=3)

        self._refresh_preview(completed=False)

    def _default_out_dir(self) -> str:
        """Return the absolute compiled_articles directory, creating it if needed."""
        d = os.path.abspath("compiled_articles")
        os.makedirs(d, exist_ok=True)
        return d

    # --------- Tab builders --------- #
    def _stretch(self, frame: ctk.CTkFrame) -> None:
        """Make column 1 stretchable inside a CTkFrame for neat layout."""
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=0)

    def _build_setup_tab(self) -> None:
        """Build the Setup tab with email, API key, query, options, and output path."""
        tab = self.tabs.tab("Setup")

        # Credentials
        creds = ctk.CTkFrame(tab)
        creds.pack(fill="x", padx=10, pady=(10, 6))
        self._stretch(creds)

        ctk.CTkLabel(creds, text="Entrez email").grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )
        self.email_e = ctk.CTkEntry(creds)
        self.email_e.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.email_e.insert(0, self.cfg["email"])

        ctk.CTkLabel(creds, text="Entrez API key").grid(
            row=1, column=0, padx=10, pady=(0, 10), sticky="w"
        )
        self.key_e = ctk.CTkEntry(creds, show="*")
        self.key_e.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")
        self.key_e.insert(0, self.cfg["api_key"])

        # Query
        qf = ctk.CTkFrame(tab)
        qf.pack(fill="x", padx=10, pady=6)
        self._stretch(qf)

        ctk.CTkLabel(qf, text="PubMed query").grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )
        self.query_e = ctk.CTkEntry(qf)
        self.query_e.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.query_e.insert(0, self.cfg["query"])

        # Options
        opts = ctk.CTkFrame(tab)
        opts.pack(fill="x", padx=10, pady=6)
        self._stretch(opts)

        ctk.CTkLabel(opts, text="Batch size").grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )
        self.batch_e = ctk.CTkEntry(opts)
        self.batch_e.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.batch_e.insert(0, str(self.cfg["batch_size"]))

        ctk.CTkLabel(opts, text="Wait between batches (s)").grid(
            row=1, column=0, padx=10, pady=(0, 10), sticky="w"
        )
        self.wait_e = ctk.CTkEntry(opts)
        self.wait_e.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")
        self.wait_e.insert(0, str(self.cfg["wait_time"]))

        # Output path
        out = ctk.CTkFrame(tab)
        out.pack(fill="x", padx=10, pady=(6, 10))
        self._stretch(out)

        ctk.CTkLabel(out, text="Output CSV path").grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )
        self.out_e = ctk.CTkEntry(out)
        self.out_e.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.out_e.insert(0, self.cfg["output_csv"])
        ctk.CTkButton(out, text="Browse", command=self._pick_csv).grid(
            row=0, column=2, padx=10, pady=10
        )

        # Ensure default folder exists right away (nice UX for users who open dir)
        try:
            os.makedirs(os.path.dirname(self.cfg["output_csv"]) or ".", exist_ok=True)
        except Exception:
            pass

        # Progress bar
        self.progress = ctk.CTkProgressBar(tab)
        self.progress.pack(fill="x", padx=20, pady=(0, 10))
        self.progress.set(0)

        # Save/Load settings row
        sl = ctk.CTkFrame(tab)
        sl.pack(fill="x", padx=10, pady=(0, 10))
        sl.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(sl, text="Load settings.json", command=self.load_settings).grid(
            row=0, column=1, padx=(0, 8)
        )
        ctk.CTkButton(sl, text="Save settings.json", command=self.save_settings).grid(
            row=0, column=2
        )

    def _build_logs_tab(self) -> None:
        """Build the Logs tab for runtime messages (large monospaced font)."""
        tab = self.tabs.tab("Logs")
        self.log_box = tk.Text(tab, height=28, wrap="word", font=self._mono_font)
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_box.configure(state="disabled")

    def _build_preview_tab(self) -> None:
        """Build the Preview tab for showing configuration or table snapshot.

        Before the run: shows current configuration JSON.
        After the run: shows the first N rows of the fetched DataFrame.
        """
        tab = self.tabs.tab("Preview")
        self.preview = tk.Text(tab, height=28, wrap="none", font=self._mono_font)
        self.preview.pack(fill="both", expand=True, padx=10, pady=10)
        self.preview.configure(state="disabled")

    # --------- Helpers --------- #
    def _pick_csv(self) -> None:
        """Open a file dialog to select the output CSV path (used as base name)."""
        path = filedialog.asksaveasfilename(
            initialdir=self._default_out_dir(),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select output CSV path",
        )
        if path:
            self.out_e.delete(0, tk.END)
            self.out_e.insert(0, path)

    def load_settings(self) -> None:
        """Load settings from settings.json if available."""
        if not os.path.exists(SETTINGS_FILE):
            messagebox.showinfo("Info", f"No {SETTINGS_FILE} found.")
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in DEFAULTS:
                if k in data:
                    self.cfg[k] = data[k]
            self._apply_cfg_to_fields()
            messagebox.showinfo("Loaded", f"Loaded settings from {SETTINGS_FILE}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {e}")

    def save_settings(self) -> None:
        """Save current settings to settings.json."""
        self._collect_into_cfg()
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)
            messagebox.showinfo("Saved", f"Saved settings to {SETTINGS_FILE}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def _apply_cfg_to_fields(self) -> None:
        """Update UI fields from the current configuration."""
        self.email_e.delete(0, tk.END); self.email_e.insert(0, self.cfg.get("email", ""))
        self.key_e.delete(0, tk.END);   self.key_e.insert(0, self.cfg.get("api_key", ""))
        self.query_e.delete(0, tk.END); self.query_e.insert(0, self.cfg.get("query", ""))
        self.batch_e.delete(0, tk.END); self.batch_e.insert(0, str(self.cfg.get("batch_size", 100)))
        self.wait_e.delete(0, tk.END);  self.wait_e.insert(0, str(self.cfg.get("wait_time", 0.2)))

        # Single, correct set for output path (fixed: remove duplicate inserts)
        default_out = os.path.abspath(os.path.join("compiled_articles", "compiled_articles.csv"))
        self.out_e.delete(0, tk.END)
        self.out_e.insert(0, self.cfg.get("output_csv", default_out))

    def _collect_into_cfg(self) -> bool:
        """Collect values from UI fields into ``self.cfg`` and validate."""
        try:
            self.cfg["email"] = self.email_e.get().strip()
            self.cfg["api_key"] = self.key_e.get().strip()
            self.cfg["query"] = self.query_e.get().strip()
            self.cfg["batch_size"] = int(float(self.batch_e.get().strip()))
            self.cfg["wait_time"] = float(self.wait_e.get().strip())

            # Normalize output path: if no dir or '.' -> route to compiled_articles/
            path = (self.out_e.get().strip()
                    or os.path.abspath(os.path.join("compiled_articles", "compiled_articles.csv")))
            base_dir = os.path.dirname(path)
            if not base_dir or os.path.abspath(base_dir) == os.path.abspath("."):
                path = os.path.join(self._default_out_dir(), os.path.basename(path))
            self.cfg["output_csv"] = path
        except Exception:
            messagebox.showerror(
                "Error",
                "Please check your inputs (batch size must be an integer; wait time a number).",
            )
            return False

        if not self.cfg["email"]:
            messagebox.showerror("Error", "Entrez email is required.")
            return False
        if not self.cfg["api_key"]:
            messagebox.showerror("Error", "Entrez API key is required.")
            return False
        if not self.cfg["query"]:
            messagebox.showerror("Error", "Please enter a PubMed query.")
            return False
        return True

    # --------- File naming & autosave --------- #
    @staticmethod
    def _timestamp() -> str:
        """Return a filesystem-friendly timestamp."""
        return datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def _derive_paths(self) -> Dict[str, str]:
        """Derive unique output paths for CSV and query sidecar based on config.

        Returns
        -------
        dict
            {'csv': <csv_path>, 'query': <query_txt_path>}
        """
        base = self.cfg["output_csv"]
        base_dir = os.path.dirname(base)

        # Force compiled_articles/ if user didn't specify a directory or used "."
        if not base_dir or os.path.abspath(base_dir) == os.path.abspath("."):
            base_dir = self._default_out_dir()
        else:
            os.makedirs(base_dir, exist_ok=True)

        stem, ext = os.path.splitext(os.path.basename(base))
        if not ext:
            ext = ".csv"

        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        csv_path = os.path.join(base_dir, f"{stem}_{ts}{ext}")
        query_path = os.path.join(base_dir, f"{stem}_{ts}.query.txt")
        return {"csv": csv_path, "query": query_path}

    def _autosave_outputs(self) -> None:
        """Automatically save CSV and query sidecar without overwriting."""
        if self.result_df is None:
            return
        paths = self._derive_paths()
        try:
            os.makedirs(os.path.dirname(paths["csv"]) or ".", exist_ok=True)
            self.result_df.to_csv(paths["csv"], index=False)
            with open(paths["query"], "w", encoding="utf-8") as f:
                f.write(self.cfg["query"].strip() + "\n")
            self._log(f"Auto-saved CSV:   {paths['csv']}")
            self._log(f"Saved keywords:   {paths['query']}")
        except Exception as e:
            self._log(f"Auto-save failed: {e}")

    # --------- Logging & progress --------- #
    def _log(self, text: str) -> None:
        """Append a message to the Logs tab and keep it scrolled to the end."""
        self.log_box.configure(state="normal")
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state="disabled")
        self.update_idletasks()

    def _set_progress(self, done: int, total: int) -> None:
        """Update the progress bar given records done and total."""
        frac = 0 if total == 0 else done / total
        self.progress.set(frac)
        self.update_idletasks()

    def _refresh_preview(self, completed: bool = False) -> None:
        """Show config (before run) or a compact table snapshot (after run)."""
        self.preview.configure(state="normal")
        self.preview.delete("1.0", tk.END)
        if completed and self.result_df is not None and not self.result_df.empty:
            head = self.result_df.head(30).to_string(index=False)
            self.preview.insert(tk.END, head)
        else:
            cfg_str = json.dumps(self.cfg, indent=2)
            self.preview.insert(tk.END, cfg_str)
        self.preview.configure(state="disabled")

    # --------- Abort control --------- #
    def on_abort(self) -> None:
        """Request abortion of the running job."""
        if self._worker and self._worker.is_alive():
            self._stop_flag = True
            self._log("Abort requested … (will stop between operations)")
        else:
            self._log("Nothing to abort.")

    def _should_stop(self) -> bool:
        """Return True if an abort has been requested."""
        return self._stop_flag

    # --------- Run / Save --------- #
    def on_run(self) -> None:
        """Run the PubMed search and fetch process in a background thread."""
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Running", "A fetch job is already running.")
            return
        if not self._collect_into_cfg():
            return

        # Apply Entrez credentials
        Entrez.email = self.cfg["email"]
        Entrez.api_key = self.cfg["api_key"]

        # Clear logs at the start of every run
        self.log_box.configure(state="normal"); self.log_box.delete("1.0", tk.END); self.log_box.configure(state="disabled")

        self._stop_flag = False
        self.run_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.progress.set(0)
        self.tabs.set("Logs")
        self._log("Starting search …")

        def work():
            try:
                # 1) Search to get total count
                if self._should_stop():
                    self._log("Aborted before search.")
                    return
                search_handle = Entrez.esearch(db="pubmed", term=self.cfg["query"], retmax=0)
                search_results = Entrez.read(search_handle)
                search_handle.close()
                total_records = int(search_results.get("Count", 0))
                self._log(f"Total records found: {total_records}")

                if self._should_stop():
                    self._log("Aborted after counting.")
                    return
                if total_records == 0:
                    self._set_progress(0, 1)
                    self.result_df = pd.DataFrame()
                    return

                # 2) Retrieve all IDs
                self._log("Fetching ID list …")
                search_handle = Entrez.esearch(db="pubmed", term=self.cfg["query"], retmax=total_records)
                search_results = Entrez.read(search_handle)
                search_handle.close()
                id_list = search_results.get("IdList", [])
                if self._should_stop():
                    self._log("Aborted before fetching details.")
                    return

                # 3) Fetch details in batches (check stop between batches)
                self._log("Fetching article details …")
                tic = time.time()
                df = fetch_details_in_batches(
                    id_list=id_list,
                    batch_size=self.cfg["batch_size"],
                    wait_time=self.cfg["wait_time"],
                    on_progress=self._set_progress,
                    on_log=self._log,
                    should_stop=self._should_stop,
                )
                toc = time.time()
                self.result_df = df

                if self._should_stop():
                    self._log(f"Aborted. Collected {len(df)} records so far.")
                else:
                    self._log(f"Completed. {len(df)} records. Elapsed: {toc - tic:.1f}s")
                    # Auto-save CSV and keywords (no overwrite)
                    self._autosave_outputs()
                    self._log("Operation completed successfully.")
            except Exception as e:
                self._log("\n" + traceback.format_exc())
                messagebox.showerror("Error", f"Run failed: {e}")
            finally:
                # Switch to UI thread to finish up
                self.after(0, self._on_run_done)

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _on_run_done(self) -> None:
        """Enable buttons and show the table snapshot in Preview tab."""
        self.run_btn.configure(state="normal")
        if self.result_df is not None and not self.result_df.empty:
            self.save_btn.configure(state="normal")
        self._refresh_preview(completed=True)
        self.tabs.set("Preview")

    def on_save_csv(self) -> None:
        """Manual save: write the current DataFrame to a new timestamped file."""
        if self.result_df is None or self.result_df.empty:
            messagebox.showinfo("Nothing to save", "No data to save yet.")
            return
        paths = self._derive_paths()
        try:
            os.makedirs(os.path.dirname(paths["csv"]) or ".", exist_ok=True)
            self.result_df.to_csv(paths["csv"], index=False)
            with open(paths["query"], "w", encoding="utf-8") as f:
                f.write(self.cfg["query"].strip() + "\n")
            self._log(f"Saved CSV to: {paths['csv']}")
            self._log(f"Saved keywords to: {paths['query']}")
            messagebox.showinfo("Saved", f"Saved to {paths['csv']}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save CSV: {e}")


if __name__ == "__main__":
    app = PubMedApp()
    app.mainloop()
