# -*- coding: utf-8 -*-
"""
02_create_network.py
--------------------
Single-window customtkinter app to build and visualize a normalized co-mention network.

Pipeline
1) Presence (literal, case-insensitive) → 2) Trim by Trim % → 3) Co-mentions
4) One-time normalization:
   - node_count_norm = count / max(count)
   - edge_weight_norm = comention / max(comention)
5) Plot normalized network; all tweak params multiply normalized values.

Layouts
- 'kamada_kawai' (default) with 'Scale'
- 'spring' with 'k' and 'iterations'  (Seed removed)

Core highlighting
- Top-percentile nodes (by normalized count) + edges between them colored red.

Dependencies: customtkinter, matplotlib, networkx, pandas, numpy
"""

import os
import json
from datetime import datetime
import threading
import tkinter.filedialog as fd
import tkinter.messagebox as mb

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk
from itertools import combinations


# ---------- utils ----------
def _to_float(s, d): 
    try: return float(s)
    except: return float(d)

def _to_int(s, d): 
    try: return int(float(s))
    except: return int(d)


# ---------- core logic ----------
def build_presence_matrix_with_progress(abstracts: pd.Series,
                                        regions: pd.Series,
                                        on_progress=None) -> pd.DataFrame:
    """Abstracts×Regions boolean matrix; literal, case-insensitive."""
    df = pd.DataFrame(False, index=np.arange(len(abstracts)), columns=regions, dtype=bool)
    tot = len(df.columns)
    for i, r in enumerate(df.columns, 1):
        df[r] = abstracts.str.contains(r, regex=False, case=False)
        if on_progress: on_progress(i, tot)
    return df

def trim_regions(df_presence: pd.DataFrame, trim_pct: float):
    """Return filtered presence df + (abstracts_with_hits, unique_regions_found)."""
    abstracts_with_hits  = int((df_presence.sum(axis=1) > 0).sum())
    unique_regions_found = int((df_presence.sum(axis=0) > 0).sum())
    thr = int(abstracts_with_hits * (trim_pct / 100.0))
    kept = df_presence.columns[df_presence.sum(axis=0) > thr]
    return df_presence[kept].copy(), abstracts_with_hits, unique_regions_found

def compute_comentions_with_progress(filtered_df: pd.DataFrame, on_progress=None) -> pd.DataFrame:
    """Symmetric co-mention matrix on kept regions."""
    cols = list(filtered_df.columns)
    co = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    tot = len(cols) * (len(cols) - 1) // 2
    done = 0
    for a, b in combinations(cols, 2):
        w = int((filtered_df[a] & filtered_df[b]).sum())
        if w: co.loc[a, b] = co.loc[b, a] = w
        done += 1
        if on_progress and tot: on_progress(done, tot)
    return co

def normalize_network(filtered_df: pd.DataFrame, co: pd.DataFrame):
    """Divide counts and co-mentions by their max (safe if max=0)."""
    counts = filtered_df.sum(axis=0)
    cmax = counts.max()
    counts_norm = counts / cmax if cmax else counts.astype(float)
    vmax = co.to_numpy().max() if not co.empty else 0
    co_norm = co / vmax if vmax else co.astype(float)
    return counts_norm, co_norm

def compute_layout(G: nx.Graph, layout: str, params: dict) -> dict:
    """Positions for 'kamada_kawai'(scale) or 'spring'(k, iterations)."""
    if G.number_of_nodes() == 0: return {}
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G, scale=float(params.get("scale", 1.0)))
    return nx.spring_layout(G,
                            k=float(params.get("k", 1.0)),
                            iterations=int(params.get("iterations", 50)))

def assign_core_mask(counts_norm: pd.Series, percentile: float = 90.0) -> dict:
    """True for nodes >= percentile of normalized count."""
    if counts_norm.empty: return {}
    thr = np.percentile(counts_norm.to_numpy(), percentile)
    return {n: (v >= thr) for n, v in counts_norm.items()}


# ---------- app ----------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Create Brain Region Network")
        self.geometry("1280x760")      # shorter window to “squish” top panel
        self.minsize(1100, 700)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        try: self.tk.call("tk", "scaling", 1.15)
        except: pass

        # root grid
        self.grid_rowconfigure(0, weight=0)  # compute
        self.grid_rowconfigure(1, weight=0)  # progress
        self.grid_rowconfigure(2, weight=1)  # plot
        self.grid_columnconfigure(0, weight=1)

        # data
        self.filtered_df = None
        self.counts_norm = None
        self.co_norm = None
        self.kept_cols = None

        self._build_compute_row()
        self._build_progress_row()
        self._build_plot_row()

    # ----- top: compute -----
    def _build_compute_row(self):
        f = ctk.CTkFrame(self, corner_radius=10)
        f.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        f.grid_columnconfigure(1, weight=1)
        f.grid_columnconfigure(4, weight=0)

        ctk.CTkLabel(f, text="Compute settings", font=ctk.CTkFont(size=16, weight="bold"))\
            .grid(row=0, column=0, columnspan=5, sticky="w", padx=10, pady=(8, 2))

        ctk.CTkLabel(f, text="Abstracts CSV").grid(row=1, column=0, padx=10, pady=4, sticky="w")
        self.abs_entry = ctk.CTkEntry(f, height=32)
        self.abs_entry.grid(row=1, column=1, columnspan=3, padx=(0, 6), pady=4, sticky="ew")
        ctk.CTkButton(f, text="Browse", width=100, command=self._browse_abs)\
            .grid(row=1, column=4, padx=6, pady=4)

        ctk.CTkLabel(f, text="Brain Regions CSV").grid(row=2, column=0, padx=10, pady=4, sticky="w")
        self.reg_entry = ctk.CTkEntry(f, height=32)
        self.reg_entry.grid(row=2, column=1, columnspan=3, padx=(0, 6), pady=4, sticky="ew")
        ctk.CTkButton(f, text="Browse", width=100, command=self._browse_reg)\
            .grid(row=2, column=4, padx=6, pady=4)

        ctk.CTkLabel(f, text="Trim %").grid(row=3, column=0, padx=10, pady=4, sticky="w")
        self.trim_entry = ctk.CTkEntry(f, height=32, width=120)
        self.trim_entry.insert(0, "0.5")
        self.trim_entry.grid(row=3, column=1, padx=(0, 6), pady=4, sticky="w")

        self.compute_btn = ctk.CTkButton(f, text="Compute", height=34, command=self._on_compute)
        self.compute_btn.grid(row=3, column=4, padx=6, pady=4, sticky="e")

        # slimmer status box
        self.status = ctk.CTkTextbox(f, height=72)
        self.status.grid(row=4, column=0, columnspan=5, padx=10, pady=(4, 8), sticky="ew")
        self.status.configure(state="disabled")

        self.abs_entry.insert(0, "compiled_articles.csv")
        self.reg_entry.insert(0, "brain_regions.csv")

    # ----- middle: progress -----
    def _build_progress_row(self):
        h = ctk.CTkFrame(self, corner_radius=10)
        h.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="ew")
        h.grid_columnconfigure(0, weight=1)

        self.progress_label = ctk.CTkLabel(h, text="", anchor="center")
        self.progress_label.grid(row=0, column=0, padx=10, pady=(6, 2), sticky="ew")

        self.progress = ctk.CTkProgressBar(h)
        self.progress.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="ew")
        self.progress.set(0)

        h.grid_remove()
        self.progress_holder = h

    def _show_progress(self, msg, val=None):
        self.progress_holder.grid()
        self.progress_label.configure(text=msg)
        if val is not None: self.progress.set(max(0.0, min(1.0, val)))
        self.update_idletasks()

    def _hide_progress(self):
        self.progress_holder.grid_remove()

    # ----- bottom: plot settings + canvas -----
    def _build_plot_row(self):
        cont = ctk.CTkFrame(self, corner_radius=10)
        cont.grid(row=2, column=0, padx=10, pady=(6, 10), sticky="nsew")
        cont.grid_columnconfigure(0, weight=0)
        cont.grid_columnconfigure(1, weight=1)
        cont.grid_rowconfigure(0, weight=1)

        # left controls
        left = ctk.CTkFrame(cont, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsw", padx=(10, 8), pady=10)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="Plot settings", font=ctk.CTkFont(size=16, weight="bold"))\
            .grid(row=0, column=0, padx=10, pady=(4, 6), sticky="w")

        ctk.CTkLabel(left, text="Layout").grid(row=1, column=0, padx=10, sticky="w")
        self.layout_var = ctk.StringVar(value="kamada_kawai")
        self.layout_menu = ctk.CTkOptionMenu(left, values=["kamada_kawai", "spring"],
                                             variable=self.layout_var,
                                             command=self._on_layout_change)
        self.layout_menu.grid(row=2, column=0, padx=10, pady=4, sticky="ew")

        # KK scale
        ctk.CTkLabel(left, text="KK Scale").grid(row=3, column=0, padx=10, pady=(6, 0), sticky="w")
        self.kk_scale_var = ctk.StringVar(value="1.0")
        self.kk_scale_entry = ctk.CTkEntry(left, textvariable=self.kk_scale_var, height=32)
        self.kk_scale_entry.grid(row=4, column=0, padx=10, pady=4, sticky="ew")
        self.kk_scale_entry.bind("<Return>", self._draw_plot)

        # Spring params (no seed)
        ctk.CTkLabel(left, text="Spring k").grid(row=5, column=0, padx=10, pady=(6, 0), sticky="w")
        self.k_var = ctk.StringVar(value="1.0")
        self.k_entry = ctk.CTkEntry(left, textvariable=self.k_var, height=32)
        self.k_entry.grid(row=6, column=0, padx=10, pady=4, sticky="ew")
        self.k_entry.bind("<Return>", self._draw_plot)

        ctk.CTkLabel(left, text="Spring iters").grid(row=7, column=0, padx=10, pady=(6, 0), sticky="w")
        self.iters_var = ctk.StringVar(value="50")
        self.iters_entry = ctk.CTkEntry(left, textvariable=self.iters_var, height=32)
        self.iters_entry.grid(row=8, column=0, padx=10, pady=4, sticky="ew")
        self.iters_entry.bind("<Return>", self._draw_plot)

        # shared normalized params
        ctk.CTkLabel(left, text="Node size × (norm)").grid(row=9, column=0, padx=10, pady=(6, 0), sticky="w")
        self.node_var = ctk.StringVar(value="200")
        self.node_entry = ctk.CTkEntry(left, textvariable=self.node_var, height=32)
        self.node_entry.grid(row=10, column=0, padx=10, pady=4, sticky="ew")
        self.node_entry.bind("<Return>", self._draw_plot)

        ctk.CTkLabel(left, text="Edge width × (norm)").grid(row=11, column=0, padx=10, pady=(6, 0), sticky="w")
        self.edge_var = ctk.StringVar(value="8")
        self.edge_entry = ctk.CTkEntry(left, textvariable=self.edge_var, height=32)
        self.edge_entry.grid(row=12, column=0, padx=10, pady=4, sticky="ew")
        self.edge_entry.bind("<Return>", self._draw_plot)

        ctk.CTkLabel(left, text="Min edge (norm 0–1)").grid(row=13, column=0, padx=10, pady=(6, 0), sticky="w")
        self.minedge_var = ctk.StringVar(value="0")
        self.minedge_entry = ctk.CTkEntry(left, textvariable=self.minedge_var, height=32)
        self.minedge_entry.grid(row=14, column=0, padx=10, pady=4, sticky="ew")
        self.minedge_entry.bind("<Return>", self._draw_plot)

        self.labels_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="Show labels", variable=self.labels_var)\
            .grid(row=15, column=0, padx=10, pady=(6, 2), sticky="w")

        ctk.CTkLabel(left, text="Core percentile (e.g., 90)").grid(row=16, column=0, padx=10, pady=(6, 0), sticky="w")
        self.core_pct_var = ctk.StringVar(value="90")
        self.core_pct_entry = ctk.CTkEntry(left, textvariable=self.core_pct_var, height=32)
        self.core_pct_entry.grid(row=17, column=0, padx=10, pady=4, sticky="ew")
        self.core_pct_entry.bind("<Return>", self._draw_plot)

        btns = ctk.CTkFrame(left)
        btns.grid(row=18, column=0, padx=10, pady=(6, 8), sticky="ew")
        btns.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(btns, text="Apply plot", command=self._draw_plot, height=34)\
            .grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkButton(btns, text="Save Outputs", command=self._save_outputs, height=34)\
            .grid(row=0, column=1, padx=(6, 0), sticky="ew")

        # right canvas
        right = ctk.CTkFrame(cont, corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(7.2, 5.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # default visibility of fields
        self._on_layout_change("kamada_kawai")

    # ----- compute -----
    def _on_compute(self):
        abs_path = self.abs_entry.get().strip()
        reg_path = self.reg_entry.get().strip()
        trim_pct = _to_float(self.trim_entry.get().strip(), 0.5)

        if not os.path.isfile(abs_path) or not os.path.isfile(reg_path):
            mb.showerror("Invalid paths", "Please select valid CSV files.")
            return

        self.compute_btn.configure(state="disabled")
        self._show_progress("Reading CSV files…", 0.0)

        def worker():
            try:
                papers = pd.read_csv(abs_path)
                abstracts = papers["Abstract"].fillna("").astype(str)
                regions = pd.read_csv(reg_path).iloc[:, 0].dropna().astype(str)

                # presence
                def p1(d, t): self.after(0, self._show_progress,
                                         "Scanning abstracts…", 0.5 * d / max(1, t))
                df_presence = build_presence_matrix_with_progress(abstracts, regions, on_progress=p1)

                # trim
                filtered_df, abs_hits, uniq_regions = trim_regions(df_presence, trim_pct)

                # co-mentions
                def p2(d, t): self.after(0, self._show_progress,
                                         "Building co-mentions…", 0.5 + 0.5 * d / max(1, t))
                co_raw = compute_comentions_with_progress(filtered_df, on_progress=p2)

                # normalize
                counts_norm, co_norm = normalize_network(filtered_df, co_raw)
                regions_kept = filtered_df.shape[1]
                abs_post = int((filtered_df.sum(axis=1) > 0).sum())

                def finalize():
                    self.filtered_df = filtered_df
                    self.kept_cols = filtered_df.columns
                    self.counts_norm = counts_norm
                    self.co_norm = co_norm

                    lines = [
                        f"Abstracts found (≥1 region): {abs_hits}",
                        f"Unique regions found (pre-filter): {uniq_regions}",
                        f"Regions kept after filtering: {regions_kept}",
                        f"Abstracts post filtering (≥1 kept region): {abs_post}",
                    ]
                    self.status.configure(state="normal"); self.status.delete("1.0", "end")
                    self.status.insert("end", "\n".join(lines)); self.status.configure(state="disabled")
                    print("\n".join(lines))

                    self._hide_progress(); self.compute_btn.configure(state="normal")
                    self._draw_plot()

                self.after(0, finalize)

            except Exception as e:
                def err(e):
                    self._hide_progress(); self.compute_btn.configure(state="normal")
                    mb.showerror("Compute error", str(e))
                self.after(0, err)

        threading.Thread(target=worker, daemon=True).start()

    # ----- layout field toggling -----
    def _on_layout_change(self, value: str):
        kk = (value == "kamada_kawai")
        self.kk_scale_entry.configure(state="normal" if kk else "disabled")
        self.k_entry.configure(state="disabled" if kk else "normal")
        self.iters_entry.configure(state="disabled" if kk else "normal")

    # ----- draw plot -----
    def _draw_plot(self, event=None):
        if self.counts_norm is None or self.co_norm is None or self.kept_cols is None:
            self.ax.clear(); self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, "Compute first (top panel).",
                         ha="center", va="center", fontsize=12)
            self.canvas.draw(); return

        layout = self.layout_var.get()
        params = {"scale": _to_float(self.kk_scale_var.get(), 1.0)} if layout == "kamada_kawai" else {
            "k": _to_float(self.k_var.get(), 1.0),
            "iterations": _to_int(self.iters_var.get(), 50),
        }
        node_x   = _to_float(self.node_var.get(), 200.0)
        edge_x   = _to_float(self.edge_var.get(), 8.0)
        min_edge = _to_float(self.minedge_var.get(), 0.0)
        show_lbl = self.labels_var.get()
        core_pct = _to_float(self.core_pct_var.get(), 90.0)

        self.ax.clear(); self.ax.set_axis_off()
        G = nx.Graph()
        for r, c in self.counts_norm.items():
            if r in self.kept_cols and c > 0: G.add_node(r, size_norm=float(c))
        for a, b in combinations(self.kept_cols, 2):
            if a in self.co_norm.index and b in self.co_norm.columns:
                w = float(self.co_norm.loc[a, b])
                if w > min_edge and a in G.nodes and b in G.nodes:
                    G.add_edge(a, b, weight_norm=w)

        pos = compute_layout(G, layout, params)
        if not G.number_of_nodes():
            self.ax.text(0.5, 0.5, "No nodes to display.\nLower Min edge.", ha="center", va="center", fontsize=12)
            self.canvas.draw(); return

        # core coloring
        core_mask = assign_core_mask(self.counts_norm.loc[list(G.nodes)], percentile=core_pct)
        CORE, NON = "#ad2f3e", "#5495ba"

        node_sizes = [G.nodes[n]['size_norm'] * node_x for n in G.nodes()]
        node_colors = [CORE if core_mask.get(n, False) else NON for n in G.nodes()]
        edge_widths = [G[u][v]['weight_norm'] * edge_x for u, v in G.edges()]
        edge_colors = [CORE if (core_mask.get(u, False) and core_mask.get(v, False)) else NON
                       for u, v in G.edges()]

        for n, s, col in zip(G.nodes(), node_sizes, node_colors):
            nx.draw_networkx_nodes(G, pos, nodelist=[n], node_size=s, node_color=col, alpha=0.9, ax=self.ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.45, ax=self.ax)
        if show_lbl:
            nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=self.ax)

        self.ax.set_title(f"Brain Region Co-mention Network | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}",
                          fontsize=13)
        self.canvas.draw()

    # ----- save -----
    def _save_outputs(self):
        path = fd.asksaveasfilename(defaultextension=".png",
                                    filetypes=[("PNG", "*.png")],
                                    title="Save outputs")
        if not path: 
            return
        try:
            base, _ = os.path.splitext(path)

            # save images
            self.fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
            self.fig.savefig(base + ".svg", dpi=300, bbox_inches="tight")

            # collect parameters
            params = {
                "abstracts_csv": self.abs_entry.get().strip(),
                "regions_csv": self.reg_entry.get().strip(),
                "trim_pct": _to_float(self.trim_entry.get(), 0.5),
                "layout": self.layout_var.get(),
                "kk_scale": _to_float(self.kk_scale_var.get(), 1.0),
                "spring_k": _to_float(self.k_var.get(), 1.0),
                "spring_iters": _to_int(self.iters_var.get(), 50),
                "node_mult": _to_float(self.node_var.get(), 200),
                "edge_mult": _to_float(self.edge_var.get(), 8),
                "min_edge": _to_float(self.minedge_var.get(), 0.0),
                "show_labels": self.labels_var.get(),
                "core_percentile": _to_float(self.core_pct_var.get(), 90),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
            with open(base + "_params.json", "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)

            mb.showinfo("Saved", f"Saved:\n{base}.png\n{base}.svg\n{base}_params.json")

        except Exception as e:
            mb.showerror("Save error", str(e))

    # ----- browse -----
    def _browse_abs(self):
        p = fd.askopenfilename(title="Select Abstracts CSV",
                               filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p: self.abs_entry.delete(0, "end"); self.abs_entry.insert(0, p)

    def _browse_reg(self):
        p = fd.askopenfilename(title="Select Brain Regions CSV",
                               filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p: self.reg_entry.delete(0, "end"); self.reg_entry.insert(0, p)


if __name__ == "__main__":
    app = App()
    app.mainloop()
