# Brain Region Co-Mention Network

This repository provides a pipeline to **collect PubMed abstracts**, **extract brain region mentions**, and **build a normalized co-mention network** with interactive visualization.

The workflow is divided into three main scripts:

1. **`_create_brain_region_list.py`** – Generates a curated list of brain region terms.
2. **`01_collect_abstracts.py`** – Fetches PubMed abstracts using a graphical interface.
3. **`02_create_network.py`** – Builds and visualizes a brain region co-mention network.

---

## Features

* Automatic creation of a **brain region dictionary** from multiple atlases (Harvard-Oxford, AAL, AAL3, Allen, fMRI).
* GUI for **PubMed search & batch fetching**, with progress tracking and export to CSV.
* GUI for **network construction & visualization**:

  * Abstracts × Brain regions presence matrix
  * Region trimming by prevalence
  * Co-mention analysis
  * Normalized network with customizable layouts (`kamada_kawai`, `spring`)
  * Adjustable plot parameters (node size, edge width, min edge threshold, labels, percentile highlighting)
  * Export plots as **PNG/SVG** and parameters as JSON.

---

## Installation

Clone the repository and create the conda environment from the provided `environment.yml` file:

```bash
git clone <your-repo-url>
cd <your-repo>
conda env create -f environment.yml
conda activate brain-network-env
```

---

## Usage

### 1. Create brain region list

Generates `brain_regions.csv` from multiple atlases.

```bash
python _create_brain_region_list.py
```

Output:

* `brain_regions.csv` → curated list of brain region names.

---

### 2. Collect PubMed abstracts

Launch GUI for querying and downloading abstracts.

```bash
python 01_collect_abstracts.py
```

* Enter your **NCBI Entrez email** and **API key**.
* Enter a **PubMed query** (default is focused on functional connectivity & postpartum depression).
* Run search, fetch abstracts in batches, and auto-save results.
* Outputs:

  * `compiled_articles/compiled_articles_<timestamp>.csv`
  * `compiled_articles/compiled_articles_<timestamp>.query.txt`

---

### 3. Create brain region co-mention network

Launch GUI to compute and visualize the co-mention network.

```bash
python 02_create_network.py
```

* Select the abstracts CSV (from step 2).
* Select `brain_regions.csv` (from step 1).
* Adjust parameters:

  * **Trim %** – filters rarely mentioned regions.
  * **Layout** – `kamada_kawai` or `spring`.
  * **Scaling, Node size, Edge width, Min edge threshold**.
  * **Core percentile** – highlights high-degree nodes & edges in red.
* Save outputs as `.png`, `.svg`, and `_params.json`.

---

## Example Workflow

1. Generate a list of brain regions.
2. Query PubMed for `"functional connectivity" AND depression`.
3. Fetch \~2000 abstracts into `compiled_articles.csv`.
4. Run the network builder, trim at 0.5%, and visualize the network.
5. Save the resulting **brain region co-mention network**.

---

## Repository Structure

```
.
├── _create_brain_region_list.py   # builds brain_regions.csv
├── 01_collect_abstracts.py        # GUI for PubMed fetching
├── 02_create_network.py           # GUI for network visualization
├── brain_regions.csv              # example brain region list
├── brain_regions/                 # atlas CSVs (AAL3, Allen, fMRI)
├── environment.yml                # conda environment specification
└── compiled_articles/             # auto-created for PubMed results
```

---

## Notes

* NCBI requires both **email** and **API key** for PubMed API usage.
* Long-running fetch jobs are performed in a **background thread** to keep the UI responsive.
* Default PubMed output folder is `./compiled_articles/`.
