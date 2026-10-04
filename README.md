<h1 align="center">🎬 Letterboxd Movie Recommender</h1>

<p align="center">
  <strong>From the films you love to the ones you might.</strong><br>
  A content-based recommendation lab that turns Letterboxd ratings, likes, and diary entries into personal movie suggestions.
</p>

<p align="center">
  <a href="main.ipynb"><strong>Explore the notebook ↗</strong></a> ·
  <a href="#results">Results</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#gallery">Gallery</a>
</p>

<p align="center" style="text-align: center;">
  <img src="docs/movie-rec-sys.svg" alt="movie-rec-sys logo with orange, green, and blue dots" width="480" style="display: block; margin: 0 auto; max-width: 100%; height: auto;">
</p>

---

| 🎞️ Processed catalog | 🧠 Approaches | 📊 Experiments | 🖼️ Original charts |
| :---: | :---: | :---: | :---: |
| **12,801 films** | **3 model families** | **5 K-NN configurations** | **23 extracted outputs** |

*Figures refer to the saved outputs in [main.ipynb](main.ipynb).*

## The idea

Two films can be similar because they share genres, feature the same topics, or tell semantically related stories. This project explores all three perspectives and compares them using the same Letterboxd profile.

The notebook covers the entire workflow: data integration, feature engineering, recommendations, evaluation, and vector space analysis. Each K-NN suggestion also retains the favorite film that generated it, making the connection between personal taste and results easier to understand.

**What's inside:** rating-weighted profiles, hard and soft clustering, lexical and semantic ranking, reranking with TMDB ratings, and a final experiment combining the three engines.

## From data to watchlist

```text
LETTERBOXD EXPORT                        THE MOVIES DATASET
ratings · loved_films · diary            metadata · credits · keywords
              │                                      │
              └─────────── title + year ──────────────┘
                                  │
                         PROFILE + CATALOG
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
           STRUCTURED          LEXICAL           SEMANTIC
         genres + numbers     TF-IDF soup      natural language
         K-Means / K-NN      sparse vectors   MiniLM embeddings
                │                 │                 │
                └─────────────────┼─────────────────┘
                                  ▼
                    RANKING · FILTERS · COMPARISON
                                  │
                                  ▼
                     PERSONALIZED RECOMMENDATIONS
```

The catalog joins metadata, cast, directors, and keywords using TMDB IDs. Preprocessing filters out records with popularity ≤ 3 or synopses of ≤ 25 characters, removes duplicates, and prepares two content representations:

- **`soup_tfidf`**: synopsis text and tokens for genres, keywords, director, and cast; metadata tokens are repeated to increase their weight.
- **`soup_transformer`**: a structured natural-language description containing the title, plot, genres, director, and cast.

The personal profile is matched to the catalog using normalized titles and release years. Ratings are standardized to weight the movie vectors; K-NN queries start from films with **both a 5-star rating and a like**.

## Three ways to find your next film

| Engine | Representation | How it generates recommendations |
| :--- | :--- | :--- |
| 🟢 **Structured** | One-hot genres, popularity, average rating, and year | K-NN with cosine distance; compares original features against hard/soft K-Means compression with **6 clusters**. |
| 🟠 **Lexical** | TF-IDF with up to **4,096 features** | Similarity to the weighted user profile, or nearest-neighbor searches around favorite films. |
| 🔵 **Semantic** | Multilingual embeddings with **384 dimensions** | The same retrieval strategies, applied to the semantic content of movie descriptions. |

The engine labeled **“BERT” in the notebook** actually uses `paraphrase-multilingual-MiniLM-L12-v2` through Sentence Transformers.

In the user-vector rankings, films already present in the mapped profile are excluded. The top 20 candidates are then reranked using a score combining **50% similarity** and **50% normalized average TMDB rating**.

## Results

### Comparing K-NN configurations

![Comparison of the five K-NN models: like rate, five-star rate, and Model Performance Score](docs/images/model-benchmark.png)

| Configuration | Likes among hits | 5-star ratings among hits | MPS ↑ |
| :--- | ---: | ---: | ---: |
| Structured K-NN | 17.07% | 29.27% | 44.59 |
| **Hard Cluster K-NN** | 24.53% | **47.17%** | **50.72** |
| Soft Cluster K-NN | 15.38% | 35.90% | 43.49 |
| TF-IDF K-NN | **28.57%** | 21.43% | 50.00 |
| Semantic K-NN | 0.00% | 0.00% | 31.33 |

In these outputs, **Hard Cluster K-NN** achieves the highest MPS and the largest share of five-star films among hits. **TF-IDF K-NN** achieves the highest share of likes.

### How much does list length matter?

![Vector Performance Score sensitivity to the Top-N cutoff: TF-IDF reaches 80.9 points at Top-20](docs/images/top-n-sensitivity.png)

The TF-IDF profile reaches **80.9 VPS points at Top-20**, then drops to **52.8 at Top-100**. The semantic profile stays between roughly 31 and 37 points across the displayed windows: in this experiment, greater model complexity does not automatically produce better rankings.

<details>
<summary><strong>Understanding the metrics and their limitations</strong></summary>

**Hits** are matches between recommendation candidates and the user's rated history. The like and five-star percentages in the table are calculated over these matches, rather than the entire catalog.

```text
MPS = 0.60 × like_hit_rate + 0.40 × (mean_rating / 5 × 100)
VPS = 0.40 × bracket_hit_rate + 0.40 × like_hit_rate
    + 0.20 × (mean_rating / 5 × 100)
```

`bracket_hit_rate` measures the percentage of Top-N items that appear in the user's history. MPS and VPS are composite scores defined in the notebook on a 0–100 scale; they measure different aspects and should not be compared directly.

Evaluation reuses the history that contributes to building the profile, without an independent train/test split. K-NN pools can also contain the same film more than once, and evaluation does not deduplicate them. These results therefore describe an **exploratory experiment on a single profile**, rather than an estimate of performance on unseen users or films.

The saved mapping reports **374 matched rows out of 782 (47.83%)**: unmatched titles and films missing from the catalog remain outside the mapped profile and its watched-film filter. Rows do not necessarily represent unique films, partly because of rewatches in the diary.

The figures reported here come from the **saved outputs**, which differ from the notebook's concluding commentary in some places. No new training session was run to generate this README.

</details>

### A taste of the recommendations

Some results from the final experiment intersecting the three engines:

| Film | Year | Director |
| :--- | :---: | :--- |
| **John Rabe** | 2009 | Florian Gallenberger |
| **Carlito's Way** | 1993 | Brian De Palma |
| **The Firm** | 1993 | Sydney Pollack |
| **Amadeus** | 1984 | Miloš Forman |
| **The Battle of Algiers** | 1966 | Gillo Pontecorvo |

The saved intersection contains **121 candidates**. It combines the Soft Cluster K-NN pool with the unseen-film catalogs from both text engines, then ranks the results using **40% structural distance, 30% TF-IDF dissimilarity, and 30% semantic dissimilarity**. This is not a vote across three independent Top-10 lists.

## Gallery

All images below are **original PNGs extracted from the outputs of `main.ipynb`**, without recreating the charts.

### The geometry of the five spaces

![Cosine similarity distributions across five spaces: structured, hard cluster, soft cluster, TF-IDF, and semantic](docs/images/spaces-cosine.png)

<details>
<summary><strong>Behind the scenes: scaling, clusters, sparsity, and distances</strong></summary>

#### Before and after normalization

![Feature distributions before normalization](docs/images/feature-scales-raw.png)
![Feature distributions after normalization](docs/images/feature-scales-normalized.png)

#### Choosing the number of clusters

![Elbow method and silhouette scores for different cluster counts](docs/images/cluster-selection.png)
![Silhouette analysis of the six clusters](docs/images/cluster-silhouette.png)

#### How sparse is the lexical space?

![Visualization of TF-IDF matrix sparsity](docs/images/tfidf-sparsity.png)

#### Comparing Euclidean distances

![Euclidean distance distributions across the five representations](docs/images/spaces-euclidean.png)

</details>

The remaining charts are available in [`docs/images/`](docs/images/). The [manifest](docs/images/manifest.json) maps each PNG to its source cell and output using zero-based indices.

## Getting started

### 1. Set up the environment

From the project directory, with Python and `pip` available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy matplotlib seaborn scikit-learn \
  deep-translator ipywidgets sentence-transformers tqdm jupyterlab
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of `source`. The repository does not include a dependency lockfile: these commands set up the environment but do not reproduce pinned versions from the original session.

### 2. Add the data

Download **The Movies Dataset**, already referenced in the notebook, from its [Kaggle dataset page](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and arrange the three CSV files as follows:

```text
datasets/kaggle/
├── movies_metadata.csv
├── credits.csv
└── keywords.csv
```

`datasets/kaggle/` is excluded from Git. Preprocessing generates `movies_metadata_en.csv` by attempting to translate synopses into English for films whose original language is not `en`; this step uses a remote service and may take time.

To use your own profile, export your data from **Letterboxd → Settings → Data**, create `datasets/<username>/` containing `ratings.csv`, `loved_films.csv`, and `diary.csv`, then update the notebook variable:

```python
username = "your_username"
```

The repository already includes an example profile in `datasets/danieledisp/`.

### 3. Open the notebook

```bash
jupyter lab main.ipynb
```

Run the sections in order, paying attention to the two checkpoint steps:

- **Personal profile:** when switching users, rebuild the mapping and weights. Save the new `df_profile_mapped.pkl` before the **Shortcut** section, or skip the cell that loads the existing checkpoint to avoid overwriting the new profile.
- **Embeddings:** choose between generation (**ONLY FIRST TIME**) and loading (**IF ALREADY SAVED**). Reuse `bert_matrix_checkpoint.npy` only if the catalog's contents and row order match those used to generate it; otherwise, regenerate it. The first generation requires downloading the model.

## Repository map

```text
movie-rec-sys/
├── README.md                         # Overview and guide
├── main.ipynb                        # Pipeline, experiments, and outputs
├── docs/
│   ├── movie-rec-sys.svg              # Project logo
│   └── images/                       # Cover + 23 charts + manifest
└── datasets/
    ├── danieledisp/                  # Example Letterboxd export
    ├── df_profile_mapped.pkl         # Mapped profile checkpoint
    ├── bert_matrix_checkpoint.npy    # Embedding checkpoint
    └── kaggle/                       # Local dataset, excluded from Git
```

**Stack:** Python · pandas · NumPy · scikit-learn · Sentence Transformers · Matplotlib · Seaborn · Jupyter.

## Previewing the README

In **VS Code**, open `README.md` and press **⇧⌘V** on macOS or **Ctrl+Shift+V** on Windows/Linux. To keep the preview alongside the source, use **Markdown: Open Preview to the Side** from the command palette.

On **GitHub**, the README is rendered automatically on the repository page. Include `docs/images/` when uploading your changes so image assets remain available. Reading the README does not require running the notebook.

---

<p align="center">
  <strong>Your taste is the starting point. Your next film is the discovery.</strong><br>
  <sub>An independent experimental project based on Letterboxd exports and The Movies Dataset.</sub>
</p>

## AI usage

This `README.md` was written with the assistance of **OpenAI Codex**, based on the **GPT-6** model. The AI usage disclosed in this note is limited **exclusively to writing the README.md**.
