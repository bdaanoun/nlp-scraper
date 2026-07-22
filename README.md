# 📰 NLP News Enrichment Pipeline

An end-to-end pipeline that **scrapes live BBC News articles**, then enriches each one with four NLP layers: **entity detection**, **topic classification**, **sentiment analysis**, and **environmental scandal detection**.

---

## 🧠 What It Does

```
BBC RSS Feeds
      ↓
  scraper_news.py          →  Fetches articles & stores them in SQLite
      ↓
  nlp_enriched_news.py     →  Runs 4 NLP stages on every article
      ↓
  results/enhanced_news.csv  →  Final enriched dataset
```

### The 4 NLP Stages

| Stage | Method | Output |
|---|---|---|
| **1. Entity Detection** | spaCy `en_core_web_sm` (NER) | List of organisations mentioned |
| **2. Topic Classification** | TF-IDF + Logistic Regression (trained on BBC dataset) | `tech`, `business`, `sport`, `politics`, `entertainment` |
| **3. Sentiment Analysis** | VADER (`nltk`) | Compound score + `positive / neutral / negative` label |
| **4. Scandal Detection** | Sentence-BERT embeddings + cosine similarity vs. scandal keywords | Float score 0–1 (higher = more scandal-like) |

The top 10 articles with the highest scandal score are flagged automatically.

---

## 📁 Project Structure

```
nlp-scraper/
├── scraper_news.py           # Step 1 — scrape & store articles
├── nlp_enriched_news.py      # Step 2 — NLP enrichment pipeline
├── EDA.ipynb                 # Exploratory data analysis notebook
├── requirements.txt          # All Python dependencies
├── data/
│   ├── news.db               # SQLite database (auto-created)
│   ├── bbc_news_train.csv    # Training data for the topic model
│   └── bbc_news_tests.csv    # Test data for the topic model
├── results/
│   ├── enhanced_news.csv     # Final enriched output
│   ├── learning_curves.png   # Topic model training curves
│   └── training_model.py     # Script to retrain the topic classifier
└── topic_classifier.pkl      # Pre-trained topic classifier
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/bdaanoun/nlp-scraper.git
cd nlp-scraper
```

### 2. Create a virtual environment

```bash
python -m venv nlp
source nlp/bin/activate        # Linux / macOS
nlp\Scripts\activate           # Windows
```

### 3. Install dependencies

> ⚠️ PyTorch's CPU build is **not** hosted on the default PyPI index — it must be
> installed first, from PyTorch's own index, otherwise pip will either fail to find
> the `+cpu` version or silently resolve the much larger GPU/CUDA build instead.

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --no-cache-dir
```
### 4. Download NLTK data (one-time)

```python
import nltk
nltk.download("vader_lexicon")
```

---

## 🚀 Usage

### Step 1 — Scrape articles

```bash
python scraper_news.py
```

Reads 7 BBC RSS feeds (Technology, Business, World, Politics, Entertainment, Science, Sport), fetches the full article body for each item, and stores them in `data/news.db`. Let it run until at least 300 articles are collected (stop manually once enough data has been retrieved, if needed).

### Step 2 — Train the topic classifier (if not already trained)

```bash
python results/training_model.py
```

Trains a TF-IDF + Logistic Regression pipeline on the labelled BBC dataset (`data/bbc_news_train.csv` / `data/bbc_news_tests.csv`), evaluates it on the held-out test set, and saves:
- `topic_classifier.pkl` — the trained pipeline + label encoder
- `results/learning_curves.png` — proof of correct fit (see below)

The script enforces a **hard accuracy threshold of 95%** on the test set via an assertion — it will fail loudly rather than silently save an underperforming model.

### Step 3 — Enrich articles

```bash
python nlp_enriched_news.py
```

Loads every article from `data/news.db`, runs all 4 NLP stages, prints progress to the console for each article, and writes the results to `results/enhanced_news.csv`.

---

## 📈 Overfitting Check — Learning Curves

**What is overfitting?** A model overfits when it learns the training data too closely — including its noise and idiosyncrasies — rather than the general patterns that transfer to new, unseen data. An overfit model shows very high accuracy on training data but performs noticeably worse on data it hasn't seen before.

**How this is checked here:** `results/learning_curves.png` plots training accuracy and 5-fold cross-validation accuracy against increasing amounts of training data (`sklearn.model_selection.learning_curve`). The plot shows:
- A small, stable gap between the training and cross-validation curves at full data size (~1.0 vs ~0.985) — a large, persistent gap would indicate overfitting.
- The cross-validation curve **rising and converging** toward the training curve as more data is added, rather than diverging away from it.
- The cross-validation curve **plateauing** past ~1000 examples, and a shrinking variance band across folds — both signs the model's performance is stable and not the result of a lucky split.

Together, this is the evidence that the classifier generalises well rather than having memorised the training set.

---

## 🕵️ Scandal Detection — How It Works

```
Define keywords → "pollution", "deforestation", "toxic waste", ...
        ↓
Convert keywords to sentence embeddings  (Sentence-BERT: all-MiniLM-L6-v2)
        ↓
Split article body into sentences containing an ORG entity
        ↓
Convert those sentences to embeddings
        ↓
Compute cosine similarity between sentence embeddings and keyword embeddings
        ↓
Max similarity score = scandal score  (close to 1 → likely scandal)
```

### Why these choices?

**Embeddings — Sentence-BERT (`all-MiniLM-L6-v2`)**: unlike word-level embeddings (e.g. Word2Vec), Sentence-BERT produces a single vector representing the overall meaning of a *full sentence*, which fits the need here to compare whole sentences — not isolated words — against scandal-related keywords. `all-MiniLM-L6-v2` specifically was chosen because it is lightweight and fast enough to run on CPU only (no GPU required, important since this project runs on CPU-only PyTorch), while still performing strongly on semantic textual similarity benchmarks — a good accuracy/speed tradeoff for processing hundreds of articles.

**Distance metric — cosine similarity**: sentence embeddings encode meaning primarily in the *direction* of the vector rather than its magnitude. Cosine similarity measures the angle between two vectors and is unaffected by differences in magnitude, which makes it the standard, recommended metric for comparing sentence embeddings — as opposed to Euclidean distance, which is sensitive to vector length and less suited to this kind of normalised embedding space.

> **Naming note**: despite being called `scandal_distance` (matching the subject's naming), the value stored is actually a **cosine similarity** score in the range 0–1, where a **higher** value means the article is **more** semantically similar to the scandal keywords — i.e. a higher score means a higher likelihood of scandal, not a greater "distance" from it. This naming was kept for consistency with the subject's specified column name.

### Avoiding false positives

Keywords were chosen to avoid ambiguous terms that carry non-environmental meanings in common usage (e.g. avoiding a bare word like "spill" which can refer to a coffee spill as easily as an oil spill). Keywords focus on more specific, less ambiguous phrasing (e.g. "toxic waste," "deforestation," "oil spill") to reduce false positives from unrelated contexts.

---

## 🏷️ Topic Classifier

The classifier is a **scikit-learn pipeline** (TF-IDF → Logistic Regression) trained on the BBC News dataset.

- **Test accuracy**: ≥ 95% (enforced by an assertion in training)
- **Categories**: `business`, `entertainment`, `politics`, `sport`, `tech`

**Pipeline configuration**:
- `TfidfVectorizer`: unigrams + bigrams, English stop words removed, sublinear TF scaling (log-dampens the influence of highly repeated words so a handful of frequent terms don't dominate a document's feature vector).
- `LogisticRegression`: `C=5.0` (regularisation strength, chosen empirically — moderate value, appropriate for this dataset's relatively clean, well-separated class vocabulary), solver `lbfgs` (efficient for multi-class problems at this dataset size).

To retrain the model with new data:

```bash
python results/training_model.py
```

This saves a new `topic_classifier.pkl` and learning curve plot to `results/learning_curves.png`.

---

## 📊 Output Schema

The final `results/enhanced_news.csv` contains one row per article (300+ rows), with the following columns. Column names use lowercase snake_case for consistency with pandas conventions, and correspond 1:1 with the subject's `Org`, `Topics`, `Sentiment`, `Scandal_distance`, and `Top_10` fields:

| Column | Type | Description | Corresponds to subject field |
|---|---|---|---|
| `id` | `str` (uuid) | Unique article identifier | Unique ID |
| `url` | `str` | Source article URL | URL |
| `date` | `date` | Date scraped | Date scraped |
| `headline` | `str` | Article headline | Headline |
| `body` | `str` | Full article text | Body |
| `org` | `list[str]` | Organisations detected via spaCy NER | Org |
| `topic` | `list[str]` | Predicted topic category (single-element list) | Topics |
| `sentiment` | `float` | VADER compound score (−1 to +1) | Sentiment |
| `scandal_distance` | `float` | Cosine similarity scandal score (0–1, higher = more scandal-like) | Scandal_distance |
| `top_10` | `bool` | `True` if among the 10 highest scandal scores in the dataset | Top_10 |

---

## 🔧 RSS Feeds Scraped

| Feed | URL |
|---|---|
| Technology | `feeds.bbci.co.uk/news/technology/rss.xml` |
| Business | `feeds.bbci.co.uk/news/business/rss.xml` |
| World | `feeds.bbci.co.uk/news/world/rss.xml` |
| Politics | `feeds.bbci.co.uk/news/politics/rss.xml` |
| Entertainment | `feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml` |
| Science | `feeds.bbci.co.uk/news/science_and_environment/rss.xml` |
| Sport | `feeds.bbci.co.uk/sport/rss.xml` |

---

## 🛠️ Key Dependencies

| Library | Purpose |
|---|---|
| `requests` + `beautifulsoup4` | Web scraping |
| `spacy` (`en_core_web_sm`) | Named entity recognition |
| `nltk` (VADER) | Sentiment analysis |
| `sentence-transformers` | Sentence embeddings for scandal detection |
| `scikit-learn` | Topic classifier (TF-IDF + Logistic Regression) |
| `pandas` | Data handling & CSV export |
| `sqlite3` | Local article database |

All exact versions are pinned in `requirements.txt` (generated via `pip freeze`).