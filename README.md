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
├── topic_classifier.pkl      # Pre-trained topic classifier
├── EDA.ipynb                 # Exploratory data analysis notebook
├── requirements.txt          # All Python dependencies
├── data/
│   ├── news.db               # SQLite database (auto-created)
│   ├── bbc_news_train.csv    # Training data for the topic model
│   └── bbc_news_tests.csv    # Test data for the topic model
└── results/
    ├── enhanced_news.csv     # Final enriched output
    ├── learning_curves.png   # Topic model training curves
    └── training_model.py     # Script to retrain the topic classifier
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/nlp-scraper.git
cd nlp-scraper
```

### 2. Create a virtual environment

```bash
python -m venv nlp
source nlp/bin/activate        # Linux / macOS
nlp\Scripts\activate           # Windows
```

### 3. Install dependencies

> **Install PyTorch (CPU) first** before the rest of the packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
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

Reads 7 BBC RSS feeds (Technology, Business, World, Politics, Entertainment, Science, Sport), fetches the full article body for each item, and stores them in `data/news.db`.

### Step 2 — Enrich articles

```bash
python nlp_enriched_news.py
```

Loads every article from the database, runs all 4 NLP stages, prints progress to the console, and writes the results to `results/enhanced_news.csv`.

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

---

## 🏷️ Topic Classifier

The classifier is a **scikit-learn pipeline** (TF-IDF → Logistic Regression) trained on the BBC News dataset.

- **Training accuracy**: ≥ 95% (enforced by an assertion in training)
- **Categories**: `business`, `entertainment`, `politics`, `sport`, `tech`

To retrain the model with new data:

```bash
python results/training_model.py
```

This saves a new `topic_classifier.pkl` and learning curve plot to `results/learning_curves.png`.

---

## 📊 Output Schema

The final `results/enhanced_news.csv` contains:

| Column | Description |
|---|---|
| `id` | Unique UUID |
| `url` | Source article URL |
| `date` | Publication date |
| `headline` | Article headline |
| `body` | Full article text |
| `org` | Detected organisations (list) |
| `topic` | Predicted topic category |
| `sentiment` | VADER compound score (−1 to +1) |
| `scandal_distance` | Cosine similarity scandal score (0–1) |
| `top_10` | `True` if in the top 10 highest scandal scores |

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

---