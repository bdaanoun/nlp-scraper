import sqlite3
import pickle
import spacy
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_sm")

DB_PATH        = "data/news.db"
MODEL_PATH     = "topic_classifier.pkl"

# Load topic model once at startup
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

topic_pipeline = model_data["pipeline"]
topic_labels   = model_data["labels"]

#  Stage 1: Entity detection
def detect_entities(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))

#  Stage 2: Topic detection 
def detect_topic(headline, body):
    combined = headline + " " + body
    idx      = topic_pipeline.predict([combined])[0]
    return topic_labels[idx]


sia = SentimentIntensityAnalyzer()

# STAGE 3: SENTIMENT ANALYSIS
def analyze_sentiment(headline, body):
    """
    Run VADER on headline + body combined.
    Returns compound score and label.
    """
    # Combine headline and body for better context
    text = headline + ". " + body

    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    # Label based on compound score
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return compound, label

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# STAGE 4: SCANDAL DETECTION

# Keywords representing environmental disasters caused by companies
SCANDAL_KEYWORDS = [
    "pollution",
    "deforestation",
    "toxic waste",
    "chemical spill",
    "oil spill",
    "contamination",
    "environmental damage",
    "illegal dumping",
    "carbon emissions",
    "ecological disaster",
]

# Embed keywords once — no need to recompute every article
keyword_embeddings = embedder.encode(SCANDAL_KEYWORDS)

def compute_scandal_score(body, orgs):
    """
    For each sentence containing an ORG entity,
    compute cosine similarity with scandal keywords.
    Return the max similarity score across all sentences.
    """
    if not orgs:
        return 0.0

    # Split body into sentences
    sentences = [s.strip() for s in body.split(".") if len(s.strip()) > 20]

    # Keep only sentences that mention at least one ORG
    org_sentences = [
        s for s in sentences
        if any(org.lower() in s.lower() for org in orgs)
    ]

    if not org_sentences:
        return 0.0

    # Embed the org sentences
    sentence_embeddings = embedder.encode(org_sentences)

    # Compute cosine similarity between each sentence and each keyword
    similarities = cosine_similarity(sentence_embeddings, keyword_embeddings)

    # Return the highest similarity score found
    return float(similarities.max())


#  Load articles
def load_articles():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("SELECT * FROM articles", conn)
    conn.close()
    return df

#  Main
def main():
    print("Loading articles from database...")
    df = load_articles()
    print(f"Found {len(df)} articles\n")

    results = []

    for i, row in df.iterrows():
        print(f"\nEnriching {row['url']}:")

        # Stage 1
        print("\n---------- Detect entities ----------")
        orgs = detect_entities(row["body"])
        if orgs:
            print(f"Detected {len(orgs)} companies which are: {', '.join(orgs)}")
        else:
            print("No organizations detected")

        # Stage 2
        print("\n---------- Topic detection ----------")
        print("Text preprocessing ...")
        topic = detect_topic(row["headline"], row["body"])
        print(f"The topic of the article is: {topic}")

        print("\n---------- Sentiment analysis ----------")
        compound, label = analyze_sentiment(row["headline"], row["body"])
        print(f"The article '{row['headline']}' has a {label} sentiment")
        print(f"Compound score: {compound}")

        # Stage 4
        print("\n---------- Scandal detection ----------")
        print("Computing embeddings and distance ...")
        scandal_score = compute_scandal_score(row["body"], orgs)
        print(f"Scandal score: {scandal_score:.4f}")

        results.append({
        "id":               row["id"],
        "url":              row["url"],
        "date":             row["date"],
        "headline":         row["headline"],
        "body":             row["body"],
        "org":              orgs,
        "topic":            topic,
        "sentiment":        compound,
        "scandal_distance": scandal_score,
        })

    df_results = pd.DataFrame(results)

    # Flag top 10 articles with highest scandal score
    threshold = df_results["scandal_distance"].nlargest(10).min()
    df_results["top_10"] = df_results["scandal_distance"] >= threshold

    # Print flagged articles
    print("\nTop 10 scandal articles:")
    top10 = df_results[df_results["top_10"]][["headline", "org", "scandal_distance"]]
    print(top10.to_string())

    # Save to CSV
    df_results.to_csv("results/enhanced_news.csv", index=False)
    print("\nSaved to results/enhanced_news.csv")

    return df_results


if __name__ == "__main__":
    df = main()
    print(df[["headline", "org", "topic"]].head(10))