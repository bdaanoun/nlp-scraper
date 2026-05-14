import sqlite3
import pickle
import spacy
import pandas as pd

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

        results.append({
            "id":       row["id"],
            "url":      row["url"],
            "date":     row["date"],
            "headline": row["headline"],
            "body":     row["body"],
            "org":      orgs,
            "topic":    topic,
        })

    df_results = pd.DataFrame(results)
    print(f"\nDone — {len(df_results)} articles enriched")
    return df_results

if __name__ == "__main__":
    df = main()
    print(df[["headline", "org", "topic"]].head(10))