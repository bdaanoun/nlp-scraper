import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load BBC News dataset
# using dataset from: http://mlg.ucd.ie/datasets/bbc.html
DATASET_PATH = "data/bbc"
MODEL_PATH   = "topic_classifier.pkl"
CURVES_PATH  = "results/learning_curves.png"

print("Loading dataset...")
dataset = load_files(DATASET_PATH, encoding="utf-8", decode_error="replace")
X, y    = dataset.data, dataset.target
labels  = dataset.target_names        # ['business','entertainment','politics','sport','tech']
print(f"  {len(X)} documents, {len(labels)} classes: {labels}")

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )),
])

print("\nTraining model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred    = pipeline.predict(X_test)
test_acc  = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {test_acc * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=labels))

assert test_acc >= 0.95, f"Accuracy {test_acc:.2%} is below the 95% threshold!"

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"pipeline": pipeline, "labels": labels}, f)
print(f"\nModel saved {MODEL_PATH}")

# Learning curves
print("\nComputing learning curves (this takes a minute)...")
train_sizes, train_scores, val_scores = learning_curve(
    pipeline,
    X, y,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_mean, "o-", color="royalblue",  label="Training score")
ax.plot(train_sizes, val_mean,   "o-", color="darkorange", label="Cross-validation score")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="royalblue")
ax.fill_between(train_sizes, val_mean   - val_std,   val_mean   + val_std,   alpha=0.15, color="darkorange")

ax.set_title("Learning Curves — Topic Classifier (TF-IDF + Logistic Regression)")
ax.set_xlabel("Training examples")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.5, 1.05)
ax.legend(loc="lower right")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

os.makedirs("results", exist_ok=True)
fig.savefig(CURVES_PATH, dpi=150)
print(f"Learning curves saved {CURVES_PATH}")