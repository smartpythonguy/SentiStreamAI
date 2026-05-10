"""
╔══════════════════════════════════════════════════════════════════╗
║           SentiStream AI Prototype  — Amazon Reviews             ║
║     Score-based Sentiment · Search · Opinion Summarization       ║
╚══════════════════════════════════════════════════════════════════╝

Dataset  : dataset.csv  (Amazon Fine Food Reviews)
Text col : Text
Score col: Score  →  4-5 = Positive | 3 = Neutral | 1-2 = Negative

Modules:
  1. DataLoader         — load CSV, map Score → sentiment label
  2. TextCleaner        — NLTK cleaning pipeline
  3. Vectorizer         — TF-IDF feature extraction
  4. SentimentModel     — Logistic Regression classifier
  5. Evaluator          — accuracy, F1, confusion matrix
  6. Predictor          — single / batch inference
  7. ReviewSearchEngine — keyword search + opinion summarizer
  8. run_pipeline()     — end-to-end orchestrator
  9. interactive_mode() — live console Q&A loop

Run:
    python sentistream_ai.py
"""

# ──────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────
import os
import re
import sys
import pickle
import random
import logging
import warnings
import textwrap
from collections import Counter

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SentiStream")


# ══════════════════════════════════════════════════════════════════
# 0.  NLTK Bootstrap
# ══════════════════════════════════════════════════════════════════
def download_nltk_resources() -> None:
    """Download NLTK data on first run — safe to call every time."""
    needed = [
        ("tokenizers/punkt",     "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
        ("corpora/wordnet",      "wordnet"),
    ]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            log.info("Downloading NLTK resource: %s", pkg)
            nltk.download(pkg, quiet=True)


# ══════════════════════════════════════════════════════════════════
# 1.  DataLoader
#     Reads dataset.csv and converts the numeric Score column into
#     human-readable sentiment labels.
# ══════════════════════════════════════════════════════════════════
class DataLoader:
    """
    Load the Amazon Reviews CSV and produce a clean DataFrame with:
        text    — raw review text (from 'Text' column)
        summary — short review title (from 'Summary' column)
        label   — Positive / Neutral / Negative (from 'Score' column)
        score   — original 1-5 star score (kept for search filters)

    Score mapping:
        4 or 5  → Positive
        3       → Neutral
        1 or 2  → Negative
    """

    def __init__(
        self,
        csv_path:    str = "dataset.csv",
        text_col:    str = "Text",
        score_col:   str = "Score",
        summary_col: str = "Summary",
        max_rows:    int = None,   # set e.g. 50_000 to limit large files
    ):
        """
        Args:
            csv_path    : Path to the Amazon reviews CSV.
            text_col    : Column name holding review text.
            score_col   : Column name holding 1-5 star scores.
            summary_col : Column name holding short review titles.
            max_rows    : If set, only the first N rows are loaded
                          (useful for quickly testing big datasets).
        """
        self.csv_path    = csv_path
        self.text_col    = text_col
        self.score_col   = score_col
        self.summary_col = summary_col
        self.max_rows    = max_rows

    @staticmethod
    def score_to_label(score) -> str:
        """
        Convert a numeric 1-5 star rating to a sentiment label.

        4-5 stars → Positive   (happy customer)
        3   stars → Neutral    (mixed / average)
        1-2 stars → Negative   (unhappy customer)
        """
        try:
            s = float(score)
        except (ValueError, TypeError):
            return "Neutral"   # fallback for bad data

        if s >= 4:
            return "Positive"
        elif s == 3:
            return "Neutral"
        else:                  # 1 or 2
            return "Negative"

    def load(self) -> pd.DataFrame:
        """
        Read the CSV and return a DataFrame with columns:
            [text, summary, label, score]

        Exits with a friendly message if the file is not found.
        """
        # ── Check file exists ─────────────────────────────────────
        if not os.path.exists(self.csv_path):
            print(f"\n{'═' * 60}")
            print(f"  ❌  File not found: '{self.csv_path}'")
            print(f"\n  Make sure your CSV file is in the same folder")
            print(f"  as sentistream_ai.py and is named 'dataset.csv'.")
            print(f"\n  Or change the csv_path variable at the bottom")
            print(f"  of this file to match your filename.")
            print(f"{'═' * 60}\n")
            sys.exit(1)

        # ── Load CSV ──────────────────────────────────────────────
        log.info("Loading dataset: %s", self.csv_path)
        try:
            df = pd.read_csv(self.csv_path, nrows=self.max_rows)
        except Exception as exc:
            print(f"\n  ❌  Could not read CSV: {exc}\n")
            sys.exit(1)

        log.info("Raw rows loaded: %d", len(df))

        # ── Validate required columns ─────────────────────────────
        missing = []
        for col in [self.text_col, self.score_col]:
            if col not in df.columns:
                missing.append(col)
        if missing:
            print(f"\n  ❌  Missing columns in CSV: {missing}")
            print(f"  Columns found: {list(df.columns)}\n")
            sys.exit(1)

        # ── Build clean DataFrame ─────────────────────────────────
        result = pd.DataFrame()
        result["text"]    = df[self.text_col].astype(str)
        result["summary"] = df[self.summary_col].astype(str) \
                            if self.summary_col in df.columns else ""
        result["score"]   = pd.to_numeric(df[self.score_col], errors="coerce")
        result["label"]   = result["score"].apply(self.score_to_label)

        # ── Drop rows with no text ────────────────────────────────
        result = result[result["text"].str.strip() != ""].reset_index(drop=True)

        # ── Print label distribution ──────────────────────────────
        dist = result["label"].value_counts()
        log.info(
            "Loaded %d reviews  |  Positive: %d  Neutral: %d  Negative: %d",
            len(result),
            dist.get("Positive", 0),
            dist.get("Neutral",  0),
            dist.get("Negative", 0),
        )
        return result


# ══════════════════════════════════════════════════════════════════
# 2.  TextCleaner
# ══════════════════════════════════════════════════════════════════
class TextCleaner:
    """
    Clean raw review text step-by-step:

        1. Lowercase everything
        2. Remove URLs (http://... or www....)
        3. Remove HTML tags (<br>, <p>, etc.)
        4. Remove punctuation, numbers, special characters
        5. Tokenize into individual words
        6. Remove common stopwords (the, is, at, ...)
        7. Lemmatize — reduce words to root form
           (running → run, better → good)
        8. Rejoin tokens into one clean string
    """

    def __init__(self):
        self.stop_words  = set(stopwords.words("english"))
        self.lemmatizer  = WordNetLemmatizer()
        # Pre-compile regex patterns for speed
        self._url_re     = re.compile(r"https?://\S+|www\.\S+")
        self._html_re    = re.compile(r"<[^>]+>")
        self._non_alpha  = re.compile(r"[^a-z\s]")

    def clean(self, text: str) -> str:
        """
        Run the full 8-step cleaning pipeline on one piece of text.

        Example:
            Input : "This product is AMAZING!!! https://amzn.com"
            Output: "product amaz"
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Step 1 — lowercase
        text = text.lower()

        # Step 2 — remove URLs
        text = self._url_re.sub(" ", text)

        # Step 3 — remove HTML tags
        text = self._html_re.sub(" ", text)

        # Step 4 — keep only letters and spaces
        text = self._non_alpha.sub(" ", text)

        # Step 5 — tokenize into word list
        tokens = word_tokenize(text)

        # Step 6 & 7 — remove stopwords + lemmatize
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]

        # Step 8 — rejoin into a single string
        return " ".join(tokens)

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Apply clean() to an entire pandas column efficiently."""
        log.info("Cleaning %d reviews …", len(series))
        cleaned = series.apply(self.clean)
        log.info("Text cleaning complete.")
        return cleaned


# ══════════════════════════════════════════════════════════════════
# 3.  Vectorizer
# ══════════════════════════════════════════════════════════════════
class Vectorizer:
    """
    Convert cleaned text into TF-IDF numerical feature vectors.

    TF-IDF = Term Frequency × Inverse Document Frequency
    It scores how important each word is across all reviews.
    Common words like "the" get low scores; unique words get high scores.

    We use both unigrams (single words) AND bigrams (word pairs):
        "dog food", "great taste", "bad quality" → bigrams
    """

    def __init__(self, max_features: int = 15_000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,       # smooth large term frequencies
            min_df=2,                # ignore words appearing in only 1 doc
        )

    def fit_transform(self, texts: pd.Series):
        """Fit the vectorizer on training text and convert to matrix."""
        log.info("Building TF-IDF vocabulary (max_features=%d) …", self.vectorizer.max_features)
        X = self.vectorizer.fit_transform(texts)
        log.info("Vocabulary size: %d unique terms", len(self.vectorizer.vocabulary_))
        return X

    def transform(self, texts: pd.Series):
        """Convert new texts using the already-fitted vocabulary."""
        return self.vectorizer.transform(texts)

    def save(self, path: str = "vectorizer.pkl") -> None:
        """Save vectorizer to disk so we don't need to re-fit every run."""
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        log.info("Vectorizer saved → %s", path)

    @staticmethod
    def load(path: str = "vectorizer.pkl") -> TfidfVectorizer:
        """Load a previously saved vectorizer."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════
# 4.  SentimentModel
# ══════════════════════════════════════════════════════════════════
class SentimentModel:
    """
    Logistic Regression classifier — predicts Positive / Negative / Neutral.

    Why Logistic Regression?
    - Fast to train (even on 500k+ reviews)
    - Works extremely well with TF-IDF text features
    - Easy to understand and explain
    - Returns class probabilities (confidence scores)
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        """
        Args:
            C        : Regularization. Lower = simpler model (less overfitting).
            max_iter : How many steps the solver takes to converge.
        """
        self.model   = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver = "lbfgs"
        )
        self.encoder = LabelEncoder()    # maps string labels ↔ integers

    def fit(self, X, y: pd.Series) -> None:
        """Train the classifier. X = TF-IDF matrix, y = label series."""
        y_enc = self.encoder.fit_transform(y)
        log.info("Training Logistic Regression on %d samples …", X.shape[0])
        self.model.fit(X, y_enc)
        log.info("Training complete.  Classes: %s", list(self.encoder.classes_))

    def predict(self, X) -> np.ndarray:
        """Return predicted labels as strings (Positive/Negative/Neutral)."""
        return self.encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, X) -> np.ndarray:
        """Return probability scores for each class (shape: n × 3)."""
        return self.model.predict_proba(X)

    def save(self, path: str = "sentiment_model.pkl") -> None:
        """Save model + encoder bundle to disk."""
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "encoder": self.encoder}, f)
        log.info("Model saved → %s", path)

    @staticmethod
    def load(path: str = "sentiment_model.pkl") -> "SentimentModel":
        """Load a previously saved model from disk."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        obj = SentimentModel.__new__(SentimentModel)
        obj.model   = bundle["model"]
        obj.encoder = bundle["encoder"]
        log.info("Model loaded ← %s", path)
        return obj


# ══════════════════════════════════════════════════════════════════
# 5.  Evaluator
# ══════════════════════════════════════════════════════════════════
class Evaluator:
    """Print accuracy metrics and confusion matrix to the console."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute and display:
            - Overall accuracy
            - Per-class precision, recall, F1-score
            - Confusion matrix
        """
        acc    = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred)

        print("\n" + "═" * 56)
        print("  📊  SentiStream AI — Model Evaluation")
        print("═" * 56)
        print(f"  Accuracy  :  {acc * 100:.2f}%")
        print("\n  Per-class Report:")
        for line in report.strip().split("\n"):
            print(f"  {line}")
        print(f"\n  Confusion Matrix:\n  {matrix}")
        print("═" * 56 + "\n")

        return {"accuracy": acc, "report": report, "matrix": matrix}


# ══════════════════════════════════════════════════════════════════
# 6.  Predictor
# ══════════════════════════════════════════════════════════════════
class Predictor:
    """
    Convenience wrapper for inference on new text.

    Usage:
        result = predictor.predict_text("This dog food is amazing!")
        # → { "label": "Positive", "probabilities": {...} }
    """

    def __init__(self, model: SentimentModel, vectorizer: TfidfVectorizer, cleaner: TextCleaner):
        self.model      = model
        self.vectorizer = vectorizer
        self.cleaner    = cleaner

    def predict_text(self, text: str) -> dict:
        """
        Predict sentiment for a single string.

        Returns:
            dict with keys: label, probabilities
        """
        cleaned = self.cleaner.clean(text)
        if not cleaned.strip():
            return {"label": "Neutral", "probabilities": {"Neutral": 100.0}}

        X      = self.vectorizer.transform([cleaned])
        label  = self.model.predict(X)[0]
        probs  = self.model.predict_proba(X)[0]
        classes = self.model.encoder.classes_

        return {
            "label": label,
            "probabilities": {
                c: round(float(p) * 100, 1)
                for c, p in zip(classes, probs)
            },
        }

    def predict_batch(self, texts: list) -> list:
        """Predict sentiment for a list of strings."""
        return [self.predict_text(t) for t in texts]


# ══════════════════════════════════════════════════════════════════
# 7.  ReviewSearchEngine
#     The core of the interactive Q&A mode.
#     Accepts a user keyword/question, searches matching reviews,
#     and prints a structured sentiment opinion summary.
# ══════════════════════════════════════════════════════════════════
class ReviewSearchEngine:
    """
    Search the Amazon reviews dataset for a keyword or phrase and
    produce a clear opinion summary:

        - Overall verdict (Mostly Positive / Mixed / Mostly Negative)
        - Positive % / Neutral % / Negative %
        - Common themes from positive reviews
        - Common themes from negative reviews
        - Sample customer quotes (3 per class)

    Example:
        engine.search_and_summarize("dog food")
    """

    def __init__(self, df: pd.DataFrame, predictor: Predictor):
        """
        Args:
            df        : The full reviews DataFrame (from DataLoader).
            predictor : Trained Predictor for re-running predictions.
        """
        self.df        = df
        self.predictor = predictor

        # Words to ignore when extracting "common themes"
        self._theme_stopwords = set(stopwords.words("english")) | {
            "product", "item", "thing", "buy", "bought", "purchase",
            "review", "star", "rating", "ordered", "amazon", "one",
            "use", "used", "using", "get", "got", "like", "really",
            "much", "also", "would", "could", "every", "even", "back",
            "still", "time", "day", "way", "first", "last", "great",
        }

    # ── Helpers ───────────────────────────────────────────────────

    def _search_reviews(self, query: str, max_results: int = 500) -> pd.DataFrame:
        """
        Return reviews whose Text or Summary contains the query keywords.

        Supports multi-word queries: all words must appear in the review.
        Case-insensitive.
        """
        keywords = query.lower().split()
        if not keywords:
            return pd.DataFrame()

        # Build a combined search field: text + summary
        combined = (self.df["text"].str.lower() + " " + self.df["summary"].str.lower())

        # All keywords must match
        mask = combined.str.contains(keywords[0], na=False)
        for kw in keywords[1:]:
            mask &= combined.str.contains(kw, na=False)

        matches = self.df[mask].copy()

        # Shuffle to avoid always showing first rows of the file
        if len(matches) > max_results:
            matches = matches.sample(n=max_results, random_state=42)

        return matches

    def _extract_keywords(self, texts: pd.Series, top_n: int = 8) -> list:
        """
        Extract the most common meaningful words from a set of reviews.
        Used to show "common themes" in positive/negative reviews.
        """
        all_words = []
        for text in texts:
            if not isinstance(text, str):
                continue
            # Simple clean: lowercase + only letters
            words = re.sub(r"[^a-z\s]", " ", text.lower()).split()
            for w in words:
                if len(w) > 3 and w not in self._theme_stopwords:
                    all_words.append(w)

        if not all_words:
            return []

        counter = Counter(all_words)
        return [word for word, _ in counter.most_common(top_n)]

    def _truncate(self, text: str, max_len: int = 120) -> str:
        """Shorten a review to a readable quote length."""
        if len(text) <= max_len:
            return text.strip()
        return text.strip()[:max_len].rsplit(" ", 1)[0] + "…"

    def _verdict(self, pos_pct: float, neg_pct: float) -> str:
        """
        Give an overall sentiment verdict based on percentages.

        Rules:
            ≥ 70% positive             → Very Positive
            ≥ 55% positive             → Mostly Positive
            ≥ 55% negative             → Mostly Negative
            ≥ 70% negative             → Very Negative
            Otherwise                  → Mixed
        """
        if pos_pct >= 70:
            return "💚  VERY POSITIVE"
        elif pos_pct >= 55:
            return "✅  MOSTLY POSITIVE"
        elif neg_pct >= 70:
            return "🔴  VERY NEGATIVE"
        elif neg_pct >= 55:
            return "❌  MOSTLY NEGATIVE"
        else:
            return "➖  MIXED / NEUTRAL"

    # ── Main public method ────────────────────────────────────────

    def search_and_summarize(self, query: str) -> None:
        """
        Run a full search + opinion summary for the given query.

        What this does:
            1. Find all reviews mentioning the query keywords
            2. Use the trained model to predict each review's sentiment
            3. Count Positive / Neutral / Negative
            4. Pick sample quotes for each group
            5. Extract common themes (keywords) from each group
            6. Print a clean, formatted summary

        Args:
            query : User's question or keyword, e.g. "dog food"
        """
        print(f"\n  🔍  Searching reviews for: \"{query}\"")
        print("  " + "─" * 50)

        # ── Step 1: find matching reviews ─────────────────────────
        matches = self._search_reviews(query)

        if matches.empty:
            print(f"\n  No reviews found matching \"{query}\".")
            print("  Try a shorter or different keyword.\n")
            return

        print(f"  Found {len(matches)} matching reviews. Analysing …\n")

        # ── Step 2: predict sentiment for each match ───────────────
        results = self.predictor.predict_batch(list(matches["text"]))
        matches = matches.copy()
        matches["predicted"] = [r["label"] for r in results]

        # ── Step 3: split into sentiment groups ────────────────────
        pos_df  = matches[matches["predicted"] == "Positive"]
        neg_df  = matches[matches["predicted"] == "Negative"]
        neu_df  = matches[matches["predicted"] == "Neutral"]

        total    = len(matches)
        pos_pct  = round(len(pos_df) / total * 100, 1)
        neg_pct  = round(len(neg_df) / total * 100, 1)
        neu_pct  = round(len(neu_df) / total * 100, 1)

        # ── Step 4: overall verdict ────────────────────────────────
        verdict = self._verdict(pos_pct, neg_pct)

        # ── Step 5: extract common themes ─────────────────────────
        pos_themes = self._extract_keywords(pos_df["text"]) if not pos_df.empty else []
        neg_themes = self._extract_keywords(neg_df["text"]) if not neg_df.empty else []

        # ── Step 6: pick sample quotes ────────────────────────────
        def sample_quotes(df: pd.DataFrame, n: int = 3) -> list:
            sample = df.head(n * 3).sample(frac=1, random_state=7)  # shuffle
            quotes = []
            for _, row in sample.iterrows():
                q = self._truncate(str(row["text"]))
                if q not in quotes:
                    quotes.append(q)
                if len(quotes) >= n:
                    break
            return quotes

        pos_quotes = sample_quotes(pos_df)
        neg_quotes = sample_quotes(neg_df)
        neu_quotes = sample_quotes(neu_df, n=2)

        # ── Step 7: print report ──────────────────────────────────
        print("═" * 56)
        print(f"  OVERALL CUSTOMER SENTIMENT")
        print(f"  {verdict}")
        print("═" * 56)

        print(f"\n  📊  Breakdown  ({total} reviews analysed)")
        print(f"  {'Positive':<12} {pos_pct:>5.1f}%   {'▓' * int(pos_pct / 4)}")
        print(f"  {'Neutral':<12} {neu_pct:>5.1f}%   {'▒' * int(neu_pct / 4)}")
        print(f"  {'Negative':<12} {neg_pct:>5.1f}%   {'░' * int(neg_pct / 4)}")

        if pos_themes:
            print(f"\n  👍  Common themes in positive reviews:")
            print(f"  {', '.join(pos_themes)}")

        if neg_themes:
            print(f"\n  👎  Common themes in negative reviews:")
            print(f"  {', '.join(neg_themes)}")

        if pos_quotes:
            print(f"\n  💬  Sample POSITIVE customer opinions:")
            for i, q in enumerate(pos_quotes, 1):
                wrapped = textwrap.fill(q, width=60, initial_indent="", subsequent_indent="      ")
                print(f"  {i}. \"{wrapped}\"")

        if neg_quotes:
            print(f"\n  💬  Sample NEGATIVE customer opinions:")
            for i, q in enumerate(neg_quotes, 1):
                wrapped = textwrap.fill(q, width=60, initial_indent="", subsequent_indent="      ")
                print(f"  {i}. \"{wrapped}\"")

        if neu_quotes:
            print(f"\n  💬  Sample NEUTRAL customer opinions:")
            for i, q in enumerate(neu_quotes, 1):
                wrapped = textwrap.fill(q, width=60, initial_indent="", subsequent_indent="      ")
                print(f"  {i}. \"{wrapped}\"")

        print("\n" + "═" * 56 + "\n")


# ══════════════════════════════════════════════════════════════════
# 8.  run_pipeline()
#     Orchestrates: Load → Clean → Vectorize → Train → Evaluate → Save
# ══════════════════════════════════════════════════════════════════
def run_pipeline(
    csv_path:    str   = "dataset.csv",
    text_col:    str   = "Text",
    score_col:   str   = "Score",
    summary_col: str   = "Summary",
    max_rows:    int   = None,
    test_size:   float = 0.20,
    model_path:  str   = "sentiment_model.pkl",
    vec_path:    str   = "vectorizer.pkl",
) -> tuple:
    """
    Full training pipeline — returns (predictor, df) tuple.

    Steps:
        1. NLTK setup
        2. Load CSV + map scores to labels
        3. Clean text
        4. Train/test split
        5. TF-IDF vectorization
        6. Logistic Regression training
        7. Evaluation
        8. Save model + vectorizer

    Returns:
        (Predictor, DataFrame) — ready for search + inference
    """
    # Step 1 — NLTK
    download_nltk_resources()

    # Step 2 — load data
    loader = DataLoader(
        csv_path=csv_path,
        text_col=text_col,
        score_col=score_col,
        summary_col=summary_col,
        max_rows=max_rows,
    )
    df = loader.load()

    # Step 3 — clean text
    cleaner     = TextCleaner()
    df["clean"] = cleaner.clean_series(df["text"])

    # Drop rows where cleaning left nothing
    df = df[df["clean"].str.strip() != ""].reset_index(drop=True)

    # Step 4 — train/test split
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df["clean"], df["label"],
            test_size=test_size,
            random_state=42,
            stratify=df["label"],  # keep class balance
        )
    except ValueError as exc:
        log.warning("Stratified split failed (%s) — retrying without stratify.", exc)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df["clean"], df["label"],
            test_size=test_size,
            random_state=42,
        )

    log.info("Train: %d samples | Test: %d samples", len(X_train_raw), len(X_test_raw))

    # Step 5 — TF-IDF
    vec_obj = Vectorizer()
    X_train = vec_obj.fit_transform(X_train_raw)
    X_test  = vec_obj.transform(X_test_raw)

    # Step 6 — train
    model = SentimentModel()
    model.fit(X_train, y_train)

    # Step 7 — evaluate
    y_pred    = model.predict(X_test)
    evaluator = Evaluator()
    evaluator.evaluate(y_test.values, y_pred)

    # Step 8 — save
    model.save(model_path)
    vec_obj.save(vec_path)

    # Build predictor
    predictor = Predictor(
        model=model,
        vectorizer=vec_obj.vectorizer,
        cleaner=cleaner,
    )

    return predictor, df


# ══════════════════════════════════════════════════════════════════
# 9.  Interactive Mode
#     Two sub-modes:
#       (a) Product search  — type a keyword to get opinion summary
#       (b) Predict mode    — type any review text to get a label
# ══════════════════════════════════════════════════════════════════
def interactive_mode(predictor: Predictor, df: pd.DataFrame) -> None:
    """
    Console Q&A loop with two modes:

    Mode 1 — SEARCH  (default)
        User types a product keyword or question.
        System searches matching reviews and returns opinion summary.

        Example inputs:
            dog food
            cat treats
            coffee
            what do people think about chips

    Mode 2 — PREDICT
        User types a full review sentence.
        System returns predicted sentiment + confidence.

        Example inputs:
            predict: This coffee tastes amazing and smells great!

    Commands:
        search <keyword>  — search reviews (default mode)
        predict: <text>   — predict a single text
        help              — show help
        quit              — exit
    """
    engine = ReviewSearchEngine(df=df, predictor=predictor)

    print("\n" + "═" * 56)
    print("  🤖  SentiStream AI — Interactive Mode")
    print("═" * 56)
    print("""
  How to use:
  ───────────────────────────────────────────────────
  Type a product keyword to see what customers think:
    → dog food
    → coffee beans
    → vitamin supplement
    → cat treats

  Or type 'predict:' followed by a review sentence:
    → predict: This product is absolutely amazing!

  Other commands:  help  |  quit
  ───────────────────────────────────────────────────
""")

    emoji_map = {"Positive": "✅", "Negative": "❌", "Neutral": "➖"}

    while True:
        try:
            user_input = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye! 👋\n")
            break

        # ── Empty input ───────────────────────────────────────────
        if not user_input:
            continue

        # ── Quit ──────────────────────────────────────────────────
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! 👋\n")
            break

        # ── Help ──────────────────────────────────────────────────
        if user_input.lower() in ("help", "?"):
            print("""
  Commands:
    <keyword>          Search reviews for this product/topic
    predict: <text>    Predict sentiment for a single review
    quit               Exit the program
""")
            continue

        # ── Predict mode ──────────────────────────────────────────
        if user_input.lower().startswith("predict:"):
            text   = user_input[len("predict:"):].strip()
            if not text:
                print("  Please provide text after 'predict:'\n")
                continue
            result = predictor.predict_text(text)
            emoji  = emoji_map.get(result["label"], "❓")
            print(f"\n  Result  : {emoji}  {result['label']}")
            print(f"  Confidence: {result['probabilities']}\n")
            continue

        # ── Natural language question handling ────────────────────
        # Strip question prefixes so "what do people think about X"
        # becomes a clean keyword search for X.
        query = user_input
        for prefix in [
            "what do people think about",
            "what do customers think about",
            "what is the opinion on",
            "tell me about",
            "show reviews for",
            "search for",
            "search",
            "find reviews about",
            "how is",
            "how are",
        ]:
            if query.lower().startswith(prefix):
                query = query[len(prefix):].strip(" ?!")
                break

        # ── Search mode ───────────────────────────────────────────
        engine.search_and_summarize(query)


# ══════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           SentiStream AI Prototype  v3.0                         ║
║     Amazon Reviews · Score-based Sentiment Analysis              ║
║     Search · Predict · Summarize Customer Opinions               ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # ──────────────────────────────────────────────
    # CONFIGURATION — edit these if needed
    # ──────────────────────────────────────────────
    #
    # csv_path  : Your CSV file name. Default: "dataset.csv"
    #             Place it in the same folder as this .py file.
    #
    # max_rows  : How many rows to load. None = load everything.
    #             Set to 50_000 for faster testing on large files:
    #               max_rows = 50_000
    #
    # ──────────────────────────────────────────────
    predictor, df = run_pipeline(
        csv_path    = "dataset.csv",   # ← your CSV file name
        text_col    = "Text",          # ← column with review text
        score_col   = "Score",         # ← column with 1-5 star ratings
        summary_col = "Summary",       # ← column with review titles
        max_rows    = None,            # ← None = load all rows
    )

    # Start the interactive Q&A console
    interactive_mode(predictor, df)
