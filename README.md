# SentiStream AI — Real-Time Reddit Sentiment Backend

> Fetch live Reddit posts → clean text → predict sentiment → aggregate opinions

---

## Project structure

```
SentiStream_Backend/
│
├── backend/                    ← all logic lives here
│   ├── __init__.py
│   ├── config.py               ← settings + credential loader
│   ├── cleaner.py              ← NLTK text cleaning pipeline
│   ├── model_loader.py         ← loads .pkl files from disk
│   ├── predictor.py            ← TF-IDF + Logistic Regression inference
│   ├── reddit_fetcher.py       ← PRAW Reddit API + MockFetcher
│   ├── aggregator.py           ← percentages, keywords, opinions
│   └── analyzer.py             ← main orchestrator (call this)
│
├── train_model.py              ← Step 1: train on your CSV
├── main.py                     ← Step 2: run interactive console
├── streamlit_app.py            ← Step 3 (optional): Streamlit UI stub
│
├── .env.example                ← copy to .env and fill in API keys
├── requirements.txt
└── README.md
```

Place your Amazon reviews CSV here and rename it `dataset.csv`.

---

## Quick start (3 steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train_model.py
```
This reads `dataset.csv`, trains the classifier, and saves:
- `sentiment_model.pkl`
- `vectorizer.pkl`

For large datasets, limit rows for faster testing:
```bash
python train_model.py --rows 100000
```

### Step 3 — Set up Reddit credentials (5 minutes, free)

1. Go to **https://www.reddit.com/prefs/apps**
2. Click **"create another app"**
3. Fill in:
   - Name: `SentiStreamAI`
   - Type: **script**
   - Redirect URI: `http://localhost:8080`
4. Click **"create app"**
5. Copy the values:
   - `client_id` → the short string under your app name
   - `client_secret` → the "secret" field

6. Copy `.env.example` → `.env` and fill in:
```
REDDIT_CLIENT_ID=paste_your_client_id
REDDIT_CLIENT_SECRET=paste_your_client_secret
REDDIT_USER_AGENT=SentiStreamAI/1.0 by YourUsername
```

### Step 4 — Run
```bash
python main.py
```

No Reddit credentials yet? Run with mock data:
```bash
python main.py --mock
```

---

## Using the console

```
Topic: electric vehicles
→ Fetches 50 Reddit posts, analyses sentiment, prints report

Topic: coffee beans
→ Overall verdict, Positive/Neutral/Negative %, keywords, sample opinions

Topic: predict: This product completely changed my life!
→ Quick single-text prediction (no Reddit fetch)

Topic: help   → show instructions
Topic: quit   → exit
```

---

## Using the backend in your own code

```python
from backend.analyzer import SentimentAnalyzer

# Create analyzer (use_mock=True skips Reddit API)
analyzer = SentimentAnalyzer(use_mock=False)

# Analyse a topic
report = analyzer.analyze("oat milk")

print(report["verdict"])          # "Mostly Positive"
print(report["percentages"])      # {"Positive": 68.2, "Negative": 14.1, "Neutral": 17.7}
print(report["all_keywords"])     # ["taste", "brand", "price", "creamy", ...]
print(report["top_opinions"])     # {"Positive": [...], "Negative": [...], ...}

# Single-text prediction (no Reddit fetch)
result = analyzer.predict_single("This is amazing!")
print(result["label"])            # "Positive"
print(result["probabilities"])    # {"Positive": 92.3, "Negative": 4.1, "Neutral": 3.6}
```

---

## Streamlit frontend (optional)

The `streamlit_app.py` file contains a ready-to-uncomment Streamlit UI.
To activate it:
```bash
pip install streamlit
# uncomment the code in streamlit_app.py
streamlit run streamlit_app.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `sentiment_model.pkl not found` | Run `python train_model.py` first |
| `dataset.csv not found` | Place your CSV in this folder |
| `praw not installed` | Run `pip install -r requirements.txt` |
| Reddit auth error | Check `.env` credentials match your Reddit app |
| No results returned | Try a shorter/more common keyword |
| Slow on large CSV | Add `--rows 50000` to `train_model.py` |
