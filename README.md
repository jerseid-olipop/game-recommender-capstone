# 🎮 Game Recommender — ML Capstone Project

**Live Demo:** https://game-recommender-capstone.web.app

A production-grade machine learning application that recommends video games based on cosine similarity. Built as the Step 11 deployment capstone for the UMass Amherst / SpringBoard ML Engineering & AI Bootcamp (2026).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                         │
│              https://game-recommender-capstone.web.app      │
└─────────────────────────┬───────────────────────────────────┘
                          │  HTTP POST (JSON)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           Firebase Cloud Function (Python 3.11)             │
│           recommend()  ·  us-central1  ·  2nd Gen           │
│                                                             │
│  1. Load model artifacts from Cloud Storage (lazy/cached)   │
│  2. Encode query → scaled feature vector                    │
│  3. NearestNeighbors cosine similarity search               │
│  4. Log request to Firestore                                │
│  5. Return top-K recommendations as JSON                    │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐    ┌─────────────────────────────────┐
│   Cloud Storage      │    │         Firestore               │
│   models/v1/         │    │   recommendation_logs           │
│   ├── scaler.pkl     │    │   (audit trail + latency data)  │
│   ├── nn_model.pkl   │    └─────────────────────────────────┘
│   ├── games_df.pkl   │
│   └── feature_       │
│       columns.pkl    │
└──────────────────────┘
```

---

## ML Models

| Component | Algorithm | Notes |
|-----------|-----------|-------|
| Feature encoding | `StandardScaler` | Ordinal, binary, one-hot encoding of 26 features |
| Recommendation | `NearestNeighbors` (cosine, brute) | k=11, on-demand similarity search |
| Dataset | [Video Game Reviews & Ratings](https://www.kaggle.com/datasets/jahnavipaliwal/video-game-reviews-and-ratings) | 47,774 games, 18 features |

### Features Used
- Numeric: User Rating, Price, Release Year, Game Length (Hours), Min Number of Players
- Ordinal: Graphics Quality, Soundtrack Quality, Story Quality, Age Group Targeted
- Binary: Multiplayer, Game Mode
- One-hot: Genre (10 categories), Platform (5 categories)

---

## Scaling Analysis (Step 8)

Benchmarked three model pairs at 47K → 1M rows:

| Model Pair | Winner at Scale | Quality Trade-off |
|-----------|----------------|-------------------|
| KMeans vs MiniBatchKMeans | MiniBatchKMeans (10× faster) | Negligible |
| Full cosine matrix vs On-demand | On-demand (O(n) memory) | None |
| LogisticRegression vs SGDClassifier | SGDClassifier (2× faster) | ~0.1% F1 |
| SGDClassifier vs Keras NN | SGDClassifier (40× faster) | Keras ~0.3% better F1 |

**Production decision:** On-demand cosine similarity with `NearestNeighbors` — identical results to full matrix, scales to any dataset size.

---

## API Reference

**Endpoint:** `https://recommend-oylnz64ghq-uc.a.run.app`

### Search by Game Title

```bash
curl -X POST https://recommend-oylnz64ghq-uc.a.run.app \
  -H "Content-Type: application/json" \
  -d '{
    "game_title": "Grand Theft Auto V",
    "top_k": 5
  }'
```

### Search by Features

```bash
curl -X POST https://recommend-oylnz64ghq-uc.a.run.app \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "Genre": "Action",
      "Platform": "PC",
      "User Rating": 45,
      "Price": 59.99,
      "Game Length (Hours)": 50,
      "Multiplayer": "Yes",
      "Graphics Quality": "Ultra",
      "Age Group Targeted": "Adults"
    },
    "top_k": 10
  }'
```

### Response

```json
{
  "seed_game": "Grand Theft Auto V",
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "game_title": "Red Dead Redemption 2",
      "similarity_score": 0.987432,
      "genre": "Action",
      "platform": "PlayStation",
      "user_rating": 48.2,
      "price": 59.99,
      "game_length_hours": 60.0,
      "multiplayer": "Yes",
      "age_group_targeted": "Adults"
    }
  ],
  "latency_ms": 142.3
}
```

### Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Missing or invalid request body |
| 404 | Game title not found in dataset |
| 503 | Model artifacts unavailable (retry) |

---

## Project Structure

```
capstone/
├── firebase.json                 # Firebase project configuration
├── firestore.rules               # Firestore security rules
├── firestore.indexes.json        # Firestore index definitions
├── functions/
│   ├── main.py                   # Cloud Function — recommendation engine
│   └── requirements.txt          # Python dependencies
└── public/
    └── index.html                # Web UI (hosted on Firebase Hosting)
```

---

## Local Development

### Prerequisites
- Python 3.11
- Node.js (for Firebase CLI)
- Firebase CLI: `npm install -g firebase-tools`

### Setup

```bash
# Clone the repo
git clone https://github.com/jerseid-olipop/game-recommender-capstone
cd game-recommender-capstone

# Log in to Firebase
firebase login

# Set up the Python venv
cd functions
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..
```

### Deploy

```bash
# Deploy everything
firebase deploy --only functions,hosting

# Deploy only the function
firebase deploy --only functions

# Deploy only the UI
firebase deploy --only hosting
```

### Model Artifacts

The pkl files are stored in Firebase Cloud Storage under `models/v1/`. To regenerate them, run the serialization cell at the bottom of `Game Recommender.ipynb` in Google Colab, then upload the output files to Cloud Storage:

| File | Description |
|------|-------------|
| `scaler.pkl` | Fitted `StandardScaler` |
| `nn_model.pkl` | Fitted `NearestNeighbors` (cosine, k=11) |
| `games_df.pkl` | Original DataFrame with all game metadata |
| `feature_columns.pkl` | Ordered list of 26 feature column names |

---

## Monitoring

Every API request is logged to Firestore (`recommendation_logs` collection) with:
- Query parameters
- Recommended game titles
- Similarity scores
- Response latency (ms)
- Server timestamp

View logs at: [Firebase Console → Firestore](https://console.firebase.google.com/project/game-recommender-capstone/firestore)

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `Game Recommender.ipynb` | Main model — data cleaning, feature engineering, NearestNeighbors, serialization |
| `Scale_Your_Prototype_with_Large_Scale_Data.ipynb` | Step 8 — scaling benchmarks at 47K → 1M rows |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn (StandardScaler, NearestNeighbors) |
| Backend | Firebase Cloud Functions (Python 3.11, 2nd Gen) |
| Storage | Firebase Cloud Storage |
| Database | Firebase Firestore |
| Frontend | Firebase Hosting (vanilla HTML/CSS/JS) |
| Dataset | Kaggle — Video Game Reviews & Ratings |
