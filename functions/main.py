"""
Game Recommender - Firebase Cloud Function v2
Serves cosine-similarity-based game recommendations via HTTP.

Artifacts loaded from Cloud Storage:
  - models/scaler.pkl          : StandardScaler fitted on training data
  - models/nn_model.pkl        : NearestNeighbors (cosine, brute, k=11)
  - models/games_df.pkl        : Original DataFrame with Game Title etc.
  - models/feature_columns.pkl : Ordered list of feature column names
"""

import os
import time
import logging
import json

from firebase_functions import https_fn
from firebase_admin import initialize_app

# ── Firebase init ─────────────────────────────────────────────────────────────
initialize_app()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Lazy-loaded model cache (persists between warm invocations) ───────────────
_MODELS = {}

BUCKET_NAME = "game-recommender-capstone.firebasestorage.app"
MODEL_PATH   = "models/v1"

ORDINAL_MAPS = {
    "Graphics Quality":    {"Low": 0, "Medium": 1, "High": 2, "Ultra": 3},
    "Soundtrack Quality":  {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3},
    "Story Quality":       {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3},
    "Age Group Targeted":  {"Kids": 0, "Teens": 1, "All Ages": 2, "Adults": 3},
}
BINARY_MAPS = {
    "Multiplayer": {"No": 0, "Yes": 1},
    "Game Mode":   {"Offline": 0, "Online": 1},
}
ONE_HOT_PREFIXES = ["Genre", "Platform"]
DROP_COLS = ["Game Title", "Developer", "User Review Text",
             "Publisher", "Requires Special Device"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_artifact(filename: str, local_path: str):
    """Download a single pkl file from Cloud Storage if not cached locally."""
    from firebase_admin import storage  # lazy import
    if not os.path.exists(local_path):
        bucket = storage.bucket(BUCKET_NAME)
        blob   = bucket.blob(f"{MODEL_PATH}/{filename}")
        blob.download_to_filename(local_path)
        logger.info("Downloaded %s → %s", filename, local_path)


def _load_models():
    """Load all model artifacts (cached after first call)."""
    global _MODELS
    if _MODELS:
        return _MODELS

    import joblib  # lazy import — keeps startup fast

    tmp = "/tmp/models"
    os.makedirs(tmp, exist_ok=True)

    artifacts = {
        "scaler":          "scaler.pkl",
        "nn_model":        "nn_model.pkl",
        "games_df":        "games_df.pkl",
        "feature_columns": "feature_columns.pkl",
    }

    for key, filename in artifacts.items():
        local = os.path.join(tmp, filename)
        _download_artifact(filename, local)
        _MODELS[key] = joblib.load(local)
        logger.info("Loaded %s", key)

    return _MODELS


def _encode_query(query_game: dict, feature_columns: list, scaler) -> "np.ndarray":
    """
    Encode a single query dict into a scaled feature vector.
    """
    import numpy as np  # lazy import
    row = {}

    # Ordinal
    for col, mapping in ORDINAL_MAPS.items():
        val = query_game.get(col, list(mapping.keys())[len(mapping) // 2])
        row[col] = mapping.get(val, list(mapping.values())[len(mapping) // 2])

    # Binary
    for col, mapping in BINARY_MAPS.items():
        val = query_game.get(col, "No")
        row[col] = mapping.get(val, 0)

    # Numeric pass-through
    row["User Rating"]           = float(query_game.get("User Rating", 3.0))
    row["Price"]                 = float(query_game.get("Price", 39.99))
    row["Release Year"]          = float(query_game.get("Release Year", 2016))
    row["Game Length (Hours)"]   = float(query_game.get("Game Length (Hours)", 30))
    row["Min Number of Players"] = float(query_game.get("Min Number of Players", 1))

    # One-hot: Genre
    genre_val = query_game.get("Genre", "Action")
    for col in feature_columns:
        if col.startswith("Genre_"):
            row[col] = 1 if col == f"Genre_{genre_val}" else 0

    # One-hot: Platform
    platform_val = query_game.get("Platform", "PC")
    for col in feature_columns:
        if col.startswith("Platform_"):
            row[col] = 1 if col == f"Platform_{platform_val}" else 0

    # Build vector in correct column order
    vec = np.array([[row.get(c, 0) for c in feature_columns]], dtype=float)
    vec_scaled = scaler.transform(vec)
    return vec_scaled


def _log_request(query: dict, indices: list, scores: list, latency_ms: float):
    """Write an audit record to Firestore."""
    try:
        from firebase_admin import firestore  # lazy import
        db = firestore.client()
        db.collection("recommendation_logs").add({
            "query":      query,
            "result_idx": indices,
            "scores":     scores,
            "latency_ms": latency_ms,
            "timestamp":  firestore.SERVER_TIMESTAMP,
        })
    except Exception as exc:
        logger.warning("Firestore log failed: %s", exc)


# ── Cloud Function ────────────────────────────────────────────────────────────

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "3600",
}


def _cors(response: https_fn.Response) -> https_fn.Response:
    for k, v in CORS_HEADERS.items():
        response.headers[k] = v
    return response


@https_fn.on_request()
def recommend(req: https_fn.Request) -> https_fn.Response:
    """
    HTTP endpoint — returns top-N game recommendations.

    Accepts POST with JSON body:
      {
        "game_title":  "Grand Theft Auto V",   // optional: lookup by title
        "query":       { ... feature dict ... }, // optional: query by features
        "top_k":       10                        // optional, default 10
      }

    Either game_title OR query must be provided.
    If game_title is given it looks up that game's features automatically.
    """
    # ── CORS preflight ────────────────────────────────────────────────────────
    if req.method == "OPTIONS":
        return _cors(https_fn.Response("", status=204))

    t_start = time.time()

    # ── Parse request ─────────────────────────────────────────────────────────
    try:
        body   = req.get_json(silent=True) or {}
        top_k  = int(body.get("top_k", 10))
        top_k  = min(max(top_k, 1), 50)           # clamp 1-50
    except Exception as exc:
        return _cors(https_fn.Response(
            json.dumps({"error": f"Invalid request body: {exc}"}),
            status=400, mimetype="application/json"
        ))

    # ── Load models ───────────────────────────────────────────────────────────
    try:
        models          = _load_models()
        scaler          = models["scaler"]
        nn_model        = models["nn_model"]
        games_df        = models["games_df"]
        feature_columns = models["feature_columns"]
    except Exception as exc:
        logger.error("Model load failed: %s", exc, exc_info=True)
        return _cors(https_fn.Response(
            json.dumps({"error": "Model unavailable. Please try again shortly."}),
            status=503, mimetype="application/json"
        ))

    # ── Resolve query vector ──────────────────────────────────────────────────
    game_title = body.get("game_title", "").strip()
    query_dict = body.get("query", {})

    if not game_title and not query_dict:
        return _cors(https_fn.Response(
            json.dumps({"error": "Provide game_title or query."}),
            status=400, mimetype="application/json"
        ))

    seed_title = None
    if game_title:
        match = games_df[games_df["Game Title"].str.lower() == game_title.lower()]
        if match.empty:
            # Fuzzy fallback: partial match
            match = games_df[
                games_df["Game Title"].str.lower().str.contains(game_title.lower(), na=False)
            ]
        if match.empty:
            return _cors(https_fn.Response(
                json.dumps({"error": f"Game '{game_title}' not found in dataset."}),
                status=404, mimetype="application/json"
            ))
        seed_row   = match.iloc[0]
        seed_title = seed_row["Game Title"]
        query_dict = seed_row.to_dict()

    try:
        vec = _encode_query(query_dict, feature_columns, scaler)
    except Exception as exc:
        logger.error("Encoding failed: %s", exc, exc_info=True)
        return _cors(https_fn.Response(
            json.dumps({"error": f"Feature encoding error: {exc}"}),
            status=400, mimetype="application/json"
        ))

    # ── Run NearestNeighbors ──────────────────────────────────────────────────
    n_neighbors = top_k + (1 if seed_title else 0)   # +1 to exclude seed itself
    n_neighbors = min(n_neighbors, len(games_df))

    distances, indices = nn_model.kneighbors(vec, n_neighbors=n_neighbors)
    distances = distances[0].tolist()
    indices   = indices[0].tolist()

    # Similarity score = 1 - cosine distance
    scores = [round(1 - d, 6) for d in distances]

    # Build result list, skipping the seed game
    results = []
    for idx, score in zip(indices, scores):
        row  = games_df.iloc[idx]
        title = row["Game Title"]
        if seed_title and title == seed_title:
            continue
        results.append({
            "rank":                  len(results) + 1,
            "game_title":            title,
            "similarity_score":      score,
            "genre":                 row.get("Genre", ""),
            "platform":              row.get("Platform", ""),
            "user_rating":           round(float(row.get("User Rating", 0)), 3),
            "price":                 round(float(row.get("Price", 0)), 2),
            "game_length_hours":     float(row.get("Game Length (Hours)", 0)),
            "multiplayer":           row.get("Multiplayer", ""),
            "age_group_targeted":    row.get("Age Group Targeted", ""),
        })
        if len(results) >= top_k:
            break

    latency_ms = round((time.time() - t_start) * 1000, 1)

    # ── Audit log ─────────────────────────────────────────────────────────────
    _log_request(
        query={"game_title": seed_title or None, "features": query_dict},
        indices=[r["game_title"] for r in results],
        scores=[r["similarity_score"] for r in results],
        latency_ms=latency_ms,
    )

    # ── Response ──────────────────────────────────────────────────────────────
    payload = {
        "seed_game": seed_title,
        "top_k":     top_k,
        "results":   results,
        "latency_ms": latency_ms,
    }
    return _cors(https_fn.Response(
        json.dumps(payload, indent=2),
        status=200, mimetype="application/json"
    ))
