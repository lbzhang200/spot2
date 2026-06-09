from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Features derived purely from track metadata (no audio features / genres needed)
FEATURE_COLS = [
    "rank_norm",          # 0=top ranked, 1=bottom of list
    "release_year_norm",  # 0=old, 1=new (normalised 2000–2026)
    "duration_norm",      # 0=short, 1=long
    "is_explicit",        # 0 or 1
    "is_single",          # 0 or 1
    "artist_top_rank",    # 1=top artist, 0=not in top artists list
    "name_energy",        # 0–1 rough energy inferred from track/artist name
]

# Keywords for rough text-based energy scoring
_ENERGY_HIGH = re.compile(
    r"\b(fire|wild|hype|party|crazy|rage|live|dance|night|bang|loud|energy|banger|"
    r"lit|turn up|go|fast|hard|heavy|electric|power|rush|storm|beast|bounce|jump)\b",
    re.IGNORECASE,
)
_ENERGY_LOW = re.compile(
    r"\b(quiet|slow|peace|calm|rest|sleep|dream|soft|chill|ease|gentle|still|"
    r"night|sad|alone|lost|miss|cry|broken|tender|mellow|acoustic|breathe|wait)\b",
    re.IGNORECASE,
)

# Each label maps to the feature dimension it "wins" on (highest or lowest centroid value).
# direction: "high" = assign to cluster with highest centroid on that feature
#            "low"  = assign to cluster with lowest centroid on that feature
MOOD_LABEL_DIMENSIONS: list[tuple[str, str, str]] = [
    ("Your Anthems",   "rank_norm",       "low"),   # lowest rank_norm = most listened
    ("Artist Staples", "artist_top_rank", "high"),  # highest top-artist affinity
    ("Long Jams",      "duration_norm",   "high"),  # longest tracks
    ("Raw & Real",     "is_explicit",     "high"),  # most explicit content
    ("Hit Singles",    "is_single",       "high"),  # most singles
]

MOOD_EMOJI = {
    "Your Anthems":   "🏆",
    "Artist Staples": "🎤",
    "Long Jams":      "🎸",
    "Raw & Real":     "🔥",
    "Hit Singles":    "🎯",
    "Cluster":        "🎵",  # fallback prefix
}


def _name_energy(name: str) -> float:
    high = len(_ENERGY_HIGH.findall(name))
    low  = len(_ENERGY_LOW.findall(name))
    score = 0.5 + 0.15 * high - 0.15 * low
    return float(max(0.0, min(1.0, score)))


def _label_cluster(centroid: dict) -> str:
    for mood, rules in MOOD_RULES:
        if all(lo <= centroid.get(feat, 0) <= hi for feat, (lo, hi) in rules.items()):
            return mood
    return "Wild Cards"


def build_track_dataframe(
    tracks: list[dict],
    top_artist_ids: set[str],
    rank_offset: int = 0,
) -> pd.DataFrame:
    """
    Build feature dataframe from track metadata alone.
    top_artist_ids: set of artist IDs that appear in the user's top artists list.
    """
    now_year = 2026
    rows = []
    n = len(tracks)
    for idx, t in enumerate(tracks):
        tid = t.get("id")
        if not tid:
            continue

        # Release year
        rd = (t.get("album") or {}).get("release_date", "2010")
        try:
            year = int(str(rd)[:4])
        except (ValueError, TypeError):
            year = 2010
        release_year_norm = max(0.0, min(1.0, (year - 2000) / (now_year - 2000)))

        # Duration (normalise 60s – 600s)
        duration_s = t.get("duration_ms", 210_000) / 1000
        duration_norm = max(0.0, min(1.0, (duration_s - 60) / 540))

        # Artist in top-artists?
        track_artist_ids = {a["id"] for a in t.get("artists", []) if a.get("id")}
        overlap = track_artist_ids & top_artist_ids
        artist_top_rank = 1.0 if overlap else 0.0

        # is_single
        album_type = (t.get("album") or {}).get("album_type", "album")
        is_single = 1.0 if album_type == "single" else 0.0

        name_energy = _name_energy(t.get("name", "") + " " + " ".join(a["name"] for a in t.get("artists", [])))

        img_url = ((t.get("album") or {}).get("images") or [{}])[0].get("url", "")

        rows.append({
            "id": tid,
            "name": t.get("name", ""),
            "artist": ", ".join(a["name"] for a in t.get("artists", [])),
            "album": (t.get("album") or {}).get("name", ""),
            "release_year": year,
            "image_url": img_url,
            "uri": t.get("uri", ""),
            "external_url": t.get("external_urls", {}).get("spotify", ""),
            "rank_norm": (idx + rank_offset) / max(n - 1, 1),
            "release_year_norm": release_year_norm,
            "duration_norm": duration_norm,
            "is_explicit": float(bool(t.get("explicit"))),
            "is_single": is_single,
            "artist_top_rank": artist_top_rank,
            "name_energy": name_energy,
        })
    return pd.DataFrame(rows)


def cluster_tracks(df: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, list[dict], list[str]]:
    if df.empty or len(df) < k:
        k = max(1, len(df))

    X = df[FEATURE_COLS].copy().fillna(0.5)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=30, random_state=42)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)

    # Build centroid dicts
    centroids: list[dict] = []
    for i in range(k):
        rows = df[df["cluster"] == i]
        centroids.append(rows[FEATURE_COLS].mean().to_dict())

    # Assign labels relatively: each label goes to the cluster that best fits its dimension.
    # Greedy assignment in priority order; no cluster gets two labels.
    labels: list[str] = [f"Cluster {i+1}" for i in range(k)]
    assigned_clusters: set[int] = set()

    for mood_name, feat, direction in MOOD_LABEL_DIMENSIONS:
        if len(assigned_clusters) == k:
            break
        vals = [
            (centroids[i].get(feat, 0), i)
            for i in range(k)
            if i not in assigned_clusters
        ]
        if not vals:
            break
        if direction == "high":
            best_idx = max(vals, key=lambda x: x[0])[1]
        else:
            best_idx = min(vals, key=lambda x: x[0])[1]
        labels[best_idx] = mood_name
        assigned_clusters.add(best_idx)

    df["mood"] = df["cluster"].map(lambda i: labels[i])
    return df, centroids, labels


def aggregate_artists(tracks: list[dict], top_artist_ids: set[str]) -> pd.DataFrame:
    """Count how many top tracks each artist appears in."""
    counts: dict[str, dict] = {}
    for t in tracks:
        for a in t.get("artists", []):
            aid = a.get("id", "")
            if aid not in counts:
                counts[aid] = {"name": a["name"], "count": 0, "is_top_artist": aid in top_artist_ids}
            counts[aid]["count"] += 1

    df = pd.DataFrame(list(counts.values())).sort_values("count", ascending=False)
    return df


def dominant_mood(df: pd.DataFrame) -> str:
    return df["mood"].value_counts().idxmax() if not df.empty else "Unknown"


def top_tracks_per_mood(df: pd.DataFrame, n: int = 5) -> dict[str, pd.DataFrame]:
    return {mood: group.head(n) for mood, group in df.groupby("mood")}


def top_artist_names_for_mood(df: pd.DataFrame, mood: str, n: int = 5) -> list[str]:
    subset = df[df["mood"] == mood]
    names: list[str] = []
    seen: set[str] = set()
    for _, row in subset.iterrows():
        for name in row["artist"].split(", "):
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
            if len(names) >= n:
                return names
    return names
