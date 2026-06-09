from __future__ import annotations

import random
import spotipy
import pandas as pd

from spotify_client import create_playlist, add_tracks_to_playlist
from analyzer import top_artist_names_for_mood


def _search_tracks(
    sp: spotipy.Spotify,
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> list[dict]:
    try:
        res = sp.search(q=query, type="track", limit=limit, offset=offset)
        return res.get("tracks", {}).get("items", [])
    except Exception:
        return []


def find_tracks_for_mood(
    sp: spotipy.Spotify,
    df: pd.DataFrame,
    mood: str,
    num_tracks: int,
    exclude_ids: set[str],
    seed: int | None = None,
) -> list[dict]:
    """
    Find varied tracks for a mood cluster. Each call with a different seed
    returns a different selection.
    """
    rng = random.Random(seed)

    artist_names = top_artist_names_for_mood(df, mood, n=20)
    if not artist_names:
        return []

    # Shuffle artist order so different artists get picked each run
    rng.shuffle(artist_names)

    seen_ids: set[str] = set(exclude_ids)
    candidates: list[dict] = []

    # Strategy 1: search by artist name with a random offset for variety
    for name in artist_names:
        offset = rng.choice([0, 10, 20, 30, 40])
        results = _search_tracks(sp, f'artist:"{name}"', limit=10, offset=offset)
        for t in results:
            tid = t.get("id")
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                candidates.append(t)
        if len(candidates) >= num_tracks * 4:
            break

    # Strategy 2: search by track names from the cluster as query terms
    mood_track_names = df[df["mood"] == mood]["name"].tolist()
    rng.shuffle(mood_track_names)
    for track_name in mood_track_names[:5]:
        # Use first couple words of the track name as a search query
        keywords = " ".join(track_name.split()[:3])
        if len(keywords) > 3:
            results = _search_tracks(sp, keywords, limit=10, offset=rng.choice([0, 10]))
            for t in results:
                tid = t.get("id")
                if tid and tid not in seen_ids:
                    seen_ids.add(tid)
                    candidates.append(t)
        if len(candidates) >= num_tracks * 5:
            break

    if not candidates:
        return []

    # Shuffle the full pool, then pick num_tracks
    rng.shuffle(candidates)
    return candidates[:num_tracks]


def build_and_create_playlist(
    sp: spotipy.Spotify,
    user_id: str,
    df: pd.DataFrame,
    mood: str,
    playlist_name: str,
    num_tracks: int,
    exclude_ids: set[str],
    seed: int | None = None,
) -> tuple[str | None, list[dict]]:
    selected = find_tracks_for_mood(sp, df, mood, num_tracks, exclude_ids, seed=seed)

    if not selected:
        raise ValueError("Could not find any candidate tracks. Try a different cluster.")

    # Try to create the playlist — silently skip if API tier blocks it
    playlist_url = None
    try:
        playlist_id = create_playlist(
            sp, user_id, playlist_name,
            description=f"Auto-generated {mood} playlist by Spotify Mood Analyzer",
        )
        uris = [t["uri"] for t in selected]
        add_tracks_to_playlist(sp, playlist_id, uris)
        playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
    except Exception:
        pass

    return playlist_url, selected
