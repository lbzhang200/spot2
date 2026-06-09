"""Last.fm API client — used for artist listener counts (obscurity data)."""

from __future__ import annotations

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")
BASE_URL = "https://ws.audioscrobbler.com/2.0/"


def _get(params: dict) -> dict:
    params = {"api_key": LASTFM_API_KEY, "format": "json", **params}
    try:
        r = requests.get(BASE_URL, params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def get_artist_listeners(artist_name: str) -> int | None:
    """Return total unique listeners for an artist, or None if not found."""
    data = _get({"method": "artist.getInfo", "artist": artist_name, "autocorrect": "1"})
    try:
        return int(data["artist"]["stats"]["listeners"])
    except (KeyError, TypeError, ValueError):
        return None


def get_listeners_for_artists(
    artist_names: list[str], delay: float = 0.1
) -> dict[str, int | None]:
    """Fetch listener counts for a list of artists. Returns {name: listeners}."""
    result: dict[str, int | None] = {}
    for name in artist_names:
        result[name] = get_artist_listeners(name)
        time.sleep(delay)  # be polite to Last.fm rate limits
    return result
