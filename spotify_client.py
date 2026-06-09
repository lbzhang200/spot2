from __future__ import annotations

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheHandler
from dotenv import load_dotenv

load_dotenv()

SCOPES = " ".join([
    "user-read-recently-played",
    "user-top-read",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-library-read",
])


def _cfg(key: str) -> str:
    """Read from Streamlit secrets first, fall back to env var."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, "")


class _SessionCacheHandler(CacheHandler):
    """Stores the OAuth token in Streamlit session_state (works on cloud)."""

    def __init__(self, session_state: dict) -> None:
        self._state = session_state

    def get_cached_token(self):
        return self._state.get("_spotify_token")

    def save_token_to_cache(self, token_info: dict) -> None:
        self._state["_spotify_token"] = token_info


def _make_oauth(redirect_uri: str, cache_handler: CacheHandler) -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=_cfg("SPOTIFY_CLIENT_ID"),
        client_secret=_cfg("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=redirect_uri,
        scope=SCOPES,
        cache_handler=cache_handler,
        open_browser=False,
        show_dialog=False,
    )


def get_auth_url(redirect_uri: str, session_state: dict) -> str:
    """Return the Spotify authorization URL for the login button."""
    handler = _SessionCacheHandler(session_state)
    oauth = _make_oauth(redirect_uri, handler)
    return oauth.get_authorize_url()


def get_spotify_from_code(
    code: str, redirect_uri: str, session_state: dict
) -> spotipy.Spotify:
    """Exchange auth code for token, store in session_state, return client."""
    handler = _SessionCacheHandler(session_state)
    oauth = _make_oauth(redirect_uri, handler)
    token = oauth.get_access_token(code, as_dict=True, check_cache=False)
    handler.save_token_to_cache(token)
    return spotipy.Spotify(auth_manager=oauth)


def get_spotify_from_session(
    redirect_uri: str, session_state: dict
) -> spotipy.Spotify | None:
    """Return an authenticated client using a cached session token, or None."""
    handler = _SessionCacheHandler(session_state)
    if not handler.get_cached_token():
        return None
    oauth = _make_oauth(redirect_uri, handler)
    # Refresh token if expired
    token = handler.get_cached_token()
    if oauth.is_token_expired(token):
        token = oauth.refresh_access_token(token["refresh_token"])
        handler.save_token_to_cache(token)
    return spotipy.Spotify(auth_manager=oauth)


def get_spotify_client_local() -> spotipy.Spotify:
    """Local-only shortcut using file-based cache (for scripts/testing)."""
    user_tag = os.getenv("SPOTIFY_USER", "default")
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=_cfg("SPOTIFY_CLIENT_ID"),
            client_secret=_cfg("SPOTIFY_CLIENT_SECRET"),
            redirect_uri=_cfg("SPOTIFY_REDIRECT_URI") or "http://127.0.0.1:8888",
            scope=SCOPES,
            cache_path=f".spotify_cache_{user_tag}",
            open_browser=True,
        )
    )


# ── Data fetch helpers ─────────────────────────────────────────────────────────

def get_current_user(sp: spotipy.Spotify) -> dict:
    return sp.current_user()


def get_top_tracks(sp: spotipy.Spotify, time_range: str, limit: int = 50) -> list[dict]:
    return sp.current_user_top_tracks(limit=limit, time_range=time_range).get("items", [])


def get_top_artists(sp: spotipy.Spotify, time_range: str, limit: int = 50) -> list[dict]:
    return sp.current_user_top_artists(limit=limit, time_range=time_range).get("items", [])


def get_recently_played(sp: spotipy.Spotify, limit: int = 50) -> list[dict]:
    tracks = []
    result = sp.current_user_recently_played(limit=min(limit, 50))
    tracks.extend(result.get("items", []))
    while result.get("next") and len(tracks) < limit:
        cursor = result["cursors"]["before"] if result.get("cursors") else None
        if not cursor:
            break
        result = sp.current_user_recently_played(limit=50, before=cursor)
        tracks.extend(result.get("items", []))
    seen, unique = set(), []
    for item in tracks:
        tid = item["track"]["id"]
        if tid and tid not in seen:
            seen.add(tid)
            unique.append(item["track"])
    return unique[:limit]


def create_playlist(sp: spotipy.Spotify, user_id: str, name: str, description: str = "") -> str:
    playlist = sp.user_playlist_create(user=user_id, name=name, public=False, description=description)
    return playlist["id"]


def add_tracks_to_playlist(sp: spotipy.Spotify, playlist_id: str, track_uris: list[str]) -> None:
    for i in range(0, len(track_uris), 100):
        sp.playlist_add_items(playlist_id, track_uris[i: i + 100])
