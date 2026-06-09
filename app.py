"""Spotify Mood & Genre Analyzer — Streamlit app."""

from __future__ import annotations

import datetime
import random
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import os
from spotify_client import (
    get_spotify_client_local,
    get_current_user,
    get_top_tracks,
    get_top_artists,
    get_recently_played,
)
from lastfm_client import get_listeners_for_artists
from analyzer import (
    build_track_dataframe,
    cluster_tracks,
    aggregate_artists,
    dominant_mood,
    top_tracks_per_mood,
    FEATURE_COLS,
    MOOD_EMOJI,
    MOOD_LABEL_DIMENSIONS,
)
from playlist_builder import build_and_create_playlist

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Mood Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body, .stApp { background-color: #121212; color: #FFFFFF; }
    .stSidebar { background-color: #000000; }
    h1, h2, h3, h4 { color: #FFFFFF; }
    .stButton > button {
        background-color: #1DB954; color: #000000;
        font-weight: 700; border-radius: 500px;
        border: none; padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { background-color: #1ed760; }
    .stSelectbox label, .stSlider label, .stTextInput label { color: #b3b3b3; }
    .stTabs [data-baseweb="tab"] { color: #b3b3b3; }
    .stTabs [aria-selected="true"] { color: #1DB954 !important; border-bottom-color: #1DB954 !important; }
    .track-card {
        display: flex; align-items: center; gap: 12px;
        background: #282828; border-radius: 8px;
        padding: 10px 14px; margin-bottom: 8px;
    }
    .track-card img { border-radius: 4px; }
    .track-title { font-weight: 600; font-size: 14px; }
    .track-sub { color: #b3b3b3; font-size: 12px; }
    a { color: #1DB954 !important; text-decoration: none; }
    .api-warning {
        background: #2a1f00; border-left: 4px solid #FF9800;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;
        font-size: 13px; color: #FFD580;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Auth ────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_sp():
    return get_spotify_client_local()

def _require_auth():
    try:
        sp = _get_sp()
        return sp, get_current_user(sp)
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.stop()

sp, user = _require_auth()

# ── Header ──────────────────────────────────────────────────────────────────────
col_img, col_name = st.columns([1, 8])
images = user.get("images") or []
if images:
    col_img.image(images[0]["url"], width=60)
col_name.markdown(f"### 👋 Welcome, **{user.get('display_name', 'Spotify User')}**")

# API restriction banner
st.markdown(
    '<div class="api-warning">⚠️ <strong>Spotify API note:</strong> Apps created after Nov 2024 '
    'have restricted access to audio features, genres, and recommendations. '
    'This app uses listening behaviour and metadata for analysis. '
    'To restore full features, request <strong>Extended Quota Mode</strong> at '
    '<a href="https://developer.spotify.com/dashboard" style="color:#FFD580" target="_blank">'
    'developer.spotify.com/dashboard</a> → your app → Settings → Request extended access.</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar nav ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Spotify Analyzer")
    section = st.radio(
        "Navigate",
        ["My Listening", "Mood Analysis", "Generate Playlist"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("<small style='color:#b3b3b3'>Data cached for 1 hour</small>", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────────
TIME_RANGES = {
    "Last 4 weeks": "short_term",
    "Last 6 months": "medium_term",
    "All time": "long_term",
    "Recent plays": "recent",
}

GREEN = "#1DB954"
RADAR_COLORS = [GREEN, "#E91E63", "#2196F3", "#FF9800", "#9C27B0"]

FEATURE_LABELS = {
    "rank_norm":          "Top-of-list",
    "release_year_norm":  "Recency",
    "duration_norm":      "Song length",
    "is_explicit":        "Explicit",
    "is_single":          "Singles",
    "artist_top_rank":    "Top-artist track",
    "name_energy":        "Name energy",
}


def _track_card(track: dict, rank: int | None = None) -> None:
    img = track.get("image_url") or ((track.get("album") or {}).get("images") or [{}])[0].get("url", "")
    name = track.get("name", "Unknown")
    artist = track.get("artist") or ", ".join(a["name"] for a in track.get("artists", []))
    url = track.get("external_url") or track.get("external_urls", {}).get("spotify", "#")
    year = track.get("release_year", "")
    prefix = f"{rank}. " if rank else ""
    img_tag = f'<img src="{img}" width="48" height="48">' if img else ""
    st.markdown(
        f'<div class="track-card">{img_tag}'
        f'<div><div class="track-title">{prefix}<a href="{url}" target="_blank">{name}</a></div>'
        f'<div class="track-sub">{artist} {f"· {year}" if year else ""}</div></div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600)
def fetch_listening_data(time_range_key: str):
    if time_range_key == "recent":
        tracks = get_recently_played(sp, limit=50)
        top_artists = get_top_artists(sp, "short_term", limit=50)
    else:
        tracks = get_top_tracks(sp, time_range_key, limit=50)
        top_artists = get_top_artists(sp, time_range_key, limit=50)

    top_artist_ids = {a["id"] for a in top_artists if a.get("id")}
    return tracks, top_artists, top_artist_ids


@st.cache_data(ttl=3600)
def fetch_cluster_data(time_range_key: str):
    tracks, top_artists, top_artist_ids = fetch_listening_data(time_range_key)
    df = build_track_dataframe(tracks, top_artist_ids)
    if df.empty:
        return df, [], []
    df, centroids, labels = cluster_tracks(df)
    return df, centroids, labels


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — My Listening
# ══════════════════════════════════════════════════════════════════════════════
if section == "My Listening":
    st.header("🎧 My Listening")

    range_label = st.selectbox("Time range", list(TIME_RANGES.keys()))
    range_key = TIME_RANGES[range_label]

    with st.spinner("Fetching your listening data…"):
        tracks, top_artists, top_artist_ids = fetch_listening_data(range_key)

    if not tracks:
        st.warning("No data found for this time range.")
        st.stop()

    # Artist track count chart
    st.subheader("Your Most-Represented Artists in Top Tracks")
    artist_df = aggregate_artists(tracks, top_artist_ids)
    top15 = artist_df.head(15)

    colors = [GREEN if row["is_top_artist"] else "#535353" for _, row in top15.iterrows()]
    fig = go.Figure(go.Bar(
        x=top15["count"],
        y=top15["name"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x} track(s)<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#121212", plot_bgcolor="#121212",
        font_color="#FFFFFF",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=10, b=0), height=420,
        xaxis_title="Tracks in your top 50",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"🟢 Green = also in your top artists list for this period")

    # Release era breakdown
    st.subheader("Music by Release Era")
    df_tmp = build_track_dataframe(tracks, top_artist_ids)
    if not df_tmp.empty:
        bins = [0, 2000, 2010, 2016, 2020, 2023, 2100]
        labels_era = ["Pre-2000", "2000s", "2010–15", "2016–19", "2020–22", "2023+"]
        df_tmp["era"] = pd.cut(df_tmp["release_year"], bins=bins, labels=labels_era, right=True)
        era_counts = df_tmp["era"].value_counts().sort_index().reset_index()
        era_counts.columns = ["era", "count"]
        fig2 = px.bar(era_counts, x="era", y="count", color_discrete_sequence=[GREEN],
                      labels={"era": "", "count": "Tracks"})
        fig2.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212",
                           font_color="#FFFFFF", margin=dict(t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Obscurity chart ──────────────────────────────────────────────────────
    st.subheader("🔍 Artist Obscurity")

    lastfm_key = os.getenv("LASTFM_API_KEY", "")
    if not lastfm_key or lastfm_key == "your_lastfm_api_key_here":
        st.info(
            "Add your **Last.fm API key** to `.env` as `LASTFM_API_KEY` to see real listener counts. "
            "Get one free at [last.fm/api](https://www.last.fm/api/account/create)."
        )
    else:
        artist_names = [a["name"] for a in top_artists[:20]]

        @st.cache_data(ttl=3600)
        def _fetch_listeners(names: tuple[str, ...]) -> dict[str, int | None]:
            return get_listeners_for_artists(list(names))

        with st.spinner("Fetching listener counts from Last.fm…"):
            listeners_map = _fetch_listeners(tuple(artist_names))

        rows = []
        for name, count in listeners_map.items():
            if count is not None:
                rows.append({"artist": name, "listeners": count})

        if not rows:
            st.warning("Last.fm returned no data — check your API key or try again.")
        else:
            obs_df = pd.DataFrame(rows).sort_values("listeners")

            # Obscurity score: invert and normalise 0–100
            max_l = obs_df["listeners"].max()
            obs_df["obscurity_score"] = ((1 - obs_df["listeners"] / max_l) * 100).round(1)
            avg_listeners = int(obs_df["listeners"].mean())
            most_obscure = obs_df.iloc[0]
            most_mainstream = obs_df.iloc[-1]

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg listeners across your artists", f"{avg_listeners:,}")
            m2.metric("Most obscure", most_obscure["artist"],
                      delta=f"{most_obscure['listeners']:,} listeners", delta_color="off")
            m3.metric("Most mainstream", most_mainstream["artist"],
                      delta=f"{most_mainstream['listeners']:,} listeners", delta_color="off")

            # Bar chart — sorted by listener count ascending (most obscure at top)
            fig_obs = go.Figure(go.Bar(
                x=obs_df["listeners"],
                y=obs_df["artist"],
                orientation="h",
                marker=dict(
                    color=obs_df["listeners"],
                    colorscale=[[0, "#1DB954"], [0.3, "#4FC3F7"], [1.0, "#E91E63"]],
                    showscale=True,
                    colorbar=dict(
                        title="Listeners",
                        tickformat=".2s",
                        bgcolor="#121212",
                        tickfont=dict(color="#FFFFFF"),
                        titlefont=dict(color="#FFFFFF"),
                    ),
                ),
                hovertemplate="<b>%{y}</b><br>%{x:,} listeners<extra></extra>",
                text=[f"{v:,}" for v in obs_df["listeners"]],
                textposition="outside",
                textfont=dict(color="#FFFFFF", size=11),
            ))
            fig_obs.update_layout(
                paper_bgcolor="#121212", plot_bgcolor="#121212",
                font_color="#FFFFFF",
                yaxis=dict(autorange="reversed"),
                xaxis=dict(tickformat=".2s"),
                margin=dict(l=0, r=80, t=10, b=0),
                height=max(350, len(rows) * 28),
            )
            st.plotly_chart(fig_obs, use_container_width=True)
            st.caption("🟢 Green = obscure  →  🔴 Pink = mainstream  (by Last.fm total unique listeners)")

    # Top artists & tracks
    col_a, col_t = st.columns(2)

    with col_a:
        st.subheader("Top 5 Artists")
        for i, artist in enumerate(top_artists[:5], 1):
            img_url = (artist.get("images") or [{}])[0].get("url", "")
            url = artist.get("external_urls", {}).get("spotify", "#")
            img_tag = f'<img src="{img_url}" width="48" height="48" style="border-radius:50%">' if img_url else ""
            st.markdown(
                f'<div class="track-card">{img_tag}'
                f'<div><div class="track-title">{i}. <a href="{url}" target="_blank">{artist["name"]}</a></div>'
                f'<div class="track-sub">Top artist #{i}</div></div></div>',
                unsafe_allow_html=True,
            )

    with col_t:
        st.subheader("Top 5 Tracks")
        for i, track in enumerate(tracks[:5], 1):
            _track_card(track, rank=i)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Mood Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Mood Analysis":
    st.header("🧠 Mood Analysis")
    st.caption("Clusters are based on listening rank, release era, song length, and artist affinity.")

    range_label = st.selectbox("Time range", list(TIME_RANGES.keys()))
    range_key = TIME_RANGES[range_label]

    with st.spinner("Clustering your tracks…"):
        df, centroids, labels = fetch_cluster_data(range_key)

    if df.empty:
        st.warning("Not enough tracks found for this time range.")
        st.stop()

    dom = dominant_mood(df)
    emoji = MOOD_EMOJI.get(dom.split()[0], "🎵")
    n_dom = df[df["mood"] == dom].shape[0]
    st.success(f"{emoji} Your dominant cluster is **{dom}** ({n_dom} tracks)")

    # Radar chart of cluster profiles
    st.subheader("Cluster Profiles")
    radar_features = [f for f in FEATURE_COLS if f not in ("is_explicit", "is_single")]
    radar_labels = [FEATURE_LABELS.get(f, f) for f in radar_features]

    fig = go.Figure()
    for i, (label, centroid) in enumerate(zip(labels, centroids)):
        vals = [centroid.get(f, 0) for f in radar_features]
        vals += vals[:1]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=f"{MOOD_EMOJI.get(label.split()[0], '🎵')} {label}",
            line_color=RADAR_COLORS[i % len(RADAR_COLORS)],
            opacity=0.75,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#282828",
            radialaxis=dict(visible=True, range=[0, 1], color="#b3b3b3", gridcolor="#333"),
            angularaxis=dict(color="#b3b3b3", gridcolor="#333"),
        ),
        paper_bgcolor="#121212", font_color="#FFFFFF",
        legend=dict(bgcolor="#121212"),
        margin=dict(l=40, r=40, t=40, b=40), height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tracks per cluster
    st.subheader("Tracks per Cluster")
    mood_tracks = top_tracks_per_mood(df, n=5)
    tabs = st.tabs([f"{MOOD_EMOJI.get(m.split()[0], '🎵')} {m}" for m in labels])
    for tab, label in zip(tabs, labels):
        with tab:
            subset = mood_tracks.get(label, pd.DataFrame())
            if subset.empty:
                st.write("No tracks in this cluster.")
            else:
                for _, row in subset.iterrows():
                    _track_card(row.to_dict())

    # Cluster size bar
    st.subheader("Cluster Sizes")
    counts = df["mood"].value_counts().reset_index()
    counts.columns = ["mood", "count"]
    fig2 = px.bar(counts, x="mood", y="count", color_discrete_sequence=[GREEN],
                  labels={"mood": "", "count": "Tracks"})
    fig2.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212",
                       font_color="#FFFFFF", margin=dict(t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Generate Playlist
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Generate Playlist":
    st.header("🎛️ Generate Playlist")
    st.caption("Finds new tracks from the same artists as your selected cluster.")

    range_label = st.selectbox("Base time range", list(TIME_RANGES.keys()))
    range_key = TIME_RANGES[range_label]

    with st.spinner("Loading clusters…"):
        df, centroids, labels = fetch_cluster_data(range_key)

    if df.empty:
        st.warning("Not enough data. Try a different time range.")
        st.stop()

    selected_mood = st.selectbox(
        "Select a cluster",
        [f"{MOOD_EMOJI.get(l.split()[0], '🎵')} {l}" for l in labels],
    )
    clean_mood = " ".join(selected_mood.split()[1:]) if selected_mood else labels[0]

    num_songs = st.slider("Number of songs", min_value=10, max_value=50, value=25, step=5)

    month_year = datetime.datetime.now().strftime("%B %Y")
    default_name = f"{clean_mood} Mix — {month_year}"
    playlist_name = st.text_input("Playlist name", value=default_name)

    st.divider()

    # Each click gets a fresh random seed → different results every time
    if "playlist_seed" not in st.session_state:
        st.session_state.playlist_seed = random.randint(0, 999_999)

    col_btn1, col_btn2 = st.columns([2, 1])
    generate_clicked = col_btn1.button("🎵 Find Tracks")
    reshuffle_clicked = col_btn2.button("🔀 Reshuffle")

    if reshuffle_clicked:
        st.session_state.playlist_seed = random.randint(0, 999_999)

    if generate_clicked or reshuffle_clicked:
        known_ids = set(df["id"].tolist())
        with st.spinner("Searching for tracks…"):
            try:
                playlist_url, rec_tracks = build_and_create_playlist(
                    sp=sp,
                    user_id=user["id"],
                    df=df,
                    mood=clean_mood,
                    playlist_name=playlist_name,
                    num_tracks=num_songs,
                    exclude_ids=known_ids,
                    seed=st.session_state.playlist_seed,
                )
                # Rotate seed so next generate is different even without reshuffle
                st.session_state.playlist_seed = random.randint(0, 999_999)

                if playlist_url:
                    st.success(f"✅ Playlist **{playlist_name}** created!")
                    st.markdown(
                        f'<a href="{playlist_url}" target="_blank">'
                        f'<button style="background:#1DB954;color:#000;font-weight:700;border:none;'
                        f'border-radius:500px;padding:0.5rem 1.5rem;cursor:pointer;font-size:15px;">'
                        f'🔗 Open in Spotify</button></a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="api-warning">⚠️ <strong>Playlist auto-creation is blocked</strong> by '
                        'Spotify\'s Essential API tier for new apps. '
                        'Request <strong>Extended Quota Mode</strong> at '
                        '<a href="https://developer.spotify.com/dashboard" style="color:#FFD580" target="_blank">'
                        'developer.spotify.com/dashboard</a> to enable it. '
                        'In the meantime, click any track below to open it in Spotify.</div>',
                        unsafe_allow_html=True,
                    )

                # Spotify URI list for manual copy-paste into desktop app
                uris = [t.get("uri", "") for t in rec_tracks if t.get("uri")]
                if uris:
                    st.text_area(
                        "📋 Copy these Spotify URIs — paste into Spotify desktop (File → Import playlist from URL or drag into a playlist)",
                        value="\n".join(uris),
                        height=120,
                    )

                st.subheader(f"Recommended tracks ({len(rec_tracks)})")
                for track in rec_tracks:
                    _track_card(track)

            except Exception as e:
                st.error(f"Failed to find tracks: {e}")
