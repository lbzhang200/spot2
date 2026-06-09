# Spotify Mood & Genre Analyzer

A personal Spotify analytics tool that clusters your listening history into mood profiles and generates new playlists directly in your Spotify account.

---

## Setup

### 1. Create a Spotify Developer App

1. Go to [https://developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
2. Click **Create app**
3. Fill in any name/description
4. Under **Redirect URIs**, add: `http://localhost:8501`
5. Save. Copy your **Client ID** and **Client Secret**.

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8501
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Recommended: use a virtual environment
> ```bash
> python -m venv .venv && source .venv/bin/activate
> ```

### 4. Run the app

```bash
streamlit run app.py
```

The first run will open a browser tab asking you to authorize the app with your Spotify account. After clicking **Agree**, you'll be redirected back and the app will load.

---

## Features

| Section | What it does |
|---|---|
| **My Listening** | Top genres (weighted bar chart), top 5 artists & tracks for any time range |
| **Mood Analysis** | KMeans clusters your tracks into 5 moods, radar chart of audio profiles, example tracks per mood |
| **Generate Playlist** | Pick a mood → generate recommendations → create a private playlist in your Spotify account |

### Time ranges

- **Last 4 weeks** — short-term Spotify listening data
- **Last 6 months** — medium-term
- **All time** — long-term
- **Recent plays** — last ~200 recently played tracks (deduplicated)

### Mood labels

| Mood | Dominant characteristics |
|---|---|
| Energetic | High energy + fast tempo |
| Happy | High valence + high danceability |
| Melancholy | Low valence + high acousticness |
| Focus | Low tempo + low energy + high instrumentalness |
| Chill | Mid-range across all features |

---

## Project structure

```
spotify_analyzer/
├── app.py               # Main Streamlit UI
├── spotify_client.py    # Auth + all Spotify API calls
├── analyzer.py          # Genre aggregation + KMeans mood clustering
├── playlist_builder.py  # Recommendation fetching + playlist creation
├── .env                 # Your credentials (never commit this)
├── .env.example         # Template
└── requirements.txt
```

---

## Notes

- API responses are cached for **1 hour** via `@st.cache_data`.
- Tracks without audio features are skipped automatically.
- The OAuth token is cached in `.spotify_cache` — delete it to re-authenticate.
