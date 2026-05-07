# 📚 StudyDeck — YouTube → Flashcards & Quiz

A Streamlit app that turns any YouTube video into flashcards and a quiz.  
Uses **YouTube captions** when available, and falls back to **Docling ASR (Whisper)** when there are none.

---

## How it works

```
YouTube URL
    │
    ├─ Has captions? ──YES──► youtube-transcript-api ──► transcript text
    │
    └─ No captions?  ──NO───► yt-dlp (download audio) ──► Docling ASR (Whisper) ──► transcript text
                                                                │
                                                          OpenRouter LLM (via LangChain)
                                                                │
                                                    ┌───────────┴───────────┐
                                               Flashcards               Quiz questions
                                                    │                       │
                                              Streamlit UI (public URL)
```

---

## 🚀 Deploy to Streamlit Cloud (free, public URL)

### Step 1 — Push this project to GitHub

1. Create a new GitHub repo (e.g. `studydeck`)
2. Upload these 3 files:
   - `app.py`
   - `requirements.txt`
   - `README.md`

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **Deploy**

That's it! You get a public URL like:  
`https://your-username-studydeck-app-xxxx.streamlit.app`

### Step 3 — Use the app

1. Open the app
2. Paste your **OpenRouter API key** in the sidebar
3. Choose a free model (e.g. `google/gemma-3-27b-it:free`)
4. Paste any YouTube URL
5. Click **Generate Study Materials**

---

## Get an OpenRouter API key (free)

1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up → Dashboard → API Keys → Create key
3. Many models are **completely free** to use

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies for Streamlit Cloud |
| `README.md` | This file |

---

## For the M2 assignment

This fulfills all task requirements:
- ✅ Uses LangChain (`langchain-openai`, `langchain-core`) for LLM integration
- ✅ Fetches YouTube transcripts via `youtube-transcript-api` (LangChain YouTube transcript integration equivalent)
- ✅ Falls back to **Docling ASR Pipeline** (Whisper) when no captions exist
- ✅ Non-trivial use case: AI-powered flashcard + quiz generation
- ✅ Public shareable URL via Streamlit Cloud
