import streamlit as st
import re
import os
import json
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyDeck – YouTube → Flashcards & Quiz",
    page_icon="📚",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

h1 { font-family: 'Playfair Display', serif !important; }

.card-front, .card-back {
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
}
.card-front {
    background: #1e2530;
    border: 1px solid #2e3a4a;
    color: #e8dcc8;
}
.card-back {
    background: #1a2410;
    border: 1px solid #3a5020;
    color: #a8d080;
}
.tag {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #7a8b6e;
    margin-bottom: 0.5rem;
}
.question-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #e8dcc8;
    line-height: 1.5;
}
.correct-option {
    background: #1a3010 !important;
    border: 1px solid #4a8030 !important;
    color: #90c060 !important;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
}
.wrong-option {
    background: #301010 !important;
    border: 1px solid #803030 !important;
    color: #c06060 !important;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
}
.explanation-box {
    background: #0d1117;
    border-left: 3px solid #4a8030;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #90a880;
    margin-top: 0.5rem;
}
.score-display {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #f0c060;
}
.section-divider {
    border: none;
    border-top: 1px solid #2e3a4a;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from any YouTube URL format."""
    patterns = [
        r"(?:v=|/embed/|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def get_transcript_from_captions(video_id: str) -> str | None:
    """Try to fetch existing captions from YouTube. Returns None if unavailable."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(chunk["text"] for chunk in transcript_list)
    except (NoTranscriptFound, TranscriptsDisabled):
        # Try any available language
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcripts.find_generated_transcript(
                [t.language_code for t in transcripts]
            )
            return " ".join(chunk["text"] for chunk in transcript.fetch())
        except Exception:
            return None
    except Exception:
        return None


def get_transcript_via_asr(video_id: str) -> str:
    """Download audio and run Docling ASR pipeline (Whisper) when no captions exist."""
    from docling.document_converter import DocumentConverter
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path.replace(".mp3", ""),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        # Find the actual output file (yt-dlp may add extension)
        for fname in os.listdir(tmpdir):
            if fname.endswith(".mp3"):
                audio_path = os.path.join(tmpdir, fname)
                break

        converter = DocumentConverter()
        result = converter.convert(audio_path)
        return result.document.export_to_text()


def get_transcript(video_id: str) -> tuple[str, str]:
    """
    Returns (transcript_text, method_used).
    Tries captions first; falls back to Docling ASR.
    """
    transcript = get_transcript_from_captions(video_id)
    if transcript:
        return transcript, "YouTube captions"
    else:
        transcript = get_transcript_via_asr(video_id)
        return transcript, "Docling ASR (Whisper)"


def build_llm(api_key: str, model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.4,
    )


def generate_flashcards(transcript: str, llm, n: int, difficulty: str) -> list[dict]:
    prompt = ChatPromptTemplate.from_template("""
You are an expert study assistant. Based on the transcript below, generate exactly {n} flashcards.
Difficulty level: {difficulty}

Rules:
- Each card must test a real concept or fact from the transcript
- Keep answers concise (1-3 sentences)
- Higher difficulty = deeper understanding required, not just recall
- Return ONLY valid JSON, no markdown, no extra text

Format:
{{"flashcards": [{{"question": "...", "answer": "..."}}]}}

Transcript:
{transcript}
""")
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"transcript": transcript, "n": n, "difficulty": difficulty})
    return result.get("flashcards", [])


def generate_quiz(transcript: str, llm, n: int, difficulty: str) -> list[dict]:
    prompt = ChatPromptTemplate.from_template("""
You are an expert study assistant. Based on the transcript below, generate exactly {n} multiple-choice quiz questions.
Difficulty level: {difficulty}

Rules:
- 4 options per question labeled A, B, C, D
- Exactly one correct answer
- All wrong options must be plausible (not obviously wrong)
- Include a short explanation for why the correct answer is right
- Return ONLY valid JSON, no markdown, no extra text

Format:
{{"quiz": [{{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "correct": 0, "explanation": "..."}}]}}

(correct is 0-indexed: 0=A, 1=B, 2=C, 3=D)

Transcript:
{transcript}
""")
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"transcript": transcript, "n": n, "difficulty": difficulty})
    return result.get("quiz", [])


# ── Session state init ─────────────────────────────────────────────────────────
for key in ["flashcards", "quiz", "transcript", "method", "card_idx",
            "show_answer", "score", "answered", "quiz_answers"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["flashcards", "quiz", "quiz_answers"] else (
            0 if key in ["card_idx", "score", "answered"] else
            False if key == "show_answer" else ""
        )


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("# 📚 StudyDeck")
st.markdown("**YouTube → Flashcards & Quiz** · Powered by OpenRouter + Docling ASR")
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-...",
        help="Get yours at openrouter.ai",
    )

    model = st.selectbox(
        "Model",
        options=[
            "google/gemma-3-27b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "anthropic/claude-3-haiku",
            "openai/gpt-4o-mini",
        ],
        help="Free models work great for this task",
    )

    n_cards = st.slider("Number of flashcards", 3, 15, 8)
    n_quiz = st.slider("Number of quiz questions", 3, 10, 5)
    difficulty = st.select_slider(
        "Difficulty",
        options=["easy", "medium", "hard"],
        value="medium",
    )

    st.markdown("---")
    st.caption("Made with ❤️ for IAU M2 · StudyDeck uses YouTube captions when available, and Docling Whisper ASR as fallback.")

# Main input
url = st.text_input(
    "🔗 YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

generate_btn = st.button("⚡ Generate Study Materials", type="primary", use_container_width=True)

if generate_btn:
    if not api_key:
        st.error("Please enter your OpenRouter API key in the sidebar.")
    elif not url:
        st.error("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Could not parse a YouTube video ID from that URL. Please check it.")
        else:
            with st.spinner("Fetching transcript..."):
                try:
                    transcript, method = get_transcript(video_id)
                    st.session_state.transcript = transcript
                    st.session_state.method = method
                except Exception as e:
                    st.error(f"Failed to get transcript: {e}")
                    st.stop()

            llm = build_llm(api_key, model)

            with st.spinner("Generating flashcards with AI..."):
                try:
                    st.session_state.flashcards = generate_flashcards(
                        transcript, llm, n_cards, difficulty
                    )
                except Exception as e:
                    st.error(f"Failed to generate flashcards: {e}")
                    st.stop()

            with st.spinner("Generating quiz questions with AI..."):
                try:
                    st.session_state.quiz = generate_quiz(
                        transcript, llm, n_quiz, difficulty
                    )
                except Exception as e:
                    st.error(f"Failed to generate quiz: {e}")
                    st.stop()

            # Reset state
            st.session_state.card_idx = 0
            st.session_state.show_answer = False
            st.session_state.score = 0
            st.session_state.answered = 0
            st.session_state.quiz_answers = [None] * len(st.session_state.quiz)
            st.rerun()


# ── Results tabs ───────────────────────────────────────────────────────────────
if st.session_state.flashcards:
    st.success(f"✅ Transcript fetched via: **{st.session_state.method}**")

    tab_flash, tab_quiz, tab_raw = st.tabs(["🃏 Flashcards", "🧠 Quiz", "📄 Transcript"])

    # ── Flashcards tab
    with tab_flash:
        cards = st.session_state.flashcards
        idx = st.session_state.card_idx
        card = cards[idx]

        st.markdown(f"**Card {idx + 1} of {len(cards)}**")

        # Question
        st.markdown(f"""
        <div class="card-front">
            <div class="tag">Question</div>
            <div class="question-text">{card['question']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Reveal toggle
        if st.button("🔄 Reveal / Hide Answer", use_container_width=True):
            st.session_state.show_answer = not st.session_state.show_answer
            st.rerun()

        if st.session_state.show_answer:
            st.markdown(f"""
            <div class="card-back">
                <div class="tag">Answer</div>
                {card['answer']}
            </div>
            """, unsafe_allow_html=True)

        # Navigation
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("← Previous", disabled=(idx == 0), use_container_width=True):
                st.session_state.card_idx -= 1
                st.session_state.show_answer = False
                st.rerun()
        with col_next:
            if st.button("Next →", disabled=(idx == len(cards) - 1), use_container_width=True):
                st.session_state.card_idx += 1
                st.session_state.show_answer = False
                st.rerun()

    # ── Quiz tab
    with tab_quiz:
        quiz = st.session_state.quiz
        answered_all = all(a is not None for a in st.session_state.quiz_answers)

        score = sum(
            1 for i, q in enumerate(quiz)
            if st.session_state.quiz_answers[i] == q["correct"]
        )
        answered = sum(1 for a in st.session_state.quiz_answers if a is not None)

        col_s, col_p = st.columns(2)
        with col_s:
            st.markdown(f'<div class="score-display">{score} / {answered}</div>', unsafe_allow_html=True)
            st.caption("Score so far")
        with col_p:
            if answered > 0:
                pct = round(score / answered * 100)
                st.metric("Accuracy", f"{pct}%")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        for i, q in enumerate(quiz):
            st.markdown(f"**Q{i+1}.** {q['question']}")
            letters = ["A", "B", "C", "D"]
            chosen = st.session_state.quiz_answers[i]

            for j, opt in enumerate(q["options"]):
                label = opt if opt.startswith(("A.", "B.", "C.", "D.")) else f"{letters[j]}. {opt}"

                if chosen is None:
                    if st.button(label, key=f"opt_{i}_{j}", use_container_width=True):
                        st.session_state.quiz_answers[i] = j
                        st.rerun()
                else:
                    if j == q["correct"]:
                        st.markdown(f'<div class="correct-option">✅ {label}</div>', unsafe_allow_html=True)
                    elif j == chosen and chosen != q["correct"]:
                        st.markdown(f'<div class="wrong-option">❌ {label}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="padding:0.6rem 1rem;margin:0.3rem 0;border-radius:8px;border:1px solid #2e3a4a;color:#888">{label}</div>', unsafe_allow_html=True)

            if chosen is not None:
                st.markdown(f'<div class="explanation-box">💡 {q["explanation"]}</div>', unsafe_allow_html=True)

            st.markdown("")

        if answered_all:
            st.balloons()
            final_pct = round(score / len(quiz) * 100)
            if final_pct == 100:
                st.success(f"🏆 Perfect score! {score}/{len(quiz)}")
            elif final_pct >= 70:
                st.success(f"🎉 Great job! {score}/{len(quiz)} ({final_pct}%)")
            else:
                st.warning(f"📖 Keep studying! {score}/{len(quiz)} ({final_pct}%) — try reviewing the flashcards")

        if st.button("🔁 Reset Quiz", use_container_width=True):
            st.session_state.quiz_answers = [None] * len(quiz)
            st.session_state.score = 0
            st.session_state.answered = 0
            st.rerun()

    # ── Transcript tab
    with tab_raw:
        st.caption(f"Source: {st.session_state.method}")
        st.text_area(
            "Raw transcript",
            value=st.session_state.transcript,
            height=400,
            disabled=True,
        )
        word_count = len(st.session_state.transcript.split())
        st.caption(f"{word_count:,} words · {len(st.session_state.transcript):,} characters")
