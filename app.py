import streamlit as st
import re
import os
import json
import tempfile
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="StudyDeck – YouTube → Flashcards & Quiz",
    page_icon="📚",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1 { font-family: 'Playfair Display', serif !important; }

.card-front, .card-back {
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 0.5rem;
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
.rtl-text {
    direction: rtl;
    text-align: right;
    font-family: 'DM Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    patterns = [r"(?:v=|/embed/|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})"]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def detect_language(text: str) -> str:
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "ar" if arabic_chars / max(len(text), 1) > 0.2 else "en"


def get_transcript_langchain(url: str, language: str) -> tuple[str, str]:
    lang_codes = [language, "en", "ar"] if language != "en" else ["en", "ar", "en-US", "en-GB"]
    for lang in lang_codes:
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,
                language=[lang],
                transcript_format=TranscriptFormat.TEXT,
            )
            docs = loader.load()
            if docs:
                return " ".join(d.page_content for d in docs), f"LangChain YoutubeLoader ({lang})"
        except Exception:
            continue
    # Last resort: try without language filter
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        docs = loader.load()
        if docs:
            return " ".join(d.page_content for d in docs), "LangChain YoutubeLoader (auto)"
    except Exception:
        pass
    raise Exception("No captions found via LangChain YoutubeLoader")


def get_transcript_asr(video_id: str) -> tuple[str, str]:
    """Fallback: shown when no captions found on cloud."""
    raise Exception(
        "This video has no captions. On the cloud version, YouTube blocks audio download. "
        "Please try a video WITH captions (Khan Academy, TED Talks, MIT lectures, etc.)"
    )


def get_transcript(url: str, video_id: str, language: str) -> tuple[str, str]:
    try:
        return get_transcript_langchain(url, language)
    except Exception:
        st.warning("⚠️ No captions found on this video.")
        st.info("💡 Try a video with captions — Khan Academy, TED Talks, or any popular YouTube video work great!")
        raise Exception("No captions available for this video on the cloud version.")


def build_llm(api_key: str, model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.4,
    )


def safe_parse(raw: str, key: str) -> list:
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        clean = raw[start:end]
        return json.loads(clean).get(key, [])
    except Exception as e:
        raise ValueError(f"Could not parse model response as JSON: {e}\n\nRaw output:\n{raw[:300]}")


def generate_flashcards(transcript: str, llm, n: int, difficulty: str, lang: str) -> list[dict]:
    lang_instruction = "Respond in Arabic. Use Arabic for all questions and answers." if lang == "ar" else "Respond in English."
    prompt = ChatPromptTemplate.from_template("""
You are an expert study assistant. Based on the transcript below, generate exactly {n} flashcards.
Difficulty: {difficulty}
Language instruction: {lang_instruction}

CRITICAL: Return ONLY the raw JSON object. No introduction, no bismillah, no markdown, no extra text before or after. Start your response with {{ and end with }}.

Format:
{{"flashcards": [{{"question": "...", "answer": "..."}}]}}

Transcript:
{transcript}
""")
    chain = prompt | llm
    raw = chain.invoke({
        "transcript": transcript,
        "n": n,
        "difficulty": difficulty,
        "lang_instruction": lang_instruction
    }).content
    return safe_parse(raw, "flashcards")


def generate_quiz(transcript: str, llm, n: int, difficulty: str, lang: str) -> list[dict]:
    lang_instruction = "Respond in Arabic. Use Arabic for all questions, options, and explanations." if lang == "ar" else "Respond in English."
    prompt = ChatPromptTemplate.from_template("""
You are an expert study assistant. Generate exactly {n} multiple-choice questions.
Difficulty: {difficulty}
Language instruction: {lang_instruction}

CRITICAL: Return ONLY the raw JSON object. No introduction, no bismillah, no markdown, no extra text before or after. Start your response with {{ and end with }}.

Format:
{{"quiz": [{{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "correct": 0, "explanation": "..."}}]}}

Transcript:
{transcript}
""")
    chain = prompt | llm
    raw = chain.invoke({
        "transcript": transcript,
        "n": n,
        "difficulty": difficulty,
        "lang_instruction": lang_instruction
    }).content
    return safe_parse(raw, "quiz")


# ── Session state ─────────────────────────────────────────────────────────────
for key in ["flashcards", "quiz", "transcript", "method", "card_idx",
            "show_answer", "score", "answered", "quiz_answers", "detected_lang"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["flashcards", "quiz", "quiz_answers"] else (
            0 if key in ["card_idx", "score", "answered"] else
            False if key == "show_answer" else ""
        )


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 📚 StudyDeck")
st.markdown("**YouTube → Flashcards & Quiz** · يدعم العربية والإنجليزية")
st.markdown('<p style="font-size:0.8rem; color:#f0c060; letter-spacing:1px;">🎓 SDAIA · Applied AI Bootcamp</p>', unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-...",
    )

    model = st.selectbox(
        "Model",
        options=[
            "openrouter/free",
            "── Specific Free Models ──",
            "deepseek/deepseek-r1:free",
            "deepseek/deepseek-v3:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-235b-a22b:free",
            "nvidia/llama-3.1-nemotron-70b-instruct:free",
            "── Paid ──",
            "openai/gpt-4o-mini",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3.5-sonnet",
        ],
    )

    if "──" in model:
        st.warning("Please select a model, not a separator!")
        st.stop()

    language = st.radio(
        "Transcript & output language",
        options=["Auto-detect", "English", "Arabic / عربي"],
        index=0,
    )

    n_cards = st.slider("Flashcards", 3, 15, 8)
    n_quiz = st.slider("Quiz questions", 3, 10, 5)
    difficulty = st.select_slider("Difficulty", options=["easy", "medium", "hard"], value="medium")

    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem; color:#f0c060; text-align:center; letter-spacing:0.5px;">🎓 SDAIA · Applied AI Bootcamp</p>', unsafe_allow_html=True)
    st.caption("Uses LangChain YoutubeLoader · Docling ASR (Whisper) as local fallback")

# Main input
url = st.text_input("🔗 YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
generate_btn = st.button("⚡ Generate Study Materials", type="primary", use_container_width=True)

if generate_btn:
    if not api_key:
        st.error("Please enter your OpenRouter API key in the sidebar.")
    elif not url:
        st.error("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Could not parse a YouTube video ID. Please check the URL.")
        else:
            lang_code = "ar" if language == "Arabic / عربي" else "en"

            with st.spinner("Fetching transcript..."):
                try:
                    transcript, method = get_transcript(url, video_id, lang_code)
                    st.session_state.transcript = transcript
                    st.session_state.method = method
                    if language == "Auto-detect":
                        lang_code = detect_language(transcript)
                        st.session_state.detected_lang = "Arabic 🇸🇦" if lang_code == "ar" else "English 🇬🇧"
                    else:
                        st.session_state.detected_lang = language
                except Exception as e:
                    st.error(f"Failed to get transcript: {e}")
                    st.stop()

            llm = build_llm(api_key, model)

            with st.spinner("Generating flashcards..."):
                try:
                    st.session_state.flashcards = generate_flashcards(
                        transcript, llm, n_cards, difficulty, lang_code
                    )
                except Exception as e:
                    st.error(f"Flashcard generation failed: {e}")
                    st.stop()

            with st.spinner("Generating quiz..."):
                try:
                    st.session_state.quiz = generate_quiz(
                        transcript, llm, n_quiz, difficulty, lang_code
                    )
                except Exception as e:
                    st.error(f"Quiz generation failed: {e}")
                    st.stop()

            st.session_state.card_idx = 0
            st.session_state.show_answer = False
            st.session_state.score = 0
            st.session_state.answered = 0
            st.session_state.quiz_answers = [None] * len(st.session_state.quiz)
            st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.flashcards:
    st.success(f"✅ Transcript via: **{st.session_state.method}** | Language: **{st.session_state.detected_lang}**")

    tab_flash, tab_quiz, tab_raw = st.tabs(["🃏 Flashcards", "🧠 Quiz", "📄 Transcript"])

    with tab_flash:
        cards = st.session_state.flashcards
        idx = st.session_state.card_idx
        card = cards[idx]
        is_arabic = st.session_state.detected_lang == "Arabic 🇸🇦"
        rtl = 'class="rtl-text"' if is_arabic else ''

        st.markdown(f"**Card {idx + 1} of {len(cards)}**")
        st.markdown(f"""
        <div class="card-front">
            <div class="tag">{"سؤال" if is_arabic else "Question"}</div>
            <div class="question-text" {rtl}>{card['question']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Reveal / Hide Answer", use_container_width=True):
            st.session_state.show_answer = not st.session_state.show_answer
            st.rerun()

        if st.session_state.show_answer:
            st.markdown(f"""
            <div class="card-back">
                <div class="tag">{"إجابة" if is_arabic else "Answer"}</div>
                <div {rtl}>{card['answer']}</div>
            </div>
            """, unsafe_allow_html=True)

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

    with tab_quiz:
        quiz = st.session_state.quiz
        is_arabic = st.session_state.detected_lang == "Arabic 🇸🇦"
        rtl = 'class="rtl-text"' if is_arabic else ''

        score = sum(
            1 for i, q in enumerate(quiz)
            if st.session_state.quiz_answers[i] == q["correct"]
        )
        answered = sum(1 for a in st.session_state.quiz_answers if a is not None)

        col_s, col_p = st.columns(2)
        with col_s:
            st.markdown(f'<div class="score-display">{score} / {answered}</div>', unsafe_allow_html=True)
            st.caption("Score")
        with col_p:
            if answered > 0:
                st.metric("Accuracy", f"{round(score/answered*100)}%")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        for i, q in enumerate(quiz):
            st.markdown(f'<div {rtl}><b>Q{i+1}.</b> {q["question"]}</div>', unsafe_allow_html=True)
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
                        st.markdown(f'<div class="correct-option" {rtl}>✅ {label}</div>', unsafe_allow_html=True)
                    elif j == chosen and chosen != q["correct"]:
                        st.markdown(f'<div class="wrong-option" {rtl}>❌ {label}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="padding:0.6rem 1rem;margin:0.3rem 0;border-radius:8px;border:1px solid #2e3a4a;color:#888" {rtl}>{label}</div>', unsafe_allow_html=True)

            if chosen is not None:
                st.markdown(f'<div class="explanation-box" {rtl}>💡 {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")

        answered_all = all(a is not None for a in st.session_state.quiz_answers)
        if answered_all:
            st.balloons()
            final_pct = round(score / len(quiz) * 100)
            if final_pct == 100:
                st.success(f"🏆 Perfect! {score}/{len(quiz)}")
            elif final_pct >= 70:
                st.success(f"🎉 Great job! {score}/{len(quiz)} ({final_pct}%)")
            else:
                st.warning(f"📖 Keep studying! {score}/{len(quiz)} ({final_pct}%)")

        if st.button("🔁 Reset Quiz", use_container_width=True):
            st.session_state.quiz_answers = [None] * len(quiz)
            st.rerun()

    with tab_raw:
        st.caption(f"Source: {st.session_state.method}")
        st.text_area("Raw transcript", value=st.session_state.transcript, height=400, disabled=True)
        st.caption(f"{len(st.session_state.transcript.split()):,} words")