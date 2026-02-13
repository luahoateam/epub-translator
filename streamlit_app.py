import hashlib
import re
import time
from io import BytesIO
from dataclasses import dataclass
from typing import List

import streamlit as st
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT

import google.generativeai as genai

# ── Config ───────────────────────────────────────────────────────────────
DEFAULT_GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")

HTML2MD_MODEL = "gemini-2.0-flash"
TRANSLATE_MODEL = "gemini-2.0-flash"

GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 8192,
}

SAFETY_SETTINGS = {
    "HATE": "BLOCK_NONE",
    "HARASSMENT": "BLOCK_NONE",
    "SEXUAL": "BLOCK_NONE",
    "DANGEROUS": "BLOCK_NONE",
}

HTML_TO_MD_PROMPT = """Convert the following HTML from an EPUB chapter into clean Markdown.

Rules:
- Preserve structure: headings, paragraphs, lists, blockquotes, tables, emphasis.
- Keep the reading order exactly.
- Remove navigation/boilerplate if it is clearly not chapter content.
- Do NOT add commentary. Output ONLY Markdown.
- If images exist, represent as Markdown image: ![alt](src).

HTML:
{html}
"""

MD_TRANSLATE_PROMPT = """Translate the following Markdown from English to Vietnamese.

Rules:
- Keep ALL Markdown structure unchanged (headings, lists, links, code fences, inline code).
- Do not change URLs.
- Do not add or remove blank lines.
- Output ONLY the translated Markdown.

Markdown:
{md}
"""


# ── Data ─────────────────────────────────────────────────────────────────
@dataclass
class Chapter:
    idx: int
    title: str
    html: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    body = soup.body if soup.body else soup
    for tag in body.find_all():
        if tag.name in ["span", "div"] and not tag.get_text(strip=True) and not tag.find(["img", "br"]):
            tag.decompose()
    return str(body)


def infer_title(html: str, fallback: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for h in soup.find_all(["h1", "h2", "h3"], limit=1):
        t = h.get_text(" ", strip=True)
        if t:
            return t
    return fallback


@st.cache_data(show_spinner=False)
def extract_chapters(epub_bytes: bytes) -> List[Chapter]:
    book = epub.read_epub(BytesIO(epub_bytes))
    items = list(book.get_items_of_type(ITEM_DOCUMENT))
    chapters: List[Chapter] = []
    for i, item in enumerate(items):
        raw = item.get_content()
        html = raw.decode("utf-8", errors="ignore")
        html = clean_html(html)
        fallback = re.sub(r"\.(xhtml|html)$", "", item.get_name() or f"chapter_{i+1}")
        title = infer_title(html, fallback)
        chapters.append(Chapter(idx=i, title=title, html=html))
    return chapters


@st.cache_resource(show_spinner=False)
def get_model(model_name: str):
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=GENERATION_CONFIG,
        safety_settings=SAFETY_SETTINGS,
    )


def gemini_call(model_name: str, prompt: str, max_attempts: int = 5) -> str:
    model = get_model(model_name)
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            resp = model.generate_content(prompt)
            text = (resp.text or "").strip()
            if not text:
                raise RuntimeError("Empty response from Gemini")
            return text
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Unreachable")


def html_to_markdown(html: str) -> str:
    return gemini_call(HTML2MD_MODEL, HTML_TO_MD_PROMPT.format(html=html))


def translate_markdown(md: str) -> str:
    return gemini_call(TRANSLATE_MODEL, MD_TRANSLATE_PROMPT.format(md=md))


def cache_key(ch: Chapter, prefix: str) -> str:
    h = hashlib.sha256(ch.html.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}:{ch.idx}:{h}"


# ── Streamlit UI ─────────────────────────────────────────────────────────
st.set_page_config(page_title="EPUB → MD → Vietnamese", layout="wide")
st.title("📖 EPUB → Markdown → Vietnamese Translator")

with st.sidebar:
    st.header("⚙️ Cài đặt")
    api_key = st.text_input("Gemini API Key", value=DEFAULT_GEMINI_KEY, type="password")
    st.caption(f"Model: `{HTML2MD_MODEL}`")

if not api_key:
    st.warning("Vui lòng nhập Gemini API Key")
    st.stop()

genai.configure(api_key=api_key)

uploaded = st.file_uploader("📁 Upload file EPUB", type=["epub"])
if not uploaded:
    st.info("Upload một file EPUB để bắt đầu.")
    st.stop()

epub_bytes = uploaded.read()
epub_hash = sha256_bytes(epub_bytes)

if "epub_hash" not in st.session_state or st.session_state.epub_hash != epub_hash:
    st.session_state.epub_hash = epub_hash
    st.session_state.md_cache = {}
    st.session_state.vi_cache = {}

chapters = extract_chapters(epub_bytes)
st.success(f"Đã trích xuất **{len(chapters)}** chapter từ EPUB.")

# ── Step 1: Chọn chapters ───────────────────────────────────────────────
st.subheader("1️⃣ Chọn chapters để xử lý")
labels = [f"{c.idx+1:03d} — {c.title}" for c in chapters]
selected_labels = st.multiselect("Chapters", options=labels, default=labels[:min(5, len(labels))])
selected = [chapters[labels.index(lbl)] for lbl in selected_labels]

if not selected:
    st.stop()

# ── Step 2 & 3: Convert & Translate ─────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("2️⃣ HTML → Markdown")
    if st.button("🔄 Chuyển đổi sang Markdown", type="primary"):
        prog = st.progress(0, text="Đang chuyển đổi...")
        for i, ch in enumerate(selected):
            key = cache_key(ch, "md")
            if key not in st.session_state.md_cache:
                with st.spinner(f"Converting: {ch.title[:40]}..."):
                    st.session_state.md_cache[key] = html_to_markdown(ch.html)
            prog.progress((i + 1) / len(selected), text=f"{i+1}/{len(selected)}")
        st.success("✅ Hoàn tất chuyển đổi Markdown!")

with col2:
    st.subheader("3️⃣ Markdown → Tiếng Việt")
    if st.button("🇻🇳 Dịch sang Tiếng Việt"):
        prog = st.progress(0, text="Đang dịch...")
        for i, ch in enumerate(selected):
            md_key = cache_key(ch, "md")
            vi_key = cache_key(ch, "vi")
            md = st.session_state.md_cache.get(md_key)
            if not md:
                st.warning(f"Chưa có Markdown cho: {ch.title} — hãy chuyển đổi trước!")
                continue
            if vi_key not in st.session_state.vi_cache:
                with st.spinner(f"Translating: {ch.title[:40]}..."):
                    st.session_state.vi_cache[vi_key] = translate_markdown(md)
            prog.progress((i + 1) / len(selected), text=f"{i+1}/{len(selected)}")
        st.success("✅ Hoàn tất dịch thuật!")

# ── Step 4: Review ───────────────────────────────────────────────────────
st.subheader("4️⃣ Xem kết quả")
ch_preview = st.selectbox("Chọn chapter để xem", selected, format_func=lambda x: f"{x.idx+1:03d} — {x.title}")

if ch_preview:
    md_key = cache_key(ch_preview, "md")
    vi_key = cache_key(ch_preview, "vi")

    tab_html, tab_md, tab_vi = st.tabs(["📄 HTML gốc", "📝 Markdown", "🇻🇳 Tiếng Việt"])

    with tab_html:
        st.code(ch_preview.html[:3000], language="html")
        if len(ch_preview.html) > 3000:
            st.caption(f"(Hiển thị 3000/{len(ch_preview.html)} ký tự)")

    with tab_md:
        md_val = st.session_state.md_cache.get(md_key, "")
        if md_val:
            st.markdown(md_val)
            with st.expander("Xem/Sửa source Markdown"):
                new_md = st.text_area("Markdown", value=md_val, height=300, key=f"edit_{md_key}")
                if new_md != md_val:
                    st.session_state.md_cache[md_key] = new_md
                    if vi_key in st.session_state.vi_cache:
                        del st.session_state.vi_cache[vi_key]
                    st.info("Đã cập nhật Markdown. Bản dịch cũ đã bị xóa, hãy dịch lại.")
        else:
            st.info("Chưa chuyển đổi. Nhấn nút 'Chuyển đổi sang Markdown' ở trên.")

    with tab_vi:
        vi_val = st.session_state.vi_cache.get(vi_key, "")
        if vi_val:
            st.markdown(vi_val)
            with st.expander("Xem source Markdown tiếng Việt"):
                st.code(vi_val, language="markdown")
        else:
            st.info("Chưa dịch. Nhấn nút 'Dịch sang Tiếng Việt' ở trên.")

# ── Step 5: Download ─────────────────────────────────────────────────────
st.subheader("5️⃣ Tải xuống")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    md_parts = []
    for ch in selected:
        md = st.session_state.md_cache.get(cache_key(ch, "md"))
        if md:
            md_parts.append(f"# {ch.title}\n\n{md}")
    if md_parts:
        md_combined = "\n\n---\n\n".join(md_parts)
        st.download_button(
            "📥 Tải Markdown (English)",
            data=md_combined.encode("utf-8"),
            file_name=re.sub(r"[^a-zA-Z0-9_\-]+", "_", uploaded.name) + "_en.md",
            mime="text/markdown",
        )

with col_dl2:
    vi_parts = []
    for ch in selected:
        vi = st.session_state.vi_cache.get(cache_key(ch, "vi"))
        if vi:
            vi_parts.append(f"# {ch.title}\n\n{vi}")
    if vi_parts:
        vi_combined = "\n\n---\n\n".join(vi_parts)
        st.download_button(
            "📥 Tải Markdown (Tiếng Việt)",
            data=vi_combined.encode("utf-8"),
            file_name=re.sub(r"[^a-zA-Z0-9_\-]+", "_", uploaded.name) + "_vi.md",
            mime="text/markdown",
        )
    missing = len(selected) - len(vi_parts)
    if missing > 0:
        st.caption(f"⚠️ {missing} chapter chưa được dịch.")
