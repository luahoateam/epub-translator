import hashlib
import math
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
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

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

PDF_TO_MD_PROMPT = """Convert this PDF page into clean Markdown.

Rules:
- Preserve structure: headings, paragraphs, lists, blockquotes, tables, emphasis.
- Keep the reading order exactly.
- Remove headers/footers/page numbers if they are clearly not content.
- Do NOT add commentary. Output ONLY Markdown.
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
    content_type: str = "html"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    body = soup.body if soup.body else soup
    for tag in body.find_all():
        if (
            tag.name in ["span", "div"]
            and not tag.get_text(strip=True)
            and not tag.find(["img", "br"])
        ):
            tag.decompose()
    return str(body)


def infer_title(html: str, fallback: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for h in soup.find_all(["h1", "h2", "h3"], limit=1):
        t = h.get_text(" ", strip=True)
        if t:
            return t
    return fallback


# ── EPUB extraction ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_chapters_epub(file_bytes: bytes) -> List[Chapter]:
    book = epub.read_epub(BytesIO(file_bytes))
    items = list(book.get_items_of_type(ITEM_DOCUMENT))
    chapters: List[Chapter] = []
    for i, item in enumerate(items):
        raw = item.get_content()
        html = raw.decode("utf-8", errors="ignore")
        html = clean_html(html)
        fallback = re.sub(r"\.(xhtml|html)$", "", item.get_name() or f"chapter_{i + 1}")
        title = infer_title(html, fallback)
        chapters.append(Chapter(idx=i, title=title, html=html, content_type="html"))
    return chapters


# ── PDF extraction ───────────────────────────────────────────────────────
def estimate_pdf_pages(file_bytes: bytes) -> int:
    content = file_bytes.decode("latin-1", errors="ignore")
    matches = re.findall(r"/Count\s+(\d+)", content)
    if matches:
        return max(int(m) for m in matches)
    page_count = content.count("/Page") - content.count("/Pages")
    return max(1, page_count)


@st.cache_data(show_spinner=False)
def extract_chapters_pdf(file_bytes: bytes, pages_per_chunk: int = 5) -> List[Chapter]:
    total_pages = estimate_pdf_pages(file_bytes)
    num_chunks = max(1, math.ceil(total_pages / pages_per_chunk))
    chapters: List[Chapter] = []
    for i in range(num_chunks):
        start = i * pages_per_chunk + 1
        end = min((i + 1) * pages_per_chunk, total_pages)
        title = f"Pages {start}–{end}"
        chapters.append(
            Chapter(
                idx=i,
                title=title,
                html=f"pages:{start}-{end}",
                content_type="pdf",
            )
        )
    return chapters


# ── Gemini calls ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(model_name: str):
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=GENERATION_CONFIG,
        safety_settings=SAFETY_SETTINGS,
    )


def gemini_call(model_name: str, content, max_attempts: int = 5) -> str:
    model = get_model(model_name)
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            resp = model.generate_content(content)
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


def html_to_markdown(html: str, model_name: str) -> str:
    return gemini_call(model_name, HTML_TO_MD_PROMPT.format(html=html))


def pdf_pages_to_markdown(
    pdf_bytes: bytes, start_page: int, end_page: int, model_name: str
) -> str:
    pdf_part = {"mime_type": "application/pdf", "data": pdf_bytes}
    prompt = (
        PDF_TO_MD_PROMPT
        + f"\n\nProcess only pages {start_page} to {end_page}. Output combined Markdown."
    )
    return gemini_call(model_name, [pdf_part, prompt])


def translate_markdown(md: str, model_name: str) -> str:
    return gemini_call(model_name, MD_TRANSLATE_PROMPT.format(md=md))


def cache_key(ch: Chapter, prefix: str) -> str:
    h = hashlib.sha256(ch.html.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}:{ch.idx}:{h}"


# ── Streamlit UI ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Book → MD → Vietnamese", layout="wide")
st.title("📖 Book → Markdown → Vietnamese Translator")

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")

    default_key = st.secrets.get("GEMINI_API_KEY", "")
    if default_key:
        st.success("🔑 API Key đã được cấu hình")
        use_custom_key = st.checkbox("Dùng API Key khác")
        if use_custom_key:
            api_key = st.text_input("Gemini API Key", type="password")
        else:
            api_key = default_key
    else:
        api_key = st.text_input("Gemini API Key", type="password")

    st.divider()
    selected_model = st.selectbox("🤖 Model", options=GEMINI_MODELS, index=0)
    st.caption(
        "💡 `2.5-flash` thông minh nhất · `2.5-flash-lite` rẻ · `2.0-flash` nhanh"
    )

if not api_key:
    st.warning("⚠️ Vui lòng cung cấp Gemini API Key trong sidebar")
    st.stop()

genai.configure(api_key=api_key)

# ── Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📁 Upload file EPUB hoặc PDF", type=["epub", "pdf"])
if not uploaded:
    st.info("Upload một file EPUB hoặc PDF để bắt đầu.")
    st.stop()

file_bytes = uploaded.read()
file_hash = sha256_bytes(file_bytes)
file_ext = uploaded.name.rsplit(".", 1)[-1].lower()

if "file_hash" not in st.session_state or st.session_state.file_hash != file_hash:
    st.session_state.file_hash = file_hash
    st.session_state.file_bytes = file_bytes
    st.session_state.file_ext = file_ext
    st.session_state.md_cache = {}
    st.session_state.vi_cache = {}

# ── Extract chapters ─────────────────────────────────────────────────────
if file_ext == "epub":
    chapters = extract_chapters_epub(file_bytes)
    st.success(f"📚 Đã trích xuất **{len(chapters)}** chapter từ EPUB.")
elif file_ext == "pdf":
    with st.sidebar:
        pages_per_chunk = st.slider("📄 Số trang / chunk", 1, 20, 5)
    chapters = extract_chapters_pdf(file_bytes, pages_per_chunk)
    st.success(f"📄 PDF chia thành **{len(chapters)}** chunk để xử lý.")
else:
    st.error("Định dạng không được hỗ trợ.")
    st.stop()

# ── Step 1: Chọn chapters ───────────────────────────────────────────────
st.subheader("1️⃣ Chọn chapters / chunks để xử lý")
labels = [f"{c.idx + 1:03d} — {c.title}" for c in chapters]
selected_labels = st.multiselect(
    "Chapters", options=labels, default=labels[: min(5, len(labels))]
)
selected = [chapters[labels.index(lbl)] for lbl in selected_labels]

if not selected:
    st.stop()

# ── Step 2 & 3: Convert & Translate ─────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("2️⃣ Chuyển sang Markdown")
    if st.button("🔄 Chuyển đổi sang Markdown", type="primary"):
        prog = st.progress(0, text="Đang chuyển đổi...")
        for i, ch in enumerate(selected):
            key = cache_key(ch, "md")
            if key not in st.session_state.md_cache:
                with st.spinner(f"Converting: {ch.title[:40]}..."):
                    if ch.content_type == "pdf":
                        m = re.match(r"pages:(\d+)-(\d+)", ch.html)
                        start_p, end_p = int(m.group(1)), int(m.group(2))
                        st.session_state.md_cache[key] = pdf_pages_to_markdown(
                            st.session_state.file_bytes,
                            start_p,
                            end_p,
                            selected_model,
                        )
                    else:
                        st.session_state.md_cache[key] = html_to_markdown(
                            ch.html, selected_model
                        )
            prog.progress((i + 1) / len(selected), text=f"{i + 1}/{len(selected)}")
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
                    st.session_state.vi_cache[vi_key] = translate_markdown(
                        md, selected_model
                    )
            prog.progress((i + 1) / len(selected), text=f"{i + 1}/{len(selected)}")
        st.success("✅ Hoàn tất dịch thuật!")

# ── Step 4: Review ───────────────────────────────────────────────────────
st.subheader("4️⃣ Xem kết quả")
ch_preview = st.selectbox(
    "Chọn chapter để xem",
    selected,
    format_func=lambda x: f"{x.idx + 1:03d} — {x.title}",
)

if ch_preview:
    md_key = cache_key(ch_preview, "md")
    vi_key = cache_key(ch_preview, "vi")

    tab_src, tab_md, tab_vi = st.tabs(
        ["📄 Nội dung gốc", "📝 Markdown", "🇻🇳 Tiếng Việt"]
    )

    with tab_src:
        if ch_preview.content_type == "pdf":
            st.info(f"PDF chunk: {ch_preview.title}")
        else:
            st.code(ch_preview.html[:3000], language="html")
            if len(ch_preview.html) > 3000:
                st.caption(f"(Hiển thị 3000/{len(ch_preview.html)} ký tự)")

    with tab_md:
        md_val = st.session_state.md_cache.get(md_key, "")
        if md_val:
            st.markdown(md_val)
            with st.expander("Xem/Sửa source Markdown"):
                new_md = st.text_area(
                    "Markdown", value=md_val, height=300, key=f"edit_{md_key}"
                )
                if new_md != md_val:
                    st.session_state.md_cache[md_key] = new_md
                    if vi_key in st.session_state.vi_cache:
                        del st.session_state.vi_cache[vi_key]
                    st.info(
                        "Đã cập nhật Markdown. Bản dịch cũ đã bị xóa, hãy dịch lại."
                    )
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
base_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", uploaded.name.rsplit(".", 1)[0])

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
            file_name=f"{base_name}_en.md",
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
            file_name=f"{base_name}_vi.md",
            mime="text/markdown",
        )
    missing = len(selected) - len(vi_parts)
    if missing > 0:
        st.caption(f"⚠️ {missing} chapter chưa được dịch.")
