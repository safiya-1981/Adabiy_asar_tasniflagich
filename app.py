# app.py
import os, re, csv, json, datetime, joblib, numpy as np
import streamlit as st
from pathlib import Path

from utils import (
    load_text_from_upload, transliterate_to_latin,
    predict_text_grade, conf_comment, has_cyrillic
)

# --- Sozlamalar ---
DEFAULT_CORPUS_DIR = r"D:\2025-2026-oquv yili\YANGIDAN_DISSERTATSIYA YOZDIM\matn_tahlili\darsliklar"
LIBRARY_DIR = "library"
ADMIN_PIN_DEFAULT = "1234"

st.set_page_config(
    page_title="🇺🇿 Adabiy Matn Tavsiyachi (5–9-sinf)",
    page_icon="🇺🇿",
    layout="centered"
)

# --- Session kalitlari (reset uchun) ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# --- Flag chizig'i ---
flag_css = """
<style>
.flagbar {height:18px; width:100%; margin:-16px 0 8px 0;}
.flagbar .b{background:#1EB1E7; height:6px;}
.flagbar .r1{background:#CE1126; height:2px;}
.flagbar .w{background:#FFFFFF; height:6px;}
.flagbar .r2{background:#CE1126; height:2px;}
.flagbar .g{background:#1EB53A; height:6px;}
</style>
<div class="flagbar">
  <div class="b"></div><div class="r1"></div><div class="w"></div><div class="r2"></div><div class="g"></div>
</div>
"""
st.markdown(flag_css, unsafe_allow_html=True)

st.title("📘 Adabiy Matn Tavsiyachi — 5–9-sinflar")
st.caption("O‘quvchilarning intellektual salohiyatiga mos **adabiy matn/asarning** darajasini taxmin qilish (TF-IDF + Logistic Regression, chunklash).")

# --- Model artefaktlarini yuklash ---
def load_models(models_dir: str = "models"):
    vpath = os.path.join(models_dir, "vectorizer.pkl")
    lpath = os.path.join(models_dir, "logreg.pkl")
    jpath = os.path.join(models_dir, "labels.json")
    if not (os.path.exists(vpath) and os.path.exists(lpath) and os.path.exists(jpath)):
        return None, None, None
    vec = joblib.load(vpath)
    lr = joblib.load(lpath)
    with open(jpath, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return vec, lr, labels

VEC, LR, LABELS = load_models()

# --- Barcha sinflar bo'yicha ehtimolliklarni hisoblash ---
_TOKEN_RE = re.compile(r"\w+", flags=re.U)

def _basic_clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _to_tokens(s: str):
    return _TOKEN_RE.findall(s)

def _make_chunks(tokens, size=400, stride=200):
    out = []
    for i in range(0, max(len(tokens) - size + 1, 0), stride):
        block = tokens[i:i + size]
        if len(block) == size:
            out.append(" ".join(block))
    return out

def get_all_class_probs(text: str, vec, lr, labels, size=400, stride=200):
    if not (vec and lr and labels) or not text.strip():
        return [], 0
    toks = _to_tokens(_basic_clean(text))
    chunks = _make_chunks(toks, size=size, stride=stride)
    if not chunks:
        return [], 0
    Xv = vec.transform(chunks)
    P = lr.predict_proba(Xv)              # (n_chunks, n_classes)
    p_mean = np.asarray(P).mean(axis=0)   # (n_classes,)
    pairs = list(zip(lr.classes_, p_mean))
    name_map = {c: c for c in lr.classes_}
    pairs_named = [(name_map.get(c, str(c)), float(p)) for c, p in pairs]
    pairs_named.sort(key=lambda x: x[1], reverse=True)
    return pairs_named, len(chunks)

# --- Sidebar: rejimlar, admin va tozalash ---
with st.sidebar:
    st.header("⚙️ Rejim va sozlamalar")
    mode = st.radio("Baholash usuli:", ["Matn yozish", "Fayl yuklash", "Kutubxonadan tanlash"])
    auto_translit = st.checkbox("Kirilni avtomatik lotinga o‘girish", value=True)
    st.markdown("---")

    st.subheader("👩‍🏫 O‘qituvchi")
    admin_pin_secret = st.secrets.get("ADMIN_PIN", ADMIN_PIN_DEFAULT)
    pin = st.text_input("PIN", type="password", placeholder="••••")
    is_admin = (pin == str(admin_pin_secret))

    if is_admin:
        chunk_tokens = st.number_input("Chunk (so‘z)", 100, 1000, 400, 50)
        stride_tokens = st.number_input("Stride (overlap)", 50, 800, 200, 50)
        st.caption("Tavsiya: 400 / 200")
        st.markdown("**Model statusi:** " + ("✅ Yuklangan" if VEC else "❗ Topilmadi"))
        if st.button("🛠 Modelni qayta o‘qitish (korpusdan)"):
            from train_and_save import train_and_save
            try:
                with st.spinner("Trening..."):
                    info = train_and_save(DEFAULT_CORPUS_DIR, models_dir="models")
                st.success(f"Model yangilandi. Namuna soni: {info['n_samples']}.")
                st.json(info)
                VEC, LR, LABELS = load_models()
            except Exception as e:
                st.error(f"Treningda xato: {e}")
    else:
        chunk_tokens, stride_tokens = 400, 200
        st.caption("Talaba rejimi (faqat baholash).")

    st.markdown("---")
    if st.button("🧹 Yangi tekshiruv / Tozalash"):
        for k in ["input_text", "src_text", "selected_fname", "all_probs", "last_result"]:
            st.session_state.pop(k, None)
        st.session_state.uploader_key += 1
        st.experimental_rerun()

if VEC is None:
    st.warning("Model topilmadi. O‘qituvchi PIN’i bilan 'Modelni qayta o‘qitish' tugmasidan foydalaning yoki `models/`ga artefaktlarni joylang.")

# --- Kiruvchi matn ---
src_text = ""

if mode == "Matn yozish":
    st.subheader("✍️ Matn yozish")
    txt = st.text_area(
        "Matn",
        height=220,
        placeholder="Matnni kiriting...",
        key="input_text",
        value=st.session_state.get("input_text", "")
    )
    if txt and txt.strip():
        src_text = txt.strip()

elif mode == "Fayl yuklash":
    st.subheader("📄 Fayl yuklash (.txt, .pdf, .docx)")
    up = st.file_uploader(
        "Faylni tanlang",
        type=["txt", "pdf", "docx"],
        key=f"up_{st.session_state.uploader_key}"
    )
    if up is not None:
        try:
            src_text = load_text_from_upload(up)
            # Rasmli/bo‘sh PDF holatini ushlab qolish
            if not src_text or not src_text.strip():
                st.error(
                    "Bu fayldan matn olinmadi. Ehtimol, PDF skaner qilingan (rasmlar). "
                    "OCR orqali matnga aylantirib yoki DOCX/TXT fayl yuklang."
                )
                st.stop()
        except Exception as e:
            st.error(f"Fayl o‘qishda xato: {e}")
            st.stop()

else:
    st.subheader("📚 Kutubxonadan tanlash")
    if not os.path.isdir(LIBRARY_DIR):
        st.info("Kutubxona topilmadi. `library/5 ... library/9` ichiga .txt fayllar qo‘ying.")
    else:
        sinf = st.selectbox("Sinf:", ["5", "6", "7", "8", "9"])
        folder = os.path.join(LIBRARY_DIR, sinf)
        files = [f for f in os.listdir(folder)] if os.path.isdir(folder) else []
        files = [f for f in files if f.lower().endswith(".txt")]
        if not files:
            st.info(f"`{folder}` ichida .txt topilmadi.")
        else:
            fname = st.selectbox("Matn fayli:", files)
            if st.button("Fayldan matnni yuklash"):
                try:
                    with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                        src_text = f.read()
                    st.session_state.input_text = ""
                    st.session_state.uploader_key += 1
                except Exception as e:
                    st.error(f"Fayl o‘qishda xato: {e}")

# --- Avto-transliteratsiya (xabar faqat haqiqatan o'zgarganda) ---
if src_text and auto_translit and has_cyrillic(src_text):
    converted = transliterate_to_latin(src_text)
    if converted != src_text:
        src_text = converted
        st.info("Kiril matn aniqlandi va **lotinga o‘girildi**.")

# --- Baholash va natija ---
st.markdown("### ✅ Baholash")
if not src_text:
    st.info("Avval matn kiriting/yuklang yoki kutubxonadan tanlang.")
else:
    disabled = (VEC is None)
    if st.button("Sinfni aniqlash", disabled=disabled):

        # ⛔ Oldindan tekshiruv: chunk chiqadimi?
        toks = _to_tokens(_basic_clean(src_text))
        chunks_preview = _make_chunks(toks, size=chunk_tokens, stride=stride_tokens)
        if not chunks_preview:
            st.error(
                "Matndan bo‘lak (chunk) hosil bo‘lmadi. "
                "Matn juda qisqa yoki PDF skaner (rasm) ko‘rinishida. "
                "Uzoqroq matn kiriting yoki faylni OCR orqali matnga aylantiring."
            )
            st.stop()

        with st.spinner("Hisoblanmoqda..."):
            out = predict_text_grade(src_text, VEC, LR, LABELS, chunk_tokens, stride_tokens)
            all_probs, n_chunks_full = get_all_class_probs(
                src_text, VEC, LR, LABELS, size=chunk_tokens, stride=stride_tokens
            )

        # Sarlavha: Top-1
        top_label = all_probs[0][0] if all_probs else out.get("label", "—")
        st.subheader(f"Eng mos sinf: **{top_label}**")

        # Progress bar: eng katta proba
        if all_probs:
            st.progress(min(1.0, max(0.0, all_probs[0][1])))

        # Barcha sinflar bo‘yicha ehtimolliklar
        st.markdown("### 📊 Har bir sinf uchun tavsiya darajasi:")
        if all_probs:
            for idx, (lab, p) in enumerate(all_probs):
                box = "🟩" if idx == 0 else "🟥"
                st.write(f"{box} **{lab}** — {p:.2f}")
        else:
            st.info("Ehtimolliklarni hisoblashning imkoni bo‘lmadi.")

        # Xizmat ko‘rsatkichlari
        nch = out.get("n_chunks", n_chunks_full)
        st.caption(f"Chunk: {chunk_tokens}, stride: {stride_tokens} — bo‘laklar soni: {nch}.")

# --- Fikr-mulohaza ---
st.markdown("---")
st.markdown("#### Fikr-mulohaza")
fb = st.text_area("Dastur haqida fikringiz (xatolar, takliflar)...", height=150)
if st.button("✉️ Fikrni yuborish"):
    Path("feedback").mkdir(exist_ok=True)
    with open("feedback/feedback.csv", "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.datetime.now().isoformat(), fb.replace("\n", " ")])
    st.success("Rahmat! Fikringiz saqlandi (feedback/feedback.csv).")

# --- Muallif bloki (footer) ---
st.markdown("---")
st.markdown(
    """
<div style="text-align:center; line-height:1.6;">
  <b>Dastur muallifi:</b> Sattarova Sapura Beknazarovna<br/>
  <i>Al-Beruniy nomidagi Urganch davlat universiteti</i><br/>
  "Kompyuter ilmlari va Sun’iy intellekt texnologiyalari" kafedrasi<br/><br/>
  Model: <b>TF-IDF + Logistik Regressiya</b> — Cross-Validation: <b>85%</b><br/>
  © 2025 — <b>Adabiy Matn Tavsiyachi</b>
</div>
    """,
    unsafe_allow_html=True,
)
