
import os, re, tempfile
import numpy as np
from typing import List

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    from transliterate import translit
except Exception:
    translit = None


def basic_clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_tokens(s: str) -> List[str]:
    return re.findall(r"\w+", s, flags=re.U)


def make_chunks(tokens: List[str], size: int = 400, stride: int = 200) -> List[str]:
    out = []
    for i in range(0, max(len(tokens) - size + 1, 0), stride):
        block = tokens[i:i + size]
        if len(block) == size:
            out.append(" ".join(block))
    if not out and len(tokens) > 0:
        out = [" ".join(tokens)]
    return out


def has_cyrillic(text: str) -> bool:
    return bool(re.search(r'[\u0400-\u04FF]', text))


def transliterate_to_latin(text: str) -> str:
    if translit is None:
        return text
    try:
        return translit(text, 'ru', reversed=True)
    except Exception:
        return text


def load_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 o‘rnatilmagan: PDF o‘qib bo‘lmadi.")
        reader = PyPDF2.PdfReader(uploaded_file)
        txt = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t:
                txt.append(t)
        return "\n".join(txt)
    elif name.endswith(".docx") or name.endswith(".doc"):
        if docx2txt is None:
            raise RuntimeError("docx2txt o‘rnatilmagan: DOC/DOCX o‘qib bo‘lmadi.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            path = tmp.name
        try:
            return docx2txt.process(path) or ""
        finally:
            try: os.unlink(path)
            except Exception: pass
    else:
        raise RuntimeError("Qo‘llab-quvvatlanadigan formatlar: .txt, .pdf, .docx, .doc")


def predict_text_grade(text: str, VEC, LR, LABELS, chunk_tokens=400, stride_tokens=200):
    txt = basic_clean(text)
    toks = to_tokens(txt)
    chunks = make_chunks(toks, chunk_tokens, stride_tokens)
    Xq = VEC.transform(chunks)

    if hasattr(LR, "predict_proba"):
        proba = LR.predict_proba(Xq)
        mean_proba = proba.mean(axis=0)
        best_idx = int(np.argmax(mean_proba))
        best_lab = LABELS[best_idx]
        best_p = float(mean_proba[best_idx])
        order = np.argsort(-mean_proba)[:3]
        top3 = [(LABELS[i], float(mean_proba[i])) for i in order]
        return dict(label=best_lab, confidence=best_p, top3=top3, n_chunks=len(chunks))
    else:
        preds = LR.predict(Xq)
        vals, cnts = np.unique(preds, return_counts=True)
        best_lab = vals[np.argmax(cnts)]
        return dict(label=best_lab, confidence=None, top3=None, n_chunks=len(chunks))


def conf_comment(p: float) -> str:
    if p >= 0.85:   return "Yuqori ishonch"
    if p >= 0.70:   return "O‘rtacha ishonch"
    return "Past ishonch — ko‘proq matn kiriting yoki tekshiring."
