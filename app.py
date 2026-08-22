"""
Asisten AI Hukum Pertanahan (ATR/BPN & Ombudsman)
- Database RAG dari file lokal: database_rag_masykur.json.gz
- Pencarian semantik (SentenceTransformer) + jawaban via Gemini
- UI Streamlit
"""
import os
import json
import gzip
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from google.genai import Client

# ===================== KONFIGURASI =====================
DB_PATH = os.environ.get("RAG_DB_PATH", "database_rag_masykur.json.gz")

st.set_page_config(page_title="AI Hukum Pertanahan", page_icon="🤖", layout="centered")
st.title("🤖 Asisten AI Hukum Pertanahan")
st.write("Sumber Data: Database RAG JDIH ATR/BPN & Ombudsman (file lokal)")


# ===================== LOAD DATABASE =====================
@st.cache_resource
def load_system():
    # --- Baca database dari file lokal (.gz) ---
    db_file = DB_PATH
    if not os.path.exists(db_file):
        st.error(
            f"File database tidak ditemukan di: {db_file}\n"
            "Pastikan 'database_rag_masykur.json.gz' sudah ada di folder repo "
            "(atau set env RAG_DB_PATH ke path-nya)."
        )
        st.stop()

    try:
        if db_file.endswith(".gz"):
            with gzip.open(db_file, "rt", encoding="utf-8") as f:
                database_rag = json.load(f)
        else:
            with open(db_file, "r", encoding="utf-8") as f:
                database_rag = json.load(f)
        st.success(f"Database dimuat: {len(database_rag):,} dokumen")
    except Exception as e:
        st.error(f"Gagal memuat database. Pastikan file adalah JSON valid. Detail: {e}")
        st.stop()

    # --- Memuat model vektor ---
    try:
        model_vektor = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Gagal memuat model vektor AI. Detail Error: {e}")
        st.stop()

    # --- Membuat matriks vektor (pencarian semantik) ---
    semua_teks = [item["text_for_embedding"] for item in database_rag]
    with st.spinner("Membuat indeks vektor pencarian..."):
        matriks_vektor = model_vektor.encode(semua_teks, show_progress_bar=False)

    return database_rag, model_vektor, matriks_vektor


# ===================== INIT (jalankan sekali) =====================
try:
    database_rag, model_vektor, matriks_vektor = load_system()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses database. Error: {e}")
    st.stop()

# ===================== API KEY GEMINI =====================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    client = Client(api_key=GEMINI_API_KEY)
except Exception:
    st.error("API Key Gemini belum terpasang di menu Secrets Streamlit Cloud. Silakan isi terlebih dahulu.")
    st.stop()

# ===================== FUNGSI RAG =====================
def cari_pasal(pertanyaan_user, jumlah_target=3):
    vektor_pertanyaan = model_vektor.encode([pertanyaan_user])
    skor = np.dot(matriks_vektor, vektor_pertanyaan) / (
        np.linalg.norm(matriks_vektor, axis=1) * np.linalg.norm(vektor_pertanyaan)
    )
    idx_terbaik = np.argsort(skor)[::-1][:jumlah_target]

    konteks = []
    for idx in idx_terbaik:
        item = database_rag[idx]
        meta = item["metadata"]
        info = (
            f"[{meta.get('jenis_aturan','')} No. {meta.get('nomor','')} "
            f"Tahun {meta.get('tahun','')} Pasal {meta.get('pasal','')} "
            f"Ayat ({meta.get('ayat','')}) tentang {meta.get('tentang','')}]"
        )
        konteks.append(f"Sumber: {info}\nIsi Aturan: {item['raw_text']}")
    return "\n\n".join(konteks)


# ===================== CHAT UI =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan aturan pertanahan atau pelayanan publik di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI sedang mencocokkan dokumen hukum..."):
            dokumen_pendukung = cari_pasal(prompt)

            instruksi_sistem = (
                "Anda adalah AI Ahli Hukum Pertanahan ATR/BPN yang cerdas dan "
                "hanya menjawab berdasarkan dokumen pendukung di bawah ini. "
                "Jika jawabannya tidak ada di dokumen, katakan jujur: "
                "'Maaf, informasi tersebut tidak ditemukan dalam database hukum saya.' "
                "Sebutkan nomor pasal, jenis aturan, dan tahun secara lengkap.\n\n"
                f"=== DOKUMEN PENDUKUNG (SUMBER SAH) ===\n{dokumen_pendukung}"
            )

            try:
                respon = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"system_instruction": instruksi_sistem, "temperature": 0.1},
                )
                answer = respon.text
            except Exception as e:
                answer = f"Maaf, server AI Gemini mengalami kendala. Detail: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
