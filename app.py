import streamlit as st
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from google.genai import Client

# 1. Konfigurasi Tampilan Halaman Web Utama
st.set_page_config(page_title="AI Hukum Pertanahan", page_icon="🤖", layout="centered")
st.title("🤖 Asisten AI Hukum Pertanahan")
st.write("Sumber Data: Database RAG JDIH ATR/BPN & Ombudsman (Terhubung Langsung ke Cloud)")

# 2. Fungsi Load Data dari Google Drive Menggunakan ID File Anda
@st.cache_resource
def load_system():
    # Menggunakan ID file dari tautan Google Drive publik Anda
    FILE_ID = "1kDVc28gwX4Da7lo7jCR2M4QLSWwFuJne"
    url_download = f"https://google.com{FILE_ID}"
    
    # Menarik file JSON 112 MB dari Drive langsung ke memori server Streamlit
    response = requests.get(url_download)
    if response.status_code == 200:
        database_rag = response.json()
    else:
        st.error("Gagal mendownload database dari Google Drive. Pastikan hak akses file Anda diatur ke 'Anyone with the link'.")
        st.stop()
        
    # Memuat Model Vektor
    model_vektor = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Membuat matriks vektor untuk pencarian semantik berdasarkan makna kata
    semua_teks_embedding = [item["text_for_embedding"] for item in database_rag]
    matriks_vektor = model_vektor.encode(semua_teks_embedding, show_progress_bar=False)
    
    return database_rag, model_vektor, matriks_vektor

# Menjalankan pemuatan database hukum pertama kali
try:
    database_rag, model_vektor, matriks_vektor = load_system()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses database. Error: {e}")
    st.stop()

# 3. Inisialisasi API Key Gemini Keamanan Tinggi
# Menggunakan st.secrets agar kunci rahasia Anda tidak bocor ke publik GitHub
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
client = Client(api_key=GEMINI_API_KEY)

# 4. Fungsi Pencarian RAG (Legal-Aware Semantic Search)
def cari_pasal(pertanyaan_user, jumlah_target=3):
    vektor_pertanyaan = model_vektor.encode([pertanyaan_user])
    skor_kemiripan = np.dot(matriks_vektor, vektor_pertanyaan) / (
        np.linalg.norm(matriks_vektor, axis=1) * np.linalg.norm(vektor_pertanyaan)
    )
    indeks_terbaik = np.argsort(skor_kemiripan)[::-1][:jumlah_target]
    
    konteks_terpilih = []
    for idx in indeks_terbaik:
        item = database_rag[idx]
        meta = item["metadata"]
        info_sumber = f"[{meta['jenis_aturan']} No. {meta['nomor']} Tahun {meta['tahun']} Pasal {meta['pasal']} Ayat ({meta['ayat']}) tentang {meta['tentang']}]"
        konteks_terpilih.append(f"Sumber: {info_sumber}\nIsi Aturan: {item['raw_text']}")
    return "\n\n".join(konteks_terpilih)

# 5. Mengelola Sesi Riwayat Obrolan Chatbot (Memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan balon obrolan riwayat sebelumnya jika ada
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Area Kotak Ketik Chat Interaktif
if prompt := st.chat_input("Tanyakan aturan pertanahan atau pelayanan publik di sini..."):
    # Tampilkan chat ketikan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Proses AI RAG mencocokkan dokumen hukum
    with st.chat_message("assistant"):
        with st.spinner("AI sedang mencocokkan dokumen hukum dari Google Drive..."):
            dokumen_pendukung = cari_pasal(prompt)
            
            instruksi_sistem = (
                "Anda adalah AI Ahli Hukum Pertanahan ATR/BPN yang cerdas dan kaku mengikuti dokumen. "
                "Tugas Anda adalah menjawab pertanyaan user HANYA DAN HARUS berdasarkan dokumen pendukung yang diberikan di bawah ini. "
                "Jika jawabannya tidak ada di dalam dokumen pendukung tersebut, katakan sejujurnya: 'Maaf, informasi tersebut tidak ditemukan dalam database hukum saya.' "
                "Sebutkan nomor pasal, jenis aturan, dan tahun secara lengkap di dalam atau di akhir jawaban Anda.\n\n"
                f"=== DOKUMEN PENDUKUNG (SUMBER SAH) ===\n{dokumen_pendukung}"
            )
            
            respon = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={'system_instruction': instruksi_sistem, 'temperature': 0.1}
            )
            
            answer = respon.text
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
