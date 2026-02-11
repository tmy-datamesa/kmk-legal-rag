import streamlit as st
import time
from src.ingestion import ingest_data
from src.rag import LegalRAG
from src import config

# Sayfa Ayarları
st.set_page_config(page_title="Legal-RAG v2", page_icon="⚖️", layout="centered")
st.title("⚖️ Kat Mülkiyeti Kanunu Asistanı")

# --- 1. Ayarlar (Dinamik Sidebar) ---
st.sidebar.header("⚙️ Ayarlar")

# Config'den provider listesini çek ({"local": "Local...", "openai": "OpenAI..."})
provider_options = {k: v["label"] for k, v in config.EMBEDDING_PROVIDERS.items()}
selected_label = st.sidebar.selectbox(
    "Embedding Modeli:",
    options=list(provider_options.values())
)

# Label'dan Key'i bul (Ters işlem)
provider_code = next(k for k, v in provider_options.items() if v == selected_label)

# Provider değişirse sistemi resetle
if "current_provider" in st.session_state and st.session_state.current_provider != provider_code:
    st.session_state.pop("rag_system", None)
st.session_state.current_provider = provider_code

# --- 2. Başlatma ---
if "rag_system" not in st.session_state:
    with st.spinner(f"Sistem hazırlanıyor... ({selected_label})"):
        try:
            # Sadece local ise otomatik ingest dene, cloud ise veri var varsay
            # (veya her durumda ingest_data check ettiği için çağırabiliriz)
            ingest_data(force_recreate=False, provider=provider_code)
            
            st.session_state.rag_system = LegalRAG(provider=provider_code)
            st.success("Sistem Hazır!")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Sistem başlatılamadı: {e}")
            st.stop()

# --- 3. Sohbet ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Sorunuzu bekliyorum."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("İnceleniyor..."):
            try:
                cevap, kaynaklar = st.session_state.rag_system.generate_answer(prompt)
                st.markdown(cevap)
                with st.expander("📚 Kaynaklar"):
                    for i, doc in enumerate(kaynaklar):
                        st.markdown(f"**Kaynak {i+1}:**\n> {doc[:200]}...")
                st.session_state.messages.append({"role": "assistant", "content": cevap})
            except Exception as e:
                st.error(f"Hata: {e}")
