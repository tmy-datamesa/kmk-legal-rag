import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# ==============================================================================
# API VE İSTEMCİ AYARLARI
# ==============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Chroma Cloud Ayarları (Varsa Cloud, yoksa Local çalışır)
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT_ID = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE_NAME = os.getenv("CHROMA_DATABASE")

# ==============================================================================
# MODEL & RAG YAPILANDIRMASI
# ==============================================================================
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
TOP_K = 5

# ==============================================================================
# EMBEDDING (VEKTÖRLEŞTİRME) AYARLARI
# ==============================================================================
# Sadece OpenAI Embedding kullanılıyor (Cloud Native)
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
COLLECTION_NAME = "kat_mulkiyeti_v2"


# ==============================================================================
# DOSYA YOLLARI (PATHS)
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(PROJECT_ROOT, "data", "kat-mulkiyeti-kanunu.pdf")

# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================
SYSTEM_PROMPT = """
Sen Türkiye Cumhuriyeti Kat Mülkiyeti Kanunu konusunda uzman, yardımcı bir hukuk asistanısın.
Görev: Kullanıcının sorusunu SADECE aşağıda verilen "KANUN METNİ"ni kullanarak cevapla.

Kurallar:
1. Asla kendi genel bilgini kullanma. Sadece verilen metne sadık kal.
2. Eğer verilen metinde sorunun cevabı yoksa, kesinlikle uydurma ve "Bu konuda verilen metinlerde bilgi bulunmamaktadır." de.
3. Cevabına ilgili madde numarasını belirterek başla (Örn: "Kat Mülkiyeti Kanunu Madde 34...").
4. Hukuki terimleri sadeleştirerek, herkesin anlayacağı dilde açıkla.
5. Resmi ve yardımsever bir ton kullan.
"""
