import chromadb
from chromadb.utils import embedding_functions
from src import config

def get_chroma_client():
    """
    ChromaDB Cloud istemcisini başlatır.
    
    Bu fonksiyon projenin 'Vector Database'e bağlanmasını sağlar.
    Eğer .env dosyasında API anahtarı yoksa hata verir.
    """
    if not config.CHROMA_API_KEY:
        raise ValueError("HATA: .env dosyasında CHROMA_API_KEY eksik.")

    try:
        client = chromadb.CloudClient(
            api_key=config.CHROMA_API_KEY,
            tenant=config.CHROMA_TENANT_ID,
            database=config.CHROMA_DATABASE_NAME
        )
        return client
    except Exception as e:
        print(f"HATA: Bulut veritabanına bağlanılamadı. Detay: {e}")
        raise e

def get_embedding_function():
    """
    OpenAI Embedding fonksiyonunu döndürür.
    Proje artık sadece OpenAI (Cloud) kullanıyor.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("HATA: .env dosyasında OPENAI_API_KEY eksik.")

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.EMBEDDING_MODEL_NAME
    )


