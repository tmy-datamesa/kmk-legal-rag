import chromadb
from chromadb.utils import embedding_functions
from src import config

def get_chroma_client():
    """
    ChromaDB Cloud istemcisini başlatır.
    
    Bu fonksiyon projenin 'Vector Database'e bağlanmasını sağlar.
    Eğer .env dosyasında API anahtarı yoksa hata fırlatır.
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

def get_embedding_provider(provider_key="local"):
    """
    Seçilen stratejiye göre Embedding Fonksiyonunu ve Koleksiyon Adını hazırlar.
    (Factory Design Pattern)
    
    Girdi:
        provider_key (str): 'local' veya 'openai'
    
    Çıktı:
        embedding_fn: Metni vektöre çeviren fonksiyon.
        collection_name: Vektörlerin saklanacağı veritabanı tablosunun adı.
    """
    if provider_key not in config.EMBEDDING_PROVIDERS:
        raise ValueError(f"Geçersiz Sağlayıcı: {provider_key}")

    # Config dosyasından ilgili ayarları çek
    provider_config = config.EMBEDDING_PROVIDERS[provider_key]
    model_type = provider_config["type"]
    model_name = provider_config["model_name"]
    collection_name = provider_config["collection_name"]

    # Model tipine göre fonksiyonu oluştur
    if model_type == "openai":
        embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            model_name=model_name
        )
    elif model_type == "sentence_transformer":
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    else:
        raise ValueError(f"Tanınmayan Model Tipi: {model_type}")

    return embedding_fn, collection_name

