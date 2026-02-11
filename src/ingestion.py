from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from src import config, utils

def load_pdf(file_path):
    """PDF dosyasını okur ve ham metni çıkarır."""
    print(f"Bilgi: PDF okunuyor -> {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text):
    """
    Ham metni anlamsal bütünlüğü koruyarak küçük parçalara (chunk) böler.
    
    Bu işlem RAG başarısı için kritiktir. Metni 'Madde' başlıklarından
    bölmeye öncelik veriyoruz.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # Hukuki metin bütünlüğünü korumak için özel ayırıcılar (Önem sırasına göre)
        separators=[
            "\nBÖLÜM ",         # Bölümler en büyük ayrım
            "\nMadde ",         # Normal maddeler
            "\nEk Madde ",      # Ek maddeler
            "\nGeçici Madde ",  # Geçici maddeler
            "\n", " ", ""       # Standart ayrımlar
        ],
        is_separator_regex=True
    )
    chunks = text_splitter.split_text(text)
    print(f"Bilgi: Metin {len(chunks)} parçaya bölündü.")
    return chunks

def ingest_data(force_recreate=False, provider="local"):
    """
    ETL Süreci (Extract, Transform, Load):
    1. Extract: PDF'den metni oku.
    2. Transform: Metni parçala ve Vektöre çevir (Embedding).
    3. Load: Vektörleri ChromaDB veritabanına yükle.
    """
    client = utils.get_chroma_client()
    
    # Seçilen stratejiye göre ayarları al
    embedding_fn, collection_name = utils.get_embedding_provider(provider)
    print(f"İşlem Başlıyor: {provider.upper()} modu | Hedef Tablo: {collection_name}")

    # Temiz başlangıç isteniyorsa eski tabloyu sil
    if force_recreate:
        try:
            client.delete_collection(name=collection_name)
            print(f"Bilgi: '{collection_name}' tablosu temizlendi.")
        except Exception:
            pass 

    # Tabloyu getir veya oluştur
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    # Veri zaten varsa tekrar yükleme
    if collection.count() > 0 and not force_recreate:
        print(f"Bilgi: Veritabanı güncel ({collection.count()} kayıt). İşlem yapılmadı.")
        return

    # Veri Yükleme Akışı
    text = load_pdf(config.PDF_PATH)
    chunks = chunk_text(text)
    
    # Her parça için benzersiz bir ID ve metadata oluştur
    ids = [f"id_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "kat-mulkiyeti-kanunu"} for _ in chunks]
    
    # Toplu kayıt (Batch insert)
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"Başarılı: {len(chunks)} kayıt veritabanına işlendi.")

if __name__ == "__main__":
    # Test amaçlı doğrudan çalıştırma
    ingest_data(force_recreate=True, provider="local")

