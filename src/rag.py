from openai import OpenAI
from src import config, utils

class LegalRAG:
    """
    RAG (Retrieval-Augmented Generation) Motoru.
    Görevi: Kullanıcı sorusunu alıp, veritabanından ilgili bilgiyi bulmak ve LLM'e cevaplatmak.
    """
    def __init__(self, provider="local"):
        """
        Sistemi başlatır.
        provider: 'local' (HuggingFace) veya 'openai' (Cloud Embedding)
        """
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.chroma_client = utils.get_chroma_client()
        
        # Hangi embedding modelini ve tabloyu kullanacağımızı belirle
        self.embedding_fn, self.collection_name = utils.get_embedding_provider(provider)
        
        print(f"Bilgi: RAG sistemi '{self.collection_name}' tablosuna bağlandı.")
        self.collection = self.chroma_client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
    
    def retrieve(self, query):
        """
        Adım 1: Retrieval (Geri Getirme)
        Soruyu vektöre çevirir ve veritabanında en benzer metinleri arar.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=config.TOP_K  # En alakalı 5 parçayı getir
        )
        return results['documents'][0]

    def generate_answer(self, query):
        """
        Adım 2 & 3: Augmentation & Generation (Zenginleştirme ve Üretme)
        Bulunan metinleri LLM'e verip cevabı üretir.
        """
        # 1. Dokümanları bul
        relevant_docs = self.retrieve(query)
        
        if not relevant_docs:
            return "Üzgünüm, bu konuyla ilgili kanun metninde bilgi bulamadım.", []

        # 2. Bağlamı (Context) birleştir
        context_text = "\n\n---\n\n".join(relevant_docs)
        
        # 3. LLM'e sor (Prompt Mühendisliği)
        response = self.openai_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                # Burada context'i ve soruyu LLM'e sunuyoruz
                {"role": "user", "content": f"KANUN METNİ:\n{context_text}\n\nSORU: {query}"}
            ],
            temperature=0.3 # Tutarlı cevaplar için düşük sıcaklık
        )
        
        answer = response.choices[0].message.content
        return answer, relevant_docs

