from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorRetriever:
    def __init__(self, persist_dir: str, model_name: str):
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding
        )

    def search(self, query: str, k: int = 5):
        docs = self.db.similarity_search(query, k=k)
        return [(d.page_content, 1.0) for d in docs]
