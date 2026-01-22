from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class KeywordRetriever:
    def __init__(self, documents: list[str]):
        self.vectorizer = TfidfVectorizer()
        self.docs = documents
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def search(self, query: str, k: int = 5):
        q_vec = self.vectorizer.transform([query])
        scores = (self.doc_vectors @ q_vec.T).toarray().ravel()
        top = np.argsort(scores)[-k:][::-1]
        return [(self.docs[i], scores[i]) for i in top]
