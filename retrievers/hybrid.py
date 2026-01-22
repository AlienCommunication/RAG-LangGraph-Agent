class HybridRetriever:
    def __init__(self, keyword, vector, alpha=0.5):
        self.keyword = keyword
        self.vector = vector
        self.alpha = alpha

    def search(self, query: str, k: int = 5):
        kw = self.keyword.search(query, k)
        vec = self.vector.search(query, k)

        fused = {}
        for doc, score in kw:
            fused[doc] = fused.get(doc, 0) + self.alpha * score
        for doc, score in vec:
            fused[doc] = fused.get(doc, 0) + (1 - self.alpha) * score

        return [d for d, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]]
