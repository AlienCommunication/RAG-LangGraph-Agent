def recall_at_k(retrieved, ground_truth):
    hits = sum(1 for d in retrieved if d in ground_truth)
    return hits / max(len(ground_truth), 1)
