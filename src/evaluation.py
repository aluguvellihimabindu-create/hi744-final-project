def eval_at_k(retrieved, relevant, k=5):
    retrieved_k = retrieved[:k]
    if not relevant:
        return {"precision_5": 0.0, "recall_5": 0.0}

    hits = sum(1 for x in retrieved_k if x in relevant)
    precision = hits / float(k)
    recall = hits / float(len(relevant))
    return {"precision_5": round(precision, 4), "recall_5": round(recall, 4)}
