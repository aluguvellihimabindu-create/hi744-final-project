import argparse
import json
from pathlib import Path

from src.data_io import load_patients
from src.preprocess import preprocess_text, tokenize
from src.representations import build_tfidf_matrix, build_word2vec_doc_matrix
from src.retrieval import top_k_similar
from src.evaluation import eval_at_k


def main():
    parser = argparse.ArgumentParser(description="HI 744 Final Project: Patient Similarity Retrieval")
    parser.add_argument("data_dir", type=str, help="Path to input data folder")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K similar patients")
    parser.add_argument("--w2v_dim", type=int, default=100, help="Word2Vec vector size")
    parser.add_argument("--w2v_window", type=int, default=5, help="Word2Vec window size")
    parser.add_argument("--w2v_min_count", type=int, default=1, help="Word2Vec min_count")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = load_patients(data_dir)
    ids = [p["patient_id"] for p in patients]

    texts = [preprocess_text(p["text"]) for p in patients]
    tokenized = [tokenize(p["text"]) for p in patients]

    tfidf_mat = build_tfidf_matrix(texts)
    w2v_mat = build_word2vec_doc_matrix(
        tokenized,
        vector_size=args.w2v_dim,
        window=args.w2v_window,
        min_count=args.w2v_min_count,
        seed=0
    )

    tfidf_topk = top_k_similar(tfidf_mat, ids, k=args.top_k)
    w2v_topk = top_k_similar(w2v_mat, ids, k=args.top_k)

    task1 = {}
    for pid in ids:
        task1[pid] = {
            "similar_patients_TFIDF": tfidf_topk[pid],
            "similar_patients_word2vec": w2v_topk[pid],
        }
    (out_dir / "task1_similar_patients.json").write_text(json.dumps(task1, indent=2))

    relevance = {p["patient_id"]: p["relevant_ids"] for p in patients}
    task2 = {}
    for pid in ids:
        task2[pid] = {
            "TF_IDF": eval_at_k(tfidf_topk[pid], relevance.get(pid, set()), args.top_k),
            "word2vec": eval_at_k(w2v_topk[pid], relevance.get(pid, set()), args.top_k),
        }
    (out_dir / "task2_eval_metrics.json").write_text(json.dumps(task2, indent=2))

    print("Wrote outputs to:", out_dir.resolve())


if __name__ == "__main__":
    main()
