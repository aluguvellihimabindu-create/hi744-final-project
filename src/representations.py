import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

def build_tfidf_matrix(texts):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    return vectorizer.fit_transform(texts)

def build_word2vec_doc_matrix(tokenized_texts, vector_size=100, window=5, min_count=1, workers=2, seed=0):
    w2v = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed
    )

    doc_vecs = []
    for tokens in tokenized_texts:
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        if len(vecs) == 0:
            doc_vecs.append(np.zeros(vector_size, dtype=np.float32))
        else:
            doc_vecs.append(np.mean(vecs, axis=0).astype(np.float32))

    return np.vstack(doc_vecs)
