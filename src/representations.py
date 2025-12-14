import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_matrix(texts):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    return vectorizer.fit_transform(texts)

def build_dummy_word2vec_matrix(texts, dim: int = 50):
    """
    Placeholder so the pipeline runs end-to-end.
    We'll replace this with a real embedding method next.
    """
    rng = np.random.default_rng(0)
    return rng.normal(size=(len(texts), dim))
