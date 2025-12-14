# HI 744 Final Project – Patient Similarity Retrieval

## Task Description
This project implements an information retrieval system that identifies the **top-K most similar patient records** based on free-text clinical descriptions.

For each patient, the system:
1. Represents patient text using **TF-IDF**
2. Represents patient text using **Word2Vec document embeddings** (average of word vectors)
3. Computes **cosine similarity** between patients
4. Retrieves the **top-K most similar patients**
5. Evaluates retrieval quality using **Precision@K** and **Recall@K**

The system outputs results in **JSON format**, as required by the course assignment.

---

## Dataset Format

The input data directory must follow this structure:

- Each patient file contains a free-text clinical description.
- `relevance.json` maps each patient ID to a list of known similar patients.

Example:
```json
{
  "P1": ["P2"],
  "P2": ["P1"],
  "P3": []
}

## How to Run the Code

Run the following commands from the repository root directory:

```bash
pip install -r requirements.txt
python -m src.run demo_data --out_dir outputs --top_k 5
