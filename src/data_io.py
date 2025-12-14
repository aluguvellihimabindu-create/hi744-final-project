from pathlib import Path
import json

def load_patients(data_dir: Path):
    """
    Expected input format:
      data_dir/
        patients/
          <patient_id>.txt
        labels/
          relevance.json   (optional)
            {
              "<patient_id>": ["<relevant_id1>", "<relevant_id2>", ...]
            }
    """
    patients_dir = data_dir / "patients"
    labels_path = data_dir / "labels" / "relevance.json"

    if not patients_dir.exists():
        raise FileNotFoundError(f"Missing folder: {patients_dir}")

    relevance = {}
    if labels_path.exists():
        relevance = json.loads(labels_path.read_text(encoding="utf-8"))

    patients = []
    for fp in sorted(patients_dir.glob("*.txt")):
        pid = fp.stem
        text = fp.read_text(encoding="utf-8", errors="ignore")
        relevant_ids = set(relevance.get(pid, []))
        patients.append({"patient_id": pid, "text": text, "relevant_ids": relevant_ids})

    if not patients:
        raise FileNotFoundError(f"No .txt files found in: {patients_dir}")

    return patients
