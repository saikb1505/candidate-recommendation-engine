import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import config
from config.schema import ResumeSchema, ResumeChunk


_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("  Loading embedding model (first time only)...")
        _model = SentenceTransformer("all-mpnet-base-v2")
        print("  Model loaded.")
    return _model


def embed_text(text: str) -> list[float]:
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_resume(resume: ResumeSchema) -> list[float]:
    text = resume.to_embedding_text()

    if not text.strip():
        model = get_model()
        dim = model.get_embedding_dimension()
        return [0.0] * dim

    return embed_text(text)


def embed_chunks_batch(chunks: list[ResumeChunk]) -> list[list[float]]:
    model = get_model()
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_resumes_batch(resumes: list[ResumeSchema]) -> list[list[float]]:
    model = get_model()

    texts = []
    for resume in resumes:
        text = resume.to_embedding_text()
        texts.append(text if text.strip() else "empty resume")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


def load_all_resumes() -> list[ResumeSchema]:
    """Load all extracted JSON files from the output directory."""
    json_dir = config.json_dir
    resumes = []

    if not json_dir.exists():
        print(f"  No JSON directory found: {json_dir}")
        return resumes

    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            resume = ResumeSchema(**data)
            resumes.append(resume)
        except Exception as e:
            print(f"  Failed to load {json_file.name}: {e}")

    print(f"  Loaded {len(resumes)} resumes from {json_dir}")
    return resumes


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
