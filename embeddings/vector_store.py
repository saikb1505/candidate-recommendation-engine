import chromadb
from pathlib import Path

from config.settings import config
from config.schema import ResumeSchema
from embeddings.embedder import embed_text, embed_resume


_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "resumes"


def get_collection() -> chromadb.Collection:
    global _client, _collection

    if _collection is None:
        db_path = str(config.output_dir / "chromadb")
        _client = chromadb.PersistentClient(path=db_path)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  ChromaDB collection '{COLLECTION_NAME}': "
              f"{_collection.count()} documents")

    return _collection


def add_resume(resume: ResumeSchema, resume_id: str) -> bool:
    collection = get_collection()

    try:
        embedding = embed_resume(resume)
        metadata = resume.to_metadata()
        document = resume.to_embedding_text()

        collection.upsert(
            ids=[resume_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document],
        )
        return True

    except Exception as e:
        print(f"  Failed to add {resume_id}: {e}")
        return False


def add_resumes_batch(
    resumes: list[ResumeSchema],
    resume_ids: list[str],
) -> int:
    collection = get_collection()

    from embeddings.embedder import embed_resumes_batch

    try:
        embeddings = embed_resumes_batch(resumes)
        metadatas = [r.to_metadata() for r in resumes]
        documents = [r.to_embedding_text() for r in resumes]

        collection.upsert(
            ids=resume_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        print(f"  Added {len(resumes)} resumes to vector store")
        return len(resumes)

    except Exception as e:
        print(f"  Batch insert failed: {e}")
        return 0


def search(
    query_text: str,
    top_k: int = 10,
    filters: dict | None = None,
) -> list[dict]:
    collection = get_collection()

    query_embedding = embed_text(query_text)

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }

    if filters:
        query_args["where"] = filters

    results = collection.query(**query_args)

    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "id": results["ids"][0][i],
            "score": round(1 - results["distances"][0][i], 4),
            "metadata": results["metadatas"][0][i],
            "document": results["documents"][0][i],
        })

    return formatted


def search_with_skills(
    query_text: str,
    required_skills: list[str] | None = None,
    min_years: float | None = None,
    location: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Fetch more candidates than needed from ChromaDB, then filter in Python.
    ChromaDB 1.5+ doesn't support substring matching on string metadata fields,
    so skill/location filtering has to happen post-retrieval.
    """
    fetch_k = min(top_k * 10, 100)

    db_filters = None
    if min_years is not None:
        db_filters = {"total_years_experience": {"$gte": min_years}}

    results = search(query_text, top_k=fetch_k, filters=db_filters)

    filtered = []
    for result in results:
        metadata = result["metadata"]

        if required_skills:
            resume_skills = metadata.get("skills", "").lower()
            if not all(skill.lower() in resume_skills for skill in required_skills):
                continue

        if location:
            resume_location = metadata.get("location", "").lower()
            if location.lower() not in resume_location:
                continue

        filtered.append(result)

        if len(filtered) >= top_k:
            break

    return filtered


def get_count() -> int:
    return get_collection().count()


def delete_all():
    global _collection
    if _client is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = None
        print(f"  Deleted collection '{COLLECTION_NAME}'")
