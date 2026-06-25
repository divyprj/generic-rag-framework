"""Quick smoke test for the RAG pipeline components.

Usage:
    python smoke_test.py                 Test with default dataset
    python smoke_test.py --dataset mars  Test with specific dataset
"""

from __future__ import annotations

import argparse
import sys

from src import config
from src.document_loader import chunk_documents, load_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def main() -> None:
    """Run smoke test against a specific dataset."""
    parser = argparse.ArgumentParser(
        description="Smoke test for the RAG pipeline.",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=config.DEFAULT_DATASET,
        help=(
            "Dataset to test "
            f"(default: '{config.DEFAULT_DATASET}')"
        ),
    )
    args = parser.parse_args()

    # Load dataset config
    try:
        ds_cfg = config.load_dataset_config(args.dataset)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Dataset  : {ds_cfg.name} ({ds_cfg.dataset_id})")
    print(f"Docs dir : {ds_cfg.data_dir}")
    print()

    # Test document loading
    docs = load_documents(ds_cfg.data_dir)
    print(f"Loaded {len(docs)} documents")
    for d in docs:
        print(
            f"  {d.metadata['source']}: "
            f"{len(d.content)} chars"
        )

    # Test chunking
    chunks = chunk_documents(
        docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP,
    )
    print(
        f"\nCreated {len(chunks)} chunks "
        f"(size={config.CHUNK_SIZE}, "
        f"overlap={config.CHUNK_OVERLAP})"
    )

    # Test embeddings
    print("\nLoading BGE embedding model...")
    model = EmbeddingModel()
    vec = model.embed_text("test query")
    print(f"Embedding dim: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")

    # Test vector store (using a test collection)
    test_collection = f"smoke_test_{ds_cfg.dataset_id}"
    vs = VectorStore(
        persist_dir=config.CHROMA_PERSIST_DIR,
        collection_name=test_collection,
        embedding_model=model,
    )
    vs.build_index(chunks)
    count = vs.collection_count()
    print(f"\nVector store: {count} chunks indexed")

    # Test retrieval
    results = vs.query(docs[0].content[:50], top_k=3)
    print("\nTop 3 retrieval results:")
    for i, r in enumerate(results, 1):
        src = r.metadata.get("source", "?")
        print(f"  {i}. [{src}] score={r.similarity_score:.4f}")
        print(f"     {r.content[:100]}...")

    # Clean up test collection
    vs.reset()
    print(f"\nSmoke test PASSED ({ds_cfg.name})")


if __name__ == "__main__":
    main()
