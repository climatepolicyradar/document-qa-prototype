from tests.vector_store import vector_store

assert vector_store is not None  # so that pyright doesn't remove the import...


def test_vector_store(vector_store):
    docs = vector_store.similarity_search("climate", 5)

    assert "text_block_ids" in docs[0].metadata.keys()
    assert all(["climate" in doc.page_content.lower() for doc in docs])
