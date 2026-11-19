import numpy as np
import pytest

from app.retrieval.embeddings import EmbeddingError, EmbeddingProvider


def test_output_shape_dtype_and_normalization(provider):
    vectors = provider.embed_documents(["first text here", "second text there", "third"])
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32
    assert vectors.flags["C_CONTIGUOUS"]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_query_embedding_is_single_normalized_vector(provider):
    vector = provider.embed_query("where is the project")
    assert vector.shape == (384,) and vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_arabic_text_supported(provider):
    assert provider.embed_documents(["مشروع سكني افتراضي"]).shape == (1, 384)


def test_batching_splits_requests(hashing_model_class):
    model = hashing_model_class()
    provider = EmbeddingProvider("m", 384, batch_size=3, model_factory=lambda name: model)
    vectors = provider.embed_documents([f"text {i}" for i in range(7)])
    assert vectors.shape == (7, 384)
    assert model.batch_sizes == [3, 3, 1]


def test_empty_input_returns_empty_array_without_loading_model():
    calls = []
    provider = EmbeddingProvider("m", 384, model_factory=lambda name: calls.append(name))
    assert provider.embed_documents([]).shape == (0, 384)
    assert calls == [] and not provider.is_loaded


@pytest.mark.parametrize("bad", ["", "   ", None, 5])
def test_invalid_texts_rejected(provider, bad):
    with pytest.raises(ValueError):
        provider.embed_documents(["ok text", bad])


def test_single_string_rejected(provider):
    with pytest.raises(TypeError):
        provider.embed_documents("not a list")


def test_empty_query_rejected(provider):
    with pytest.raises(ValueError):
        provider.embed_query("  ")


def test_dimension_mismatch_detected(hashing_model_class):
    provider = EmbeddingProvider("m", 384, model_factory=lambda name: hashing_model_class(dim=128))
    with pytest.raises(EmbeddingError, match="shape"):
        provider.embed_documents(["some text"])


def test_zero_vector_rejected():
    class ZeroModel:
        def passage_embed(self, texts, **_):
            for _text in texts:
                yield np.zeros(384, dtype=np.float32)

    provider = EmbeddingProvider("m", 384, model_factory=lambda name: ZeroModel())
    with pytest.raises(EmbeddingError, match="zero vector"):
        provider.embed_documents(["text"])


def test_model_loaded_lazily_and_once(hashing_model_class):
    created = []

    def factory(name):
        created.append(name)
        return hashing_model_class()

    provider = EmbeddingProvider("model-x", 384, model_factory=factory)
    assert not provider.is_loaded and created == []
    provider.embed_documents(["one"])
    provider.embed_query("two")
    assert created == ["model-x"] and provider.is_loaded


def test_load_failure_becomes_embedding_error():
    def failing(name):
        raise OSError("no network")

    provider = EmbeddingProvider("m", 384, model_factory=failing)
    with pytest.raises(EmbeddingError, match="Could not load"):
        provider.embed_documents(["text"])


def test_inference_failure_becomes_embedding_error():
    class Broken:
        def passage_embed(self, texts, **_):
            raise RuntimeError("onnx failure")

    provider = EmbeddingProvider("m", 384, model_factory=lambda name: Broken())
    with pytest.raises(EmbeddingError, match="inference failed"):
        provider.embed_documents(["text"])


def test_registry_lists_configured_model():
    pytest.importorskip("fastembed")
    EmbeddingProvider(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384
    ).verify_registry()
    with pytest.raises(EmbeddingError, match="not supported"):
        EmbeddingProvider("nonexistent/model", 384).verify_registry()
    with pytest.raises(EmbeddingError, match="dimension"):
        EmbeddingProvider(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 768
        ).verify_registry()
