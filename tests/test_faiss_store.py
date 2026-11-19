import numpy as np
import pytest

from app.retrieval.faiss_store import FaissStore, FaissStoreError

DIM = 8


def unit(index: int) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


@pytest.fixture
def store() -> FaissStore:
    store = FaissStore(DIM)
    store.add([100, 200, 300, 400], np.stack([unit(0), unit(1), unit(2), unit(0)]),
              ["darglobal", "wasalt", "wasalt", "darglobal"])
    return store


def test_empty_index_search_is_safe():
    empty = FaissStore(DIM)
    assert empty.size == 0 and empty.search(unit(0), 5) == []


def test_add_and_size(store):
    assert store.size == 4 and store.vector_ids == [100, 200, 300, 400]
    assert store.company_of(200) == "wasalt"


def test_search_returns_explicit_ids_not_positions(store):
    results = store.search(unit(1), 1)
    assert results[0][0] == 200 and results[0][1] == pytest.approx(1.0)


def test_top_k_limits_and_orders_by_score(store):
    query = (unit(0) * 0.9 + unit(1) * 0.1)
    query /= np.linalg.norm(query)
    results = store.search(query, 3)
    assert len(results) == 3
    assert [r[0] for r in results][:2] in ([100, 400], [400, 100])
    assert results[0][1] >= results[1][1] >= results[2][1]
    assert len(store.search(unit(0), 50)) == 4


def test_company_filter_excludes_other_companies(store):
    results = store.search(unit(0), 4, companies=["wasalt"])
    assert {r[0] for r in results} == {200, 300}
    assert store.search(unit(0), 4, companies=["darglobal"])[0][0] in (100, 400)
    assert store.search(unit(0), 4, companies=[]) == []
    assert store.search(unit(0), 4, companies=["nobody"]) == []


def test_company_filter_widens_search_when_needed():
    big = FaissStore(DIM)
    ids = list(range(60))
    vectors = np.stack([unit(0) for _ in range(59)] + [unit(1)])
    big.add(ids, vectors, ["darglobal"] * 59 + ["wasalt"])
    results = big.search(unit(0), 1, companies=["wasalt"])
    assert [r[0] for r in results] == [59]


def test_duplicate_ids_rejected(store):
    with pytest.raises(FaissStoreError, match="already exists"):
        store.add([100], unit(3).reshape(1, -1), ["darglobal"])
    with pytest.raises(FaissStoreError, match="Duplicate"):
        store.add([1, 1], np.stack([unit(3), unit(4)]), ["darglobal", "darglobal"])


def test_dimension_mismatch_rejected(store):
    with pytest.raises(FaissStoreError, match="columns"):
        store.add([5], np.ones((1, DIM + 1), dtype=np.float32), ["darglobal"])
    with pytest.raises(FaissStoreError, match="columns"):
        store.search(np.ones(DIM + 1, dtype=np.float32), 1)


def test_unnormalized_and_invalid_input_rejected(store):
    with pytest.raises(FaissStoreError, match="normalized"):
        store.add([5], np.full((1, DIM), 2.0, dtype=np.float32), ["darglobal"])
    with pytest.raises(FaissStoreError, match="non-finite"):
        store.add([5], np.full((1, DIM), np.nan, dtype=np.float32), ["darglobal"])
    with pytest.raises(FaissStoreError, match="non-negative"):
        store.add([-1], unit(3).reshape(1, -1), ["darglobal"])
    with pytest.raises(FaissStoreError, match="same length"):
        store.add([5, 6], unit(3).reshape(1, -1), ["darglobal"])
    with pytest.raises(ValueError):
        store.search(unit(0), 0)


def test_save_and_reload_roundtrip(store, tmp_path):
    path = tmp_path / "nested" / "faiss.index"
    store.save(path)
    assert path.exists() and not path.with_name("faiss.index.tmp").exists()
    companies = {i: store.company_of(i) for i in store.vector_ids}
    loaded = FaissStore.load(path, companies)
    assert loaded.size == 4 and loaded.dimension == DIM
    assert loaded.search(unit(2), 1)[0][0] == 300
    assert loaded.search(unit(0), 4, companies=["wasalt"]) == store.search(unit(0), 4, companies=["wasalt"])


def test_load_rejects_missing_company_metadata(store, tmp_path):
    path = tmp_path / "faiss.index"
    store.save(path)
    with pytest.raises(FaissStoreError, match="no chunk metadata"):
        FaissStore.load(path, {100: "darglobal"})


def test_load_rejects_corrupt_or_missing_file(tmp_path):
    bad = tmp_path / "bad.index"
    bad.write_bytes(b"not an index")
    with pytest.raises(FaissStoreError):
        FaissStore.load(bad, {})
    with pytest.raises(FaissStoreError):
        FaissStore.load(tmp_path / "missing.index", {})
