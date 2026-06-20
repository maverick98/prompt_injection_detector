import numpy as np

from prompt_injection_detector.models.semantic import (
    contrastive_similarity_score,
    l2_normalize,
    select_semantic_threshold,
    top_k_mean_similarity,
)


def test_l2_normalize_handles_zero_vectors():
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]])

    normalized = l2_normalize(matrix)

    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [0.0, 0.0])


def test_top_k_mean_similarity_prefers_nearest_reference():
    query = np.array([[1.0, 0.0]])
    references = np.array([[1.0, 0.0], [0.0, 1.0]])

    scores = top_k_mean_similarity(query, references, top_k=1)

    assert np.allclose(scores, [1.0])


def test_contrastive_similarity_score_is_higher_for_attack_like_embedding():
    queries = np.array([[1.0, 0.0], [0.0, 1.0]])
    attacks = np.array([[1.0, 0.0]])
    clean = np.array([[0.0, 1.0]])

    scores = contrastive_similarity_score(queries, attacks, clean, top_k=1)

    assert scores[0] > scores[1]
    assert scores[0] > 0.9
    assert scores[1] < 0.1


def test_select_semantic_threshold_is_recall_first():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.6, 0.8])

    threshold = select_semantic_threshold(y_true, scores)

    assert 0.3 < threshold <= 0.6
