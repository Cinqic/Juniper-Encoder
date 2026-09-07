import pytest

torch = pytest.importorskip("torch")

from juniper_encoder.train.protocol import MaskingProtocol
from juniper_encoder.train.runner import reranker_listwise_loss, retrieval_contrastive_loss


def test_masking_is_deterministic_and_excludes_structural_tokens():
    protocol = MaskingProtocol(seed=1729, mask_rate=1.0)
    first = protocol.mask_detailed([2, 5, 7, 8, 3, 0], "example", 0)
    second = protocol.mask_detailed([2, 5, 7, 8, 3, 0], "example", 0)
    assert first == second
    masked, positions, targets = first
    assert positions == [2, 3]
    assert targets == [7, 8]
    assert masked[:2] == [2, 5]
    assert masked[4:] == [3, 0]


def test_frozen_retrieval_and_listwise_losses_are_finite():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert float(retrieval_contrastive_loss(query, candidates)) < 1e-4
    assert float(reranker_listwise_loss(torch.tensor([[3.0, 0.0], [0.0, 3.0]]), torch.tensor([0, 1]))) < 0.1
