import pytest

torch = pytest.importorskip("torch")

from juniper_encoder.constants import PARAMETER_COUNTS
from juniper_encoder.model import DeploymentModel, actual_parameter_counts


def test_actual_deployment_parameter_counts_and_pad_row():
    model = DeploymentModel(seed=1729)
    counts = actual_parameter_counts(model)
    assert counts["physical_named_parameters"] == PARAMETER_COUNTS["reranked_pipeline_deployed_total"]
    assert counts["shared_backbone"] == PARAMETER_COUNTS["shared_backbone"]
    assert counts["fast_pipeline_unique"] == PARAMETER_COUNTS["fast_pipeline_unique"]
    assert counts["pad_row_max_abs"] == 0.0


def test_adjacent_pair_rope_uses_positive_rotation():
    from juniper_encoder.model import AdjacentPairRoPE

    rope = AdjacentPairRoPE()
    value = torch.zeros((1, 1, 1, 64), dtype=torch.float32)
    value[..., 0] = 1.0
    rotated = rope(value, torch.tensor([1.0]))
    assert rotated[..., 0].item() == pytest.approx(torch.cos(torch.tensor(1.0)).item())
    assert rotated[..., 1].item() == pytest.approx(torch.sin(torch.tensor(1.0)).item())


def test_pad_position_ids_are_zeroed():
    from juniper_encoder.model import AdjacentPairRoPE

    rope = AdjacentPairRoPE()
    value = torch.zeros((1, 2, 1, 64), dtype=torch.float32)
    value[..., 0] = 1.0
    rotated = rope(value, torch.tensor([[0.0, 0.0]]))
    assert torch.equal(rotated[..., 0], torch.ones_like(rotated[..., 0]))
    assert torch.equal(rotated[..., 1], torch.zeros_like(rotated[..., 1]))


def test_padding_does_not_change_pooled_embedding():
    model = DeploymentModel(seed=1729).eval()
    short = torch.tensor([[7, 8, 9, 0]], dtype=torch.long)
    longer = torch.tensor([[7, 8, 9, 0, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        first = model.retrieval_vector(short)
        second = model.retrieval_vector(longer)
    assert torch.equal(first, second)
