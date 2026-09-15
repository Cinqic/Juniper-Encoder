from juniper_encoder.constants import *
from juniper_encoder.model import ModelConfig


def test_frozen_model_config():
    config = ModelConfig()
    config.validate()
    assert config.width == 512
    assert config.q_heads * config.head_dim == config.width
    assert GEGLU_WIDTH == 1536
    assert ROPE_SCALE == 1 / 8


def test_dropout_sites_are_declared_once():
    assert DROPOUT_P == 0.1
    assert LN_EPS == 1e-5
