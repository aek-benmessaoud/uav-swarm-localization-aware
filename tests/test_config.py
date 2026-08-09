"""
tests/test_config.py — Locked V4 constants must stay stable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def test_locked_experiment_defaults():
    assert config.GRID_SIZE == 100
    assert config.DEFAULT_FOV_RADIUS == 5
    assert config.DEFAULT_OBSTACLE_RATIO == 0.05
    assert config.NUM_AGENTS == 6
    assert config.MAX_STEPS == 10000
    assert config.NUM_RUNS == 30
    assert config.BASE_SEED == 0
    assert config.SEED_STRIDE == 1000


def test_locked_signal_defaults():
    assert config.ENTROPY_HORIZON == 8
    assert config.ENTROPY_WINDOW is None
    assert abs(config.SENSOR_RELIABILITY - 0.9) < 1e-12
    assert abs(config.UTILITY_EPSILON - 1.0) < 1e-12
    assert abs(config.TIE_BREAK_EPS - 1e-3) < 1e-12
    assert abs(config.ADAPTIVE_K - 0.5) < 1e-12


def test_info_model_locked():
    assert config.INFO_MODEL == "comm_limited"


def test_method_names_unique():
    names = config.METHOD_ORDER
    assert len(names) == len(set(names))
    assert config.METHOD_ENTROPY == "Entropy"
    assert config.METHOD_ENTROPY_FRAC == "Entropy-Frac"
    assert config.METHOD_FRONTIER_ENTROPY == "Frontier+Entropy"
    assert config.METHOD_FRONTIER_RICHNESS == "Frontier+Richness"


def test_chao_variant_locked():
    assert config.CHAO_DEFAULT_VARIANT == "bias_cap"
