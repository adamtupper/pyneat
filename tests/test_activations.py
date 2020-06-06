"""Tests for the activations module.
"""
import pytest

from pyneat.activations import steep_sigmoid_activation


def test_steep_sigmoid_activation():
    assert steep_sigmoid_activation(0) == pytest.approx(0.5)
    assert steep_sigmoid_activation(-6) == pytest.approx(0)
    assert steep_sigmoid_activation(6) == pytest.approx(1)
