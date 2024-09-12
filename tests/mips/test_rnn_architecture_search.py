import sys
import os

import pytest
from expecttest import assert_expected_inline

import pebble.scripts.rnn_architecture_search as ras


def test_range_product():
    assert_expected_inline(ras.range_product([0, 2], [1, 3]), """""")
