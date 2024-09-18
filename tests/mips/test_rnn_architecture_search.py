import sys
import os

import pytest
from expecttest import assert_expected_inline

import mips.scripts.rnn_architecture_search as ras


def test_range_product():
    assert_expected_inline(str(ras.range_product([[0, 2], [1, 3]])), """9""")
    assert_expected_inline(str(ras.range_product([[0, 2], [1, 3], [4, 5]])), """18""")

def test_tuple_to_int():
    raise NotImplementedError
    # assert_expected_inline(str(ras.tuple_to_int((1, 2, 3))), """123""")