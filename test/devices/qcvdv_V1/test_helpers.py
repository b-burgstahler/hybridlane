import numpy as np
import pytest

from hybridlane.devices.qcvdv_V1.simulate import (
    _decimal_to_fock_string,
    _fock_string_to_decimal,
)


@pytest.mark.parametrize(
    "expected, dims, num",
    [
        ([0, 0], [2, 4], 0),
        ([1, 0], [2, 4], 4),
        ([0, 1], [2, 4], 1),
        ([1, 1], [2, 4], 5),
        ([0, 2], [2, 4], 2),
        ([1, 2], [2, 4], 6),
        ([0, 3], [2, 4], 3),
        ([1, 3], [2, 4], 7),
        ([0, 0], [4, 2], 0),
        ([0, 1], [4, 2], 1),
        ([1, 0], [4, 2], 2),
        ([1, 1], [4, 2], 3),
        ([2, 0], [4, 2], 4),
        ([2, 1], [4, 2], 5),
        ([3, 0], [4, 2], 6),
        ([3, 1], [4, 2], 7),
        ([0, 1, 2], [2, 4, 4], 6),
        ([0, 2, 1], [2, 4, 4], 9),
        ([1, 0, 2], [4, 2, 4], 10),
        ([1, 2, 0], [4, 4, 2], 12),
        ([2, 0, 1], [4, 2, 4], 17),
        ([2, 1, 0], [4, 4, 2], 18),
    ],
)
def test_decimal_to_fock_string(num, dims, expected):
    assert _decimal_to_fock_string(num, dims) == expected


@pytest.mark.parametrize(
    "fock_string, dims, expected",
    [
        # fock string in lexicographic order, dims in the same order
        # ie fock_string[0] corresponds to dims[0] and is the 'top' wire in pennylane (bottom wire in qiskit)
        # ([0, 0, 0], [2, 2, 2], 0),  # binary check
        # ([1, 0, 0], [2, 2, 2], 4),  # confirm endianess
        ([0, 0], [2, 4], 0),
        ([1, 0], [2, 4], 4),
        ([0, 1], [2, 4], 1),
        ([1, 1], [2, 4], 5),
        ([0, 2], [2, 4], 2),
        ([1, 2], [2, 4], 6),
        ([0, 3], [2, 4], 3),
        ([1, 3], [2, 4], 7),
        ([0, 0], [4, 2], 0),
        ([0, 1], [4, 2], 1),
        ([1, 0], [4, 2], 2),
        ([1, 1], [4, 2], 3),
        ([2, 0], [4, 2], 4),
        ([2, 1], [4, 2], 5),
        ([3, 0], [4, 2], 6),
        ([3, 1], [4, 2], 7),
        ([0, 1, 2], [2, 4, 4], 6),
        ([0, 2, 1], [2, 4, 4], 9),
        ([1, 0, 2], [4, 2, 4], 10),
        ([1, 2, 0], [4, 4, 2], 12),
        ([2, 0, 1], [4, 2, 4], 17),
        ([2, 1, 0], [4, 4, 2], 18),
    ],
)
def test_fock_string_to_decimal(fock_string, dims, expected):
    assert _fock_string_to_decimal(fock_string, dims) == expected


def test_decimal_to_fock_string_and_back():
    dims = [2, 4]
    for num in range(np.prod(dims)):
        fock_string = _decimal_to_fock_string(num, dims)
        dec = _fock_string_to_decimal(fock_string, dims)
        assert dec == num, f"Failed for num={num}, dims={dims}"
