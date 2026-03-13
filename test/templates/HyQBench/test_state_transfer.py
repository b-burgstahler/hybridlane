import numpy as np
import pennylane as qml
import pytest

import hybridlane as hqml
from hybridlane.templates.HyQBench.state_transfer import StateTransferCVtoDV


@pytest.fixture
def device():
    return qml.device("bosonicqiskit.hybrid", max_fock_level=256)


@pytest.mark.parametrize(
    "n_qubits, lmbda",
    [
        (4, 0.29),
    ],
)
def test_state_transfer_cv_to_dv(device, n_qubits, lmbda):
    wires = list(range(n_qubits + 1))

    @qml.qnode(device)
    def circuit():
        StateTransferCVtoDV(n_qubits, lmbda, wires=wires)
        return tuple(hqml.expval(qml.Z(i)) for i in range(n_qubits))

    exps = circuit()
    assert len(exps) == n_qubits
    assert np.allclose(exps, n_qubits * [0])
