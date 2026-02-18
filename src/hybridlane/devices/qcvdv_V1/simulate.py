from __future__ import annotations

import warnings

import numpy as np
from pennylane.tape import QuantumScript
from qcvdv.circuit import HybridCircuitV1 as HybridCircuit
from qcvdv.circuit import from_CVCircuit
from qcvdv.simulator import HybridSimulator
from qiskit.quantum_info import Statevector
from qiskit.result import Result as QiskitResult
from scipy.sparse import SparseEfficiencyWarning

from ...measurements import (
    FockTruncation,
    StateMeasurement,
)
from ..bosonic_qiskit.register_mapping import RegisterMapping
from ..bosonic_qiskit.simulate import (
    analytic_measurement_map,
    get_observable_matrix,
    make_cv_circuit,
)


def simulate(
    tape: QuantumScript,
    truncation: FockTruncation,
    *,
    hbar: float,
    simulator: HybridSimulator = HybridSimulator(method="dense"),
) -> tuple[np.ndarray]:
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

    bq_qc, regmapper = make_cv_circuit(tape, truncation)

    if tape.shots and not len(tape.shots.shot_vector) == 1:
        raise NotImplementedError("Complex shot batching is not yet supported")

    results = []
    if tape.shots:
        raise NotImplementedError(
            "Shot-based measurements are not yet supported for QCvDv"
        )
    else:
        # Compute state once and reuse across measurements to reduce simulation time
        qc = from_CVCircuit(bq_qc, hyb_circ=HybridCircuit)
        state = simulator.run(qc, shots=1)
        state = Statevector(state)
        result = None  # TODO: format this as a qiskit result?
        for m in tape.measurements:
            assert isinstance(m, StateMeasurement)
            results.append(analytic_measurement(m, state, result, regmapper, hbar=hbar))

        if len(tape.measurements) == 1:
            return results[0]

    return tuple(results)


def analytic_measurement(
    m: StateMeasurement,
    state: Statevector,
    result: QiskitResult,
    regmapper: RegisterMapping,
    *,
    hbar: float,
):
    obs = (
        get_observable_matrix(m.obs, regmapper, hbar=hbar, qiskit_order=False)
        if m.obs is not None
        else None
    )
    return (
        analytic_measurement_map.get(type(m))(state, result, obs)
        if type(m) in analytic_measurement_map
        else analytic_state(state, result, obs, regmapper, qiskit_order=False)
    )


def analytic_state(
    state: Statevector,
    result: QiskitResult,
    obs: np.ndarray,
    regmapper: RegisterMapping,
    qiskit_order: bool = True,
) -> np.ndarray:
    out_vector = -1.0 * np.ones(len(state.data), dtype=complex)
    dims = [regmapper.truncation.dim_sizes[x] for x in regmapper.wire_order]
    all_fock_strings = [
        _decimal_to_fock_string(i, dims) for i in range(len(state.data))
    ]
    order = [
        _fock_string_to_decimal(fock_string, dims) for fock_string in all_fock_strings
    ]

    for i, idx in enumerate(order):
        out_vector[int(idx)] = state.data[i]

    assert not np.any(out_vector == -1)
    return out_vector


def _fock_string_to_decimal(
    bin_list: list[int], dims: list[int], lexigocraphical: bool = True
) -> int:
    rev_dim = dims[::-1] if lexigocraphical else dims
    bases = np.cumprod([1] + rev_dim[:-1])
    return int(np.dot(bases, bin_list[::-1]))


def _decimal_to_fock_string(
    num: int, dims: list[int], lexigocraphical: bool = True
) -> list[int]:
    length = len(dims)
    rev_dim = dims[::-1] if lexigocraphical else dims
    bin_list = [0] * length
    for i in range(length):
        bin_list[length - 1 - i] = num % rev_dim[i]
        num //= rev_dim[i]
    return bin_list
