from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from pennylane.tape import QuantumScript
from pennylane.wires import Wires
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
    permute_subsystems,
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
        get_observable_matrix(m.obs, regmapper, hbar=hbar, qiskit_order=True)
        if m.obs is not None
        else None
    )
    return (
        analytic_measurement_map.get(type(m))(state, result, obs)
        if type(m) in analytic_measurement_map
        else analytic_state(state, result, obs, regmapper, qiskit_order=True)
    )


def analytic_state(
    state: Statevector,
    result: QiskitResult,
    obs: np.ndarray,
    regmapper: RegisterMapping,
    qiskit_order: bool = True,
) -> np.ndarray:
    source_wires = Wires(
        range(len(regmapper.wire_order))
    )  # auto pulling from ALL hybridlane wires
    destination_wires = (
        regmapper.wire_order
    )  # how those wires map to the bq statevector wires
    state_size = state.data.shape[0]

    out_vector = -1 * np.ones(state.data.shape, dtype=complex)

    order = permute_subsystems(
        sp.diags([range(state_size)], [0]),  # matrix
        source_wires,  # 'observable' wires
        destination_wires,  # 'statevector' wires
        regmapper,
        qiskit_order=qiskit_order,
    ).diagonal()
    for i, idx in enumerate(order):
        out_vector[int(idx)] = state.data[i]

    assert not np.any(out_vector == -1)
    return out_vector
