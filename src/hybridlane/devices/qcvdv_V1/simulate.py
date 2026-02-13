from __future__ import annotations

import warnings

import numpy as np
from pennylane.tape import QuantumScript
from qcvdv.circuit import HybridCircuitV1 as HybridCircuit
from qcvdv.circuit import from_CVCircuit
from qcvdv.simulator import HybridSimulator
from qiskit.quantum_info import Statevector
from scipy.sparse import SparseEfficiencyWarning

from ...measurements import FockTruncation, StateMeasurement
from ..bosonic_qiskit.simulate import analytic_measurement, make_cv_circuit


def simulate(
    tape: QuantumScript,
    truncation: FockTruncation,
    *,
    hbar: float,
    simulator: HybridSimulator = HybridSimulator(method="dense"),
) -> tuple[np.ndarray]:
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

    qc, regmapper = make_cv_circuit(tape, truncation)

    if tape.shots and not len(tape.shots.shot_vector) == 1:
        raise NotImplementedError("Complex shot batching is not yet supported")

    results = []
    if tape.shots:
        raise NotImplementedError(
            "Shot-based measurements are not yet supported for QCvDv"
        )
    else:
        # Compute state once and reuse across measurements to reduce simulation time
        qc = from_CVCircuit(qc, hyb_circ=HybridCircuit)
        state = simulator.run(qc, shots=1)
        state = Statevector(state)
        result = None  # TODO: format this as a qiskit result?
        for m in tape.measurements:
            assert isinstance(m, StateMeasurement)
            results.append(analytic_measurement(m, state, result, regmapper, hbar=hbar))

        if len(tape.measurements) == 1:
            return results[0]

    return tuple(results)
