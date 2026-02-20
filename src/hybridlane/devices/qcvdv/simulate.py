from __future__ import annotations

import logging
import warnings

import numpy as np
import pennylane as qml
from pennylane.exceptions import DeviceError
from pennylane.operation import Operator
from pennylane.ops.cv import CVOperation
from pennylane.tape import QuantumScript
from qcvdv.circuit import HybridCircuit
from qcvdv.simulator import HybridSimulator
from qiskit.quantum_info import Statevector
from scipy.sparse import SparseEfficiencyWarning

import hybridlane as hqml

from ... import sa
from ...measurements import (
    FockTruncation,
    StateMeasurement,
)
from ...ops.mixins import Hybrid
from ..bosonic_qiskit.simulate import (
    analytic_measurement,
)
from .gates import cv_gate_map, dv_gate_map, hybrid_gate_map, misc_gate_map
from .register_mapping import RegisterMapping

_logger = logging.getLogger(__name__)


def simulate(
    tape: QuantumScript,
    truncation: FockTruncation,
    *,
    hbar: float,
    simulator: HybridSimulator = HybridSimulator(method="dense"),
) -> tuple[np.ndarray]:
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

    qc, regmapper = make_circuit(tape, truncation)

    if tape.shots and not len(tape.shots.shot_vector) == 1:
        raise NotImplementedError("Complex shot batching is not yet supported")

    results = []
    if tape.shots:
        raise NotImplementedError(
            "Shot-based measurements are not yet supported for QCvDv"
        )
    else:
        # Compute state once and reuse across measurements to reduce simulation time

        state = simulator.run(qc, shots=1)
        state = Statevector(state)
        result = None  # TODO: format this as a qiskit result?
        for m in tape.measurements:
            assert isinstance(m, StateMeasurement)
            results.append(analytic_measurement(m, state, result, regmapper, hbar=hbar))

        if len(tape.measurements) == 1:
            return results[0]

    return tuple(results)


def make_circuit(
    tape: QuantumScript, truncation: FockTruncation
) -> tuple[HybridCircuit, RegisterMapping]:
    res = sa.analyze(tape)
    regmapper = RegisterMapping(res, truncation)
    for wire, dim in regmapper.truncation.dim_sizes.items():
        _logger.debug(f"wire {wire} has dimension {dim})")

    try:
        qc = HybridCircuit(*regmapper.regs)
    except ValueError as e:
        raise DeviceError(
            "Bosonic qiskit currently doesn't support executing circuits without a qumode."
        ) from e

    for op in tape.operations:
        # Validate that we have actual values in the parameters
        for p in op.parameters:
            if qml.math.is_abstract(p):
                raise DeviceError(
                    "Need instantiated tensors to convert to qiskit. Circuit may contain Jax or TensorFlow tracing tensors."
                )

        apply_gate(qc, regmapper, op)

    return qc, regmapper


def apply_gate(qc: HybridCircuit, regmapper: RegisterMapping, op: Operator):
    wires = op.wires

    if method := dv_gate_map.get(op.name):
        qubits = [regmapper.get(w) for w in wires]

        match op:
            # This is equivalent up to a global phase of e^{-i(φ + ω)/2}
            case qml.Rot(parameters=(phi, theta, omega)):
                getattr(qc, method)(
                    theta, phi, omega, *qubits
                )  # note the reordered parameters
            case _:
                getattr(qc, method)(*op.parameters, *qubits)

    elif isinstance(op, CVOperation) and (method := cv_gate_map.get(op.name)):
        qumodes = [regmapper.get(w) for w in wires]

        match op:
            # These gates take complex parameters or differ from bosonic qiskit
            case hqml.Displacement(parameters=(r, phi)):
                arg = r * np.exp(1j * phi)
                getattr(qc, method)(arg, *qumodes)
            case hqml.Squeezing(parameters=(r, phi)):
                arg = -r * np.exp(-1j * phi)
                getattr(qc, method)(arg, *qumodes)
            case hqml.Rotation(parameters=(theta,)):
                getattr(qc, method)(-theta, *qumodes)
            case hqml.Beamsplitter(parameters=(theta, phi)):
                new_theta = theta / 2
                new_phi = phi - np.pi / 2
                z = new_theta * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes)
            case hqml.TwoModeSqueezing(parameters=(r, phi)):
                new_phi = phi + np.pi / 2
                z = r * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes)
            case _:
                getattr(qc, method)(*op.parameters, *qumodes)

    elif isinstance(op, Hybrid) and (method := hybrid_gate_map.get(op.name)):
        wire_types = op.wire_types()

        qumodes = [regmapper.get(w) for w in op.wires if wire_types[w] == sa.Qumode()]
        qubits = [regmapper.get(w) for w in op.wires if wire_types[w] == sa.Qubit()]

        match op:
            case hqml.ConditionalRotation(parameters=(theta,)):
                getattr(qc, method)(-theta / 2, *qumodes, *qubits)
            case (
                hqml.ConditionalDisplacement(parameters=(r, phi))
                | hqml.ConditionalSqueezing(parameters=(r, phi))
            ):
                arg = r * np.exp(1j * phi)
                getattr(qc, method)(arg, *qumodes, *qubits)
            case (
                hqml.SQR(parameters=parameters, hyperparameters={"n": n})
                | hqml.SNAP(parameters=parameters, hyperparameters={"n": n})
            ):
                getattr(qc, method)(*parameters, n, *qumodes, *qubits)
            case hqml.ConditionalBeamsplitter(parameters=(theta, phi)):
                new_theta = theta / 2
                new_phi = phi - np.pi / 2
                z = new_theta * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes)
            case hqml.ConditionalTwoModeSqueezing(parameters=(r, phi)):
                new_phi = phi + np.pi / 2
                z = r * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes, *qubits)
            case _:
                getattr(qc, method)(*op.parameters, *qumodes, *qubits)

    elif method := misc_gate_map.get(op.name):
        match op:
            case qml.Barrier():
                pass  # no-op

    else:
        raise DeviceError(f"Unsupported operation {op.name}")
