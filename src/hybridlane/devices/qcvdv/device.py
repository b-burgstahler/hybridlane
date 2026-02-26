import warnings
from collections.abc import Sequence

import numpy as np
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.exceptions import DeviceError
from pennylane.tape import QuantumScript
from qcvdv.simulator import HybridSimulator
from qiskit.quantum_info import Statevector
from scipy.sparse import SparseEfficiencyWarning

from ... import sa
from ...measurements import FockTruncation, StateMeasurement
from ..bosonic_qiskit.device import _infer_truncation
from ..bosonic_qiskit.simulate import analytic_measurement
from ..qcvdv_V1.device import QCvDvDevice_V1
from .simulate import make_circuit


class QCvDvDevice(QCvDvDevice_V1):
    name = "QCvDvDevice"  # type: ignore
    short_name = "qc_vdv"
    version = "0.1"
    author = "b-burgstahler"

    def __init__(self, *args, **kwargs):
        self._backend = kwargs.pop("backend", "dense")
        assert self._backend in ("dense", "scipy", "dense_matrix_gpuv1"), (
            "Unsupported backend for QCvDvDevice"
        )
        self.__simulator = HybridSimulator(method=self._backend)
        self._dict_of_circs = {}
        super().__init__(*args, **kwargs)

    def execute(  # type: ignore
        self,
        circuits: Sequence[QuantumScript],
        execution_config: ExecutionConfig | None = None,
    ):
        # from .simulate import simulate

        execution_config = execution_config or ExecutionConfig()
        truncation = execution_config.device_options.get("truncation", self._truncation)
        max_fock_level = execution_config.device_options.get(
            "max_fock_level", self._max_fock_level
        )

        # Try to infer truncation based on circuit structure
        if truncation is None:
            sa_results = map(sa.analyze, circuits)
            truncations = list(
                map(lambda res: _infer_truncation(res, max_fock_level), sa_results)
            )
            if any(t is None for t in truncations):
                raise DeviceError(
                    "Unable to infer truncation for one of the circuits in the batch. Need to specify truncation "
                    "of qumodes through `device_options`"
                )
        else:
            truncations = [truncation] * len(circuits)

        return tuple(
            self.__simulate(
                tape, truncation, hbar=self._hbar, simulator=self.__simulator
            )
            for tape, truncation in zip(circuits, truncations)
        )

    def __simulate(
        self,
        tape: QuantumScript,
        truncation: FockTruncation,
        *,
        hbar: float,
        simulator: HybridSimulator,
    ) -> tuple[np.ndarray]:
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

        if tuple(tape.operations) in self.dict_of_circs:
            qc, regmapper = self.dict_of_circs[tuple(tape.operations)]
        else:
            qc, regmapper = make_circuit(tape, truncation)
            self.dict_of_circs[tuple(tape.operations)] = (qc, regmapper)

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
                results.append(
                    analytic_measurement(m, state, result, regmapper, hbar=hbar)
                )

            if len(tape.measurements) == 1:
                return results[0]

        return tuple(results)

    @property
    def dict_of_circs(self):
        return self._dict_of_circs
