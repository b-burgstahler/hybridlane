from collections.abc import Sequence

from pennylane.devices.execution_config import ExecutionConfig
from pennylane.exceptions import DeviceError
from pennylane.tape import QuantumScript
from qcvdv.simulator import HybridSimulator

from ... import sa
from ..bosonic_qiskit.device import BosonicQiskitDevice, _infer_truncation


class QCvDvDevice_V1(BosonicQiskitDevice):
    name = "QCvDvDevice_V1"  # type: ignore
    short_name = "qc_vdv_v1"
    version = "0.1"
    author = "b-burgstahler"

    def __init__(self, *args, **kwargs):
        self._backend = kwargs.pop("backend", "dense")
        assert self._backend in ("dense", "scipy", "dense_matrix_gpuv1"), (
            "Unsupported backend for QCvDvDevice"
        )
        self.__simulator = HybridSimulator(method=self._backend)
        super().__init__(*args, **kwargs)

    def execute(  # type: ignore
        self,
        circuits: Sequence[QuantumScript],
        execution_config: ExecutionConfig | None = None,
    ):
        from .simulate import simulate

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
            simulate(tape, truncation, hbar=self._hbar, simulator=self.__simulator)
            for tape, truncation in zip(circuits, truncations)
        )
