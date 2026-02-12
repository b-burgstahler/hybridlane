from ..bosonic_qiskit import gates as bq_gates

# Default to same mappings as the bosonic qiskit device.
# see hybridlane.devices.bosonic_qiskit.gates for details on the mapping, as well as caveats and limitations.
# We may need to customize these for use with qcvdv, but for now we will attempt to reuse them.
dv_gate_map: dict[str, str] = bq_gates.dv_gate_map
cv_gate_map: dict[str, str] = bq_gates.cv_gate_map
hybrid_gate_map: dict[str, str] = bq_gates.hybrid_gate_map
misc_gate_map: dict[str, str] = bq_gates.misc_gate_map


supported_operations = set(
    k
    for k, v in (dv_gate_map | cv_gate_map | hybrid_gate_map | misc_gate_map).items()
    if v is not None
)
