from ..devices.bosonic_qiskit.register_mapping import (
    RegisterMapping as bq_RegisterMapping,
)


class RegisterMapping(bq_RegisterMapping):
    r"""Utility class to map wires -> qcvdv registers for the qcvdv device"""

    # For now, we can just reuse the same register mapping as the bosonic qiskit device, since the requirements are the same.
    # However, if we later want to customize this for the qcvdv device, we can do so by overriding the methods of the bosonic qiskit RegisterMapping class here.
    # TODO: chat with srikar about the details here
