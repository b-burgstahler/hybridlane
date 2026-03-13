"""
Cat state benchmark.

Defines a CatState gate that prepares |cat> = N(|alpha> + |-alpha>) on a qumode
using an ancilla qubit with two conditional displacements (sigma_x and sigma_y
controlled).

The preparation has fidelity F ~ 1 - pi^2 / (64 * alpha^2).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pennylane as qml
from pennylane.decomposition.resources import adjoint_resource_rep
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

import hybridlane as hqml
from hybridlane.ops import Hybrid


# ---------------------------------------------------------------------------
# CatState gate
# ---------------------------------------------------------------------------


class CatState(Operation, Hybrid):
    r"""Cat-state preparation on a qumode.

    Prepares a cat state using an ancilla qubit and hybrid gates:

    .. math::

        \ket{\psi} = \mathcal{N}\left(\ket{\alpha} + \ket{-\alpha}\right)

    where :math:`\alpha = ae^{i\phi}`. The preparation fidelity is

    .. math::

        F \approx 1 - \frac{\pi^2}{64\alpha^2}

    The decomposition uses two conditional displacements:

    1. :math:`\sigma_x` controlled: :math:`H \; CD_z(a, \phi) \; H`
    2. :math:`\sigma_y` controlled: :math:`S^\dagger \; H \; CD_z(b, -\phi + \pi/2) \; H \; S`

    where :math:`b = \pi / (8a)` is the correction amplitude.

    The ``wires`` attribute is ``(qubit, qumode)``.
    """

    num_wires = 2
    num_params = 2
    type_signature = (hqml.sa.Qubit(), hqml.sa.Qumode())
    grad_method = None

    resource_keys = set()

    def __init__(
        self,
        a: TensorLike,
        phi: TensorLike,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        super().__init__(a, phi, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Wires = None,
        **hyperparameters: dict[str, Any],
    ) -> Sequence[Operation]:
        a, phi = params
        b = np.pi / (8 * a)
        q = wires[0]
        return [
            # sigma_x controlled displacement
            qml.H(q),
            hqml.ConditionalDisplacement(a, phi, wires),
            qml.H(q),
            # sigma_y controlled displacement
            qml.adjoint(qml.S)(q),
            qml.H(q),
            hqml.ConditionalDisplacement(b, -phi + np.pi / 2, wires),
            qml.H(q),
            qml.S(q),
        ]

    @property
    def resource_params(self):
        return {}

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "Cat", cache=cache
        )


# ---------------------------------------------------------------------------
# Decomposition registration
# ---------------------------------------------------------------------------


@qml.register_resources(
    {
        qml.H: 4,
        qml.S: 1,
        adjoint_resource_rep(qml.S): 1,
        hqml.ConditionalDisplacement: 2,
    }
)
def _catstate_decomp(*params, wires, **_):
    a, phi = params
    q = wires[0]

    qml.H(q)
    hqml.ConditionalDisplacement(a, phi, wires)
    qml.H(q)

    qml.adjoint(qml.S)(q)
    qml.H(q)
    b = np.pi / (8 * a)
    hqml.ConditionalDisplacement(b, -phi + np.pi / 2, wires)
    qml.H(q)
    qml.S(q)


qml.add_decomps(CatState, _catstate_decomp)


# ---------------------------------------------------------------------------
# Benchmark circuits
# ---------------------------------------------------------------------------


def cat_state_expval_n(alpha):
    """Prepare cat state and measure <N> on the qumode."""
    CatState(alpha, 0, wires=["q", "m"])
    return hqml.expval(hqml.N("m"))


def cat_state_expval_z(alpha):
    """Prepare cat state and measure <Z> on the ancilla qubit."""
    CatState(alpha, 0, wires=["q", "m"])
    return hqml.expval(qml.Z("q"))


def cat_state_expval_z_and_n(alpha):
    """Prepare cat state and measure both <Z> on qubit and <N> on qumode."""
    CatState(alpha, 0, wires=["q", "m"])
    return hqml.expval(qml.Z("q")), hqml.expval(hqml.N("m"))


def cat_state_qubit_bloch(alpha):
    """Prepare cat state and return (<X>, <Y>, <Z>) on the ancilla qubit.

    Qubit purity = (1 + <X>^2 + <Y>^2 + <Z>^2) / 2
    """
    CatState(alpha, 0, wires=["q", "m"])
    return (
        hqml.expval(qml.X("q")),
        hqml.expval(qml.Y("q")),
        hqml.expval(qml.Z("q")),
    )


def cat_state_sample_n(alpha):
    """Prepare cat state and sample the qumode in the Fock basis."""
    CatState(alpha, 0, wires=["q", "m"])
    return hqml.sample(hqml.N("m"))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_cat_state(alpha=2.0):
    """Draw the cat state circuit."""
    import matplotlib.pyplot as plt

    hqml.draw_mpl(cat_state_expval_n, style="sketch")(alpha)
    plt.show()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_cat_state_benchmark(alpha, max_fock_level=256):
    """
    Run cat state benchmark for a given alpha.

    Returns dict with mean_n, sigma_z, qubit_purity, fidelity, bloch_vector.
    """
    dev = qml.device("bosonicqiskit.hybrid", max_fock_level=max_fock_level)

    # <Z> and <N> together
    qnode_zn = qml.QNode(cat_state_expval_z_and_n, dev)
    sigma_z, mean_n = qnode_zn(alpha)
    sigma_z, mean_n = float(sigma_z), float(mean_n)

    # Qubit fidelity: F = (1 + <Z>) / 2
    fidelity = (1 + sigma_z) / 2

    # Qubit Bloch vector for purity
    qnode_bloch = qml.QNode(cat_state_qubit_bloch, dev)
    ex, ey, ez = qnode_bloch(alpha)
    ex, ey, ez = float(ex), float(ey), float(ez)
    qubit_purity = (1 + ex**2 + ey**2 + ez**2) / 2

    return {
        "mean_n": mean_n,
        "sigma_z": sigma_z,
        "fidelity": fidelity,
        "qubit_purity": qubit_purity,
        "bloch_vector": (ex, ey, ez),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from scipy.special import gammaln

    def cat_state_expected_n(alpha):
        """Analytic <N> for an even cat state."""
        n = np.arange(512)
        alpha_sq = alpha**2
        log_fac = gammaln(n + 1)
        log_p = -alpha_sq + n * np.log(alpha_sq) - log_fac
        log_norm = np.log(2) - np.log(1 + np.exp(-2 * alpha_sq))
        log_pn = log_norm + log_p
        log_pn[n % 2 != 0] = -np.inf
        pn = np.exp(log_pn)
        return (n * pn).sum()

    alphas = [3, 4, 5, 6]
    fock = 256

    print(f"Cat state benchmark  (max_fock_level={fock})")
    print("=" * 65)

    for a in alphas:
        res = run_cat_state_benchmark(a, max_fock_level=fock)
        expected_n = cat_state_expected_n(a)
        expected_f = 1 - np.pi**2 / (64 * a**2)
        print(f"  alpha = {a}")
        print(f"    <N>        = {res['mean_n']:.4f}  (theory: {expected_n:.4f})")
        print(f"    <Z>        = {res['sigma_z']:.4f}")
        print(f"    fidelity   = {res['fidelity']:.4f}  (theory >= {expected_f:.4f})")
        print(f"    qubit pur. = {res['qubit_purity']:.6f}")
        print(
            f"    Bloch      = ({res['bloch_vector'][0]:.4f}, "
            f"{res['bloch_vector'][1]:.4f}, {res['bloch_vector'][2]:.4f})"
        )
        print()

    draw_cat_state(4.0)
