"""First-passage-time distributions for Gaussian fermionic networks.

The module implements the one-way particle-counting construction derived in
``main.tex`` (Sec. "FCS and FPT with non-Gaussian initial state: fermions").
A Gaussian steady state is conditioned on one particle-removal jump.  The
resulting quadratic-Grassmann-polynomial-times-Gaussian kernel is propagated
exactly under the tilted quadratic Lindbladian, and the first-passage-time
(FPT) densities are recovered by Fourier inversion.

This is the fermionic mirror of ``bosonic_fpt.py``: same call signatures, same
return containers, same numerical strategy (Diffrax Tsit5 with a PID
controller, one integration per counting field, then two inverse transforms).

Conventions
-----------
Majorana operators are interleaved as ``w = (w_1, w_2, w_3, w_4, ...)`` with

    f_j = (w_{2j-1} + i w_{2j}) / 2,      {w_j, w_k} = 2 delta_jk,

so mode ``j`` (zero-based) owns the Majorana pair ``(2j, 2j+1)``.  The
quadratic Hamiltonian is

    H = sum_ij h_ij f_i^dagger f_j
        + 1/2 sum_ij (pairing_ij f_i^dagger f_j^dagger + conj(pairing_ij) f_j f_i)
      = (i/4) w^T G w + const,

where ``h = diag(frequencies) + couplings`` is Hermitian, ``pairing`` is a
complex *antisymmetric* matrix, and ``G`` is real antisymmetric.  There is no
coherent-drive sector: a term linear in the mode operators would violate
fermion parity, which is why the bosonic ``drives``/``displacement`` variables
have no counterpart here.

Each site carries thermal loss and gain channels

    sqrt(gamma_i (1 - occupation_i)) f_i,
    sqrt(gamma_i occupation_i) f_i^dagger,

with ``occupation_i`` the bath Fermi factor in ``[0, 1]``.  Only loss channels
selected by ``monitored`` are counted, so the implemented FPT formula is for a
one-way emission (particle-out) current.

The Gaussian data is the covariance ``Gamma_jk = (i/2) <[w_j, w_k]>`` (real
antisymmetric, ``Gamma = (2n - 1) K`` per mode for a thermal state with
``K = [[0, 1], [-1, 0]]``), and the non-Gaussian dressing is the antisymmetric
matrix ``T`` of ``omega = (N + (i/2) theta^T T theta) omega_G``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import diffrax
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.integrate import trapezoid
from scipy.linalg import solve_continuous_lyapunov


ArrayLike = Sequence[float] | Sequence[complex] | np.ndarray


@dataclass(frozen=True)
class FermionicNetwork:
    """Matrices defining an N-mode quadratic fermionic network."""

    frequencies: np.ndarray
    decay_rates: np.ndarray
    occupations: np.ndarray
    couplings: np.ndarray
    pairing: np.ndarray
    monitored: np.ndarray
    hamiltonian_matrix: np.ndarray
    jump_vectors: np.ndarray
    jump_modes: np.ndarray
    jump_kinds: tuple[str, ...]
    monitored_channels: np.ndarray
    b_real: np.ndarray
    b_imag: np.ndarray
    drift: np.ndarray
    source: np.ndarray

    @property
    def n_modes(self) -> int:
        return self.frequencies.size

    @property
    def dimension(self) -> int:
        return 2 * self.n_modes

    def loss_channel(self, mode: int) -> int:
        """Return the channel index of the loss jump for ``mode``."""
        if not 0 <= mode < self.n_modes:
            raise IndexError(
                f"mode must lie in [0, {self.n_modes - 1}], got {mode}"
            )
        return 2 * mode


@dataclass(frozen=True)
class GaussianSteadyState:
    """Untilted long-time Majorana covariance."""

    covariance: np.ndarray
    drift_eigenvalues: np.ndarray
    lyapunov_residual: float
    antisymmetry_residual: float
    maximum_correlation_eigenvalue: float


@dataclass(frozen=True)
class ParticleRemovedState:
    """Normalized Grassmann dressing produced by one loss jump."""

    mode: int
    channel: int
    pre_normalization_jump_rate: float
    normalization: float
    dressing: np.ndarray


@dataclass(frozen=True)
class FPTDiagnostics:
    """Numerical diagnostics associated with a Fourier inversion."""

    maximum_steps: int
    maximum_imaginary_coefficient: float
    minimum_fpt_density: float
    count_aliasing_tail: float
    fpt_aliasing_tail: float
    integrated_fpt_probabilities: np.ndarray
    maximum_covariance_norm: float


@dataclass(frozen=True)
class FPTResult:
    """Output of :func:`first_passage_time_distribution`."""

    times: np.ndarray
    passage_numbers: np.ndarray
    fpt_density: np.ndarray
    count_numbers: np.ndarray
    count_probabilities: np.ndarray
    steady_state: GaussianSteadyState
    particle_removed_state: ParticleRemovedState
    diagnostics: FPTDiagnostics


def _one_dimensional_array(
    values: ArrayLike,
    name: str,
    *,
    length: int | None = None,
    dtype: type = float,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must have length {length}, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def majorana_generator(n_modes: int) -> np.ndarray:
    """Return the single-mode generator K for interleaved Majoranas.

    ``K = [[0, 1], [-1, 0]]`` on each mode block; a thermal state of occupation
    ``n`` has covariance ``(2n - 1) K``, the vacuum ``-K``.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    block = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return np.kron(np.eye(n_modes), block)


def thermal_covariance(occupations: ArrayLike) -> np.ndarray:
    """Return the Majorana covariance of a product thermal state."""
    occupations = _one_dimensional_array(occupations, "occupations")
    if np.any(occupations < 0.0) or np.any(occupations > 1.0):
        raise ValueError("occupations must lie in [0, 1]")
    weights = np.repeat(2.0 * occupations - 1.0, 2)
    return weights[:, None] * majorana_generator(occupations.size)


def build_fermionic_network(
    frequencies: ArrayLike,
    decay_rates: ArrayLike,
    occupations: ArrayLike,
    monitored: Sequence[bool] | np.ndarray,
    couplings: np.ndarray | None = None,
    pairing: np.ndarray | None = None,
) -> FermionicNetwork:
    """Build the Hamiltonian and thermal Lindblad channels of a network.

    Parameters
    ----------
    frequencies:
        Bare on-site energies or rotating-frame detunings, one per mode.
    decay_rates:
        Positive bath-coupling rates ``gamma_i``.
    occupations:
        Bath Fermi factors ``n_i`` in ``[0, 1]``.
    monitored:
        Boolean mask.  A true entry counts the corresponding loss channel.
    couplings:
        Optional Hermitian hopping matrix.  Its diagonal must be zero because
        on-site terms are supplied through ``frequencies``.
    pairing:
        Optional complex antisymmetric matrix in
        ``(f^dagger pairing f^dagger + f conj(pairing) f) / 2``, i.e.
        superconducting/Kitaev-type pair creation between distinct modes.
    """
    frequencies = _one_dimensional_array(frequencies, "frequencies")
    n_modes = frequencies.size
    decay_rates = _one_dimensional_array(
        decay_rates, "decay_rates", length=n_modes
    )
    occupations = _one_dimensional_array(
        occupations, "occupations", length=n_modes
    )
    monitored_array = np.asarray(monitored, dtype=bool)
    if monitored_array.ndim != 1 or monitored_array.size != n_modes:
        raise ValueError(f"monitored must be a boolean array of length {n_modes}")
    if np.any(decay_rates <= 0.0):
        raise ValueError("all decay_rates must be strictly positive")
    if np.any(occupations < 0.0) or np.any(occupations > 1.0):
        raise ValueError("occupations must lie in [0, 1] for fermions")

    if couplings is None:
        couplings_array = np.zeros((n_modes, n_modes), dtype=complex)
    else:
        couplings_array = np.asarray(couplings, dtype=complex)
        if couplings_array.shape != (n_modes, n_modes):
            raise ValueError(
                f"couplings must have shape {(n_modes, n_modes)}, "
                f"got {couplings_array.shape}"
            )
        if not np.all(np.isfinite(couplings_array)):
            raise ValueError("couplings contains a non-finite value")
        if not np.allclose(couplings_array, couplings_array.conj().T):
            raise ValueError("couplings must be Hermitian")
        if not np.allclose(np.diag(couplings_array), 0.0):
            raise ValueError(
                "the diagonal of couplings must be zero; use frequencies "
                "for on-site terms"
            )

    if pairing is None:
        pairing_array = np.zeros((n_modes, n_modes), dtype=complex)
    else:
        pairing_array = np.asarray(pairing, dtype=complex)
        if pairing_array.shape != (n_modes, n_modes):
            raise ValueError(
                f"pairing must have shape {(n_modes, n_modes)}, "
                f"got {pairing_array.shape}"
            )
        if not np.all(np.isfinite(pairing_array)):
            raise ValueError("pairing contains a non-finite value")
        if not np.allclose(pairing_array, -pairing_array.T):
            raise ValueError(
                "pairing must be antisymmetric; symmetric parts annihilate "
                "against fermionic statistics"
            )

    single_particle_hamiltonian = np.diag(frequencies.astype(complex)) + couplings_array
    dimension = 2 * n_modes

    # H = sum_ab kernel_ab w_a w_b + const, then G = 4 Im(kernel) because the
    # kernel is Hermitian and only its antisymmetric part survives.
    kernel = np.zeros((dimension, dimension), dtype=complex)
    pair_kernel = np.zeros((dimension, dimension), dtype=complex)
    for i in range(n_modes):
        for j in range(n_modes):
            hij = single_particle_hamiltonian[i, j]
            kernel[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] += (hij / 4.0) * (
                np.array([[1.0, 1.0j], [-1.0j, 1.0]])
            )
            dij = pairing_array[i, j]
            pair_kernel[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] += (dij / 8.0) * (
                np.array([[1.0, -1.0j], [-1.0j, -1.0]])
            )
    kernel = kernel + pair_kernel + pair_kernel.conj().T
    hamiltonian_matrix = 4.0 * kernel.imag
    hamiltonian_matrix = 0.5 * (hamiltonian_matrix - hamiltonian_matrix.T)

    # Channels are ordered (loss_0, gain_0, loss_1, gain_1, ...).
    jump_vectors = np.zeros((2 * n_modes, dimension), dtype=complex)
    jump_modes = np.repeat(np.arange(n_modes), 2)
    jump_kinds: list[str] = []
    monitored_channels = np.zeros(2 * n_modes, dtype=bool)
    for mode in range(n_modes):
        loss_scale = np.sqrt(
            decay_rates[mode] * (1.0 - occupations[mode])
        ) / 2.0
        gain_scale = np.sqrt(decay_rates[mode] * occupations[mode]) / 2.0
        loss = 2 * mode
        gain = loss + 1
        jump_vectors[loss, 2 * mode : 2 * mode + 2] = (
            loss_scale * np.array([1.0, 1.0j])
        )
        jump_vectors[gain, 2 * mode : 2 * mode + 2] = (
            gain_scale * np.array([1.0, -1.0j])
        )
        jump_kinds.extend(("loss", "gain"))
        monitored_channels[loss] = monitored_array[mode]

    channel_matrices = np.einsum(
        "mi,mj->mij", jump_vectors, np.conjugate(jump_vectors)
    )
    b_real = channel_matrices.real
    b_imag = channel_matrices.imag
    b_real_total = np.sum(b_real, axis=0)
    b_imag_total = np.sum(b_imag, axis=0)

    drift = hamiltonian_matrix - 2.0 * b_real_total
    source = 4.0 * b_imag_total

    return FermionicNetwork(
        frequencies=frequencies,
        decay_rates=decay_rates,
        occupations=occupations,
        couplings=couplings_array,
        pairing=pairing_array,
        monitored=monitored_array,
        hamiltonian_matrix=hamiltonian_matrix,
        jump_vectors=jump_vectors,
        jump_modes=jump_modes,
        jump_kinds=tuple(jump_kinds),
        monitored_channels=monitored_channels,
        b_real=b_real,
        b_imag=b_imag,
        drift=drift,
        source=source,
    )


def steady_state(
    network: FermionicNetwork,
    *,
    stability_tolerance: float = 1e-12,
) -> GaussianSteadyState:
    """Solve the long-time Lyapunov equation for the Majorana covariance."""
    eigenvalues = np.linalg.eigvals(network.drift)
    largest_real_part = float(np.max(eigenvalues.real))
    if largest_real_part >= -stability_tolerance:
        raise ValueError(
            "the network has no unique stable Gaussian steady state: "
            f"max Re(eig(E)) = {largest_real_part:+.6e}"
        )

    covariance = solve_continuous_lyapunov(network.drift, -network.source)
    covariance = np.real_if_close(covariance).real
    antisymmetry_residual = float(
        np.linalg.norm(covariance + covariance.T, ord="fro")
    )
    covariance = 0.5 * (covariance - covariance.T)

    lyapunov_residual = float(
        np.linalg.norm(
            network.drift @ covariance
            + covariance @ network.drift.T
            + network.source,
            ord="fro",
        )
    )
    correlation_eigenvalues = np.linalg.eigvalsh(1.0j * covariance)
    maximum_correlation_eigenvalue = float(np.max(np.abs(correlation_eigenvalues)))
    if maximum_correlation_eigenvalue > 1.0 + 1e-8:
        raise RuntimeError(
            "the computed steady covariance is unphysical (a mode correlation "
            f"exceeds one: {maximum_correlation_eigenvalue})"
        )

    return GaussianSteadyState(
        covariance=covariance,
        drift_eigenvalues=eigenvalues,
        lyapunov_residual=lyapunov_residual,
        antisymmetry_residual=antisymmetry_residual,
        maximum_correlation_eigenvalue=maximum_correlation_eigenvalue,
    )


def particle_removed_state(
    network: FermionicNetwork,
    gaussian_state: GaussianSteadyState,
    mode: int,
    *,
    normalization_tolerance: float = 1e-14,
) -> ParticleRemovedState:
    """Construct the normalized state after one particle-removal jump."""
    channel = network.loss_channel(mode)
    covariance = gaussian_state.covariance
    b_real = network.b_real[channel]
    b_imag = network.b_imag[channel]

    jump_rate = np.trace(b_real) + np.trace(covariance @ b_imag)
    jump_rate = float(np.real_if_close(jump_rate))
    if jump_rate <= normalization_tolerance:
        raise ValueError(
            f"cannot condition on loss from mode {mode}: its steady-state "
            f"jump rate is {jump_rate:.6e}"
        )

    t_mu = 2.0 * (
        b_imag
        - covariance @ b_real
        - b_real @ covariance
        - covariance @ b_imag @ covariance
    )

    # omega_post = (1 + (i/2) theta^T T theta) omega_G.
    dressing = np.real_if_close(t_mu / jump_rate).real
    dressing = 0.5 * (dressing - dressing.T)
    return ParticleRemovedState(
        mode=mode,
        channel=channel,
        pre_normalization_jump_rate=jump_rate,
        normalization=1.0,
        dressing=dressing,
    )


def _inverse_counting_transform(
    samples: np.ndarray,
    phase_offset: float,
) -> np.ndarray:
    """Invert samples taken at theta_k = 2 pi (k + offset) / N."""
    coefficients = np.fft.ifft(samples, axis=0)
    indices = np.arange(samples.shape[0])
    phase = np.exp(
        2.0j * np.pi * phase_offset * indices / samples.shape[0]
    )
    return phase[:, None] * coefficients


def first_passage_time_distribution(
    network: FermionicNetwork,
    times: ArrayLike,
    *,
    initial_jump_mode: int,
    passage_numbers: Sequence[int] = (1,),
    n_counting_fields: int = 64,
    counting_phase_offset: float = 0.5,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100_000,
    progress: bool = False,
) -> FPTResult:
    """Calculate one-way emission FPT densities after particle removal.

    ``passage_numbers=(1, 2, ...)`` requests the time density for the first,
    second, ... monitored emission after the conditioning jump.  Diffrax's
    adaptive fifth-order Tsitouras solver and PID step-size controller are
    used for every counting field.

    The default ``counting_phase_offset=0.5`` places the counting fields on the
    half-step offset grid ``z_k = exp(i pi (2k + 1) / N_z)``, which keeps the
    tilted Riccati flow away from the real axis where the fermionic covariance
    develops the coordinate singularities of ``det X_t = 0``.  The diagnostic
    ``maximum_covariance_norm`` reports how close the flow came to one.
    """
    times_array = _one_dimensional_array(times, "times")
    if times_array.size < 2:
        raise ValueError("times must contain at least two points")
    if not np.isclose(times_array[0], 0.0):
        raise ValueError("times must start at zero")
    if np.any(np.diff(times_array) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if n_counting_fields < 4:
        raise ValueError("n_counting_fields must be at least 4")
    if not 0.0 <= counting_phase_offset < 1.0:
        raise ValueError("counting_phase_offset must lie in [0, 1)")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("rtol and atol must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    passage_array = np.asarray(passage_numbers, dtype=int)
    if passage_array.ndim != 1 or passage_array.size == 0:
        raise ValueError("passage_numbers must be a non-empty 1D sequence")
    if np.any(passage_array < 1):
        raise ValueError("passage_numbers must be positive")
    if np.unique(passage_array).size != passage_array.size:
        raise ValueError("passage_numbers must not contain duplicates")
    if np.max(passage_array) > n_counting_fields:
        raise ValueError(
            "each passage number must be no larger than n_counting_fields"
        )
    if not np.any(network.monitored_channels):
        raise ValueError("at least one mode must be monitored")

    gaussian_state = steady_state(network)
    removed_state = particle_removed_state(
        network, gaussian_state, initial_jump_mode
    )

    # Convert constant model data once; all ODE states are complex because the
    # unit-circle counting fields make eta_mu complex.
    complex_dtype = jnp.complex128
    drift = jnp.asarray(network.drift, dtype=complex_dtype)
    source = jnp.asarray(network.source, dtype=complex_dtype)
    b_real_all = jnp.asarray(network.b_real, dtype=complex_dtype)
    b_imag_all = jnp.asarray(network.b_imag, dtype=complex_dtype)
    save_times = jnp.asarray(times_array)
    n_channels = network.jump_vectors.shape[0]

    complex_y0 = (
        jnp.asarray(gaussian_state.covariance, dtype=complex_dtype),
        jnp.asarray(0.0, dtype=complex_dtype),
        jnp.asarray(removed_state.dressing, dtype=complex_dtype),
        jnp.asarray(1.0, dtype=complex_dtype),
    )

    # Diffrax currently labels native complex-state integration experimental.
    # Split every complex leaf into a final (real, imaginary) axis so that the
    # numerical solver itself follows a purely real system.
    def to_real_pair(value):
        return jnp.stack((jnp.real(value), jnp.imag(value)), axis=-1)

    def from_real_pair(value):
        return value[..., 0] + 1.0j * value[..., 1]

    y0 = tuple(to_real_pair(value) for value in complex_y0)

    def vector_field(t, state, eta):
        del t
        covariance, log_norm, dressing, normalization = state
        covariance_dot = drift @ covariance + covariance @ drift.T + source
        log_norm_dot = jnp.zeros_like(log_norm)
        dressing_dot = drift @ dressing + dressing @ drift.T
        normalization_dot = jnp.zeros_like(normalization)

        for channel in range(n_channels):
            eta_mu = eta[channel]
            b_real = b_real_all[channel]
            b_imag = b_imag_all[channel]
            covariance_dot = covariance_dot + 2.0 * eta_mu * (
                b_imag
                - covariance @ b_real
                - b_real @ covariance
                - covariance @ b_imag @ covariance
            )
            log_norm_dot = log_norm_dot + eta_mu * (
                jnp.trace(b_real) + jnp.trace(covariance @ b_imag)
            )
            dressing_dot = dressing_dot - 2.0 * eta_mu * (
                b_real @ dressing
                + dressing @ b_real
                + covariance @ b_imag @ dressing
                + dressing @ b_imag @ covariance
            )
            normalization_dot = normalization_dot + eta_mu * jnp.trace(
                b_imag @ dressing
            )

        return (
            covariance_dot,
            log_norm_dot,
            dressing_dot,
            normalization_dot,
        )

    def real_vector_field(t, real_state, real_eta):
        complex_state = tuple(from_real_pair(value) for value in real_state)
        complex_eta = from_real_pair(real_eta)
        complex_derivative = vector_field(t, complex_state, complex_eta)
        return tuple(to_real_pair(value) for value in complex_derivative)

    term = diffrax.ODETerm(real_vector_field)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=save_times)

    @jax.jit
    def integrate_one(eta):
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=save_times[0],
            t1=save_times[-1],
            dt0=None,
            y0=y0,
            args=eta,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=max_steps,
            throw=True,
        )
        return solution.ys, solution.stats["num_steps"]

    angles = (
        2.0
        * np.pi
        * (np.arange(n_counting_fields) + counting_phase_offset)
        / n_counting_fields
    )
    counting_fields = np.exp(1.0j * angles)
    generating_function = np.empty(
        (n_counting_fields, times_array.size), dtype=complex
    )
    flux_generating_function = np.empty_like(generating_function)
    step_counts = np.empty(n_counting_fields, dtype=int)
    covariance_norms = np.empty(n_counting_fields, dtype=float)

    monitored_indices = np.flatnonzero(network.monitored_channels)
    for field_index, counting_field in enumerate(counting_fields):
        eta = np.zeros(n_channels, dtype=complex)
        eta[monitored_indices] = counting_field ** -1 - 1.0
        real_eta = to_real_pair(jnp.asarray(eta, dtype=complex_dtype))
        states, steps = integrate_one(real_eta)
        covariance_t, log_norm_t, dressing_t, norm_t = (
            np.asarray(value)[..., 0] + 1.0j * np.asarray(value)[..., 1]
            for value in states
        )
        step_counts[field_index] = int(np.asarray(steps))
        covariance_norms[field_index] = float(
            np.max(np.linalg.norm(covariance_t, ord=2, axis=(1, 2)))
        )

        exponential_norm = np.exp(log_norm_t)
        generating_function[field_index] = norm_t * exponential_norm
        flux = np.zeros(times_array.size, dtype=complex)
        for channel in monitored_indices:
            b_real = network.b_real[channel]
            b_imag = network.b_imag[channel]
            gaussian_part = np.trace(b_real) + np.einsum(
                "tij,ji->t", covariance_t, b_imag
            )
            dressing_part = np.einsum("ij,tji->t", b_imag, dressing_t)
            flux = flux + exponential_norm * (
                norm_t * gaussian_part + dressing_part
            )
        flux_generating_function[field_index] = flux

        if progress and (
            field_index == 0
            or (field_index + 1) % max(1, n_counting_fields // 8) == 0
            or field_index + 1 == n_counting_fields
        ):
            print(
                f"counting field {field_index + 1}/{n_counting_fields}; "
                f"Diffrax steps: {step_counts[field_index]}"
            )

    count_coefficients = _inverse_counting_transform(
        generating_function, counting_phase_offset
    )
    fpt_coefficients = _inverse_counting_transform(
        flux_generating_function, counting_phase_offset
    )
    selected_indices = passage_array - 1
    fpt_density = fpt_coefficients[selected_indices].real
    count_probabilities = count_coefficients.real

    tail_start = max(1, int(0.9 * n_counting_fields))
    maximum_imaginary_coefficient = float(
        max(
            np.max(np.abs(count_coefficients.imag)),
            np.max(np.abs(fpt_coefficients.imag)),
        )
    )
    diagnostics = FPTDiagnostics(
        maximum_steps=int(np.max(step_counts)),
        maximum_imaginary_coefficient=maximum_imaginary_coefficient,
        minimum_fpt_density=float(np.min(fpt_density)),
        count_aliasing_tail=float(
            np.max(np.abs(count_coefficients[tail_start:]))
        ),
        fpt_aliasing_tail=float(
            np.max(np.abs(fpt_coefficients[tail_start:]))
        ),
        integrated_fpt_probabilities=np.asarray(
            trapezoid(fpt_density, x=times_array, axis=1)
        ),
        maximum_covariance_norm=float(np.max(covariance_norms)),
    )

    return FPTResult(
        times=times_array,
        passage_numbers=passage_array,
        fpt_density=fpt_density,
        count_numbers=np.arange(n_counting_fields),
        count_probabilities=count_probabilities,
        steady_state=gaussian_state,
        particle_removed_state=removed_state,
        diagnostics=diagnostics,
    )


__all__ = [
    "FermionicNetwork",
    "FPTDiagnostics",
    "FPTResult",
    "GaussianSteadyState",
    "ParticleRemovedState",
    "build_fermionic_network",
    "first_passage_time_distribution",
    "majorana_generator",
    "particle_removed_state",
    "steady_state",
    "thermal_covariance",
]
