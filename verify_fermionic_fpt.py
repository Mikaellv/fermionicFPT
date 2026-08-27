"""Brute-force Jordan-Wigner verification of fermionic_fpt.py."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, null_space

from fermionic_fpt import (
    build_fermionic_network,
    first_passage_time_distribution,
    particle_removed_state,
    steady_state,
)

np.set_printoptions(precision=6, suppress=True, linewidth=140)


def jw_modes(n):
    sm = np.array([[0.0, 1.0], [0.0, 0.0]])
    z = np.diag([1.0, -1.0])
    eye = np.eye(2)
    ops = []
    for j in range(n):
        factors = [z] * j + [sm] + [eye] * (n - 1 - j)
        op = factors[0]
        for f in factors[1:]:
            op = np.kron(op, f)
        ops.append(op.astype(complex))
    return ops


def majoranas(fs):
    w = []
    for f in fs:
        w.append(f + f.conj().T)
        w.append(-1.0j * (f - f.conj().T))
    return w


def build_exact(network):
    n = network.n_modes
    fs = jw_modes(n)
    w = majoranas(fs)
    dim = 2 ** n
    h = np.diag(network.frequencies.astype(complex)) + network.couplings
    ham = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        for j in range(n):
            ham += h[i, j] * fs[i].conj().T @ fs[j]
            d = network.pairing[i, j]
            ham += 0.5 * d * fs[i].conj().T @ fs[j].conj().T
            ham += 0.5 * np.conjugate(d) * fs[j] @ fs[i]
    jumps = []
    for channel in range(network.jump_vectors.shape[0]):
        l = network.jump_vectors[channel]
        jumps.append(sum(l[a] * w[a] for a in range(2 * n)))
    return fs, w, ham, jumps


def liouvillian(ham, jumps, eta):
    dim = ham.shape[0]
    eye = np.eye(dim)
    lio = -1.0j * (np.kron(ham, eye) - np.kron(eye, ham.T))
    for mu, l in enumerate(jumps):
        ldl = l.conj().T @ l
        lio += (1.0 + eta[mu]) * np.kron(l, l.conj())
        lio -= 0.5 * (np.kron(ldl, eye) + np.kron(eye, ldl.T))
    return lio


def report(label, error, tolerance):
    status = "ok " if error < tolerance else "FAIL"
    print(f"  [{status}] {label:<52s} {error:.3e}")
    return error < tolerance


def main():
    rng = np.random.default_rng(7)
    n = 3
    frequencies = np.array([0.9, -0.4, 0.25])
    decay_rates = np.array([0.7, 1.1, 0.5])
    occupations = np.array([0.3, 0.65, 0.15])
    couplings = np.zeros((n, n), dtype=complex)
    couplings[0, 1] = 0.45 + 0.2j
    couplings[1, 2] = -0.3 + 0.1j
    couplings += couplings.conj().T
    pairing = np.zeros((n, n), dtype=complex)
    pairing[0, 1] = 0.35 - 0.15j
    pairing[1, 2] = 0.2 + 0.05j
    pairing -= pairing.T
    monitored = np.array([True, False, True])

    network = build_fermionic_network(
        frequencies,
        decay_rates,
        occupations,
        monitored,
        couplings=couplings,
        pairing=pairing,
    )
    fs, w, ham, jumps = build_exact(network)
    dim = 2 ** n
    passed = []

    print("Structure checks")
    g = network.hamiltonian_matrix
    ham_majorana = 0.25j * sum(
        g[a, b] * w[a] @ w[b] for a in range(2 * n) for b in range(2 * n)
    )
    residual = ham_majorana - ham
    shift = np.trace(residual) / dim
    passed.append(
        report(
            "H = (i/4) w^T G w + const",
            np.linalg.norm(residual - shift * np.eye(dim)),
            1e-10,
        )
    )
    passed.append(
        report("G real antisymmetric", np.linalg.norm(g + g.T), 1e-12)
    )
    loss_exact = np.sqrt(decay_rates * (1.0 - occupations))
    gain_exact = np.sqrt(decay_rates * occupations)
    jump_error = 0.0
    for mode in range(n):
        jump_error = max(
            jump_error,
            np.linalg.norm(jumps[2 * mode] - loss_exact[mode] * fs[mode]),
            np.linalg.norm(
                jumps[2 * mode + 1] - gain_exact[mode] * fs[mode].conj().T
            ),
        )
    passed.append(report("L_mu = l_mu^T w reproduce sqrt(g)f, sqrt(g)f^dag", jump_error, 1e-12))

    print("\nSteady state")
    gaussian = steady_state(network)
    lio0 = liouvillian(ham, jumps, np.zeros(len(jumps)))
    kernel = null_space(lio0, rcond=1e-9)
    rho_ss = kernel[:, 0].reshape(dim, dim)
    rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
    rho_ss /= np.trace(rho_ss).real
    exact_cov = np.array(
        [
            [0.5j * np.trace(rho_ss @ (w[a] @ w[b] - w[b] @ w[a])) for b in range(2 * n)]
            for a in range(2 * n)
        ]
    ).real
    passed.append(
        report("Majorana covariance Gamma", np.linalg.norm(gaussian.covariance - exact_cov), 1e-9)
    )
    passed.append(report("Lyapunov residual", gaussian.lyapunov_residual, 1e-10))

    print("\nSingle-jump dressing")
    jump_mode = 0
    removed = particle_removed_state(network, gaussian, jump_mode)
    l0 = jumps[network.loss_channel(jump_mode)]
    rho1 = l0 @ rho_ss @ l0.conj().T
    rate_exact = np.trace(rho1).real
    passed.append(
        report(
            "jump rate <L^dag L>",
            abs(removed.pre_normalization_jump_rate - rate_exact),
            1e-9,
        )
    )
    rho1 /= rate_exact

    print("\nGenerating function and flux on the counting circle")
    n_fields = 16
    offset = 0.5
    angles = 2.0 * np.pi * (np.arange(n_fields) + offset) / n_fields
    zs = np.exp(1.0j * angles)
    times = np.linspace(0.0, 6.0, 25)
    monitored_indices = np.flatnonzero(network.monitored_channels)

    exact_m = np.empty((n_fields, times.size), dtype=complex)
    exact_flux_m = np.empty_like(exact_m)
    vec_rho1 = rho1.reshape(-1)
    for k, z in enumerate(zs):
        eta = np.zeros(len(jumps), dtype=complex)
        eta[monitored_indices] = 1.0 / z - 1.0
        lio = liouvillian(ham, jumps, eta)
        for it, t in enumerate(times):
            rho_t = (expm(lio * t) @ vec_rho1).reshape(dim, dim)
            exact_m[k, it] = np.trace(rho_t)
            flux = sum(
                np.trace(jumps[mu].conj().T @ jumps[mu] @ rho_t)
                for mu in monitored_indices
            )
            exact_flux_m[k, it] = flux

    result = first_passage_time_distribution(
        network,
        times,
        initial_jump_mode=jump_mode,
        passage_numbers=(1, 2, 3, 4),
        n_counting_fields=n_fields,
        counting_phase_offset=offset,
        rtol=1e-11,
        atol=1e-13,
    )

    # Recompute the module's raw generating data for a direct comparison.
    from fermionic_fpt import _inverse_counting_transform

    exact_counts = _inverse_counting_transform(exact_m, offset).real
    exact_fpt = _inverse_counting_transform(exact_flux_m, offset).real

    passed.append(
        report(
            "P_t(n) counting distribution",
            np.max(np.abs(result.count_probabilities - exact_counts)),
            1e-9,
        )
    )
    for idx, npass in enumerate(result.passage_numbers):
        passed.append(
            report(
                f"FPT density Pbar_{npass}(t)",
                np.max(np.abs(result.fpt_density[idx] - exact_fpt[npass - 1])),
                1e-9,
            )
        )

    print("\nDiagnostics")
    d = result.diagnostics
    print(f"  max solver steps                 {d.maximum_steps}")
    print(f"  max |Im| Fourier coefficient     {d.maximum_imaginary_coefficient:.3e}")
    print(f"  min FPT density                  {d.minimum_fpt_density:.3e}")
    print(f"  aliasing tail (counts / FPT)     {d.count_aliasing_tail:.3e} / {d.fpt_aliasing_tail:.3e}")
    print(f"  max ||Gamma(t;z)||_2             {d.maximum_covariance_norm:.3e}")
    print(f"  integrated FPT probabilities     {d.integrated_fpt_probabilities}")
    print(f"  sum_n P_t(n) at final time       {result.count_probabilities[:, -1].sum():.12f}")

    print("\nAll checks passed." if all(passed) else "\nSOME CHECKS FAILED.")


if __name__ == "__main__":
    main()
