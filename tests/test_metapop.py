import numpy as np
import pytest

from metapop import (amplitude_for_duty_cycle, build_supra, cyclic_schedule, duty_cycle,
                     lambda_mean, lambda_product, make_landscape, persistence_radius,
                     simulate, sinusoidal_radius, spectral_radius, static_lambda,
                     threshold_scenario, transition_matrix)


def random_directed_scenario(n=8, seed=0):
    rng = np.random.default_rng(seed)
    S_list = []
    for _ in range(4):
        S = (rng.random((n, n)) < 0.4) * rng.random((n, n))
        np.fill_diagonal(S, 0)
        S_list.append(S)
    e = np.full(n, 0.2)
    return lambda k: (S_list[k % 4], e)


def test_lambda_product_matches_brute_force():
    supra = build_supra(random_directed_scenario(), c=0.5, T=12)
    brute = spectral_radius(np.linalg.multi_dot(supra[::-1])) ** (1 / len(supra))
    assert lambda_product(supra) == pytest.approx(brute, rel=1e-10)


def test_lambda_product_no_overflow():
    supra = build_supra(random_directed_scenario(), c=5.0, T=5000)
    assert np.isfinite(lambda_product(supra))


def test_static_network_all_predictors_agree():
    land = make_landscape(50, 10, seed=1)
    scen = threshold_scenario(land, sinusoidal_radius(4.0, w=0.0), 0.002, 1.0, 1.0)
    supra = build_supra(scen, c=1.0, T=200)
    rho = spectral_radius(supra[0])
    assert lambda_product(supra) == pytest.approx(rho, rel=1e-10)
    assert lambda_mean(supra) == pytest.approx(rho, rel=1e-10)


def test_linearisation_matches_nonlinear_dynamics_on_directed_graph():
    """M_k must be the Jacobian of the nonlinear update (I − D + c·Sᵀ, not c·S)."""
    scen, c, T = random_directed_scenario(), 0.5, 8
    P0 = np.random.default_rng(1).random(8) * 1e-9
    P_T = simulate(scen, c, T, P0)[:, -1]
    linear = np.linalg.multi_dot(build_supra(scen, c, T)[::-1]) @ P0
    np.testing.assert_allclose(P_T, linear, rtol=1e-6)


def test_persistence_radius_brackets_threshold():
    land = make_landscape(100, 20, seed=40)
    args = (1.0, 0.001, 1.0, 1.0)
    r_p = persistence_radius(land, *args)
    assert r_p > 1.0
    assert static_lambda(land, r_p - 0.05, *args) < 1 <= static_lambda(land, r_p + 0.05, *args)


def test_duty_cycle_inverse():
    for phi in (0.1, 0.25, 0.5, 0.75):
        assert duty_cycle(amplitude_for_duty_cycle(phi, 2.0), 2.0) == pytest.approx(phi)


def test_cyclic_schedule():
    masks = ['a', 'b', 'c']
    f = cyclic_schedule(masks, [2, 1, 3])
    assert [f(k) for k in range(8)] == ['a', 'a', 'b', 'c', 'c', 'c', 'a', 'a']


def test_transition_matrix_definition():
    S = np.array([[0, 1.0], [0, 0]])          # patch 0 colonises patch 1
    M = transition_matrix(S, np.array([0.1, 0.2]), c=0.5)
    assert M[1, 0] == 0.5 and M[0, 1] == 0
