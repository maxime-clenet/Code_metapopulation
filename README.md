# Metapopulation persistence in a dynamically changing landscape

Code associated with the project "Metapopulation persistence in a dynamically changing landscape".

## Model

Discrete-time occupancy model on a landscape whose connectivity changes with time k.
`S_k[j, i]` is the colonisation pressure of source patch j on target patch i:

```
S_k[j, i] = A_j^β · K_k(d_ji) · mask_k[j, i]
q_i(k)    = ∏_j (1 − c·S_k[j, i]·P_j(k))
P_i(k+1)  = 1 − (1 − (1 − e_i)·P_i(k)) · q_i(k),       e_i = e_0·A_i^(−z)
```

Linearising at P = 0 gives the transition matrices `M_k = I − diag(e) + c·S_kᵀ`, and the
**temporal metapopulation capacity**

```
λ = ρ(M_{T−1} ··· M_0)^(1/T)        λ > 1 ⇔ persistence
```

`lambda_product` computes it exactly, renormalising the running product so it never overflows.

## Layout

```
metapop/                 library
  geometry.py            landscapes, periodic distances, percolation radius, ER/SBM masks
  model.py               nonlinear update, transition matrices, Hanski capacity
  scenarios.py           time-varying scenarios, radius signals r(k), duty cycle, r_p
  spectral.py            λ_product, λ_mean, shuffling, commutators, threshold search
  stochastic.py          Gillespie simulation
  plotting.py            shared figure helpers (figures saved to figures/)
experiments/
  static/                single static landscape (Euler, discrete, Gillespie)
  temporal/              early scenarios: baseline, α(t), modularity, seasons, spectrum
  rgn/                   oscillating random geometric network: main results
tests/                   checks of the library (pytest)
figures/                 generated PDFs
```

## Usage

```bash
pip install -e ".[test]"
pytest
python experiments/rgn/rgn_analysis.py
```
