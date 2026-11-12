import numpy as np

def simulate_gbm(S0, mu, sigma, T, steps, paths, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / steps
    Z = rng.standard_normal((paths, steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    S = S0 * np.exp(np.hstack([np.zeros((paths, 1)), log_paths]))
    return S

S = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, steps=252, paths=100_000)
print("Mean S_T:", S[:, -1].mean(), "| Expected:", 100 * np.exp(0.05))