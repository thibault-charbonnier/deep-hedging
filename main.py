from src.simulation import SABRProcess

sabr_cfg = {
    "maturity": 1.0,
    "n_steps": 252,
    "S0": 100.0,
    "mu": 0.05,
    "sigma0": 0.20,
    "nu": 0.60,
    "rho": -0.40,
}

process = SABRProcess(sabr_cfg)
paths = process.simulate_paths(n_paths=10)

print(paths["S"].shape)
print(paths["sigma"].shape)