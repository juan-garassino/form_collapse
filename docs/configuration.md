# Configuration Guide

## config.yaml

The main configuration file defines simulation parameters, attractor systems, and GAN settings.

### Global Settings

```yaml
global:
  default_sim_time: 33      # Simulation duration (seconds of simulated time)
  default_sim_steps: 9999    # Number of output points
```

### System Definition (ODE)

Each ODE system needs a function name, scale, and parameter ranges:

```yaml
systems:
  Lorenz:
    func: lorenz_system       # Must match a key in SYSTEM_FUNCTIONS
    scale: 20.0               # Initial condition scale factor
    sim_time: 50              # Optional: override global default
    sim_steps: 5000           # Optional: override global default
    params:
      sigma:
        type: uniform
        range: [9, 11]        # Random sampling range
      rho:
        type: uniform
        range: [20, 30]
      beta:
        type: uniform
        range: [2, 3]
    bifurcation_param:        # Optional: enables bifurcation diagrams
      name: rho
      start: 0
      stop: 50
      num: 200
```

### System Definition (Discrete Map)

Discrete maps use `type: map` and specify iteration count instead of simulation time:

```yaml
  Henon:
    func: henon_map
    type: map
    scale: 1.0
    n_iterations: 10000
    transient: 1000           # Steps to discard before recording
    params:
      a:
        type: uniform
        range: [1.2, 1.4]
      b:
        type: uniform
        range: [0.2, 0.4]
```

### Parameter Types

- **Range parameters**: `{type: uniform, range: [lo, hi]}` — sampled uniformly at random each run
- **Fixed parameters**: Just a scalar value (e.g., `sigma: 10.0`)

### GAN Settings

```yaml
gan_params:
  latent_dim: 100
  batch_size: 64
  num_epochs: 50
  seq_len: 64                # Trajectory segment length for Conv1d GAN
```

## Known-Good Presets

Use `--preset` to bypass random sampling. Presets are defined in `src/utils/config.py`:

| System | Parameters |
|--------|-----------|
| Lorenz | sigma=10, rho=28, beta=8/3 |
| Aizawa | a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1 |
| Rabinovich-Fabrikant | alpha=0.14, gamma=0.10 |
| Chen | a=35, b=3, c=28 |
| Halvorsen | a=1.27 |
| Newton-Leipnik | a=0.4, b=0.175 |
| Three-Scroll | a=40, b=55, c=1.833 |
| Rossler | a=0.2, b=0.2, c=5.7 |
| Anishchenko | a=1.2, b=0.5, c=0.6 |
| Arnold | omega=1.0 |
| Burke-Shaw | a=10, b=4.272, c=2.73 |
| Chen-Celikovsky | a=36, c=20, d=1.833 |
| Finance | a=0.001, b=0.2, c=1.1 |
| Qi-Chen | a=38, b=2.666, c=80 |
| Rayleigh-Benard | a=9, b=5, c=12 |
| TSUCS1 | a=40, b=0.16, c=0.65 |
| Liu-Chen | a=5, b=-10, c=-3.78, d=1 |
| Henon (map) | a=1.4, b=0.3 |
| Ikeda (map) | u=0.918 |

## Parameter Sweep (LHS)

For systematic parameter exploration, use `--sweep N` from the CLI or the Python API:

```python
from src.utils.config import get_config, generate_params_lhs

config = get_config("config.yaml")
param_sets = generate_params_lhs(config, "Lorenz", n_samples=50)
# Returns 50 parameter dicts with well-distributed samples
```

This uses `scipy.stats.qmc.LatinHypercube` for even coverage of the parameter space. Results are visualized as a classification heatmap showing which parameter regions produce strange attractors.

```bash
python main.py --attractor Lorenz --sweep 20 --mode trajectory
```

## Adding a New ODE System

1. Define the ODE function in `src/attractors/attractors.py`:

```python
def my_system(X, t, params):
    x, y, z = X
    a = params['a']
    return np.array([...])
```

2. Register it in `src/attractors/simulators.py` under `SYSTEM_FUNCTIONS`.

3. Add configuration in `config.yaml`.

4. Optionally add a preset in `src/utils/config.py` under `KNOWN_GOOD_PRESETS`.

## Adding a New Discrete Map

1. Define the map function in `src/attractors/maps.py`:

```python
def my_map(state, params):
    x, y = state[0], state[1]
    return np.array([new_x, new_y, 0.0])
```

2. Add it to `MAP_FUNCTIONS` in the same file.

3. Add config entry with `type: map`, `n_iterations`, and `transient`.
