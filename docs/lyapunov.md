# Lyapunov Exponent Estimation

## Background

The maximal Lyapunov exponent (MLE) measures the average rate of divergence of infinitesimally close trajectories. A positive MLE is the defining signature of chaos.

- **MLE > 0**: Chaotic (strange attractor)
- **MLE ~ 0**: Quasiperiodic (limit cycle / torus)
- **MLE < 0**: Dissipative (fixed point)

## Method: Rosenstein (1993)

formCollapse uses Rosenstein's algorithm for MLE estimation, chosen for robustness with short, noisy time series.

### Algorithm

1. **Build KD-tree** on the trajectory data for fast nearest-neighbor queries.

2. **Find nearest neighbors**: For each reference point `i`, find its closest neighbor `j` in phase space, excluding temporal neighbors within a Theiler window (default: 50 steps) to avoid correlated pairs.

3. **Track divergence**: For each valid pair `(i, j)`, compute the Euclidean distance `d(i+k, j+k)` at future time steps `k = 0, 1, ..., T`.

4. **Average**: Compute `<ln(d(k))>` averaged over all valid pairs.

5. **Linear fit**: The MLE is the slope of `<ln(d(k))>` vs `k * dt`.

### Performance

The implementation uses several optimizations:

- **Subsampling**: Only `max_reference_points` (default: 1000) evenly-spaced points are used as references, avoiding O(n^2) scaling.
- **Batch KD-tree queries**: All reference points are queried in a single batch call.
- **Vectorized divergence**: Distance computation at each time step is fully vectorized with numpy.

Result: ~0.1s for 10,000-point trajectories (vs ~100s for naive implementation).

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 1.0 | Time step between data points |
| `theiler_window` | 50 | Minimum temporal separation for neighbor pairs |
| `max_divergence_steps` | 200 | Number of future steps to track |
| `max_reference_points` | 1000 | Max reference points (subsampling) |

### Usage

```python
from src.attractors.simulators import compute_lyapunov_exponent

mle = compute_lyapunov_exponent(data, dt=0.0033)
```

## Classification

The `classify_attractor()` function uses the MLE along with trajectory statistics:

| Classification | Criteria |
|---------------|----------|
| `divergent` | NaN/Inf values or data range > 1e6 |
| `fixed_point` | All coordinate std < 1e-3 |
| `strange_attractor` | MLE > 0.01 |
| `limit_cycle` | Bounded, non-chaotic (everything else) |

```python
from src.attractors.simulators import classify_attractor

classification, metrics = classify_attractor(data, mle)
# classification: "strange_attractor"
# metrics: {"lyapunov_exponent": 1.26, "std": [...], "range": [...], "mean": [...]}
```

## References

- Rosenstein, M.T., Collins, J.J., De Luca, C.J. (1993). "A practical method for calculating largest Lyapunov exponents from small data sets." *Physica D*, 65(1-2), 117-134.
- Wolf, A., Swift, J.B., Swinney, H.L., Vastano, J.A. (1985). "Determining Lyapunov exponents from a time series." *Physica D*, 16(3), 285-317.
