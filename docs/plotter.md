# Pen Plotter Export

formCollapse exports attractor trajectories as SVG and G-code files optimized for pen plotters.

## SVG Export

```bash
python main.py --attractor Lorenz --mode trajectory --export svg --preset
```

Output: `results/<session>/svg/<name>_svg.svg`

Features:
- **RDP simplification**: Reduces point count while preserving shape (`--rdp-epsilon`, default 0.01)
- **Multi-segment paths**: Discontinuous trajectory sections become separate `<path>` elements
- **Stroke reorder**: Greedy nearest-neighbor TSP minimizes pen travel between segments
- **Configurable stroke width**: Via `stroke_width` parameter
- **Proper viewBox**: Resolution-independent SVG with `viewBox` attribute

## G-code Export

```bash
python main.py --attractor Lorenz --preset --export gcode
python main.py --attractor Lorenz --preset --export both
```

Output: `results/<session>/gcode/<name>_gcode.gcode`

### G-code Structure

```gcode
G21              ; Millimeters
G90              ; Absolute positioning
G0 Z5           ; Pen up
G0 X.. Y..      ; Rapid to segment start
G0 Z0           ; Pen down
G1 X.. Y.. F..  ; Draw with feed rate
...
G0 Z5           ; Pen up (between segments)
G0 X.. Y..      ; Rapid to next segment
G0 Z0           ; Pen down
...
G0 Z5           ; Pen up (final)
G0 X0 Y0        ; Return home
```

### Pen-Lift Detection

Unlike the original single pen-down approach, the plotter output now detects discontinuities in the trajectory. When the inter-point distance exceeds a threshold (5% of canvas diagonal), a pen-up / rapid move / pen-down sequence is inserted. This prevents visible drag lines between disconnected regions.

### Plotter Options

| Flag | Default | Description |
|------|---------|-------------|
| `--feed-rate F` | none | Appends `F{rate}` to G1 commands |
| `--canvas-margin` | 0.05 | Fraction of canvas reserved as margin (5%) |
| `--rdp-epsilon` | 0.01 | RDP simplification tolerance |
| `--export` | none | `svg`, `gcode`, or `both` |

### Python API

```python
from src.utils.svg_gcode import generate_gcode, save_svg

# G-code with feed rate and margin
generate_gcode(data, "my_attractor", output_dir,
               canvas_size=(200, 200), feed_rate=1000,
               canvas_margin=0.1, epsilon=0.005)

# SVG with custom stroke width
save_svg(data, "my_attractor", output_dir,
         canvas_size=(1000, 1000), stroke_width=0.5)
```

## 3D Projection

By default, the X-Y columns of 3D trajectory data are used for 2D plotter output. For better results:

```bash
# Manual projection angles
python main.py --attractor Lorenz --preset --export svg --elevation 30 --azimuth -60

# Auto-find best projection (maximizes convex hull area)
python main.py --attractor Lorenz --preset --export svg --auto-project
```

The `--auto-project` option tries 100 random elevation/azimuth combinations and picks the one that maximizes the 2D spread of the attractor.

## Multi-Attractor Composition

Layer multiple attractors into a single SVG/G-code file:

```bash
python main.py --compose Lorenz,Rossler,Chen --preset
```

Each layer gets:
- A separate `<g>` group in SVG with a distinct stroke color
- A `M6 T{n}` tool change in G-code (for multi-pen plotters)
- Independent normalization and positioning

## Tips

- **Smoothing**: Use `--smooth` (default on) for cleaner plotter output. Spline interpolation increases resolution to 10,000 points.
- **Best systems for plotting**: Lorenz, Halvorsen, Rossler, and Burke-Shaw produce the most visually striking pen plots.
- **Preset params**: Use `--preset` for reliable, well-known attractor shapes.
- **RDP epsilon**: Lower values (e.g., 0.001) preserve more detail but produce larger files. Higher values (e.g., 0.05) are faster but lose fine structure.
- **Stroke reorder**: Automatically applied — segments are reordered via nearest-neighbor to minimize pen travel distance.
