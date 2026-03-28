# Attractor Systems

formCollapse includes 19 attractor systems: 17 continuous ODE systems and 2 discrete maps.

## ODE Systems

All ODE systems follow the signature `f(X, t, params) -> dX/dt` where X is a 3D state vector.

### Lorenz System

The canonical chaotic attractor. Discovered by Edward Lorenz (1963) while modeling atmospheric convection.

**Equations:**
```
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z
```

**Known-good parameters:** sigma=10, rho=28, beta=8/3

**Behavior:** Double-scroll strange attractor. Positive maximal Lyapunov exponent (~0.9). Sensitive dependence on initial conditions ("butterfly effect").

---

### Aizawa System

A 6-parameter system producing a toroidal attractor with complex winding structure.

**Equations:**
```
dx/dt = (z - b) * x - d * y
dy/dt = d * x + (z - b) * y
dz/dt = c + a * z - z^3/3 - (x^2 + y^2)(1 + e * z) + f * z * x^3
```

**Known-good parameters:** a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1

---

### Rabinovich-Fabrikant System

Models wave interaction in plasma physics. Known for extreme sensitivity to parameters.

**Equations:**
```
dx/dt = y * (z - 1 + x^2) + gamma * x
dy/dt = x * (3z + 1 - x^2) + gamma * y
dz/dt = -2z * (alpha + x * y)
```

**Known-good parameters:** alpha=0.14, gamma=0.10

**Note:** Requires careful parameter tuning. The adaptive solver may need multiple attempts.

---

### Chen System

A dual of the Lorenz system discovered by Guanrong Chen (1999). Produces a similar double-scroll but with different topology.

**Equations:**
```
dx/dt = a * (y - x)
dy/dt = (c - a) * x - x * z + c * y
dz/dt = x * y - b * z
```

**Known-good parameters:** a=35, b=3, c=28

---

### Halvorsen System

A cyclically symmetric system producing a 3-lobed attractor.

**Equations:**
```
dx/dt = -a * x - 4y - 4z - y^2
dy/dt = -a * y - 4z - 4x - z^2
dz/dt = -a * z - 4x - 4y - x^2
```

**Known-good parameters:** a=1.27

---

### Newton-Leipnik System

A system with two strange attractors coexisting in the same phase space.

**Equations:**
```
dx/dt = -a * x + y + 10 * y * z
dy/dt = -x - 0.4 * y + 5 * x * z
dz/dt = b * z - 5 * x * y
```

**Known-good parameters:** a=0.4, b=0.175

---

### Three-Scroll System

Triple-scroll chaotic attractor with a large-scale structure.

**Known-good parameters:** a=40, b=55, c=1.833

---

### Rossler System

One of the simplest chaotic flows. Produces a band-type attractor with a single folding region.

**Known-good parameters:** a=0.2, b=0.2, c=5.7

---

### Anishchenko System

Models quasiperiodic to chaotic transition.

**Known-good parameters:** a=1.2, b=0.5, c=0.6

---

### Arnold System

Arnold's cat map continuous-time analog. **Note:** This system exhibits linear divergence, not chaos. Included as a control/test case.

**Known-good parameters:** omega=1.0

---

### Burke-Shaw System

A simplification of the Lorenz system with a distinctive attractor shape.

**Known-good parameters:** a=10, b=4.272, c=2.73

---

### Chen-Celikovsky System

A generalization bridging the Lorenz and Chen systems.

**Known-good parameters:** a=36, c=20, d=1.833

---

### Finance System

A chaotic model of financial market dynamics.

**Known-good parameters:** a=0.001, b=0.2, c=1.1

---

### Qi-Chen System

A four-wing chaotic attractor.

**Known-good parameters:** a=38, b=2.666, c=80

---

### Rayleigh-Benard System

Models thermal convection patterns.

**Known-good parameters:** a=9, b=5, c=12

---

### TSUCS1 System

T-system unified chaotic system.

**Known-good parameters:** a=40, b=0.16, c=0.65

---

### Liu-Chen System

A four-parameter chaotic system.

**Known-good parameters:** a=5, b=-10, c=-3.78, d=1

---

## Discrete Maps

Discrete maps follow the signature `f(state, params) -> next_state`. They are iterated rather than integrated.

### Henon Map

The classic 2D chaotic map.

```
x_{n+1} = 1 - a * x_n^2 + y_n
y_{n+1} = b * x_n
```

**Known-good parameters:** a=1.4, b=0.3

---

### Ikeda Map

Models light in a nonlinear optical resonator.

**Known-good parameters:** u=0.918

---

### Logistic Map

The simplest chaotic map: `x_{n+1} = r * x_n * (1 - x_n)`.

---

### Standard Map

Also known as the Chirikov-Taylor map. Models kicked rotators.

**Parameters:** K (stochasticity parameter)
