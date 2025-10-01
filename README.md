# MuJoCo Tutorial
This tutorial aims at providing example-based introduction to MuJoCo Simulator for beginners who want to use MuJoCo for robotics research. This project was inspired by the use of MuJoCo for CSHexapod project simulation and initially intended to provide guidance for other lab members in BIRDS Lab at Umich.

## A Little About MuJoCo
[MuJoCo](https://mujoco.org/) (short for **Multi-Joint dynamics with Contact**) is a free, fast, physics-based simulation engine widely used in robotics, control, and reinforcement learning research. 

MuJoCo was originally developed as a commercial product by Emo Todorov at the University of Washington. In October 2021, DeepMind acquired MuJoCo and made it freely for everyone. In May 2022, DeepMind released MuJoCo as open source under the Apache 2.0 license, hosted on [GitHub](https://github.com/google-deepmind/mujoco)

## Installing MuJoCo

MuJoCo 3.x supports both Python and C/C++ APIs. These instructions focus on the Python workflow, which is what the examples in this repository use.

### 1. Install prerequisites
- A recent Linux distribution (this repository was authored on Ubuntu 24.04)
- GPU drivers if you want hardware-accelerated rendering (optional)
- Python 3.10+

### 2. Create & activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. Install MuJoCo and supporting libraries
```bash
pip install mujoco glfw numpy
```

The package ships with prebuilt binaries for CPU and GPU rendering backends. The `glfw` extra is recommended for interactive visualization.

### 4. Validate the install
```bash
python -c "import mujoco; print(mujoco.__version__)"
```

If this prints a version number (e.g. `3.3.6`), the install succeeded. Headless servers without a GPU may need to export `MUJOCO_GL=osmesa` before running simulations.

---

## Modeling in MuJoCo

MuJoCo models are defined in MJCF XML or via the Python API. The simulator loads a `mujoco.MjModel`, which holds all static model information (bodies, joints, actuators, meshes). The companion `mujoco.MjData` object stores mutable state.

### Key MJCF concepts
- `worldbody`: hierarchical bodies with positions and inertias
- `geom`: collision/visual geometry (spheres, capsules, meshes, heightfields)
- `joint`: degrees of freedom (hinge, slide, ball, free)
- `actuator`: maps controls to joint forces/torques
- `sensor` & `site`: expose measurements and attachment frames
- `asset`: shared resources like meshes, textures, heightfields

### Inspecting provided examples
- `models/pendulum.xml`: single-joint pendulum with a torque actuator

You can load any MJCF file via:
```python
import mujoco
m = mujoco.MjModel.from_xml_path("/absolute/path/to/model.xml")
d = mujoco.MjData(m)
```

## Rendering & Visualization

MuJoCo bundles a lightweight viewer that can render via GLFW (GPU) or OSMesa (CPU). Every example chooses a backend with `MUJOCO_GL` before importing the viewer:

```python
if subprocess.run("nvidia-smi", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
    os.environ["MUJOCO_GL"] = "glfw"  # interactive window
else:
    os.environ["MUJOCO_GL"] = "osmesa"  # headless CPU renderer
```

```python
with viewer.launch_passive(m, d) as v:
    v.cam.distance = 3.0
    v.cam.lookat = [0.0, 0.0, 0.0]
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(m.opt.timestep)
```

## Controlling Models

Controllers read the current state from `d` and write to `d.ctrl` (for actuators) or directly to generalized forces (advanced). In `pendulum_control_demo.py` we start the pendulum pointing downward and use gravity-compensating torques to keep it balanced. A proportional-derivative (PD) controller computes torque as a weighted sum of the angular position error (brings the link back toward upright) and the angular velocity (adds damping so the motion settles). For a refresher on PD/PID control, see the [PID controller primer](https://en.wikipedia.org/wiki/PID_controller).

The core control loop looks like this:

```python
def control_step():
    angle_error = wrap_to_pi(np.pi - d.qpos[0])
    velocity_error = -d.qvel[0]
    torque = kp * angle_error + kd * velocity_error
    d.ctrl[0] = np.clip(torque, -torque_limit, torque_limit)
    mujoco.mj_step(m, d)
```

`qpos` stores generalized positions, `qvel` stores velocities, and `ctrl` writes actuator commands. Free joints occupy 7 entries in `qpos` and 6 in `qvel`, so index carefully.

## Reading Model & Simulation State

Commonly used arrays on `mujoco.MjData`:
- `d.qpos`: generalized positions (size `m.nq`)
- `d.qvel`: generalized velocities (size `m.nv`)
- `d.xpos`: world coordinates of body frames
- `d.cfrc_ext`: external/contact forces on each body
- `d.sensordata`: all sensor readings (if sensors defined)

You can reset or perturb state directly:
```python
d.qpos[:] = initial_configuration
mujoco.mj_forward(m, d)  # recompute derived quantities
```

---

## How does MuJoCo handle contacts?
### How are contacts identified and used in simulation
1. **Filter & qualify pairs**: MuJoCo first uses bitmasks (`contype/conaffinity`), parent-child/body rules, and a bounding-sphere (or plane-sphere) test to decide which geom pairs enter narrow-phase
2. **Narrow-phase**: Type-specific collision routines compute the contact point, frame (normal + tangent axes), and signed distance. MuJoCo supports a native multi-contact pipeline (and a legacy libccd path). Detected contacts go into `mjData.contact`, subject to a margin threshold
3. **Forces from optimization**: Then all active contacts become constraints solved by a convex optimization problem (conic for elliptic cones, QP for pyramidal), yielding the contact impulses/forces used in the step
### "Rolling" vs "Normal" Contacts
The differences between various contact types are mainly reflected in the forces produced by the contact points, which is configured by `condim` attribute. 
- **Normal** (frictionless): `condim="1"` - only a normal force preventing interpenetration.
- **Frictional sliding:** `condim="3"` (elliptic) / `condim="4"` (pyramidal) → adds tangential friction resisting slip.
- **+ Torsional friction**: `condim="4"` (elliptic) / `condim="6"` (pyramidal) → also resists twist about the contact normal; torsional coefficient has length units.
- **+ Rolling friction** (full model): `condim="6" `(elliptic) / `condim="10"` (pyramidal) → additionally resists rotation about the two tangential axes, so balls/wheels don’t roll forever; rolling coefficients also have length units. 

Friction coefficients: Geoms specify a 3-tuple `friction="sliding torsional rolling"`; contacts internally track up to five coefficients (2 tangential, 1 torsional, 2 rolling)

See example [rolling_demo](rolling_demo.py) and [rolling_demo xml](./models/ball_on_u_trough.xml)

## Further Reading
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Official Examples](https://github.com/google-deepmind/mujoco/tree/main/python)