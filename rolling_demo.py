import numpy as np
import mujoco
from mujoco import viewer
import os
import time
import subprocess

if subprocess.run("nvidia-smi", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
    print("NVIDIA GPU detected, using GLFW for rendering.")
    os.environ["MUJOCO_GL"] = "glfw"
else:
    print("No NVIDIA GPU detected, using OSMesa for rendering.")
    os.environ["MUJOCO_GL"] = "osmesa"

# Load model + data
model_path = os.path.join(os.path.dirname(__file__), "models", "ball_on_u_trough.xml")
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

# ---- Build a U-shaped heightfield: z = k * x^2 (normalized to [0,1]) ----
hf_id = 0  # first/only hfield in this model
nrow = m.hfield_nrow[hf_id]
ncol = m.hfield_ncol[hf_id]
hx, hy, hmax, hbase = m.hfield_size[hf_id]


# grid in world-scaled coordinates spanning [-hx, hx] x [-hy, hy]
xs = np.linspace(-hx, hx, ncol, dtype=np.float32)
ys = np.linspace(-hy, hy, nrow, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)

# Parabolic trough only along x (U-shape). Adjust strength via k.
k = 0.35 / (hx**2)          # so peak~0.35 m at edges before normalization
Z = k * (X**2)              # shape in meters (pre-normalization)

# Normalize to [0,1] as MuJoCo expects for hfield_data
Z -= Z.min()
Z /= max(1e-9, Z.max())

# Write into model memory (row-major), then tell the viewer to refresh
start = m.hfield_adr[hf_id]
m.hfield_data[start:start + nrow*ncol] = Z.ravel(order="C")

with viewer.launch_passive(m, d) as v:
    v.update_hfield(hf_id)  # ensure the updated terrain is drawn

    # Visualize contact points/forces/frames
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = True
    v.opt.frame = mujoco.mjtFrame.mjFRAME_CONTACT
    v.opt.label = mujoco.mjtLabel.mjLABEL_CONTACTFORCE  # label magnitudes

    v.cam.distance = 5.0
    v.cam.lookat = [0.0, 0.0, 0.0]

    # optional: small initial spin for richer contact behavior
    d.qvel[3:6] = np.array([0.0, 2.0, 0.0], dtype=np.float64)

    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(0.002)
