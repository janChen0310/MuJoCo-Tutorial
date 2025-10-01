import argparse
import os
import time
import subprocess

import mujoco
import numpy as np

from mujoco import viewer

# MuJoCo relies on OpenGL for visualization. The python bindings select a backend
# via the MUJOCO_GL environment variable. GLFW opens a native window and uses the
# GPU; OSMesa is a CPU renderer that works headless (useful on servers).
if subprocess.run("nvidia-smi", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
    print("NVIDIA GPU detected, using GLFW for rendering.")
    os.environ["MUJOCO_GL"] = "glfw"
else:
    print("No NVIDIA GPU detected, using OSMesa for rendering.")
    os.environ["MUJOCO_GL"] = "osmesa"


def simulate(target_deg=15.0, duration=6.0, render=True):
    model_path = os.path.join(os.path.dirname(__file__), "models", "pendulum.xml")
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)


    timestep = m.opt.timestep
    steps = int(duration / timestep)

    # Start slightly away from equilibrium so gravity induces motion even
    # without control. Angles are in radians.
    d.qpos[0] = np.deg2rad(target_deg)
    mujoco.mj_forward(m, d)

    if render:
        # launch_passive opens a viewer window (GLFW) or an offscreen context
        # (OSMesa) depending on MUJOCO_GL. It drives the render loop while we
        # advance physics.
        with viewer.launch_passive(m, d) as v:
            # Elevate the camera to see the pendulum
            v.cam.distance = 3.0
            v.cam.lookat = [0.0, 0.0, 0.0]
            for _ in range(steps):
                mujoco.mj_step(m, d)
                # v.sync() uploads the latest simulation state to the current
                # OpenGL context and processes window events.
                v.sync()
                # Sleeping keeps the playback close to real time. Without it the
                # viewer draws as fast as the CPU/GPU can step.
                time.sleep(timestep)
    else:
        for _ in range(steps):
            mujoco.mj_step(m, d)

    return d.qpos.copy(), d.qvel.copy()


def main():
    parser = argparse.ArgumentParser(description="Simple pendulum demo for MuJoCo")
    parser.add_argument("--target", type=float, default=15.0, help="Initial angle offset in degrees")
    parser.add_argument("--duration", type=float, default=6.0, help="Simulation duration in seconds")
    parser.add_argument("--no-render", action="store_true", help="Disable viewer rendering")
    args = parser.parse_args()

    pos, vel = simulate(target_deg=args.target, duration=args.duration, render=not args.no_render)
    print(f"Final position: {pos[0]}, Final velocity: {vel[0]}")


if __name__ == "__main__":
    main()

