"""
Balance the MuJoCo pendulum upright with a PD controller.
"""

import argparse
import os
import subprocess
import time

import mujoco
import numpy as np
from mujoco import viewer

if subprocess.run("nvidia-smi", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
    print("NVIDIA GPU detected, using GLFW for rendering.")
    os.environ["MUJOCO_GL"] = "glfw"
else:
    print("No NVIDIA GPU detected, using OSMesa for rendering.")
    os.environ["MUJOCO_GL"] = "osmesa"

def wrap_to_pi(angle_rad):
    """Map any angle to the [-pi, pi) range to avoid discontinuities."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi

def balance_pendulum(initial_deg=0.0, duration=8.0, render=True, kp=60.0, kd=12.0, torque_limit=8.0):
    """Run a PD controller that keeps the pendulum upright."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "pendulum.xml")
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    # Initialize the pendulum near upright so the controller has something to stabilize.
    d.qpos[0] = np.deg2rad(initial_deg)
    d.qvel[0] = 0.0
    mujoco.mj_forward(m, d)

    timestep = m.opt.timestep
    total_steps = int(duration / timestep)
    upright_angle = np.pi  # radians; 0 rad points down, pi rad points up.

    def control_step():
        # Compute angle error relative to the upright configuration.
        angle_error = wrap_to_pi(upright_angle - d.qpos[0])
        velocity_error = -d.qvel[0]

        # PD torque: proportional term balances gravity, derivative term damps motion.
        torque = kp * angle_error + kd * velocity_error

        # Clamp to actuator limits declared in the MJCF (±10 Nm) to keep things physical.
        d.ctrl[0] = float(np.clip(torque, -torque_limit, torque_limit))

        mujoco.mj_step(m, d)

    if render:
        with viewer.launch_passive(m, d) as v:
            v.cam.distance = 3.0
            v.cam.lookat = [0.0, 0.0, 0.8]
            for _ in range(total_steps):
                control_step()
                v.sync()
                time.sleep(timestep)
    else:
        for _ in range(total_steps):
            control_step()

    return d.qpos.copy(), d.qvel.copy()


def main():
    parser = argparse.ArgumentParser(description="Balance the MuJoCo pendulum upright with PD control")
    parser.add_argument("--initial", type=float, default=0.0, help="Initial angle (deg) from downward pose")
    parser.add_argument("--duration", type=float, default=8.0, help="Simulation time (s)")
    parser.add_argument("--kp", type=float, default=60.0, help="Proportional gain")
    parser.add_argument("--kd", type=float, default=12.0, help="Derivative gain")
    parser.add_argument("--torque-limit", type=float, default=8.0, help="Actuator torque saturation (Nm)")
    parser.add_argument("--no-render", action="store_true", help="Disable the interactive viewer")
    args = parser.parse_args()

    final_qpos, final_qvel = balance_pendulum(
        initial_deg=args.initial,
        duration=args.duration,
        render=not args.no_render,
        kp=args.kp,
        kd=args.kd,
        torque_limit=args.torque_limit,
    )

    print(f"Final angle (deg): {np.rad2deg(final_qpos[0]):.2f}")
    print(f"Final angular velocity (deg/s): {np.rad2deg(final_qvel[0]):.2f}")


if __name__ == "__main__":
    main()

