"""Tutorial script that shows how to read MuJoCo sensor and site data."""

import argparse
import os
import subprocess
import time
from collections import OrderedDict

import mujoco
import numpy as np
from mujoco import viewer

# Pick a rendering backend once so every example run behaves the same.
if subprocess.run("nvidia-smi", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
    print("NVIDIA GPU detected, using GLFW for rendering.")
    os.environ["MUJOCO_GL"] = "glfw"
else:
    print("No NVIDIA GPU detected, using OSMesa for rendering.")
    os.environ["MUJOCO_GL"] = "osmesa"


def _sensor_slices(model: mujoco.MjModel) -> "OrderedDict[str, slice]":
    """Pre-compute the slice for every sensor inside `data.sensordata`.
    Returns an ordered dict mapping sensor names to their slices.
    """
    names_to_slices: "OrderedDict[str, slice]" = OrderedDict()
    for sensor_id in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        adr = model.sensor_adr[sensor_id]
        dim = model.sensor_dim[sensor_id]
        names_to_slices[name] = slice(adr, adr + dim)
    return names_to_slices


def _site_indices(model: mujoco.MjModel, site_names: list[str]) -> dict[str, int]:
    """Resolve human-readable site names to the integer ids MuJoCo expects.
    Returns a dict mapping site names to their integer ids.
    """
    indices: dict[str, int] = {}
    for name in site_names:
        indices[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return indices


def run_demo(duration: float, print_every: float, initial_angle: float, render: bool) -> None:
    """Simulate the pendulum and periodically print sensor/site values."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "pendulum.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Initialize the pendulum away from equilibrium so the sensors produce motion.
    data.qpos[0] = np.deg2rad(initial_angle)
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    # sensor_map maps sensor names to their slices in data.sensordata.
    sensor_map = _sensor_slices(model)
    imu_sensor_names = {
        "tip_orientation",
        "tip_angular_velocity",
        "tip_linear_acceleration",
    }
    # site_ids maps site names to their integer ids.
    site_ids = _site_indices(model, ["tip"])
    # mj_objectVelocity writes into a 6-long buffer [vx, vy, vz, wx, wy, wz].
    velocity_buffer = np.zeros(6, dtype=np.float64)

    print("Registered sensors:")
    for name, slc in sensor_map.items():
        print(f"  {name:>14s} -> sensordata[{slc.start}:{slc.stop}] (dim={slc.stop - slc.start})")
    print()

    timestep = model.opt.timestep
    total_steps = int(duration / timestep)
    next_report_time = 0.0

    def report_measurements() -> str:
        lines: list[str] = []
        sensordata = data.sensordata

        # Sensor block
        lines.append("Sensor readings:")
        for name, slc in sensor_map.items():
            if name in imu_sensor_names:
                continue
            block = sensordata[slc]
            values = ", ".join(f"{v:+.4f}" for v in block)
            lines.append(f"  {name:>14s}: [{values}]")

        imu_orientation = None
        imu_angular_velocity = None
        imu_linear_acceleration = None

        if "tip_orientation" in sensor_map:
            imu_orientation = sensordata[sensor_map["tip_orientation"]]
        if "tip_angular_velocity" in sensor_map:
            imu_angular_velocity = sensordata[sensor_map["tip_angular_velocity"]]
        if "tip_linear_acceleration" in sensor_map:
            imu_linear_acceleration = sensordata[sensor_map["tip_linear_acceleration"]]

        # Site block
        tip_id = site_ids["tip"]
        position = data.site_xpos[tip_id]
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_SITE,
            tip_id,
            velocity_buffer,
            0,
        )
        # Split the six-vector into translational and rotational velocity.
        linear_velocity = velocity_buffer[:3]
        angular_velocity = velocity_buffer[3:]
        lines.append("Tip site (world frame):")
        lines.append(f"  position (m): [{position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f}]")
        lines.append(f"  linear vel (m/s): [{linear_velocity[0]:+.4f}, {linear_velocity[1]:+.4f}, {linear_velocity[2]:+.4f}]")
        lines.append(f"  angular vel (rad/s): [{angular_velocity[0]:+.4f}, {angular_velocity[1]:+.4f}, {angular_velocity[2]:+.4f}]")

        if any(value is not None for value in (imu_orientation, imu_angular_velocity, imu_linear_acceleration)):
            lines.append("Tip IMU (sensor frame):")
            if imu_orientation is not None:
                lines.append(
                    "  orientation quat (w, x, y, z): "
                    f"[{imu_orientation[0]:+.4f}, {imu_orientation[1]:+.4f}, {imu_orientation[2]:+.4f}, {imu_orientation[3]:+.4f}]"
                )
            if imu_angular_velocity is not None:
                lines.append(
                    "  angular vel (rad/s): "
                    f"[{imu_angular_velocity[0]:+.4f}, {imu_angular_velocity[1]:+.4f}, {imu_angular_velocity[2]:+.4f}]"
                )
            if imu_linear_acceleration is not None:
                lines.append(
                    "  linear acc (m/s^2): "
                    f"[{imu_linear_acceleration[0]:+.4f}, {imu_linear_acceleration[1]:+.4f}, {imu_linear_acceleration[2]:+.4f}]"
                )

        return "\n".join(lines)

    def step_once() -> None:
        mujoco.mj_step(model, data)

    if render:
        with viewer.launch_passive(model, data) as v:
            v.cam.distance = 3.0
            v.cam.lookat = [0.0, 0.0, 0.0]

            for _ in range(total_steps):
                step_once()
                if data.time >= next_report_time:
                    text = report_measurements()
                    print(f"t = {data.time:.3f} s\n{text}\n")
                    next_report_time = data.time + print_every

                # Keep the viewer responsive and upload the latest state.
                v.sync()
                time.sleep(timestep)

        v.close()
    else:
        # Headless mode simply prints measurements at the requested cadence.
        for _ in range(total_steps):
            step_once()
            if data.time >= next_report_time:
                print(f"t = {data.time:.3f} s\n{report_measurements()}\n")
                next_report_time = data.time + print_every


def main() -> None:
    """Parse CLI arguments and run the sensor inspection demo."""
    parser = argparse.ArgumentParser(description="Inspect sensor and site data in the MuJoCo pendulum.")
    parser.add_argument("--duration", type=float, default=10.0, help="Total simulation time in seconds.")
    parser.add_argument("--print-every", type=float, default=0.03, help="How often to print measurements (s).")
    parser.add_argument("--initial-angle", type=float, default=25.0, help="Initial pendulum angle in degrees.")
    parser.add_argument("--no-render", action="store_true", help="Disable the interactive viewer.")
    args = parser.parse_args()

    run_demo(
        duration=args.duration,
        print_every=args.print_every,
        initial_angle=args.initial_angle,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()

