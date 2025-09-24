# MuJoCo Tutorial
This tutorial aims at providing example-based introduction to MuJoCo Simulator for beginners who want to use MuJoCo for robotics research. This project was inspired by the use of MuJoCo for CSHexapod project simulation and initially intended to provide guidance for other lab members in BIRDS Lab at Umich.

## A Little About MuJoCo
[MuJoCo](https://mujoco.org/) (short for **Multi-Joint dynamics with Contact**) is a free, fast, physics-based simulation engine widely used in robotics, control, and reinforcement learning research. 

MuJoCo was originally developed as a commercial product by Emo Todorov at the University of Washington. In October 2021, DeepMind acquired MuJoCo and made it freely for everyone. In May 2022, DeepMind released MuJoCo as open source under the Apache 2.0 license, hosted on [GitHub](https://github.com/google-deepmind/mujoco)

## How to Set Up MuJoCo

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