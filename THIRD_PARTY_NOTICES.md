# Third-party notices

## What this repository contains

| Content | Origin | License |
|---|---|---|
| `docking_perception/`, `tools/`, `test/` | original code | MIT (see `LICENSE`) |
| `models/*.pt` (trained weights) | original work | MIT |
| `worlds/underwater_docking_physics.sdf` (docking station built from primitive shapes) | original work | MIT |
| `simulator/auv_simulator_docking.patch` | modifications and additions to [AUV-Simulator](https://github.com/Irlkidonu/AUV-Simulator), which is MIT-licensed by the same author | MIT |
| `docs/*` | screenshots rendered from the simulator (see below) | see below |

## Not included, used at runtime

These are installed separately and are **not redistributed** here:

| Component | License |
|---|---|
| ROS 2 Jazzy, `ros_gz` | Apache-2.0 |
| Gazebo Sim (Harmonic) | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| OpenCV (`cv2`, used by ROS image tools) | Apache-2.0 |

## Simulator assets referenced, not included

The docking world reuses scenery and the vehicle body from **AUV-Simulator**. Those files stay
in that repository under their own licenses and are **not copied** here. See AUV-Simulator's
`NOTICE` for the full list, including:

- **BlueROV2 mesh** from [CentraleNantesROV/bluerov2](https://github.com/CentraleNantesROV/bluerov2), Apache-2.0. BlueROV2 is a product of Blue Robotics Inc.; no affiliation or endorsement is claimed.
- **Underwater scenery models** (coral, fish, plants, jellyfish and others), CC0 and Sketchfab licenses, as recorded in AUV-Simulator's `NOTICE`.

## Images in `docs/`

The images are screenshots and a short animation rendered from the simulator. They show the
scenery models and the BlueROV2 mesh listed above as they appear in a rendered scene; they do
not contain the model files themselves.
