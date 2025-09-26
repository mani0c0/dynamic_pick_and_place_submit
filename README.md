### Dynamic Pick-and-Place (Isaac Sim)

This project demonstrates a configurable pick-and-place task using a Franka Emika Panda robot in NVIDIA Isaac Sim. The robot picks multiple objects from a grid and places them onto a moving mobile robot, adapting in real time to random movements using an RMPflow-based controller.

---

### Features
- **Franka Emika Panda** manipulator with parallel gripper
- **Moving mobile base** with randomized x-axis perturbations during runtime
- **Configurable task** via YAML files:
  - Object positions, orientations, grasp offsets, scales
  - Spawn and placement patterns (rows, columns, layers)
  - Motion speeds, gaps, and tolerances per step
- **Multiple objects** with enable/disable and priority settings

---

### Requirements
- NVIDIA Isaac Sim 4.50 (standalone)
- Python that ships with Isaac Sim (no separate install required)

> Tip: Launch from the Isaac Sim provided Python to ensure dependencies are available.

---

### Project Layout
- `pick_drop_main.py`: Entry point to run the simulation
- `pick_drop_task.py`, `pick_drop_controller.py`, `pick_drop_controller_class.py`, `rmpflow_controller.py`: Task and control logic
- `franka.py`, `mobile_bot.py`: Robot and mobile base helpers
- `task_config.yaml`: High-level task and environment configuration
- `objects_config.yaml`: Object definitions and target placements
- `geforce3080/`: USD asset and textures for the RTX 3080 object

---

### Quick Start
```bash

cd /home/{user}/isaacsim
./python.sh standalone_examples/revel_project_submit/pick_drop_main.py

```

> Note: The script sets `headless=False` so a viewer window will open. Use a local display or a remote setup with GPU + display forwarding.

---

### Configuration

#### Task configuration: `task_config.yaml`
Controls which objects are included, their grid patterns, and environment parameters.

Key fields:
- `task.name`: Task identifier
- `task.sequence_mode`: `sequential` processing of objects
- `objects[]`: Per-object settings
  - `name`: Must match a key in `objects_config.yaml`
  - `enabled`: Include/exclude the object
  - `priority`: Lower values processed first
  - `n_rows`, `n_col`, `n_layers`: Grid spawn pattern
  - `target_n_rows`, `target_n_col`: Target grid pattern on the mobile robot
  - `pick_r_space`, `pick_c_space`: Row/column spacing for picks (meters)
  - `target_r_space`, `target_c_space`, `target_l_space`: Spacing for placements (meters)
- `robot`: Robot type and offsets
- `mobile_robot.position`: Initial pose of the mobile base
- `environment`: Pallet toggles and properties

#### Object configuration: `objects_config.yaml`
Defines object assets, initial poses, and targets.

Key fields per object:
- `type`: `usd_file` or `primitive`
- `usd_path`/`shape`: Asset reference
- `initial_position`, `initial_orientation`: Start pose
- `grasp_offset_position`: Offset from object origin for grasping
- `target_position`: Where to place on the moving base
- `placing_orientation_euler` (optional): Desired orientation at placement
- `size`/`scale`: Geometry dimensions or scaling

---

### Runtime Behavior
- The mobile robot wheels are driven in short velocity bursts with random gaps that create x-axis motion while maintaining a slight positive drift.
- The controller computes actions via `forward(...)` using live observations and object-specific grasp offsets.
- When all enabled objects are processed, the world resets and the cycle restarts.

---

