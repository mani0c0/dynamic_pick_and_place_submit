from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
from isaacsim.core.api import World
import pick_drop_controller
from pick_drop_task import ConfigurablePickDrop
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.core.utils.numpy.rotations as rot_utils
# from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
task_config_path = os.path.join(current_dir, "task_config.yaml")
objects_config_path = os.path.join(current_dir, "objects_config.yaml")

my_world = World(stage_units_in_meters=1.0)

my_task = ConfigurablePickDrop(
    name="rtx_3080_pick_drop",
    task_config_path=task_config_path,
    objects_config_path=objects_config_path
)

my_world.add_task(my_task)
camera = Camera(
    prim_path="/World/camera",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=20,
    resolution=(512, 512),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
)
my_world.reset()
camera.initialize()


my_task.set_camera(camera)

task_params = my_task.get_params()
my_franka = my_world.scene.get_object(task_params["robot_name"]["value"])
my_mobile_bot = my_world.scene.get_object("MobileBot")


if hasattr(my_franka, 'initialize'):
    my_franka.initialize()
my_controller = pick_drop_controller.PickDropController(
    name="pick_drop_controller",
    gripper=my_franka.gripper,
    robot_articulation=my_franka,
    end_effector_initial_height=0.4
)
articulation_controller_franka = my_franka.get_articulation_controller()
articulation_controller_mobile_bot = None
if my_mobile_bot is not None:
    articulation_controller_mobile_bot = my_mobile_bot.get_articulation_controller()

    articulation_controller_mobile_bot.switch_control_mode(mode="velocity")
    left_wheel_idx = my_mobile_bot.get_dof_index('left_wheel_joint')
    right_wheel_idx = my_mobile_bot.get_dof_index('right_wheel_joint')
# Initialize simulation state
i = 0
reset_needed = False
current_object_index = 0  # Track which object we're currently working on
objects_completed = []
total_objects = my_task.get_object_count()
step_count = 0
rng = np.random.default_rng()
burst_steps_remaining = 0
gap_steps_remaining = 0
current_burst_velocity = 0.0

print("Configurable Pick and Drop Task")
print(f"Total objects to process: {total_objects}")
for i in range(total_objects):
    obj_name = my_task.get_current_object_name(i)
    print(f"  Object {i+1}: {obj_name}")

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        step_count += 1
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
            current_object_index = 0
            objects_completed = []
            step_count = 0
            burst_steps_remaining = 0
            gap_steps_remaining = 0
            current_burst_velocity = 0.0
            print("Simulation reset - starting over")
        # Drive the mobile robot wheels for small x-motion using controller
        if articulation_controller_mobile_bot is not None:
            # Randomized bursts with slight positive drift; keep magnitude/duration similar
            if burst_steps_remaining > 0:
                v = float(current_burst_velocity)
                burst_steps_remaining -= 1
                action = ArticulationAction(
                    joint_velocities=np.array([v, v]),
                    joint_indices=np.array([left_wheel_idx, right_wheel_idx]),
                )
                articulation_controller_mobile_bot.apply_action(action)
            else:
                if gap_steps_remaining > 0:
                    gap_steps_remaining -= 1
                    action = ArticulationAction(
                        joint_velocities=np.array([0.0, 0.0]),
                        joint_indices=np.array([left_wheel_idx, right_wheel_idx]),
                    )
                    articulation_controller_mobile_bot.apply_action(action)
                else:
                    # Start a deterministic burst; random direction with slight positive drift
                    duration = 20
                    direction = 1.0 if rng.random() < 0.6 else -1.0
                    current_burst_velocity = direction * 0.8
                    burst_steps_remaining = duration - 1

                    sigma = 0.25
                    mu = float(np.log(400.0) - 0.5 * (sigma ** 2))
                    gap = rng.lognormal(mean=mu, sigma=sigma)
                    gap_steps_remaining = int(np.clip(gap, 200, 900))
                    v = float(current_burst_velocity)
                    action = ArticulationAction(
                        joint_velocities=np.array([v, v]),
                        joint_indices=np.array([left_wheel_idx, right_wheel_idx]),
                    )
                    articulation_controller_mobile_bot.apply_action(action)


        if current_object_index >= total_objects:
            print("All objects completed! Resetting simulation...")
            reset_needed = True
            continue
        
        observations = my_world.get_observations()


        current_object_name = my_task.get_current_object_name(current_object_index)
        if current_object_name is None:
            print("Error: Could not get current object name")
            break

        if current_object_name not in observations:
            print(f"Warning: Object {current_object_name} not found in observations")
            current_object_index += 1
            continue

        picking_position = observations[current_object_name]["position"]
        placing_position = observations[current_object_name]["target_position"]
        grasp_offset_pos = observations[current_object_name]["grasp_offset_pos"]

        
        actions = my_controller.forward(
            picking_position=picking_position + grasp_offset_pos,
            placing_position=placing_position,
            current_joint_positions=observations[
                task_params["robot_name"]["value"]
            ]["joint_positions"],
            end_effector_offset=np.array([0, 0.005, 0]),
            end_effector_orientation=observations[current_object_name].get("placing_orientation"),
            current_end_effector_position=observations[
                task_params["robot_name"]["value"]
            ]["end_effector_position"] + np.array([0, 0.005, 0]),
        )

        if my_controller.is_done():
            print(f"Done picking and placing {current_object_name}!")
            objects_completed.append(current_object_name)
            current_object_index += 1
            my_controller.reset()
    
            if current_object_index >= total_objects:
                print(f"All {total_objects} objects completed!")

        if actions is not None:
            articulation_controller_franka.apply_action(actions)
simulation_app.close()
