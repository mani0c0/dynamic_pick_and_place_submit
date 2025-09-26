# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional, List, Dict
import copy

import numpy as np
import yaml
import os
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import get_stage_units, add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.prims import RigidPrim
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.rotations import euler_angles_to_quat


class TaskObject():
    """Unified object representation for pick and place tasks.
    
    Supports both primitive objects (cubes, spheres) and USD file objects
    """
    
    def __init__(
        self,
        name: str,
        obj_type: str = "primitive",  # "primitive" or "usd_file"
        usd_path: Optional[str] = None,
        obj_initial_position: Optional[np.ndarray] = None,
        obj_initial_orientation: Optional[np.ndarray] = None,
        grasp_offset_pos: Optional[np.ndarray] = None,
        obj_size: Optional[np.ndarray] = None,
        obj_scale: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
        placing_orientation_euler: Optional[np.ndarray] = None,
    ) -> None:
        self.name = name
        self.obj_type = obj_type
        self.usd_path = usd_path
        self._obj_initial_position = obj_initial_position
        self._obj_initial_orientation = obj_initial_orientation
        self._grasp_offset_pos = grasp_offset_pos
        self._obj_size = obj_size
        self._obj_scale = obj_scale
        self._target_position = target_position
        self._color = color
        self._scene_object = None  # Will store the actual scene object
        self._placing_orientation_euler = placing_orientation_euler
        
        # Set defaults
        if self._obj_size is None:
            self._obj_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        if self._obj_scale is None:
            self._obj_scale = np.array([1.0, 1.0, 1.0])
        if self._obj_initial_position is None:
            self._obj_initial_position = np.array([0.5, 0.3, 0.3]) / get_stage_units()
        if self._obj_initial_orientation is None:
            self._obj_initial_orientation = np.array([1, 0, 0, 0])
        if self._grasp_offset_pos is None:
            self._grasp_offset_pos = np.array([0, 0, 0]) / get_stage_units()
        if self._target_position is None:
            self._target_position = np.array([-0.3, -0.3, 0.4]) / get_stage_units()
        if self._color is None:
            self._color = np.array([0, 1, 0])
        if self._placing_orientation_euler is None:
            self._placing_orientation_euler = np.array([0.18 * np.pi, np.pi, 0])

    @property 
    def initial_position(self) -> np.ndarray:
        return self._obj_initial_position

    @property
    def initial_orientation(self) -> np.ndarray:
        return self._obj_initial_orientation

    @property
    def grasp_offset_pos(self)-> np.ndarray:
        return self._grasp_offset_pos

    @property
    def target_position(self) -> np.ndarray:
        return self._target_position

    @property
    def size(self) -> np.ndarray:
        return self._obj_size
    
    @property
    def scale(self) -> np.ndarray:
        return self._obj_scale

    @property
    def color(self) -> np.ndarray:
        return self._color

    @property
    def placing_orientation_euler(self) -> np.ndarray:
        return self._placing_orientation_euler

    def set_scene_object(self, scene_object):
        """Set the actual scene object reference."""
        self._scene_object = scene_object

    def get_scene_object(self):
        """Get the actual scene object reference."""
        return self._scene_object

    def get_pose(self):
        """Get current pose of the object."""
        if self._scene_object:
            return self._scene_object.get_local_pose()
        return self._obj_initial_position, self._obj_initial_orientation

    def set_pose(self, position=None, orientation=None):
        """Set pose of the object."""
        if self._scene_object:
            self._scene_object.set_local_pose(translation=position, orientation=orientation)


class RigidPrimWrapper:
    """Wrapper to make RigidPrim compatible with the expected single-object API."""
    
    def __init__(self, rigid_prim):
        self._rigid_prim = rigid_prim
        
    def get_world_pose(self):
        """Get single object world pose from RigidPrim's multi-object API."""
        positions, orientations = self._rigid_prim.get_world_poses()
        return positions[0], orientations[0]
    
    def get_local_pose(self):
        """Get single object local pose from RigidPrim's multi-object API."""
        positions, orientations = self._rigid_prim.get_local_poses()
        return positions[0], orientations[0]
    
    def set_local_pose(self, translation=None, orientation=None):
        """Set single object local pose using RigidPrim's multi-object API."""
        if translation is not None:
            translation = translation.reshape(1, -1)
        if orientation is not None:
            orientation = orientation.reshape(1, -1)
        self._rigid_prim.set_local_poses(translations=translation, orientations=orientation)
    
    def set_world_pose(self, position=None, orientation=None):
        """Set single object world pose using RigidPrim's multi-object API."""
        if position is not None:
            position = position.reshape(1, -1)
        if orientation is not None:
            orientation = orientation.reshape(1, -1)
        self._rigid_prim.set_world_poses(positions=position, orientations=orientation)
    
    def set_default_state(self, position=None, orientation=None, linear_velocity=None, angular_velocity=None):
        """Set single object default state using RigidPrim's multi-object API."""
        positions = None
        orientations = None
        linear_velocities = None
        angular_velocities = None
        
        if position is not None:
            positions = position.reshape(1, -1)
        if orientation is not None:
            orientations = orientation.reshape(1, -1)
        if linear_velocity is not None:
            linear_velocities = linear_velocity.reshape(1, -1)
        if angular_velocity is not None:
            angular_velocities = angular_velocity.reshape(1, -1)
            
        self._rigid_prim.set_default_state(
            positions=positions,
            orientations=orientations, 
            linear_velocities=linear_velocities,
            angular_velocities=angular_velocities
        )
    
    @property
    def name(self):
        """Get the name of the wrapped RigidPrim."""
        return self._rigid_prim.name
    
    def __getattr__(self, name):
        """Delegate other attributes to the wrapped RigidPrim."""
        return getattr(self._rigid_prim, name)


class ObjectFactory:
    """Factory class to create TaskObject instances and their corresponding scene objects."""
    
    @staticmethod
    def create_task_object_from_config(name: str, config: dict) -> TaskObject:
        """Create a TaskObject from configuration dictionary."""

        initial_position = None
        if config.get("initial_position"):
            initial_position = np.array(config["initial_position"]) / get_stage_units()
            
        initial_orientation = None  
        if config.get("initial_orientation"):
            initial_orientation = np.array(config["initial_orientation"])
        
        grasp_offset_position = None
        if config.get("grasp_offset_position"):
            grasp_offset_position = np.array(config["grasp_offset_position"]) / get_stage_units()
            
        target_position = None
        if config.get("target_position"):
            target_position = np.array(config["target_position"]) / get_stage_units()
            
        size = None
        if config.get("size"):
            size = np.array(config["size"]) / get_stage_units()
            
        scale = None
        if config.get("scale"):
            scale = np.array(config["scale"])
            
        color = None
        if config.get("color"):
            color = np.array(config["color"])

        placing_orientation_euler = None
        if config.get("placing_orientation_euler"):
            placing_orientation_euler = np.array(config["placing_orientation_euler"])  # radians
        
        return TaskObject(
            name=name,
            obj_type=config.get("type", "primitive"),
            usd_path=config.get("usd_path"),
            obj_initial_position=initial_position,
            obj_initial_orientation=initial_orientation,
            grasp_offset_pos=grasp_offset_position,
            obj_size=size,
            obj_scale=scale,
            target_position=target_position,
            color=color,
            placing_orientation_euler=placing_orientation_euler,
        )
    
    @staticmethod
    def create_scene_object(task_obj: TaskObject, scene: Scene) -> object:
        """Create and add the actual scene object based on TaskObject configuration."""
        if task_obj.obj_type == "usd_file":
            return ObjectFactory._create_usd_object(task_obj, scene)
        else:
            return ObjectFactory._create_primitive_object(task_obj, scene)
    
    @staticmethod
    def _create_usd_object(task_obj: TaskObject, scene: Scene) -> RigidPrim:
        """Create a USD-based object (like RTX 3080)."""
        prim_path = find_unique_string_name(
            initial_name=f"/World/{task_obj.name}", 
            is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        
        # Add USD file to stage
        add_reference_to_stage(
            usd_path=task_obj.usd_path,
            prim_path=prim_path
        )
        
        # Fix hierarchy issue by disabling child rigid bodies and optimize physics
        from pxr import Usd, UsdPhysics, PhysxSchema
        from isaacsim.core.utils.stage import get_current_stage
        
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        
        if root_prim:
            # Traverse USD hierarchy and disable rigid body API on child meshes
            for prim in Usd.PrimRange(root_prim):
                # Skip the root prim, only disable children
                if prim.GetPath() != root_prim.GetPath():
                    if UsdPhysics.RigidBodyAPI.CanApply(prim):
                        # Remove rigid body API from child prims to fix hierarchy conflict
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

        rigid_prim = scene.add(
            RigidPrim(
                prim_paths_expr=prim_path,
                name=task_obj.name,
                positions=task_obj.initial_position.reshape(1, -1),
                orientations=task_obj.initial_orientation.reshape(1, -1),
                scales=task_obj.scale.reshape(1, -1)
            )
        )
        
        return RigidPrimWrapper(rigid_prim)
    
    @staticmethod  
    def _create_primitive_object(task_obj: TaskObject, scene: Scene) -> DynamicCuboid:
        """Create a primitive object (cube, sphere, etc.)."""
        prim_path = find_unique_string_name(
            initial_name=f"/World/{task_obj.name}", 
            is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        

        scene_object = scene.add(
            DynamicCuboid(
                name=task_obj.name,
                position=task_obj.initial_position,
                orientation=task_obj.initial_orientation,
                prim_path=prim_path,
                scale=task_obj.size,
                size=1.0,
                color=task_obj.color,
            )
        )
        
        return scene_object




class ConfigurablePickDropClass(BaseTask):
    """Configurable pick and drop task that can handle multiple objects from configuration files.
  
    Args:
        name (str): Name of the task.
        task_config_path (str): Path to the task configuration YAML file.
        objects_config_path (str): Path to the objects configuration YAML file.
        offset (Optional[np.ndarray], optional): Offset applied to the task. Defaults to None.
    """
    
    def __init__(
        self,
        name: str = "configurable_pick_drop",
        task_config_path: str = "task_config.yaml",
        objects_config_path: str = "objects_config.yaml", 
        offset: Optional[np.ndarray] = None,
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)

        self.task_config_path = task_config_path
        self.objects_config_path = objects_config_path

        self.task_config = self._load_config(task_config_path)
        self.objects_config = self._load_config(objects_config_path)

        self.task_objects: List[TaskObject] = []
        self._robot = None
        self._camera = None  # Camera sensor for observations
        # Cache mobile robot config position for reuse
        mobile_cfg = self.task_config.get('mobile_robot', {})
        self._mobile_robot_config_position = np.array(
            mobile_cfg.get('position', [0.3, -0.5, 0])
        ) / get_stage_units()
        
        # Create TaskObjects from configuration
        self._create_task_objects()
        
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file {config_path}: {e}")
    
    def _create_task_objects(self):
        """Create TaskObject instances from configuration."""
        enabled_objects = [obj for obj in self.task_config['objects'] if obj.get('enabled', True)]
        
        # Sort by priority if specified
        enabled_objects.sort(key=lambda x: x.get('priority', 999))
        
        for obj_config in enabled_objects:
            obj_name = obj_config['name']
            if obj_name not in self.objects_config:
                raise ValueError(f"Object '{obj_name}' not found in objects configuration")
            
            # Create TaskObject(s) from configuration, supporting grid spawning

            # Pick grid configuration
            n_rows = int(obj_config.get('n_rows', 1))
            n_col = int(obj_config.get('n_col', 1))
            n_layers = int(obj_config.get('n_layers', 1))
            pick_r_space = float(obj_config.get('pick_r_space', 0.2))
            pick_c_space = float(obj_config.get('pick_c_space', 0.2))
            
            # Target grid configuration (independent of pick grid)
            target_n_rows = int(obj_config.get('target_n_rows', n_rows))
            target_n_col = int(obj_config.get('target_n_col', n_col))
            target_r_space = float(obj_config.get('target_r_space', 0.2))
            target_c_space = float(obj_config.get('target_c_space', 0.2))
            target_l_space = float(obj_config.get('target_l_space', 0.2))

            # Linear object counter for target placement mapping
            obj_linear_index = 0

            # Pick objects are only in 2D grid (no layers in picking)
            for i in range(0, n_rows):
                for j in range(0, n_col):
                    # Deep-copy the per-object config to avoid in-place mutation
                    grid_cfg = copy.deepcopy(self.objects_config[obj_name])

                    # Apply XY offset for pick placement (meters)
                    # Pick positions are always flat (single layer)
                    pick_offset_vec = np.array(
                        [i * pick_r_space, j * pick_c_space, 0.0]
                    )
                    
                    # Calculate target grid indices independently from pick indices
                    # n_layers only affects target arrangement (stacking)
                    target_i = ((obj_linear_index //
                                 (target_n_col * n_layers)) % target_n_rows)
                    target_j = (obj_linear_index // n_layers) % target_n_col
                    target_k = obj_linear_index % n_layers
                    
                    # Apply XY and Z (layer) offsets for target placement (meters)
                    # Uses target grid indices
                    target_offset_vec = np.array(
                        [target_i * target_r_space, target_j * target_c_space,
                         0.0]
                    )
                    target_layer_offset_vec = np.array(
                        [0.0, 0.0, target_k * target_l_space])

                    if ('initial_position' in grid_cfg and 
                            grid_cfg['initial_position'] is not None):
                        init_pos = (np.array(grid_cfg['initial_position']) +
                                    pick_offset_vec).tolist()
                        grid_cfg['initial_position'] = init_pos

                    if 'target_position' in grid_cfg and grid_cfg['target_position'] is not None:
                        base_target = np.array(grid_cfg['target_position'])
                        adjusted_target = (
                            base_target
                            - target_offset_vec
                            + target_layer_offset_vec
                        ).tolist()
                        grid_cfg['target_position'] = adjusted_target

                    # Ensure unique names per grid instance to avoid observation key collisions
                    # Use linear index to guarantee uniqueness
                    instance_name = f"{obj_name}_{obj_linear_index}"

                    task_obj = ObjectFactory.create_task_object_from_config(
                        name=instance_name,
                        config=grid_cfg
                    )
                    self.task_objects.append(task_obj)
                    
                    obj_linear_index += 1

    
    def set_up_scene(self, scene: Scene) -> None:
        """Set up the scene with all configured objects."""
        super().set_up_scene(scene)

        assets_root_path = get_assets_root_path()
        warehouse_usd_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        add_reference_to_stage(usd_path=warehouse_usd_path, prim_path="/World/Warehouse")

        # Add pallet if configured
        if self.task_config.get('environment', {}).get('include_pallet', False):
            self._add_pallet(scene)
        
        # Create scene objects for all task objects
        for task_obj in self.task_objects:
            scene_obj = ObjectFactory.create_scene_object(task_obj, scene)
            task_obj.set_scene_object(scene_obj)
            # Register the object with BaseTask's _task_objects
            # Keep the wrapper so BaseTask can use get_world_pose() method
            self._task_objects[task_obj.name] = scene_obj

        self._robot = self.set_robot()
        self._mobile_bot = self.set_mobile_bot()

        scene.add(self._robot)
        scene.add(self._mobile_bot)
        self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._mobile_bot.name] = self._mobile_bot
        
        self._move_task_objects_to_their_frame()
    
    def _add_pallet(self, scene: Scene):
        """Add pallet to the scene if configured."""
        pallet_prim_path = find_unique_string_name(
            initial_name="/World/Pallet", 
            is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        
        add_reference_to_stage(
            usd_path='/home/mani/Downloads/plastic_pallet/plastic_pallet.usd',
            prim_path=pallet_prim_path
        )
        
        # Fix the hierarchy issue by disabling child rigid bodies
        # This prevents the physics conflicts that cause jerky motion and errors
        from pxr import Usd, UsdPhysics
        from isaacsim.core.utils.stage import get_current_stage
        
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(pallet_prim_path)
        
        if root_prim:
            # Traverse the USD hierarchy and disable rigid body API on child meshes
            for prim in Usd.PrimRange(root_prim):
                # Skip the root prim, only disable children
                if prim.GetPath() != root_prim.GetPath():
                    if UsdPhysics.RigidBodyAPI.CanApply(prim):
                        # Remove rigid body API from child prims to fix hierarchy conflict
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        
        pallet_position = self.task_config.get('environment', {}).get('pallet_position', [-0.5, -0.5, 0])
        pallet_prim = scene.add(
            RigidPrim(
                prim_paths_expr=pallet_prim_path,
                name="pallet",
                positions=np.array([pallet_position]) / get_stage_units(),
                scales=np.array(self.task_config.get('environment', {}).get('pallet_scale', [0.8, 0.8, 0.8])).reshape(1, -1)
            )
        )
        
        # Enable physics but make pallet static to avoid unwanted movement
        pallet_prim.enable_rigid_body_physics()
    
    def set_robot(self):
        """Set up the robot. Override this in subclasses."""
        raise NotImplementedError("Subclasses must implement set_robot()")
    
    def set_mobile_bot(self):
        raise NotImplementedError("Subclasses must implement set_mobile_bot()")
    
    def set_camera(self, camera):
        """Set the camera sensor for image observations."""
        self._camera = camera
    
    def get_params(self) -> dict:
        """Get task parameters for all configured objects."""
        params = {}
        
        # Add robot params
        params["robot_name"] = {"value": self._robot.name, "modifiable": False}
        
        # Add parameters for each object
        for task_obj in self.task_objects:
            scene_obj = task_obj.get_scene_object()
            if scene_obj:
                position, orientation = scene_obj.get_local_pose()
                prefix = task_obj.name
                
                params[f"{prefix}_position"] = {"value": position, "modifiable": True}
                params[f"{prefix}_orientation"] = {"value": orientation, "modifiable": True}
                params[f"{prefix}_grasp_offset"] = {"value": task_obj.grasp_offset_pos, "modifiable": True}
                params[f"{prefix}_target_position"] = {"value": task_obj.target_position, "modifiable": True}
                params[f"{prefix}_name"] = {"value": task_obj.name, "modifiable": False}
        
        return params
    
    def get_observations(self) -> dict:
        """Get observations for all configured objects."""
        observations = {}
        
        # Robot observations
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_world_pose()
        observations[self._robot.name] = {
            "joint_positions": joints_state.positions,
            "end_effector_position": end_effector_position,
        }
        
        # Object observations
        for task_obj in self.task_objects:
            scene_obj = task_obj.get_scene_object()
            if scene_obj:
                task_obj.target_position_f = (
                    task_obj.target_position
                    + (self._mobile_bot.get_world_pose()[0] - self._mobile_robot_config_position)
                )
                position, orientation = scene_obj.get_local_pose()
                observations[task_obj.name] = {
                    "position": position,
                    "orientation": orientation, 
                    "grasp_offset_pos": task_obj.grasp_offset_pos,
                    "target_position": task_obj.target_position_f,
                    "placing_orientation": euler_angles_to_quat(task_obj.placing_orientation_euler),
                }
        
        # Camera image observation
#        if self._camera is not None:
#            observations["image"] = self._camera.get_current_frame()
        
        return observations
    
    def get_current_object_name(self, object_index: int = 0) -> str:
        """Get the name of the object at the specified index."""
        if 0 <= object_index < len(self.task_objects):
            return self.task_objects[object_index].name
        return None
    
    def get_object_count(self) -> int:
        """Get the total number of configured objects."""
        return len(self.task_objects)
    
    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """Pre-step processing."""
        return
    
    def post_reset(self) -> None:
        """Post-reset processing."""
        from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
        
        if isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        return
    
    def calculate_metrics(self) -> dict:
        """Calculate task metrics."""
        # Can be implemented by subclasses
        return {}
    
    def is_done(self) -> bool:
        """Check if task is complete."""
        # Can be implemented by subclasses
        return False