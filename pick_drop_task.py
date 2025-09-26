# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional
import numpy as np
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from franka import Franka
from mobile_bot import MobileBot
from pick_drop_class import ConfigurablePickDropClass


class ConfigurablePickDrop(ConfigurablePickDropClass):
    """Franka-specific implementation of configurable pick and drop task.

    This class provides the Franka robot implementation for the configurable
    pick and drop system, allowing it to work with any objects defined in
    the configuration files.

    Args:
        name (str, optional): Task name.
            Defaults to "franka_configurable_pick_drop".
        task_config_path (str, optional): Path to task configuration.
            Defaults to "task_config.yaml".
        objects_config_path (str, optional): Path to objects configuration.
            Defaults to "objects_config.yaml".
        offset (Optional[np.ndarray], optional): Task offset.
            Defaults to None.
    """

    def __init__(
        self,
        name: str = "franka_configurable_pick_drop",
        task_config_path: str = "task_config.yaml",
        objects_config_path: str = "objects_config.yaml",
        offset: Optional[np.ndarray] = None,
    ) -> None:
        ConfigurablePickDropClass.__init__(
            self,
            name=name,
            task_config_path=task_config_path,
            objects_config_path=objects_config_path,
            offset=offset,
        )

    def set_robot(self) -> Franka:
        """Set up the Franka robot for the configurable pick and drop task.

        Returns:
            Franka: The configured Franka robot instance.
        """
        franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka",
            is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        franka_robot_name = find_unique_string_name(
            initial_name="my_franka",
            is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        return Franka(prim_path=franka_prim_path, name=franka_robot_name)

    def set_mobile_bot(self) -> MobileBot:
        mobile_bot_prim_path = find_unique_string_name(
            initial_name="/World/MobileBot",
            is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        mobile_bot_name = find_unique_string_name(
            initial_name="MobileBot",
            is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        # Load mobile robot position from task config; default if missing
        mobile_cfg = (
            self.task_config.get('mobile_robot', {})
            if hasattr(self, 'task_config') else {}
        )
        mobile_pos = (
            np.array(mobile_cfg.get('position', [0.3, -0.5, 0]), dtype=float)
            / get_stage_units()
        )
        return MobileBot(
            prim_path=mobile_bot_prim_path,
            name=mobile_bot_name,
            position=mobile_pos,
        )
