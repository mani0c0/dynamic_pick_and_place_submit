from typing import List, Optional

import carb
import numpy as np

from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.storage.native import get_assets_root_path


class MobileBot(Robot):
    def __init__(
        self,
        prim_path: str,
        name: str = "pallet_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        
        prim = get_prim_at_path(prim_path)

        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                usd_path = assets_root_path + "/Isaac/Robots/Idealworks/iw_hub.usd"
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, scale=[0.8, 0.8, 0.8],
            articulation_controller=None
        )

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        return

    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()

        return