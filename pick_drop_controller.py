# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import List, Optional

import isaacsim.robot.manipulators.controllers as manipulators_controllers
from isaacsim.core.prims import SingleArticulation
import rmpflow_controller
from isaacsim.robot.manipulators.grippers.parallel_gripper import (
    ParallelGripper,
)

from pick_drop_controller_class import PickDropControllerClass


class PickDropController(PickDropControllerClass):
    """[summary]

    Args:
        name (str): [description]
        gripper (ParallelGripper): [description]
        robot_articulation (SingleArticulation): [description]
        end_effector_initial_height (Optional[float], optional): [description].
            Defaults to None.
        events_dt (Optional[List[float]], optional): [description].
            Defaults to None.
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
        motion_events_dt: Optional[List[float]] = None,
    ) -> None:
        if events_dt is None:
            events_dt = [
                0.01,       #0
                0.0005,     #1
                1,          #2
                0.1,        #3
                0.05,       #4
                0.01,       #5
                0.0025,     #6
                0.1,        #7
                0.006,      #8
                0.004,      #9
            ]
        if motion_events_dt is None:
            motion_events_dt = [
                0.005,      #0
                0.005,      #1
                1,          #2
                0.1,        #3
                0.05,       #4
                0.005,      #5
                0.005,      #6
                1,          #7
                0.008,      #8
                0.08,       #9
            ]
        manipulators_controllers.PickDropController.__init__(
            self,
            name=name,
            cspace_controller=rmpflow_controller.RMPFlowController(
                name=name + "_cspace_controller",
                robot_articulation=robot_articulation,
            ),
            gripper=gripper,
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
            motion_events_dt=motion_events_dt,
        )
        return
