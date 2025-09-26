# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import typing

import numpy as np
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper


class PickDropControllerClass(BaseController):
    """
    - Phase 0: Move end_effector above the cube center at the
      'end_effector_initial_height'.
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight
      (lifting the block).
    - Phase 5: Smoothly move the end_effector toward the goal xy,
      keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the
      'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the
      'end_effector_initial_height'
    - Phase 9: Move end_effector towards the old xy position.

    Args:
        name (str): Name id of the controller
        cspace_controller (BaseController): a cartesian space controller
            that returns an ArticulationAction type
        gripper (Gripper): a gripper controller for open/ close actions.
        end_effector_initial_height (typing.Optional[float], optional):
            end effector initial picking height to start from (more info
            in phases above). If not defined, set to 0.3 meters.
            Defaults to None.
        events_dt (typing.Optional[typing.List[float]], optional): Dt of
            each phase/ event step. 10 phases dt has to be defined.
            Defaults to None.

        motion_events_dt (typing.Optional[typing.List[float]], optional): used
            to parametrize the motion.

    Raises:
        Exception: events dt need to be list or numpy array
        Exception: events dt need have length of 10
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        motion_events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._tau = 0.0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.3 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [
                0.008,
                0.005,
                0.1,
                0.1,
                0.0025,
                0.001,
                0.0025,
                1,
                0.008,
                0.08,
            ]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(
                self._events_dt, list
            ):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 10:
                raise Exception("events dt length must be 10")
        # Motion progress increments used only for trajectory
        # parameterization (tau)
        self._motion_events_dt = motion_events_dt
        if self._motion_events_dt is None:
            # Derive defaults from events_dt; zero for non-motion phases
            # (2, 3, 7)
            self._motion_events_dt = self._events_dt.copy()
            for _idx in [2, 3, 7]:
                self._motion_events_dt[_idx] = 0.0
        else:
            if (
                not isinstance(self._motion_events_dt, np.ndarray)
                and not isinstance(self._motion_events_dt, list)
            ):
                raise Exception(
                    "motion events dt need to be list or numpy array"
                )
            elif isinstance(self._motion_events_dt, np.ndarray):
                self._motion_events_dt = self._motion_events_dt.tolist()
            if len(self._motion_events_dt) != 10:
                raise Exception("motion events dt length must be 10")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        # Position-gated advancement settings for approach phases
        # (events 0 and 1)
        self._pos_gating_enabled = True

        if not hasattr(self, "_pos_gating_enabled"):
            self._pos_gating_enabled = True
        if not hasattr(self, "_gating_position_tolerance"):
            self._gating_position_tolerance = 0.008 / get_stage_units()
        if not hasattr(self, "_gating_xy_tolerance"):
            self._gating_xy_tolerance = 0.05 / get_stage_units()
        if not hasattr(self, "_gating_z_tolerance"):
            self._gating_z_tolerance = 0.07 / get_stage_units()
        # Require N consecutive in-tolerance steps (hysteresis)
        self._gating_hysteresis_steps = 5
        self._gate_counter = 0
        self._gate_event_id = -1
        return

    def is_paused(self) -> bool:
        """

        Returns:
            bool: True if the state machine is paused. Otherwise False.
        """
        return self._pause

    def get_current_event(self) -> int:
        """

        Returns:
            int: Current event/ phase of the state machine
        """
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        current_end_effector_position: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            picking_position (np.ndarray): The object's position to be
                picked in local frame.
            placing_position (np.ndarray):  The object's position to be
                placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions
                of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional):
                offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional):
                end effector orientation while picking and placing.
                Defaults to None.
            current_end_effector_position (typing.Optional[np.ndarray],
                optional): current end effector position for
                position-gated progression in approach phases.
                Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the
            ArticulationController
        """
        # Lazy init for gating attributes

        if not hasattr(self, "_gating_hysteresis_steps"):
            self._gating_hysteresis_steps = 2
        if not hasattr(self, "_gate_counter"):
            self._gate_counter = 0
        if not hasattr(self, "_gate_event_id"):
            self._gate_event_id = -1

        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [
                None
            ] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._event == 2:
            target_joint_positions = ArticulationAction(
                joint_positions=[
                    None
                ] * current_joint_positions.shape[0]
            )
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 7:
            target_joint_positions = self._gripper.forward(action="open")
        else:
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]

            interpolated_xy = self._get_interpolated_xy(
                placing_position[0],
                placing_position[1],
                self._current_target_x,
                self._current_target_y,
            )
            target_height = self._get_target_hs(placing_position[2])
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            # Orientation policy:
            # - For placing phases (6-8): use provided orientation if any,
            #   otherwise fall back to the controller's hardcoded value.
            # - For other phases: keep provided orientation if given,
            #   otherwise use the default approach orientation.
            if self._event in [6, 7, 8]:
                if end_effector_orientation is None:
                    end_effector_orientation = euler_angles_to_quat(
                        np.array([0.18 * np.pi, np.pi, 0])
                    )
            else:
                end_effector_orientation = euler_angles_to_quat(
                    np.array([0, np.pi, 0])
                )
            # Compensate for articulation uniform scale so that world targets
            # remain invariant when the robot prim is visually scaled
            # (e.g., ScaledFranka). Ignore if unscaled

            position_target_controller = position_target
            try:
                amp = getattr(
                    self._cspace_controller,
                    "_articulation_motion_policy",
                )
                articulation = amp._robot_articulation
                base_pos, _ = articulation.get_world_pose()
                world_scale = articulation.get_world_scale()
                # Handle scalar/array; assume uniform scaling if provided as
                # a 3-vector
                if world_scale is not None:
                    if isinstance(world_scale, np.ndarray):
                        scale_factor = float(np.mean(world_scale))
                    else:
                        scale_factor = float(world_scale)
                    if abs(scale_factor - 1.0) > 1e-6:
                        # De-scale the target around the base position
                        position_target_controller = (
                            base_pos
                            + (position_target - base_pos) / scale_factor
                        )
            except Exception:
                # If anything fails, fall back to the unmodified target
                pass
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target_controller,
                target_end_effector_orientation=end_effector_orientation,
            )
        print("end effector pos:", current_end_effector_position)
        # Position-gated advancement for approach phases (0 and 1), with
        # hysteresis and timeout fallback
        advance_by_position = False
        if (
            self._pos_gating_enabled
            and self._event in [0, 1, 5]
        ):
            if self._gate_event_id != self._event:
                self._gate_event_id = self._event
                self._gate_counter = 0
            final_z = self._h1 if self._event in [0, 5] else self._h0

            if self._event == 5:
                gx = placing_position[0] + end_effector_offset[0]
                gy = placing_position[1] + end_effector_offset[1]
            else:
                gx = self._current_target_x + end_effector_offset[0]
                gy = self._current_target_y + end_effector_offset[1]
            gating_target = np.array(
                [
                    gx, gy, final_z + end_effector_offset[2],
                ]
            )
            # Use world-space end-effector position for gating to avoid
            # scale-related inconsistencies when the robot prim is scaled.
            try:
                amp = getattr(
                    self._cspace_controller,
                    "_articulation_motion_policy",
                )
                articulation = amp._robot_articulation
                ee_pos_world, _ = articulation.end_effector.get_world_pose()
                ee_for_gating = ee_pos_world
            except Exception:
                # Fallback to provided observation if available
                ee_for_gating = current_end_effector_position
            delta = gating_target - ee_for_gating
            xy_error = np.linalg.norm(delta[:2])
            z_error = abs(delta[2])
            print("ERRORS: ", xy_error, z_error)
            print(
                "xy  z  tolerance: ",
                self._gating_xy_tolerance,
                self._gating_z_tolerance,
            )
            if (
                xy_error <= self._gating_xy_tolerance
                and z_error <= self._gating_z_tolerance
            ):
                self._gate_counter += 1
                print('inc_gate_counter:', self._gate_counter)
            else:
                self._gate_counter = 0
            if self._gate_counter >= self._gating_hysteresis_steps:
                advance_by_position = True

        # Update motion progress (tau) for motion phases; timeout always
        # accumulates
        motion_phase = self._event in [0, 1, 4, 5, 6, 8, 9]
        if motion_phase:
            self._tau = min(
                1.0,
                self._tau + self._motion_events_dt[self._event],
            )
        # Always advance by time as a fallback (timeout)
        self._t += self._events_dt[self._event]

        # Determine advancement conditions
        ready_by_position = advance_by_position if motion_phase else False
        ready_by_timeout = self._t >= 1.0

        if ready_by_position or ready_by_timeout:
            self._event += 1
            self._t = 0
            self._tau = 0.0
            self._gate_counter = 0

        print("EVENT: ", self._event)
        return target_joint_positions

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        base_xy = (1 - alpha) * np.array([current_x, current_y])
        target_xy = alpha * np.array([target_x, target_y])
        xy_target = base_xy + target_xy
        return xy_target

    def _get_alpha(self):
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._tau)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            # Return from placing XY back to picking XY smoothly
            return 1.0 - self._mix_sin(self._tau)
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._tau))
            h = self._combine_convex(self._h1, self._h0,  a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._tau))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        elif self._event == 6:
            h = self._combine_convex(
                self._h1, target_height, self._mix_sin(self._tau)
            )
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(
                target_height, self._h1, self._mix_sin(self._tau)
            )
        elif self._event == 9:
            h = self._h1
        else:
            raise ValueError()
        return h

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            end_effector_initial_height (typing.Optional[float], optional):
                end effector initial picking height to start from. If not
                defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional): Dt of
                each phase/ event step. 10 phases dt has to be defined.
                Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._tau = 0.0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False

        if hasattr(self, "_gate_counter"):
            self._gate_counter = 0
        if hasattr(self, "_gate_event_id"):
            self._gate_event_id = -1
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(
                self._events_dt, list
            ):
                raise Exception(
                    "event velocities need to be list or numpy array"
                )
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 10:
                raise Exception("events dt length must be 10")
        return

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase.
            Otherwise False.
        """
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        """Pauses the state machine's time and phase."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase."""
        self._pause = False
        return
