"""Move execution primitives for synchronized dual-arm control.

This module owns low-level execution details:
- arm_controller IK updates
- optional serial transmission
- optional visualization
- waiting for movement completion
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    from . import arm_controller, arm_serial
except ImportError:
    import arm_controller
    import arm_serial

def _as_vec(value: Sequence[float], expected_len: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (expected_len,):
        raise ValueError(f"{name} must have exactly {expected_len} values")
    return arr


@dataclass
class MoveRuntime:
    """Runtime state for executing Moves."""

    do_serial: bool = False
    do_visual: bool = False
    serial_port: str = "/dev/cu.usbmodem11401"
    left_arm_offset_y: float = arm_controller.LEFT_OFFSET_Y
    external_fig: object = None
    external_ax: object = None

    def __post_init__(self):
        self.right_controller = arm_controller.Arm("right_arm")
        self.left_controller = arm_controller.Arm("left_arm", [0, self.left_arm_offset_y, 0])

        self.current_right_pose = self.right_controller.pose.copy()
        self.current_left_pose = self.left_controller.pose.copy()
        self.current_right_gripper_closed = bool(self.right_controller.gripper_closed)
        self.current_left_gripper_closed = bool(self.left_controller.gripper_closed)

        if self.do_serial:
            self.ser = arm_serial.ArmSerial(port=self.serial_port)
            time.sleep(1.0) # wait for Arduino
            # Send home position
            self.ser.send_ik(
                self.right_controller.motor_angles,
                self.left_controller.motor_angles,
                self.current_right_gripper_closed,
                self.current_left_gripper_closed,
                movement_time=2.0,
            )
        else:
            self.ser = None

        if self.do_visual:
            if self.external_fig is not None:
                self.fig = self.external_fig
                self.ax = self.external_ax
            else:
                import matplotlib.pyplot as plt

                plt.ion()
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection="3d")
            self.refresh_visual()
        else:
            self.fig = None
            self.ax = None

        if self.do_serial:
            print("Waiting to home")
            time.sleep(2.0) # Wait for arms to home
            print("Wait done")

    def refresh_visual(self):
        if not self.do_visual:
            return

        ax = self.ax
        ax.clear()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(0.0, 0.85)
        ax.set_ylim(-0.45, 0.45)
        ax.set_zlim(0.0, 0.55)

        self.left_controller.arm_kinematics.arm_chain.plot(
            self.left_controller.motor_angles, ax, show=False)

        self.right_controller.arm_kinematics.arm_chain.plot(
            self.right_controller.motor_angles, ax, show=False)

        self.fig.canvas.draw()
        ax.figure.canvas.flush_events()

    def close(self):
        if self.ser is not None:
            try:
                self.ser.ser.close()
            except Exception:
                pass
            self.ser = None

        if self.fig is not None and self.external_fig is None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self.fig)
            except Exception:
                pass
        self.fig = None
        self.ax = None

    def current_move(self) -> "Move":
        return Move(
            right_x=float(self.current_right_pose[0]),
            right_y=float(self.current_right_pose[1]),
            right_z=float(self.current_right_pose[2]),
            right_roll=float(self.current_right_pose[3]),
            right_pitch=float(self.current_right_pose[4]),
            right_yaw=float(self.current_right_pose[5]),
            right_gripper_closed=self.current_right_gripper_closed,
            left_x=float(self.current_left_pose[0]),
            left_y=float(self.current_left_pose[1]),
            left_z=float(self.current_left_pose[2]),
            left_roll=float(self.current_left_pose[3]),
            left_pitch=float(self.current_left_pose[4]),
            left_yaw=float(self.current_left_pose[5]),
            left_gripper_closed=self.current_left_gripper_closed,
            movement_time=3.0,
        )


@dataclass(frozen=True)
class Move:
    """Represents one synchronized dual-arm command (15 parameters total)."""

    right_x: Optional[float] = None
    right_y: Optional[float] = None
    right_z: Optional[float] = None
    right_roll: Optional[float] = None
    right_pitch: Optional[float] = None
    right_yaw: Optional[float] = None
    right_gripper_closed: Optional[bool] = None

    left_x: Optional[float] = None
    left_y: Optional[float] = None
    left_z: Optional[float] = None
    left_roll: Optional[float] = None
    left_pitch: Optional[float] = None
    left_yaw: Optional[float] = None
    left_gripper_closed: Optional[bool] = None

    movement_time: float = 2.5

    def with_fallback(self, previous_move: "Move") -> "Move":
        if previous_move is None:
           raise ValueError("previous_move is required to resolve partial move fields")

        return Move(
            right_x=self.right_x if self.right_x is not None else previous_move.right_x,
            right_y=self.right_y if self.right_y is not None else previous_move.right_y,
            right_z=self.right_z if self.right_z is not None else previous_move.right_z,
            right_roll=self.right_roll if self.right_roll is not None else previous_move.right_roll,
            right_pitch=self.right_pitch if self.right_pitch is not None else previous_move.right_pitch,
            right_yaw=self.right_yaw if self.right_yaw is not None else previous_move.right_yaw,
            right_gripper_closed=(
                self.right_gripper_closed
                if self.right_gripper_closed is not None
                else previous_move.right_gripper_closed
            ),
            left_x=self.left_x if self.left_x is not None else previous_move.left_x,
            left_y=self.left_y if self.left_y is not None else previous_move.left_y,
            left_z=self.left_z if self.left_z is not None else previous_move.left_z,
            left_roll=self.left_roll if self.left_roll is not None else previous_move.left_roll,
            left_pitch=self.left_pitch if self.left_pitch is not None else previous_move.left_pitch,
            left_yaw=self.left_yaw if self.left_yaw is not None else previous_move.left_yaw,
            left_gripper_closed=(
                self.left_gripper_closed
                if self.left_gripper_closed is not None
                else previous_move.left_gripper_closed
            ),
            movement_time=float(self.movement_time),
        )

    def as_pose_vectors(self) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        # Configuration Constants
        Z_SLOPE_L = 0.019
        Z_SLOPE_R = 0.0185
        Z_INTERCEPT_L = -0.007
        Z_INTERCEPT_R = -0.0055
        XY_SCALE_L = 0.05
        XY_SCALE_R = 0.035
        XY_OFFSET_R = -0.03
        XY_OFFSET_L = -0.03

#         Z_SLOPE_L = 0
#         Z_SLOPE_R = 0
#         Z_INTERCEPT_L = 0
#         Z_INTERCEPT_R = 0
#         XY_SCALE_L = 0
#         XY_SCALE_R = 0
#         XY_OFFSET_R = 0
#         XY_OFFSET_L = 0

        required = (
            self.right_x, self.right_y, self.right_z, self.right_roll, self.right_pitch, self.right_yaw, self.right_gripper_closed,
            self.left_x, self.left_y, self.left_z, self.left_roll, self.left_pitch, self.left_yaw, self.left_gripper_closed,
        )
        if any(value is None for value in required):
            raise ValueError("Move is not fully specified; resolve missing fields first")

        def get_adjusted_z(x, y, z, left=False):
            # 1. Calculate Euclidean distance in XY plane from arm origin
            if left:
                dist_xy = np.linalg.norm([x, y - arm_controller.LEFT_OFFSET_Y])
                z_offset = (Z_SLOPE_L * dist_xy) + Z_INTERCEPT_L
            else:
                dist_xy = np.linalg.norm([x, y])
                z_offset = (Z_SLOPE_R * dist_xy) + Z_INTERCEPT_R
            # 2. Calculate offset: offset = (m * dist) + b
            # Calculate x, y scale
            if dist_xy != 0:
                if left:
                    xy_offset = np.array([x, y - arm_controller.LEFT_OFFSET_Y]) * XY_SCALE_L + np.array([x, y - arm_controller.LEFT_OFFSET_Y]) / dist_xy * XY_OFFSET_L
                else:
                    xy_offset = np.array([x, y]) * XY_SCALE_R + np.array([x, y]) / dist_xy * XY_OFFSET_R
            else:
                xy_offset = 0
            new_xy = np.array([x, y]) + xy_offset
            return float(new_xy[0]), float(new_xy[1]), float(z + z_offset)

        # Calculate adjusted Z for both arms
        rx, ry, rz = get_adjusted_z(self.right_x, self.right_y, self.right_z, left=False)
        lx, ly, lz = get_adjusted_z(self.left_x, self.left_y, self.left_z, left=True)

        right_pose = np.array(
            [rx, ry, rz, self.right_roll, self.right_pitch, self.right_yaw],
            dtype=float,
        )
        left_pose = np.array(
            [lx, ly, lz, self.left_roll, self.left_pitch, self.left_yaw],
            dtype=float,
        )
        
        return right_pose, left_pose, bool(self.right_gripper_closed), bool(self.left_gripper_closed)

    def to_send_ik_kwargs(self, right_ik: Sequence[float], left_ik: Sequence[float]) -> dict:
        if self.right_gripper_closed is None or self.left_gripper_closed is None:
            raise ValueError("Gripper states must be resolved before serial conversion")

        return {
            "ik_right": _as_vec(right_ik, 7, "right_ik"),
            "ik_left": _as_vec(left_ik, 7, "left_ik"),
            "grip_right_closed": bool(self.right_gripper_closed),
            "grip_left_closed": bool(self.left_gripper_closed),
            "movement_time": float(self.movement_time),
        }

    def execute(self, runtime: MoveRuntime) -> "Move":
        """Resolve missing fields, run IK, send serial command, and wait for completion."""
        resolved = self.with_fallback(runtime.current_move())
        right_pose, left_pose, right_grip, left_grip = resolved.as_pose_vectors()

        right_ik = runtime.right_controller.update_from_pose(right_pose, right_grip)
        left_ik = runtime.left_controller.update_from_pose(left_pose, left_grip)

        runtime.current_right_pose = right_pose.copy()
        runtime.current_left_pose = left_pose.copy()
        runtime.current_right_gripper_closed = right_grip
        runtime.current_left_gripper_closed = left_grip

        runtime.refresh_visual()

        if runtime.ser is not None:
            runtime.ser.send_ik(
                right_ik,
                left_ik,
                right_grip,
                left_grip,
                movement_time=resolved.movement_time,
            )

        time.sleep(float(resolved.movement_time))
        return resolved
