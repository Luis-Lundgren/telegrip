"""
TelegripRobot - XLerobot variant without head motors.

This subclass of XLerobot is for robots that have:
- Left arm: motors 1-6 (no head motors 7, 8)
- Right arm: motors 1-6 + wheels (configurable: 2-wheel differential or 3-wheel omni)

It inherits all the useful methods from XLerobot (send_action, connect, disconnect,
wheel kinematics, etc.) but configures the motor buses differently.
"""

import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.robot import Robot
from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig
from lerobot.robots.xlerobot.xlerobot import XLerobot

from ..config import (
    BASE_MODE,
    DIFF_LEFT_MOTOR_ID, DIFF_RIGHT_MOTOR_ID,
    OMNI_LEFT_MOTOR_ID, OMNI_BACK_MOTOR_ID, OMNI_RIGHT_MOTOR_ID,
)

logger = logging.getLogger(__name__)


class TelegripRobot(XLerobot):
    """
    XLerobot variant without head motors.

    Motor configuration:
    - Bus 1 (port1): Left arm motors 1-6
    - Bus 2 (port2): Right arm motors 1-6 + wheel motors 7, 8, 9
    """

    name = "telegrip_robot"

    def __init__(self, config: XLerobotConfig):
        # Skip XLerobot.__init__ and call Robot.__init__ directly
        # This avoids creating the buses with head motors
        Robot.__init__(self, config)

        self.config = config
        self.teleop_keys = config.teleop_keys

        # Speed levels (inherited behavior)
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.3, "theta": 90},
        ]
        self.speed_index = 0

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # Calibration for bus1 (left arm only, no head)
        if self.calibration.get("left_arm_shoulder_pan") is not None:
            calibration1 = {
                "left_arm_shoulder_pan": self.calibration.get("left_arm_shoulder_pan"),
                "left_arm_shoulder_lift": self.calibration.get("left_arm_shoulder_lift"),
                "left_arm_elbow_flex": self.calibration.get("left_arm_elbow_flex"),
                "left_arm_wrist_flex": self.calibration.get("left_arm_wrist_flex"),
                "left_arm_wrist_roll": self.calibration.get("left_arm_wrist_roll"),
                "left_arm_gripper": self.calibration.get("left_arm_gripper"),
                # NO head motors
            }
        else:
            calibration1 = self.calibration

        # Bus 1: Left arm only (NO head motors 7, 8)
        self.bus1 = FeetechMotorsBus(
            port=self.config.port1,
            motors={
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration1,
        )

        # Store base mode for later use
        self.base_mode = BASE_MODE
        logger.info(f"TelegripRobot using base mode: {self.base_mode}")

        # Build wheel motor configuration based on base mode
        if self.base_mode == "differential":
            # 2-wheel differential drive
            wheel_motors = {
                "base_left_wheel": Motor(DIFF_LEFT_MOTOR_ID, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(DIFF_RIGHT_MOTOR_ID, "sts3215", MotorNormMode.RANGE_M100_100),
            }
            wheel_calibration_keys = ["base_left_wheel", "base_right_wheel"]
        else:
            # 3-wheel omnidirectional drive
            wheel_motors = {
                "base_left_wheel": Motor(OMNI_LEFT_MOTOR_ID, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(OMNI_BACK_MOTOR_ID, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(OMNI_RIGHT_MOTOR_ID, "sts3215", MotorNormMode.RANGE_M100_100),
            }
            wheel_calibration_keys = ["base_left_wheel", "base_back_wheel", "base_right_wheel"]

        # Calibration for bus2 (right arm + wheels)
        if self.calibration.get("right_arm_shoulder_pan") is not None:
            calibration2 = {
                "right_arm_shoulder_pan": self.calibration.get("right_arm_shoulder_pan"),
                "right_arm_shoulder_lift": self.calibration.get("right_arm_shoulder_lift"),
                "right_arm_elbow_flex": self.calibration.get("right_arm_elbow_flex"),
                "right_arm_wrist_flex": self.calibration.get("right_arm_wrist_flex"),
                "right_arm_wrist_roll": self.calibration.get("right_arm_wrist_roll"),
                "right_arm_gripper": self.calibration.get("right_arm_gripper"),
            }
            # Add wheel calibrations
            for key in wheel_calibration_keys:
                calibration2[key] = self.calibration.get(key)
        else:
            calibration2 = self.calibration

        # Build bus2 motors dict (right arm + wheels)
        bus2_motors = {
            "right_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
            "right_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
            "right_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
            "right_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
            "right_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
            "right_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
        bus2_motors.update(wheel_motors)

        # Bus 2: Right arm + wheels
        self.bus2 = FeetechMotorsBus(
            port=self.config.port2,
            motors=bus2_motors,
            calibration=calibration2,
        )

        # Motor groupings
        self.left_arm_motors = [m for m in self.bus1.motors if m.startswith("left_arm")]
        self.right_arm_motors = [m for m in self.bus2.motors if m.startswith("right_arm")]
        self.head_motors = []  # No head motors
        self.base_motors = [m for m in self.bus2.motors if m.startswith("base")]

        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot, automatically loading calibration if available.

        Unlike XLerobot.connect(), this method does NOT prompt for confirmation
        when a calibration file exists - it just loads it directly.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus1.connect()
        self.bus2.connect()

        # Load calibration automatically if file exists (no prompt)
        if self.calibration_fpath.is_file() and self.calibration:
            logger.info(f"Loading calibration from {self.calibration_fpath}")
            try:
                # Load calibration data into bus memory
                self.bus1.calibration = {k: v for k, v in self.calibration.items() if k in self.bus1.motors}
                self.bus2.calibration = {k: v for k, v in self.calibration.items() if k in self.bus2.motors}
                logger.info("Calibration data loaded into bus memory successfully!")

                # Write calibration data to motors
                self.bus1.write_calibration({k: v for k, v in self.calibration.items() if k in self.bus1.motors})
                self.bus2.write_calibration({k: v for k, v in self.calibration.items() if k in self.bus2.motors})
                logger.info("Calibration restored successfully from file!")
            except Exception as e:
                logger.warning(f"Failed to restore calibration from file: {e}")
                if calibrate:
                    logger.info("Proceeding with manual calibration...")
                    self.calibrate()
        elif calibrate:
            logger.info("No calibration file found, proceeding with manual calibration...")
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def get_observation(self) -> dict[str, Any]:
        """
        Get current robot state.

        Overrides XLerobot.get_observation() to skip head motor reads
        since TelegripRobot has no head motors.
        """
        import time
        from lerobot.utils.errors import DeviceNotConnectedError

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read actuators position for arms and velocity for base
        start = time.perf_counter()
        left_arm_pos = self.bus1.sync_read("Present_Position", self.left_arm_motors)
        right_arm_pos = self.bus2.sync_read("Present_Position", self.right_arm_motors)
        # NO head motors to read
        base_wheel_vel = self.bus2.sync_read("Present_Velocity", self.base_motors)

        # Convert wheel velocities to body frame based on mode
        if self.base_mode == "differential":
            from .differential_kinematics import wheel_velocities_to_body as diff_wheel_to_body
            x_vel, y_vel, theta_vel = diff_wheel_to_body(
                base_wheel_vel["base_left_wheel"],
                base_wheel_vel["base_right_wheel"],
            )
            base_vel = {"x.vel": x_vel, "y.vel": y_vel, "theta.vel": theta_vel}
        else:
            base_vel = self._wheel_raw_to_body(
                base_wheel_vel["base_left_wheel"],
                base_wheel_vel["base_back_wheel"],
                base_wheel_vel["base_right_wheel"],
            )

        left_arm_state = {f"{k}.pos": v for k, v in left_arm_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_arm_pos.items()}
        # Combine all arm states (no head)
        obs_dict = {**left_arm_state, **right_arm_state, **base_vel}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Override send_action to handle differential vs omnidirectional base modes.

        In differential mode, we use 2-wheel kinematics.
        In omnidirectional mode, we use the parent's 3-wheel kinematics.
        """
        from lerobot.utils.errors import DeviceNotConnectedError

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract position commands for arms
        left_arm_pos = {k: v for k, v in action.items() if k.startswith("left_arm_") and k.endswith(".pos")}
        right_arm_pos = {k: v for k, v in action.items() if k.startswith("right_arm_") and k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        # Convert body velocities to wheel commands based on mode
        if self.base_mode == "differential":
            from .differential_kinematics import body_to_wheel_velocities as diff_body_to_wheel
            x_in = base_goal_vel.get("x.vel", 0.0)
            theta_in = base_goal_vel.get("theta.vel", 0.0)
            if x_in != 0 or theta_in != 0:
                logger.debug(f"Differential input: x={x_in}, theta={theta_in}")
            base_wheel_goal_vel = diff_body_to_wheel(
                x_in,
                base_goal_vel.get("y.vel", 0.0),  # Will be ignored
                theta_in,
            )
            if x_in != 0 or theta_in != 0:
                logger.debug(f"Differential output: {base_wheel_goal_vel}")
        else:
            # Use parent's 3-wheel kinematics
            base_wheel_goal_vel = self._body_to_wheel_raw(
                base_goal_vel.get("x.vel", 0.0),
                base_goal_vel.get("y.vel", 0.0),
                base_goal_vel.get("theta.vel", 0.0),
            )

        # Strip the .pos suffix for sync_write
        left_arm_pos_raw = {k.replace(".pos", ""): v for k, v in left_arm_pos.items()}
        right_arm_pos_raw = {k.replace(".pos", ""): v for k, v in right_arm_pos.items()}

        # Write to motors
        if left_arm_pos_raw:
            self.bus1.sync_write("Goal_Position", left_arm_pos_raw)
        if right_arm_pos_raw:
            self.bus2.sync_write("Goal_Position", right_arm_pos_raw)
        if base_wheel_goal_vel:
            self.bus2.sync_write("Goal_Velocity", base_wheel_goal_vel)

        return action

    @property
    def _state_ft(self) -> dict[str, type]:
        """Override to exclude head motors from state features."""
        return dict.fromkeys(
            (
                "left_arm_shoulder_pan.pos",
                "left_arm_shoulder_lift.pos",
                "left_arm_elbow_flex.pos",
                "left_arm_wrist_flex.pos",
                "left_arm_wrist_roll.pos",
                "left_arm_gripper.pos",
                "right_arm_shoulder_pan.pos",
                "right_arm_shoulder_lift.pos",
                "right_arm_elbow_flex.pos",
                "right_arm_wrist_flex.pos",
                "right_arm_wrist_roll.pos",
                "right_arm_gripper.pos",
                # No head motors
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )
