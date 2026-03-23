r"""
2-wheel differential drive kinematics for the mobile base.

Wheel layout (looking from above):
       Front (X+)
         ^
         |
    L----+----R   <- Left (motor 10) and Right (motor 9) wheels
         |

Body frame:
- X: Forward (positive = robot moves forward)
- Theta: Counter-clockwise rotation (positive = robot rotates CCW)

Note: Differential drive cannot strafe (no Y movement).
"""

import numpy as np
from typing import Dict, Tuple

from ..config import DIFF_WHEEL_RADIUS, DIFF_WHEEL_BASE_WIDTH


def body_to_wheel_velocities(
    x_vel: float,
    y_vel: float,  # Ignored in differential drive
    theta_vel: float,
    wheel_radius: float = DIFF_WHEEL_RADIUS,
    wheel_base_width: float = DIFF_WHEEL_BASE_WIDTH,
    max_raw: int = 3000,
) -> Dict[str, int]:
    """
    Convert desired body-frame velocities into wheel raw velocity commands.

    For differential drive, y_vel (strafe) is ignored since 2-wheel robots
    cannot move sideways.

    Parameters:
        x_vel: Linear velocity in x (m/s), positive = forward
        y_vel: Ignored (differential drive cannot strafe)
        theta_vel: Rotational velocity (deg/s), positive = counter-clockwise
        wheel_radius: Radius of each wheel (meters)
        wheel_base_width: Distance between the two wheels (meters)
        max_raw: Maximum allowed raw command (steps/s) per wheel

    Returns:
        Dictionary with wheel raw velocity commands:
        {"base_left_wheel": value, "base_right_wheel": value}
    """
    # Convert rotational velocity from deg/s to rad/s
    theta_rad = theta_vel * (np.pi / 180.0)

    # Empirically determined: the motor response on this robot requires
    # swapping the roles of x_vel (linear) and theta (angular) compared
    # to standard differential drive formulas
    half_width = wheel_base_width / 2.0

    # Linear velocities of each wheel (m/s)
    # Swapped: theta controls both wheels same direction, x controls differential
    v_left = half_width * theta_rad - x_vel
    v_right = half_width * theta_rad + x_vel

    # Convert to angular velocities (rad/s)
    omega_left = v_left / wheel_radius
    omega_right = v_right / wheel_radius

    # Convert to deg/s
    degps_left = omega_left * (180.0 / np.pi)
    degps_right = omega_right * (180.0 / np.pi)

    # Scale down if any wheel exceeds max_raw
    steps_per_deg = 4096.0 / 360.0
    raw_floats = [abs(degps_left) * steps_per_deg, abs(degps_right) * steps_per_deg]
    max_raw_computed = max(raw_floats)
    if max_raw_computed > max_raw:
        scale = max_raw / max_raw_computed
        degps_left *= scale
        degps_right *= scale

    # Convert each wheel's angular speed (deg/s) to raw integer
    raw_left = _degps_to_raw(degps_left)
    raw_right = _degps_to_raw(degps_right)

    return {
        "base_left_wheel": raw_left,
        "base_right_wheel": raw_right,
    }


def _degps_to_raw(degps: float) -> int:
    """
    Convert angular velocity in deg/s to raw motor command (steps/s).

    Parameters:
        degps: Angular velocity in degrees per second

    Returns:
        Raw motor command as signed integer
    """
    steps_per_deg = 4096.0 / 360.0
    speed_in_steps = degps * steps_per_deg
    speed_int = int(round(speed_in_steps))

    # Cap to signed 16-bit range (-32768 to 32767)
    if speed_int > 0x7FFF:
        speed_int = 0x7FFF
    elif speed_int < -0x8000:
        speed_int = -0x8000

    return speed_int


def wheel_velocities_to_body(
    left_wheel_raw: int,
    right_wheel_raw: int,
    wheel_radius: float = DIFF_WHEEL_RADIUS,
    wheel_base_width: float = DIFF_WHEEL_BASE_WIDTH,
) -> Tuple[float, float, float]:
    """
    Convert wheel raw velocities back to body-frame velocities (inverse kinematics).

    Parameters:
        left_wheel_raw: Left wheel raw velocity
        right_wheel_raw: Right wheel raw velocity
        wheel_radius: Radius of each wheel (meters)
        wheel_base_width: Distance between the two wheels (meters)

    Returns:
        Tuple of (x_vel, y_vel, theta_vel) in (m/s, m/s, deg/s)
        Note: y_vel is always 0 for differential drive
    """
    # Convert raw to deg/s
    degps_left = _raw_to_degps(left_wheel_raw)
    degps_right = _raw_to_degps(right_wheel_raw)

    # Convert to rad/s then to linear speed (m/s)
    omega_left = degps_left * (np.pi / 180.0)
    omega_right = degps_right * (np.pi / 180.0)

    v_left = omega_left * wheel_radius
    v_right = omega_right * wheel_radius

    # Inverse kinematics:
    # x_vel = (v_left + v_right) / 2
    # theta_rad = (v_right - v_left) / wheel_base_width
    x_vel = (v_left + v_right) / 2.0
    theta_rad = (v_right - v_left) / wheel_base_width
    theta_vel = theta_rad * (180.0 / np.pi)  # Convert to deg/s

    # y_vel is always 0 for differential drive
    return x_vel, 0.0, theta_vel


def _raw_to_degps(raw_speed: int) -> float:
    """
    Convert raw motor command to angular velocity in deg/s.

    Parameters:
        raw_speed: Raw motor command (steps/s)

    Returns:
        Angular velocity in degrees per second
    """
    steps_per_deg = 4096.0 / 360.0
    return raw_speed / steps_per_deg
