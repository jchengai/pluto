#   Heavily borrowed from:
#   https://github.com/autonomousvision/tuplan_garage (Apache License 2.0)
# & https://github.com/motional/nuplan-devkit (Apache License 2.0)

from typing import Dict, Any

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
from shapely import Polygon, LineString

from .enum import CollisionType, StateIndex
from nuplan.planning.simulation.observation.idm.utils import is_agent_behind


def compute_agents_vertices(
    center: np.ndarray,
    angle: np.ndarray,
    shape: np.ndarray,
) -> np.ndarray:
    """
    Args:
        position: (N, T, 2)
        angle: (N, T)
        shape: (N, 2) [width, length]
    Returns:
        4 corners of oriented box (FL, RL, RR, FR)
        vertices: (N, T, 4, 2)
    """
    # Extracting dimensions
    N, T = center.shape[0], center.shape[1]

    # Reshaping the arrays for calculations
    center = center.reshape(N * T, 2)
    angle = angle.reshape(N * T)

    if shape.ndim == 2:
        shape = (shape / 2).repeat(T, axis=0)
    else:
        shape = (shape / 2).reshape(N * T, 2)

    # Calculating half width and half_l
    half_w = shape[:, 0]
    half_l = shape[:, 1]

    # Calculating cos and sin of angles
    cos_angle = np.cos(angle)[:, None]
    sin_angle = np.sin(angle)[:, None]
    rot_mat = np.stack([cos_angle, sin_angle, -sin_angle, cos_angle], axis=-1).reshape(
        N * T, 2, 2
    )

    offset_width = np.stack([half_w, half_w, -half_w, -half_w], axis=-1)
    offset_length = np.stack([half_l, -half_l, -half_l, half_l], axis=-1)

    vertices = np.stack([offset_length, offset_width], axis=-1)
    vertices = np.matmul(vertices, rot_mat) + center[:, None]

    # Calculating vertices
    vertices = vertices.reshape(N, T, 4, 2)

    return vertices


def ego_rear_to_center(rear_xy, heading, rear_to_center=1.461):
    direction = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
    center = rear_xy + direction * rear_to_center
    return center


def get_sub_polygon(polygon: Polygon, ratio=0.4):
    vertices = np.array(polygon.exterior.coords)
    return Polygon(
        [
            vertices[0],
            vertices[0] * (1 - ratio) + vertices[1] * ratio,
            vertices[3] * (1 - ratio) + vertices[2] * ratio,
            vertices[3],
        ]
    )


def get_collision_type(
    state: np.ndarray,
    ego_polygon: Polygon,
    object_info: Dict[str, Any],
    stopped_speed_threshold: float = 5e-02,
) -> CollisionType:
    """
    Classify collision between ego and the track.
    :param ego_state: Ego's state at the current timestamp.
    :param tracked_object: Tracked object.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return Collision type.
    """

    ego_speed = np.hypot(state[StateIndex.VELOCITY_X], state[StateIndex.VELOCITY_Y])

    is_ego_stopped = float(ego_speed) <= stopped_speed_threshold

    object_pos: np.ndarray = object_info["pose"]
    object_velocity: np.ndarray = object_info["velocity"]
    object_polygon: Polygon = object_info["polygon"]

    tracked_object_center = StateSE2(*object_pos)

    ego_rear_axle_pose: StateSE2 = StateSE2(*state[StateIndex.STATE_SE2])

    # Collisions at (close-to) zero ego speed
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION

    # Collisions at (close-to) zero track speed
    elif np.linalg.norm(object_velocity) <= stopped_speed_threshold:
        collision_type = CollisionType.STOPPED_TRACK_COLLISION

    # Rear collision when both ego and track are not stopped
    elif is_agent_behind(ego_rear_axle_pose, tracked_object_center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION

    # Front bumper collision when both ego and track are not stopped
    # elif get_sub_polygon(ego_polygon).intersects(object_polygon):
        # collision_type = CollisionType.ACTIVE_FRONT_COLLISION
    elif LineString(
        [
            ego_polygon.exterior.coords[0],
            ego_polygon.exterior.coords[3],
        ]
    ).intersects(object_polygon):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION

    # Lateral collision when both ego and track are not stopped
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type
