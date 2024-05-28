from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Set, Tuple, Type, Union

import cv2
import numba
import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

EGO_PARAMS = get_pacifica_parameters()
LANE_LAYERS = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]

HALF_WIDTH = EGO_PARAMS.half_width
FRONT_LENGTH = EGO_PARAMS.front_length
REAR_LENGTH = EGO_PARAMS.rear_length


class PolylineElements(IntEnum):
    """
    Enum for PolylineElements.
    """

    LANE = 0
    BOUNDARY = 1
    STOP_LINE = 2
    CROSSWALK = 3

    @classmethod
    def deserialize(cls, layer: str) -> PolylineElements:
        """Deserialize the type when loading from a string."""
        return PolylineElements.__members__[layer]


GlobalTypeMapping = {
    "AV": 0,
    TrackedObjectType.VEHICLE: 1,
    TrackedObjectType.PEDESTRIAN: 2,
    TrackedObjectType.BICYCLE: 3,
    PolylineElements.LANE: 4,
    PolylineElements.BOUNDARY: 5,
}


def interpolate_polyline(points: np.ndarray, t: int) -> np.ndarray:
    """copy from av2-api"""

    if points.ndim != 2:
        raise ValueError("Input array must be (N,2) or (N,3) in shape.")

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen: np.ndarray = np.linalg.norm(np.diff(points, axis=0), axis=1)  # type: ignore
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc: np.ndarray = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins: np.ndarray = np.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    chordlen[tbins - 1] = np.where(
        chordlen[tbins - 1] == 0, chordlen[tbins - 1] + 1e-6, chordlen[tbins - 1]
    )

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp: np.ndarray = anchors + offsets

    return points_interp


def get_ego_corners(rear_axle_xy: np.ndarray, heading: np.ndarray):
    """
    rear_axle_xy: [T, x, y]
    """
    ego_corners_offset = np.array(
        [
            [-REAR_LENGTH, -HALF_WIDTH],
            [-REAR_LENGTH, HALF_WIDTH],
            [FRONT_LENGTH, HALF_WIDTH],
            [FRONT_LENGTH, -HALF_WIDTH],
        ],
        dtype=np.float64,
    )
    ego_corners = rear_axle_xy[..., None, :] + ego_corners_offset[None, ...]
    rotate_mat = np.zeros((len(heading), 2, 2), dtype=np.float64)
    rotate_mat[:, 0, 0] = np.cos(heading)
    rotate_mat[:, 0, 1] = np.sin(heading)
    rotate_mat[:, 1, 0] = -np.sin(heading)
    rotate_mat[:, 1, 1] = np.cos(heading)

    ego_corners = ego_corners @ rotate_mat
    return ego_corners


@numba.njit
def rotate_round_z_axis(points: np.ndarray, angle: float):
    rotate_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        # dtype=np.float64,
    )
    # return np.matmul(points, rotate_mat)
    return points @ rotate_mat
