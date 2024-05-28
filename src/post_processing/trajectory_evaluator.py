#   Heavily borrowed from:
#   https://github.com/autonomousvision/tuplan_garage (Apache License 2.0)
# & https://github.com/motional/nuplan-devkit (Apache License 2.0)

import warnings
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import shapely.creation
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from shapely import LineString

from src.scenario_manager.occupancy_map import OccupancyMap

from .common.enum import (
    CollisionType,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from .common.geometry import (
    compute_agents_vertices,
    ego_rear_to_center,
    get_collision_type,
)
from .evaluation.comfort_metrics import ego_is_comfortable
from .forward_simulation.forward_simulator import ForwardSimulator
from .observation.world_from_prediction import WorldFromPrediction

WEIGHTED_METRICS_WEIGHTS = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.PROGRESS] = 5.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.TTC] = 5.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.SPEED_LIMIT] = 2.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.COMFORTABLE] = 2.0

DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 4.0  # [m] (driving direction)
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 12.0  # [m] (driving direction)
STOPPED_SPEED_THRESHOLD = 5e-03  # [m/s] (ttc)
PROGRESS_DISTANCE_THRESHOLD = 0.1  # [m] (progress)
MAX_OVERSPEED_VALUE_THRESHOLD = 2.23  # [m/s] (speed limit)


class TrajectoryEvaluator:
    def __init__(
        self,
        dt: float = 0.1,
        num_frames: int = 40,
    ) -> None:
        assert dt * num_frames <= 8, "dt * num_frames should be less than 8s"

        self._dt = dt
        self._num_frames = num_frames

        self._route_lane_dict = None
        self._drivable_area_map: Optional[OccupancyMap] = None
        self._world = WorldFromPrediction(dt, num_frames)
        self._forward_simulator = ForwardSimulator(dt, num_frames)

        self._init_ego_state: Optional[EgoState] = None
        self._ego_rollout: Optional[np.ndarray[np.float64]] = None
        self._ego_polygons: Optional[np.ndarray[np.object_]] = None
        self._ego_footprints: Optional[np.ndarray[np.object_]] = None
        self._ego_footprints_speed_limit: Optional[np.ndarray[np.object_]] = None
        self._ego_baseline_path: [Optional[LineString]] = None
        self._ego_progress: Optional[np.ndarray[np.float64]] = None
        self._ego_parameters = get_pacifica_parameters()
        self._ego_shape = np.array(
            [self._ego_parameters.width, self._ego_parameters.length],
            dtype=np.float64,
        )

        self._multi_metrics: Optional[np.ndarray[np.float64]] = None
        self._weighted_metrics: Optional[np.ndarray[np.float64]] = None
        self._at_fault_collision_time: Optional[np.ndarray[np.float64]] = None
        self._final_score = None

        self.progress_score = None

    def time_to_at_fault_collision(self, rollout_idx: int) -> float:
        return self._at_fault_collision_time[rollout_idx]

    def evaluate(
        self,
        candidate_trajectories: np.ndarray,
        init_ego_state: EgoState,
        detections: DetectionsTracks,
        traffic_light_data: List[TrafficLightStatusData],
        agents_info: Dict[str, np.ndarray],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        drivable_area_map: Optional[OccupancyMap],
        baseline_path: Optional[LineString],
    ):
        self._reset(
            candidate_trajectories=candidate_trajectories,
            init_ego_state=init_ego_state,
            detections=detections,
            traffic_light_data=traffic_light_data,
            agents_info=agents_info,
            route_lane_dict=route_lane_dict,
            drivable_area_map=drivable_area_map,
            baseline_path=baseline_path,
        )

        self._update_ego_footprints()

        self._evaluate_no_at_fault_collisions()
        self._evaluate_drivable_area_compliance()
        self._evaluate_driving_direction_compliance()

        self._evaluate_time_to_collision()
        self._evaluate_speed_limit_compliance()
        self._evaluate_progress()
        self._evaluate_is_comfortable()

        return self._aggregate_scores()

    def _reset(
        self,
        candidate_trajectories: np.ndarray,
        init_ego_state: EgoState,
        detections: DetectionsTracks,
        traffic_light_data: List[TrafficLightStatusData],
        agents_info: Dict[str, np.ndarray],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        drivable_area_map: Optional[OccupancyMap],
        baseline_path: Optional[LineString],
    ):
        self._num_candidates = len(candidate_trajectories)
        self._route_lane_dict = route_lane_dict
        self._drivable_area_map = drivable_area_map
        self._world.drivable_area = drivable_area_map
        self._ego_baseline_path = baseline_path

        self._update_ego_rollout(
            init_ego_state=init_ego_state,
            candidate_trajectories=candidate_trajectories,
        )
        self._world.update(
            ego_state=init_ego_state,
            detections=detections,
            traffic_light_data=traffic_light_data,
            agents_info=agents_info,
            route_lane_dict=self._route_lane_dict,
        )

        self._at_fault_collision_time = np.full((self._num_candidates,), np.inf)

        self._multi_metrics = np.zeros(
            (len(MultiMetricIndex), self._num_candidates), dtype=np.float64
        )
        self._weighted_metrics = np.zeros(
            (len(WeightedMetricIndex), self._num_candidates), dtype=np.float64
        )

    def _update_ego_rollout(
        self, init_ego_state: EgoState, candidate_trajectories: np.ndarray
    ):
        rollout_states = self._forward_simulator.forward(
            candidate_trajectories, init_ego_state
        )
        N, T, _ = rollout_states.shape
        vertices = compute_agents_vertices(
            center=ego_rear_to_center(rollout_states[..., :2], rollout_states[..., 2]),
            angle=rollout_states[..., 2],
            shape=self._ego_shape[None, :].repeat(N, axis=0),
        )

        self._init_ego_state = init_ego_state
        self._ego_rollout = rollout_states
        self._ego_vertices = vertices
        self._ego_polygons = shapely.creation.polygons(vertices)
        self._ego_footprints = np.zeros((N, T, 3), dtype=bool)
        self._ego_footprints_speed_limit = np.zeros((N, T), dtype=np.float64)

    def _update_ego_footprints(self) -> None:
        keypoints = np.concatenate(
            [self._ego_vertices, self._ego_rollout[:, :, None, :2]], axis=-2
        )

        N, T, P, _ = keypoints.shape
        keypoints = keypoints.reshape(N * T * P, 2)

        (
            in_polygons,
            speed_limit,
        ) = self._drivable_area_map.points_in_polygons_with_attribute(
            keypoints, "speed_limit"
        )

        # (N, T, num_polygons, num_points)
        in_polygons = in_polygons.reshape(
            len(self._drivable_area_map), N, T, P
        ).transpose(1, 2, 0, 3)
        speed_limit = speed_limit.reshape(
            len(self._drivable_area_map), N, T, P
        ).transpose(1, 2, 0, 3)[..., 4]

        da_on_route_idx: List[int] = [
            idx
            for idx, token in enumerate(self._drivable_area_map.tokens)
            if token in self._route_lane_dict.keys()
        ]

        corners_in_polygon, center_in_polygon = (
            in_polygons[..., :4],
            in_polygons[..., 4],
        )

        on_multi_lane_mask = corners_in_polygon.any(-1).sum(-1) > 1
        on_single_lane_mask = corners_in_polygon.all(-1).any(-1)
        on_multi_lane_mask &= ~on_single_lane_mask
        out_drivable_area_mask = (corners_in_polygon.sum(-2) > 0).sum(-1) < 4
        oncoming_traffic_mask = ~center_in_polygon[..., da_on_route_idx].any(-1)
        speed_limit[~center_in_polygon] = 0.0

        self._ego_footprints[on_multi_lane_mask, EgoAreaIndex.MULTIPLE_LANES] = True
        self._ego_footprints[out_drivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA] = (
            True
        )
        self._ego_footprints[oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC] = (
            True
        )
        self._ego_footprints_speed_limit = speed_limit.max(-1)

    def _evaluate_no_at_fault_collisions(self):
        no_collision_scores = np.ones(self._num_candidates, dtype=np.float64)
        collided_tokens = {
            i: deepcopy(self._world.collided_tokens)
            for i in range(self._num_candidates)
        }

        for i in range(1, self._num_frames + 1):
            ego_polygons = self._ego_polygons[:, i]
            intersect_indices = self._world[i].query(ego_polygons, "intersects")

            if len(intersect_indices) == 0:
                continue

            for rollout_idx, obj_idx in zip(intersect_indices[0], intersect_indices[1]):
                token = self._world[i].tokens[obj_idx]
                if token.startswith(self._world.red_light_prefix):
                    no_collision_scores[rollout_idx] = 0
                    self._at_fault_collision_time[rollout_idx] = min(
                        i * self._dt, self._at_fault_collision_time[rollout_idx]
                    )
                    continue
                elif token in collided_tokens[rollout_idx]:
                    continue

                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._ego_footprints[rollout_idx, i, EgoAreaIndex.MULTIPLE_LANES]
                    or self._ego_footprints[
                        rollout_idx, i, EgoAreaIndex.NON_DRIVABLE_AREA
                    ]
                )

                object_info = self._world.get_object_at_frame(token, i)

                collision_type = get_collision_type(
                    state=self._ego_rollout[rollout_idx, i],
                    ego_polygon=ego_polygons[rollout_idx],
                    object_info=object_info,
                )

                collisions_at_stopped_track_or_active_front = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral = (
                    collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                )

                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    no_at_fault_collision_score = (
                        0.0 if object_info["is_agent"] else 0.5
                    )
                    no_collision_scores[rollout_idx] = np.minimum(
                        no_collision_scores[rollout_idx], no_at_fault_collision_score
                    )
                    collided_tokens[rollout_idx].append(token)
                    self._at_fault_collision_time[rollout_idx] = min(
                        i * self._dt, self._at_fault_collision_time[rollout_idx]
                    )

        self._multi_metrics[MultiMetricIndex.NO_COLLISION] = no_collision_scores

    def _evaluate_time_to_collision(self):
        ttc_score = np.ones(self._num_candidates, dtype=np.float64)
        collided_tokens = {
            i: deepcopy(self._world.collided_tokens)
            for i in range(self._num_candidates)
        }

        if self._dt == 0.1:
            future_idx = np.array([2, 4, 6, 8])
        elif self._dt == 0.2:
            future_idx = np.array([2, 4, 6, 8])
        else:
            raise NotImplementedError

        n_future_steps = len(future_idx)

        ego_vertices = self._ego_vertices.copy()
        heading = self._ego_rollout[..., StateIndex.HEADING]
        speed = self._ego_rollout[..., StateIndex.VELOCITY_X]
        direction = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
        delta = (
            (direction * speed[..., None])[:, :, None, :]
            * future_idx.reshape(1, 1, n_future_steps, 1)
            * self._dt
        )

        ego_vertices_n_steps = ego_vertices[:, :, None, :] + delta[:, :, :, None]
        ego_polygon = shapely.creation.polygons(ego_vertices_n_steps)

        for t in range(1, self._num_frames + 1):
            for i, step in enumerate(future_idx):
                if t + step > self._num_frames:
                    break

                polygon_at_step = ego_polygon[:, t, i]
                intersect_indices = self._world[t].query(polygon_at_step, "intersects")

                if len(intersect_indices) == 0:
                    continue

                for rollout_idx, obj_idx in zip(
                    intersect_indices[0], intersect_indices[1]
                ):
                    token = self._world[i + step].tokens[obj_idx]
                    if (
                        token.startswith(self._world.red_light_prefix)
                        or (token in collided_tokens[rollout_idx])
                        or (speed[rollout_idx, t] < STOPPED_SPEED_THRESHOLD)
                    ):
                        continue

                    ego_in_multiple_lanes_or_nondrivable_area = (
                        self._ego_footprints[
                            rollout_idx, i, EgoAreaIndex.MULTIPLE_LANES
                        ]
                        or self._ego_footprints[
                            rollout_idx, i, EgoAreaIndex.NON_DRIVABLE_AREA
                        ]
                    )

                    object_info = self._world.get_object_at_frame(token, i)

                    collision_type = get_collision_type(
                        state=self._ego_rollout[rollout_idx, i],
                        ego_polygon=polygon_at_step[rollout_idx],
                        object_info=object_info,
                    )
                    # print(rollout_idx, t, i, collision_type, token)

                    collisions_at_stopped_track_or_active_front = collision_type in [
                        CollisionType.ACTIVE_FRONT_COLLISION,
                        CollisionType.STOPPED_TRACK_COLLISION,
                    ]
                    collision_at_lateral = (
                        collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                    )

                    if collisions_at_stopped_track_or_active_front or (
                        ego_in_multiple_lanes_or_nondrivable_area
                        and collision_at_lateral
                    ):
                        ttc_score[rollout_idx] = 0.0
                        collided_tokens[rollout_idx].append(token)

        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_score

    def _evaluate_driving_direction_compliance(self):
        displacement = np.linalg.norm(
            np.diff(self._ego_rollout[..., :2], axis=-2), axis=-1
        )
        on_coming_traffic_mask = self._ego_footprints[
            :, 1:, EgoAreaIndex.ONCOMING_TRAFFIC
        ]
        displacement[~on_coming_traffic_mask] = 0.0

        # ignores changes of driving direction
        cum_distance = displacement.sum(-1)

        scores = np.zeros(self._num_candidates, dtype=np.float64)
        scores[cum_distance < DRIVING_DIRECTION_VIOLATION_THRESHOLD] = 0.5
        scores[cum_distance < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD] = 1.0

        # forbid reverse
        reverse_mask = (self._ego_rollout[..., StateIndex.VELOCITY_X] < 0.0).sum(-1) > 5
        scores[reverse_mask] = 0.0

        self._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION] = scores

    def _evaluate_drivable_area_compliance(self):
        scores = np.ones(self._num_candidates, dtype=np.float64)
        non_da_mask = self._ego_footprints[..., EgoAreaIndex.NON_DRIVABLE_AREA]
        if non_da_mask[:, :3].any(-1).all():
            # corner case: ego is already out of drivable area at the beginning
            pass
        else:
            scores[non_da_mask.any(-1)] = 0.0

        self._multi_metrics[MultiMetricIndex.DRIVABLE_AREA] = scores

    def _evaluate_progress(self):
        progress = np.zeros(self._num_candidates, dtype=np.float64)

        start_point = shapely.Point(*self._init_ego_state.rear_axle.array)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start_progress = self._ego_baseline_path.project(start_point)

        for i in range(self._num_candidates):
            end_point = shapely.Point(*self._ego_rollout[i, -1, :2])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                end_progress = self._ego_baseline_path.project(end_point)
            progress[i] = max(end_progress - start_progress, 0.0)

        self._ego_progress = progress

    def _evaluate_is_comfortable(self):
        timestamp = np.linspace(0, self._dt * self._num_frames, self._num_frames + 1)
        is_comfortable = ego_is_comfortable(self._ego_rollout, timestamp)

        self._weighted_metrics[WeightedMetricIndex.COMFORTABLE] = is_comfortable.all(-1)

    def _evaluate_speed_limit_compliance(self):
        ego_speed = np.linalg.norm(
            self._ego_rollout[..., StateIndex.VELOCITY_2D], axis=-1
        )

        overspeed = (ego_speed - self._ego_footprints_speed_limit).clip(min=0.0)
        violation_loss = overspeed.sum(-1) / (
            MAX_OVERSPEED_VALUE_THRESHOLD * overspeed.shape[-1]
        )
        score = (1 - violation_loss).clip(min=0.0)

        self._weighted_metrics[WeightedMetricIndex.SPEED_LIMIT] = score

    def _aggregate_scores(self):
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)
        comfort_multi_score = np.ones_like(multiplicate_metric_scores)
        speed_limit_score = np.ones_like(multiplicate_metric_scores)
        comfort_multi_score[
            self._weighted_metrics[WeightedMetricIndex.COMFORTABLE] == 0
        ] = 0.5
        speed_limit_score[
            self._weighted_metrics[WeightedMetricIndex.SPEED_LIMIT] < 0.5
        ] = 0.5
        multiplicate_metric_scores *= comfort_multi_score
        multiplicate_metric_scores *= speed_limit_score

        progress = self._ego_progress * multiplicate_metric_scores
        max_progress = progress.max()
        if max_progress > PROGRESS_DISTANCE_THRESHOLD:
            progress_score = progress / max_progress
            not_making_progress_mask = self._ego_progress < PROGRESS_DISTANCE_THRESHOLD
            progress_score[not_making_progress_mask] = 0.0
        else:
            progress_score = (
                np.ones(self._num_candidates, dtype=np.float64)
                * multiplicate_metric_scores
            )
        self.progress_score = progress_score
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = progress_score

        weighted_metric_scores = (
            self._weighted_metrics * WEIGHTED_METRICS_WEIGHTS[..., None]
        ).sum(axis=0) / WEIGHTED_METRICS_WEIGHTS.sum()

        final_scores = multiplicate_metric_scores * weighted_metric_scores
        self._final_score = final_scores

        return final_scores
