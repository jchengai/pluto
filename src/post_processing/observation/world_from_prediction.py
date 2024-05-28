#   Heavily borrowed from:
#   https://github.com/autonomousvision/tuplan_garage (Apache License 2.0)
# & https://github.com/motional/nuplan-devkit (Apache License 2.0)

from typing import Dict, List, Optional

import numpy as np
import shapely.creation
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from src.scenario_manager.occupancy_map import OccupancyMap

from ..common.geometry import compute_agents_vertices


class WorldFromPrediction:
    def __init__(self, dt=0.1, num_frames=40, base_radius=50) -> None:
        self.dt = dt
        self.num_frames = num_frames
        self.interval = int(dt // 0.1)

        # todo: determined by velocity
        self.radius = max(base_radius * dt * num_frames / 4, base_radius)

        self.occupancy_map: Optional[List[OccupancyMap]] = None
        self.drivable_area: Optional[OccupancyMap] = None
        self.objects_info: Optional[Dict[str, np.ndarray]] = None

        self.red_light_prefix = "red_light"

        self.collided_tokens = None
        self._static_object_tokens = None
        self._agent_tokens = None
        self._ego_state = None

    def __getitem__(self, idx: int) -> OccupancyMap:
        assert 0 <= idx < len(self.occupancy_map), "index out of range"
        return self.occupancy_map[idx]

    def __len__(self) -> int:
        return len(self.occupancy_map)

    def update(
        self,
        ego_state: EgoState,
        detections: DetectionsTracks,
        traffic_light_data: List[TrafficLightStatusData],
        agents_info: Dict[str, np.ndarray],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ):
        self._ego_state = ego_state
        self.occupancy_map: List[OccupancyMap] = []

        tl_tokens, tl_polygons = self._get_route_red_traffic_lights(
            traffic_light_data, route_lane_dict
        )
        statics_tokens, statics_polygon = self._get_static_obstacles(
            ego_state, detections
        )
        agents_tokens, agents_vertices = self._get_dynamic_agents_from_prediction(
            agents_info
        )
        has_agents = len(agents_tokens) > 0
        agents_polygon = np.array([], dtype=np.object_)

        for i in range(self.num_frames + 1):
            if has_agents:
                agents_polygon = agents_vertices[:, i]
                agents_polygon = shapely.creation.polygons(agents_polygon)

            frame_tokens = statics_tokens + agents_tokens + tl_tokens
            frame_polygons = np.concatenate(
                [statics_polygon, agents_polygon, tl_polygons], axis=0
            )
            frame_occupancy_map = OccupancyMap(
                tokens=frame_tokens, geometries=frame_polygons
            )
            self.occupancy_map.append(frame_occupancy_map)

        # update initial ego collision status
        self.collided_tokens = []
        ego_polygon = ego_state.car_footprint.geometry
        intersect_tokens = self.occupancy_map[0].intersects(ego_polygon)
        for token in intersect_tokens:
            if token.startswith(self.red_light_prefix):
                if not ego_polygon.within(self.occupancy_map[0][token]):
                    continue
            self.collided_tokens.append(token)

        self._static_object_tokens = set(statics_tokens)
        self._agent_tokens = set(agents_tokens)

    def _get_static_obstacles(self, ego_state: EgoState, detections: DetectionsTracks):
        self._static_objects = {}
        tokens, polygons = [], []

        for static_obstacle in detections.tracked_objects.get_static_objects():
            if (
                np.linalg.norm(static_obstacle.center.array - ego_state.center.array)
                > self.radius
            ):
                continue
            if len(self.drivable_area.intersects(static_obstacle.box.geometry)) > 0:
                tokens.append(static_obstacle.track_token)
                polygons.append(static_obstacle.box.geometry)
                self._static_objects[static_obstacle.track_token] = static_obstacle

        if len(tokens) == 0:
            polygons = np.array([], dtype=np.object_)

        return tokens, polygons

    def _get_dynamic_agents_from_prediction(
        self,
        agents_info: Dict[str, np.ndarray],
    ):
        tokens = agents_info["tokens"]
        shape = agents_info["shape"]
        category = agents_info["category"]
        velocity = agents_info["velocity"]
        predictions = agents_info["predictions"][:, :: self.interval]

        agents_vertices = compute_agents_vertices(
            center=predictions[..., :2], angle=predictions[..., 2], shape=shape
        )

        self._agents_info = {}

        for i in range(len(tokens)):
            self._agents_info[tokens[i]] = {
                "shape": shape[i],
                "velocity": velocity[i],
                "prediction": predictions[i],
            }

        return tokens, agents_vertices

    def _get_route_red_traffic_lights(
        self,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ):
        tokens, polygons = [], []

        for data in traffic_light_data:
            if data.status != TrafficLightStatusType.RED:
                continue
            lane_connector_id = str(data.lane_connector_id)
            if lane_connector_id in route_lane_dict.keys():
                lane_connector = route_lane_dict[lane_connector_id]
                tokens.append(f"{self.red_light_prefix}_{lane_connector_id}")
                polygons.append(lane_connector.polygon)

        if len(tokens) == 0:
            polygons = np.array([], dtype=np.object_)

        return tokens, polygons

    def get_object_at_frame(self, token, frame_idx):
        if token in self._static_object_tokens:
            return {
                "is_agent": False,
                "pose": self._static_objects[token].center,
                "velocity": np.zeros(2),
                "polygon": self.occupancy_map[frame_idx][token],
            }
        else:
            return {
                "is_agent": True,
                "pose": self._agents_info[token]["prediction"][frame_idx],
                "velocity": self._agents_info[token]["velocity"][frame_idx],
                "polygon": self.occupancy_map[frame_idx][token],
            }
