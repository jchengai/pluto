import warnings

from typing import List

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.simulation.observation.idm.utils import (
    create_path_from_se2,
    path_to_linestring,
)
from nuplan.planning.simulation.path.utils import trim_path
from shapely.geometry import Point, Polygon
from shapely.geometry.base import CAP_STYLE
from shapely.ops import unary_union

from .occupancy_map import OccupancyMap, OccupancyType
from .route_manager import RouteManager

DRIVABLE_LAYERS = {
    SemanticMapLayer.ROADBLOCK,
    SemanticMapLayer.ROADBLOCK_CONNECTOR,
    SemanticMapLayer.CARPARK_AREA,
}

STATIC_OBJECT_TYPES = {
    TrackedObjectType.CZONE_SIGN,
    TrackedObjectType.BARRIER,
    TrackedObjectType.TRAFFIC_CONE,
    TrackedObjectType.GENERIC_OBJECT,
}

DYNAMIC_OBJECT_TYPES = {
    TrackedObjectType.VEHICLE,
    TrackedObjectType.BICYCLE,
    TrackedObjectType.PEDESTRIAN,
}


class ScenarioManager:
    def __init__(
        self,
        map_api,
        ego_state: EgoState,
        route_roadblocks_ids: List[str],
        radius=50,
    ) -> None:
        self._map_api = map_api
        self._radius = radius
        self._drivable_area_map: OccupancyMap = None  # [lanes, lane connectors]
        self._obstacle_map: OccupancyMap = None  # [agents, red lights, etc]

        self._ego_state = ego_state
        self._ego_path = None
        self._ego_path_linestring = None
        self._ego_trimmed_path = None
        self._ego_progress = None

        self._route_manager = RouteManager(map_api, route_roadblocks_ids, radius)

    @property
    def drivable_area_map(self):
        return self._drivable_area_map

    def get_route_roadblock_ids(self, process=True) -> List[str]:
        if not self._route_manager.initialized:
            self._route_manager.load_route(self._ego_state, process)
        return self._route_manager.route_roadblock_ids

    def get_route_lane_dicts(self):
        assert self._route_manager.initialized
        return self._route_manager._route_lane_dict

    def update_ego_state(self, ego_state: EgoState):
        self._ego_state = ego_state

    def update_drivable_area_map(self):
        """
        Builds occupancy map of drivable area.
        :param ego_state: EgoState
        """

        position: Point2D = self._ego_state.center.point
        drivable_area = self._map_api.get_proximal_map_objects(
            position, self._radius, DRIVABLE_LAYERS
        )

        drivable_polygons: List[Polygon] = []
        drivable_polygon_ids: List[str] = []
        lane_speed_limit = []

        for road in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
            for roadblock in drivable_area[road]:
                for lane in roadblock.interior_edges:
                    drivable_polygons.append(lane.polygon)
                    drivable_polygon_ids.append(lane.id)
                    speed_limit = lane.speed_limit_mps if lane.speed_limit_mps else 50
                    lane_speed_limit.append(speed_limit)

        for carpark in drivable_area[SemanticMapLayer.CARPARK_AREA]:
            drivable_polygons.append(carpark.polygon)
            drivable_polygon_ids.append(carpark.id)
            speed_limit = lane.speed_limit_mps if lane.speed_limit_mps else 50
            lane_speed_limit.append(speed_limit)

        self._drivable_area_map = OccupancyMap(
            drivable_polygon_ids,
            drivable_polygons,
            attribute={"speed_limit": lane_speed_limit},
        )
        self._route_manager.update_drivable_area_map(self._drivable_area_map)

    def update_obstacle_map(
        self,
        detections: TrackedObjects,
        traffic_light_status: List[TrafficLightStatusData],
    ):
        """
        Builds occupancy map of obstacles.
        :param ego_state: EgoState
        """
        tokens = []
        types = []
        polygons = []

        for obj in detections.tracked_objects:
            if (
                np.linalg.norm(self._ego_state.center.array - obj.center.array)
                < self._radius
            ):
                obj_type = (
                    OccupancyType.DYNAMIC
                    if obj.tracked_object_type in DYNAMIC_OBJECT_TYPES
                    else OccupancyType.STATIC
                )
                tokens.append(obj.track_token)
                types.append(obj_type)
                polygons.append(obj.box.geometry)

        for data in traffic_light_status:
            if (
                data.status == TrafficLightStatusType.RED
                and str(data.lane_connector_id) in self._route_manager.route_lane_ids
            ):
                tokens.append(data.lane_connector_id)
                types.append(OccupancyType.RED_LIGHT)
                polygons.append(
                    self._map_api.get_map_object(
                        str(data.lane_connector_id), SemanticMapLayer.LANE_CONNECTOR
                    ).polygon
                )

        self._obstacle_map = OccupancyMap(tokens, polygons, types)

    def update_ego_path(self, length=50):
        ego_path: List[StateSE2] = self._route_manager.get_ego_path(self._ego_state)
        self._ego_path = create_path_from_se2(ego_path)
        self._ego_path_linestring = path_to_linestring(ego_path)

        start_progress = self._ego_path.get_start_progress()
        end_progress = self._ego_path.get_end_progress()

        with warnings.catch_warnings():
            # https://github.com/shapely/shapely/issues/1796
            warnings.simplefilter("ignore")
            self._ego_progress = self._ego_path_linestring.project(
                Point([self._ego_state.center.x, self._ego_state.center.y])
            )

        trimmed_path = trim_path(
            self._ego_path,
            max(start_progress, min(self._ego_progress, end_progress)),
            min(self._ego_progress + length, end_progress),
        )
        self._ego_trimmed_path = trimmed_path

        np_path = np.array([p.point.array for p in trimmed_path])
        return np_path

    def get_leading_objects(self):
        expanded_ego_path = path_to_linestring(self._ego_trimmed_path).buffer(
            self._ego_state.car_footprint.width / 2, cap_style=CAP_STYLE.square
        )
        expanded_ego_path = unary_union(
            [expanded_ego_path, self._ego_state.car_footprint.geometry]
        )
        intersecting_objects = self._obstacle_map.intersects(expanded_ego_path)

        if len(intersecting_objects) == 0:
            return []

        leading_objects = []
        ego_polygon = self._ego_state.car_footprint.geometry

        for obj_token in intersecting_objects:
            leading_objects.append(
                (
                    obj_token,
                    self._obstacle_map.get_type(obj_token),
                    self._obstacle_map[obj_token].distance(ego_polygon),
                )
            )

        self.leading_objects = sorted(leading_objects, key=lambda x: x[2])

        return self.leading_objects

    def get_occupancy_object(self, token: str):
        return self._obstacle_map[token]

    def get_ego_path_points(self, start_progress, end_progress):
        start_progress += self._ego_progress
        end_progress += self._ego_progress
        return np.array(
            [
                np.array([p.x, p.y, p.heading], dtype=np.float64)
                for p in self._ego_trimmed_path
                if p.progress >= start_progress and p.progress <= end_progress
            ]
        )

    def get_reference_lines(self, length=100):
        return self._route_manager.get_reference_lines(self._ego_state, length=length)

    def get_cached_reference_lines(self):
        if self._route_manager.reference_lines:
            return self._route_manager.reference_lines
        else:
            raise ValueError("Reference lines not cached")

    def object_in_drivable_area(self, polygon: Polygon):
        return len(self._drivable_area_map.intersects(polygon)) > 0
