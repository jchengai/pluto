from typing import Dict, List, Set

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.patches import Polygon
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)

from src.scenario_manager.scenario_manager import ScenarioManager
from ..utils.vis import *

AGENT_COLOR_MAPPING = {
    TrackedObjectType.VEHICLE: "#001eff",
    TrackedObjectType.PEDESTRIAN: "#9500ff",
    TrackedObjectType.BICYCLE: "#ff0059",
}

TRAFFIC_LIGHT_COLOR_MAPPING = {
    TrafficLightStatusType.GREEN: "#2ca02c",
    TrafficLightStatusType.YELLOW: "#ff7f0e",
    TrafficLightStatusType.RED: "#d62728",
}


class NuplanScenarioRender:
    def __init__(
        self,
        future_horizon: float = 8,
        sample_interval: float = 0.1,
        bounds=60,
        offset=20,
        disable_agent=False,
    ) -> None:
        super().__init__()

        self.future_horizon = future_horizon
        self.future_samples = int(self.future_horizon / sample_interval)
        self.sample_interval = sample_interval
        self.ego_params = get_pacifica_parameters()
        self.length = self.ego_params.length
        self.width = self.ego_params.width
        self.bounds = bounds
        self.offset = offset
        self.disable_agent = disable_agent
        self.initialize = False
        self.scenario_manager = None
        self.need_update = False
        self.candidate_index = None
        self._history_trajectory = []
        self._expert_history_trajectory = []

        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        self.static_objects_types = [
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.BARRIER,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]
        self.road_elements = [
            # SemanticMapLayer.ROADBLOCK,
            # SemanticMapLayer.ROADBLOCK_CONNECTOR,
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            # SemanticMapLayer.CROSSWALKh
        ]

    def render_from_simulation(
        self,
        current_input: PlannerInput = None,
        initialization: PlannerInitialization = None,
        route_roadblock_ids: List[str] = None,
        scenario=None,
        iteration=None,
        planning_trajectory=None,
        candidate_trajectories=None,
        predictions=None,
        rollout_trajectories=None,
        agent_attn_weights=None,
        candidate_index=None,
        return_img=True,
    ):
        ego_state = current_input.history.ego_states[-1]
        map_api = initialization.map_api
        tracked_objects = current_input.history.observations[-1]
        traffic_light_status = current_input.traffic_light_data
        mission_goal = initialization.mission_goal
        if route_roadblock_ids is None:
            route_roadblock_ids = initialization.route_roadblock_ids

        self.candidate_index = candidate_index

        if scenario is not None:
            gt_state = scenario.get_ego_state_at_iteration(iteration)
            gt_trajectory = scenario.get_ego_future_trajectory(
                iteration=iteration,
                time_horizon=self.future_horizon,
                num_samples=self.future_samples,
            )
        else:
            gt_state, gt_trajectory = None, None

        return self.render(
            map_api=map_api,
            ego_state=ego_state,
            route_roadblock_ids=route_roadblock_ids,
            tracked_objects=tracked_objects,
            traffic_light_status=traffic_light_status,
            mission_goal=mission_goal,
            gt_state=gt_state,
            gt_trajectory=gt_trajectory,
            planning_trajectory=planning_trajectory,
            candidate_trajectories=candidate_trajectories,
            rollout_trajectories=rollout_trajectories,
            predictions=predictions,
            agent_attn_weights=agent_attn_weights,
            return_img=return_img,
        )

    def render_from_scenario(
        self,
        scenario: AbstractScenario,
        ego_state: EgoState = None,
        iteration=0,
        planning_trajectory=None,
        candidate_trajectories=None,
        rollout_trajectories=None,
        predictions=None,
        return_image=True,
    ):
        if ego_state is None:
            ego_state = scenario.get_ego_state_at_iteration(iteration)
        map_api = scenario.map_api
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        tracked_objects = scenario.get_tracked_objects_at_iteration(iteration)
        traffic_light_status = scenario.get_traffic_light_status_at_iteration(iteration)
        mission_goal = scenario.get_mission_goal()
        gt_state = scenario.get_ego_state_at_iteration(iteration)
        gt_trajectory = scenario.get_ego_future_trajectory(
            iteration=iteration,
            time_horizon=self.future_horizon,
            num_samples=self.future_samples,
        )

        return self.render(
            map_api=map_api,
            ego_state=ego_state,
            route_roadblock_ids=route_roadblock_ids,
            tracked_objects=tracked_objects,
            traffic_light_status=traffic_light_status,
            mission_goal=mission_goal,
            gt_state=gt_state,
            gt_trajectory=gt_trajectory,
            planning_trajectory=planning_trajectory,
            candidate_trajectories=candidate_trajectories,
            rollout_trajectories=rollout_trajectories,
            predictions=predictions,
            return_img=return_image,
        )

    def render(
        self,
        map_api: AbstractMap,
        ego_state: EgoState,
        route_roadblock_ids: List[str],
        tracked_objects: TrackedObjects,
        traffic_light_status: Dict[int, TrafficLightStatusData],
        mission_goal: StateSE2,
        gt_state=None,
        gt_trajectory=None,
        planning_trajectory=None,
        candidate_trajectories=None,
        rollout_trajectories=None,
        predictions=None,
        agent_attn_weights=None,
        return_img=False,
    ):
        fig, ax = plt.subplots(figsize=(10, 10))

        self._history_trajectory.append(ego_state.rear_axle.array)
        if gt_state is not None:
            self._expert_history_trajectory.append(gt_state.rear_axle.array)

        if self.scenario_manager is None:
            self.scenario_manager = ScenarioManager(
                map_api, ego_state, route_roadblock_ids
            )
            self.scenario_manager.get_route_roadblock_ids()
            self.need_update = True

        if self.need_update:
            self.scenario_manager.update_ego_state(ego_state)
            self.scenario_manager.update_drivable_area_map()
            self.scenario_manager.update_ego_path()

        self.origin = ego_state.rear_axle.array
        self.angle = ego_state.rear_axle.heading
        self.rot_mat = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ],
            dtype=np.float64,
        )

        self._plot_map(
            ax,
            map_api,
            ego_state.center.point,
            traffic_light_status,
            set(route_roadblock_ids),
        )

        self._plot_reference_lines(ax, self.scenario_manager.get_reference_lines())

        self._plot_ego(ax, ego_state)

        if gt_state is not None:
            self._plot_ego(ax, gt_state, gt=True)
            gt_trajectory = np.array([state.rear_axle.array for state in gt_trajectory])
            gt_trajectory = np.matmul(gt_trajectory - self.origin, self.rot_mat)
            ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], color="blue", alpha=0.5)

        if not self.disable_agent:
            for track in tracked_objects.tracked_objects:
                self._plot_tracked_object(ax, track, agent_attn_weights)

        if planning_trajectory is not None:
            self._plot_planning(ax, planning_trajectory)

        if candidate_trajectories is not None:
            self._plot_candidate_trajectories(ax, candidate_trajectories)

        if rollout_trajectories is not None:
            self._plot_rollout_trajectories(ax, rollout_trajectories)

        if predictions is not None:
            self._plot_prediction(ax, predictions)

        self._plot_mission_goal(ax, mission_goal)
        self._plot_history(ax)

        ax.axis("equal")
        ax.set_xlim(xmin=-self.bounds + self.offset, xmax=self.bounds + self.offset)
        ax.set_ylim(ymin=-self.bounds, ymax=self.bounds)
        ax.axis("off")
        plt.tight_layout(pad=0)

        if return_img:
            fig.canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                int(height), int(width), 3
            )
            plt.close(fig)
            return img
        else:
            plt.show()

    def _plot_map(
        self,
        ax,
        map_api: AbstractMap,
        query_point: Point2D,
        traffic_light_status: Dict[int, TrafficLightStatusData],
        route_roadblock_ids: Set[str],
    ):
        road_objects = map_api.get_proximal_map_objects(
            query_point, self.bounds + self.offset, self.road_elements
        )
        road_objects = (
            road_objects[SemanticMapLayer.LANE]
            + road_objects[SemanticMapLayer.LANE_CONNECTOR]
        )
        tls = {tl.lane_connector_id: tl.status for tl in traffic_light_status}

        for obj in road_objects:
            obj_id = int(obj.id)
            kwargs = {"color": "lightgray", "alpha": 0.4, "ec": None, "zorder": 0}
            if obj.get_roadblock_id() in route_roadblock_ids:
                kwargs["color"] = "dodgerblue"
                kwargs["alpha"] = 0.1
                kwargs["zorder"] = 1
            ax.add_artist(self._polygon_to_patch(obj.polygon, **kwargs))

            # for stopline in obj.stop_lines:
            #     if stopline.id in plotted_stopline:
            #         continue
            #     kwargs = {"color": "k", "alpha": 0.3, "ec": None, "zorder": 1}
            #     ax.add_artist(self._polygon_to_patch(stopline.polygon, **kwargs))
            #     plotted_stopline.add(stopline.id)

            cl_color, linewidth = "gray", 1.0
            if obj_id in tls:
                cl_color = TRAFFIC_LIGHT_COLOR_MAPPING.get(tls[obj_id], "gray")
                linewidth = 1
            cl = np.array([[s.x, s.y] for s in obj.baseline_path.discrete_path])
            cl = np.matmul(cl - self.origin, self.rot_mat)
            ax.plot(
                cl[:, 0],
                cl[:, 1],
                color=cl_color,
                alpha=0.5,
                linestyle="--",
                zorder=1,
                linewidth=linewidth,
            )

        crosswalks = map_api.get_proximal_map_objects(
            query_point, self.bounds + self.offset, [SemanticMapLayer.CROSSWALK]
        )
        for obj in crosswalks[SemanticMapLayer.CROSSWALK]:
            xys = np.array(obj.polygon.exterior.coords.xy).T
            xys = np.matmul(xys - self.origin, self.rot_mat)
            polygon = Polygon(
                xys, color="gray", alpha=0.4, ec=None, zorder=3, hatch="///"
            )
            ax.add_patch(polygon)

    def _plot_ego(self, ax, ego_state: EgoState, gt=False):
        kwargs = {"lw": 1.5}
        if gt:
            ax.add_patch(
                self._polygon_to_patch(
                    ego_state.car_footprint.geometry,
                    color="gray",
                    alpha=0.3,
                    zorder=9,
                    **kwargs,
                )
            )
        else:
            ax.add_patch(
                self._polygon_to_patch(
                    ego_state.car_footprint.geometry,
                    ec="#ff7f0e",
                    fill=False,
                    zorder=10,
                    **kwargs,
                )
            )

        ax.plot(
            [1.69, 1.69 + self.length * 0.75],
            [0, 0],
            color="#ff7f0e",
            linewidth=1.5,
            zorder=11,
        )

    def _plot_tracked_object(self, ax, track: TrackedObject, agent_attn_weights=None):
        center, angle = track.center.array, track.center.heading
        center = np.matmul(center - self.origin, self.rot_mat)
        angle = angle - self.angle

        direct = np.array([np.cos(angle), np.sin(angle)]) * track.box.length / 1.5
        direct = np.stack([center, center + direct], axis=0)

        color = AGENT_COLOR_MAPPING.get(track.tracked_object_type, "k")
        ax.add_patch(
            self._polygon_to_patch(
                track.box.geometry, ec=color, fill=False, alpha=1.0, zorder=4, lw=1.5
            )
        )

        if color != "k":
            ax.plot(direct[:, 0], direct[:, 1], color=color, linewidth=1, zorder=4)
        if agent_attn_weights is not None and track.track_token in agent_attn_weights:
            weight = agent_attn_weights[track.track_token]
            ax.text(
                center[0],
                center[1] + 0.5,
                f"{weight:.2f}",
                color="red",
                zorder=5,
                fontsize=7,
            )

    def _polygon_to_patch(self, polygon: shapely.geometry.Polygon, **kwargs):
        polygon = np.array(polygon.exterior.xy).T
        polygon = np.matmul(polygon - self.origin, self.rot_mat)
        return patches.Polygon(polygon, **kwargs)

    def _plot_planning(self, ax, planning_trajectory: np.ndarray):
        plot_polyline(
            ax,
            [planning_trajectory],
            linewidth=4,
            arrow=False,
            zorder=6,
            alpha=1.0,
            cmap="spring",
        )

    def _plot_candidate_trajectories(self, ax, candidate_trajectories: np.ndarray):
        for traj in candidate_trajectories:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color="gray",
                alpha=0.5,
                zorder=5,
                linewidth=2,
            )
            ax.scatter(traj[-1, 0], traj[-1, 1], color="gray", zorder=5, s=10)

    def _plot_rollout_trajectories(self, ax, candidate_trajectories: np.ndarray):
        for i, traj in enumerate(candidate_trajectories):
            kwargs = {"lw": 1.5, "zorder": 5, "color": "cyan"}
            if self.candidate_index is not None and i == self.candidate_index:
                kwargs = {"lw": 5, "zorder": 6, "color": "red"}
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, **kwargs)
            ax.scatter(traj[-1, 0], traj[-1, 1], color="cyan", zorder=5, s=10)

    def _plot_prediction(self, ax, predictions: np.ndarray):
        kwargs = {"lw": 3}
        for pred in predictions:
            pred = pred[:40, ..., :2]
            self._plot_polyline(ax, pred, cmap="winter", **kwargs)

    def _plot_polyline(self, ax, polyline, cmap="spring", **kwargs) -> None:
        arc = get_polyline_arc_length(polyline)
        polyline = polyline.reshape(-1, 1, 2)
        segment = np.concatenate([polyline[:-1], polyline[1:]], axis=1)
        norm = plt.Normalize(arc.min(), arc.max())
        lc = LineCollection(
            segment,
            cmap=cmap,
            norm=norm,
            array=arc,
            **kwargs,
        )
        ax.add_collection(lc)

    def _plot_reference_lines(self, ax, ref_lines):
        for ref_line in ref_lines:
            ref_line_pos = np.matmul(ref_line[::20, :2] - self.origin, self.rot_mat)
            ref_line_angle = ref_line[::20, 2] - self.angle
            for p, angle in zip(ref_line_pos, ref_line_angle):
                ax.arrow(
                    p[0],
                    p[1],
                    np.cos(angle) * 1.5,
                    np.sin(angle) * 1.5,
                    color="magenta",
                    width=0.2,
                    head_width=0.8,
                    zorder=6,
                    alpha=0.2,
                )

    def _plot_mission_goal(self, ax, mission_goal: StateSE2):
        point = np.matmul(mission_goal.point.array - self.origin, self.rot_mat)
        ax.plot(point[0], point[1], marker="*", markersize=5, color="gold", zorder=6)

    def _plot_history(self, ax):
        history = np.array(self._history_trajectory)
        history = np.matmul(history - self.origin, self.rot_mat)
        ax.plot(
            history[:, 0],
            history[:, 1],
            color="#ff7f0e",
            alpha=0.5,
            zorder=6,
            linewidth=2,
        )

        if len(self._expert_history_trajectory) > 0:
            expert_history = np.array(self._expert_history_trajectory)
            expert_history = np.matmul(expert_history - self.origin, self.rot_mat)
            ax.plot(
                expert_history[:, 0],
                expert_history[:, 1],
                color="blue",
                alpha=0.5,
                zorder=6,
            )
