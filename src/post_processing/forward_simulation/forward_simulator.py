import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)

from .batch_kinematic_bicycle import BatchKinematicBicycleModel
from .batch_lqr import BatchLQRTracker
from ..common.enum import StateIndex, ego_state_to_state_array


class ForwardSimulator:
    def __init__(
        self,
        dt: float = 0.1,
        num_frames: int = 40,
        estop: bool = False,
        soft_brake: bool = False,
    ) -> None:
        self.dt = dt
        self.interval = int(dt * 10)
        self.num_frames = num_frames
        self.motion_model = BatchKinematicBicycleModel()
        self.tracker = BatchLQRTracker(
            discretization_time=dt,
            tracking_horizon=int(1 / dt),
            estop=estop,
            soft_brake=soft_brake,
        )

    def forward(self, candidate_trajectories: np.ndarray, init_ego_state: EgoState):
        """
        Args:
            candidate_trajectories: (N, 80+1, S), sampled at 10 Hz
        """
        N = candidate_trajectories.shape[0]
        rollout_states = np.zeros(
            (N, self.num_frames + 1, StateIndex.size()), dtype=np.float64
        )
        rollout_states[:, 0] = ego_state_to_state_array(init_ego_state)

        t_now = init_ego_state.time_point
        delta_t = TimeDuration.from_s(self.dt)

        current_iteration = SimulationIteration(t_now, 0)
        next_iteration = SimulationIteration(t_now + delta_t, 1)

        self.tracker.update(candidate_trajectories[:, :: self.interval])

        for t in range(1, self.num_frames + 1):
            sampling_time: TimePoint = (
                next_iteration.time_point - current_iteration.time_point
            )
            command_states = self.tracker.track_trajectory(
                current_iteration,
                next_iteration,
                rollout_states[:, t - 1],
            )

            rollout_states[:, t] = self.motion_model.propagate_state(
                states=rollout_states[:, t - 1],
                command_states=command_states,
                sampling_time=sampling_time,
            )

            current_iteration = next_iteration
            next_iteration = SimulationIteration(
                current_iteration.time_point + delta_t, 1 + t
            )

        return rollout_states
