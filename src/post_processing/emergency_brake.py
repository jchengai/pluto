#   Heavily borrowed from:
#   https://github.com/autonomousvision/tuplan_garage (Apache License 2.0)
# & https://github.com/motional/nuplan-devkit (Apache License 2.0)

from typing import Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .forward_simulation.forward_simulator import ForwardSimulator
from .common.enum import StateIndex


class EmergencyBrake:
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            num_poses=80, interval_length=0.1
        ),
        time_to_infraction_threshold: float = 2.0,
        max_ego_speed: float = 8.0,
        max_long_accel: float = 2.40,
        min_long_accel: float = -2.40,
        emergency_decel: float = -4.05,
    ):
        # trajectory parameters
        self._trajectory_sampling = trajectory_sampling

        # braking parameters
        self._max_ego_speed: float = max_ego_speed  # [m/s]
        self._max_long_accel: float = max_long_accel  # [m/s^2]
        self._min_long_accel: float = min_long_accel  # [m/s^2]
        self._emergency_decel: float = emergency_decel  # [m/s^2]

        # braking condition parameters
        self._time_to_infraction_threshold: float = time_to_infraction_threshold

    def brake_if_emergency(
        self,
        ego_state: EgoState,
        time_to_at_fault_collision: float,
        ego_trajectory: npt.NDArray[np.float64],
    ) -> Optional[InterpolatedTrajectory]:
        trajectory = None
        ego_speed: float = ego_state.dynamic_car_state.speed

        time_to_infraction = time_to_at_fault_collision
        min_brake_time = max(ego_speed / abs(self._min_long_accel) + 0.5, 3.0)

        if time_to_infraction <= min_brake_time and ego_speed <= self._max_ego_speed:
            print("Emergency Brake")
            min_reaction_time = ego_speed / abs(self._emergency_decel) + 0.5
            is_soft_brake_possible = time_to_infraction > min_reaction_time
            trajectory = self._generate_ebrake_trajectory(
                ego_trajectory, ego_state, soft_brake=is_soft_brake_possible
            )

        return trajectory

    def _generate_ebrake_trajectory(
        self, origin_trajectory: np.ndarray, ego_state: EgoState, soft_brake=False
    ):
        simulator = ForwardSimulator(
            dt=self._trajectory_sampling.interval_length,
            num_frames=self._trajectory_sampling.num_poses,
            estop=True,
            soft_brake=soft_brake,
        )
        rollout = simulator.forward(origin_trajectory[None, ...], ego_state)[0]

        ego_states, current_time_point = [], ego_state.time_point
        delta_t = TimePoint(int(self._trajectory_sampling.interval_length * 1e6))

        for state in rollout:
            ego_states.append(
                EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2(*state[:3]),
                    rear_axle_velocity_2d=StateVector2D(*state[StateIndex.VELOCITY_2D]),
                    rear_axle_acceleration_2d=StateVector2D(
                        *state[StateIndex.ACCELERATION_2D]
                    ),
                    tire_steering_angle=state[StateIndex.STEERING_ANGLE],
                    time_point=current_time_point,
                    vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
                )
            )
            current_time_point += delta_t

        return InterpolatedTrajectory(ego_states)
