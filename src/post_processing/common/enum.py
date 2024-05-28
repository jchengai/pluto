#   Heavily borrowed from:
#   https://github.com/autonomousvision/tuplan_garage (Apache License 2.0)
# & https://github.com/motional/nuplan-devkit (Apache License 2.0)

from enum import IntEnum

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState


class StateIndex:
    _X = 0
    _Y = 1
    _HEADING = 2
    _VELOCITY_X = 3
    _VELOCITY_Y = 4
    _ACCELERATION_X = 5
    _ACCELERATION_Y = 6
    _STEERING_ANGLE = 7
    _STEERING_RATE = 8
    _ANGULAR_VELOCITY = 9
    _ANGULAR_ACCELERATION = 10

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
            and not attribute.startswith("__")
            and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def VELOCITY_X(cls):
        return cls._VELOCITY_X

    @classmethod
    @property
    def VELOCITY_Y(cls):
        return cls._VELOCITY_Y

    @classmethod
    @property
    def ACCELERATION_X(cls):
        return cls._ACCELERATION_X

    @classmethod
    @property
    def ACCELERATION_Y(cls):
        return cls._ACCELERATION_Y

    @classmethod
    @property
    def STEERING_ANGLE(cls):
        return cls._STEERING_ANGLE

    @classmethod
    @property
    def STEERING_RATE(cls):
        return cls._STEERING_RATE

    @classmethod
    @property
    def ANGULAR_VELOCITY(cls):
        return cls._ANGULAR_VELOCITY

    @classmethod
    @property
    def ANGULAR_ACCELERATION(cls):
        return cls._ANGULAR_ACCELERATION

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)

    @classmethod
    @property
    def VELOCITY_2D(cls):
        # assumes velocity X, Y have subsequent indices
        return slice(cls._VELOCITY_X, cls._VELOCITY_Y + 1)

    @classmethod
    @property
    def ACCELERATION_2D(cls):
        # assumes acceleration X, Y have subsequent indices
        return slice(cls._ACCELERATION_X, cls._ACCELERATION_Y + 1)


class DynamicStateIndex(IntEnum):
    ACCELERATION_X = 0
    STEERING_RATE = 1


class BBCoordsIndex(IntEnum):
    FRONT_LEFT = 0
    REAR_LEFT = 1
    REAR_RIGHT = 2
    FRONT_RIGHT = 3
    CENTER = 4


class EgoAreaIndex(IntEnum):
    MULTIPLE_LANES = 0
    NON_DRIVABLE_AREA = 1
    ONCOMING_TRAFFIC = 2


class MultiMetricIndex(IntEnum):
    NO_COLLISION = 0
    DRIVABLE_AREA = 1
    DRIVING_DIRECTION = 2


class WeightedMetricIndex(IntEnum):
    PROGRESS = 0
    SPEED_LIMIT = 1
    COMFORTABLE = 2
    TTC = 3


class CollisionType(IntEnum):
    """Enum for the types of collisions of interest."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


def ego_state_to_state_array(ego_state: EgoState):
    """
    Converts an ego state into an array representation (drops time-stamps and vehicle parameters)
    :param ego_state: ego state class
    :return: array containing ego state values
    """
    state_array = np.zeros(StateIndex.size(), dtype=np.float64)

    state_array[StateIndex.STATE_SE2] = ego_state.rear_axle.serialize()
    state_array[
        StateIndex.VELOCITY_2D
    ] = ego_state.dynamic_car_state.rear_axle_velocity_2d.array
    state_array[
        StateIndex.ACCELERATION_2D
    ] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.array

    state_array[StateIndex.STEERING_ANGLE] = ego_state.tire_steering_angle
    state_array[
        StateIndex.STEERING_RATE
    ] = ego_state.dynamic_car_state.tire_steering_rate

    state_array[
        StateIndex.ANGULAR_VELOCITY
    ] = ego_state.dynamic_car_state.angular_velocity
    state_array[
        StateIndex.ANGULAR_ACCELERATION
    ] = ego_state.dynamic_car_state.angular_acceleration

    return state_array
