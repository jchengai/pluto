#   Heavily borrowed from:
#   https://github.com/autonomousvision/tuplan_garage (Apache License 2.0)
# & https://github.com/motional/nuplan-devkit (Apache License 2.0)

from enum import Enum
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import shapely.vectorized
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import Geometry
from shapely.strtree import STRtree


class OccupancyType(Enum):
    DYNAMIC = 0, "dynamic"
    STATIC = 1, "static"
    RED_LIGHT = 2, "red_light"


class OccupancyMap:
    def __init__(
        self,
        tokens: List[str],
        geometries: npt.NDArray[np.object_],
        types: List[Enum] = None,
        node_capacity: int = 10,
        attribute: Dict[str, Any] = None,
    ):
        self._tokens: List[str] = tokens
        self._types: List[Enum] = types
        self._token_to_idx: Dict[str, int] = {
            token: idx for idx, token in enumerate(tokens)
        }

        self._geometries = geometries
        self._attribute = attribute
        self._node_capacity = node_capacity
        self._str_tree = STRtree(self._geometries, node_capacity)

    def __getitem__(self, token) -> Geometry:
        """
        Retrieves geometry of token.
        :param token: geometry identifier
        :return: Geometry of token
        """
        return self._geometries[self._token_to_idx[token]]

    def __len__(self) -> int:
        """
        Number of geometries in the occupancy map
        :return: int
        """
        return len(self._tokens)

    def get_type(self, token: str) -> Enum:
        """
        Retrieves type of token.
        :param token: geometry identifier
        :return: type of token
        """
        return self._types[self._token_to_idx[token]]

    @property
    def tokens(self) -> List[str]:
        """
        Getter for track tokens in occupancy map
        :return: list of strings
        """
        return self._tokens

    @property
    def token_to_idx(self) -> Dict[str, int]:
        """
        Getter for track tokens in occupancy map
        :return: dictionary of tokens and indices
        """
        return self._token_to_idx

    def intersects(self, geometry: Geometry) -> List[str]:
        """
        Searches for intersecting geometries in the occupancy map
        :param geometry: geometries to query
        :return: list of tokens for intersecting geometries
        """
        indices = self.query(geometry, predicate="intersects")
        return [self._tokens[idx] for idx in indices]

    def get_subset_by_intersection(self, geometry: Geometry) -> "OccupancyMap":
        indices = self.query(geometry, predicate="intersects")
        polygons = [self._geometries[i] for i in indices]

        return OccupancyMap(self._tokens, polygons)

    def get_subset_by_type(self, type):
        assert self._types is not None, "OccupancyMap: No types defined!"
        indices = [i for i, t in enumerate(self._types) if t == type]
        polygons = [self._geometries[i] for i in indices]

        return OccupancyMap(self._tokens, polygons)

    def query(self, geometry: Geometry, predicate=None):
        """
        Function to directly calls shapely's query function on str-tree
        :param geometry: geometries to query
        :param predicate: see shapely, defaults to None
        :return: query output
        """
        return self._str_tree.query(geometry, predicate=predicate)

    def points_in_polygons(
        self, points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """
        Determines wether input-points are in polygons of the occupancy map
        :param points: input-points
        :return: boolean array of shape (polygons, input-points)
        """
        output = np.zeros((len(self._geometries), len(points)), dtype=bool)
        for i, polygon in enumerate(self._geometries):
            output[i] = shapely.vectorized.contains(polygon, points[:, 0], points[:, 1])

        return output

    def points_in_polygons_with_attribute(
        self, points: npt.NDArray[np.float64], attribute_name: str
    ):
        """
        Determines wether input-points are in polygons of the occupancy map
        :param points: input-points
        :return: boolean array of shape (polygons, input-points)
        """
        output = np.zeros((len(self._geometries), len(points)), dtype=bool)
        attribute = np.zeros((len(self._geometries), len(points)))
        for i, polygon in enumerate(self._geometries):
            output[i] = shapely.vectorized.contains(polygon, points[:, 0], points[:, 1])
            attribute[i] = self._attribute[attribute_name][i]

        return output, attribute
