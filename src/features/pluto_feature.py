from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from torch.nn.utils.rnn import pad_sequence

from src.utils.utils import to_device, to_numpy, to_tensor


@dataclass
class PlutoFeature(AbstractModelFeature):
    data: Dict[str, Any]  # anchor sample
    data_p: Dict[str, Any] = None  # positive sample
    data_n: Dict[str, Any] = None  # negative sample
    data_n_info: Dict[str, Any] = None  # negative sample info

    @classmethod
    def collate(cls, feature_list: List[PlutoFeature]) -> PlutoFeature:
        batch_data = {}

        pad_keys = ["agent", "map"]
        stack_keys = ["current_state", "origin", "angle"]

        if "reference_line" in feature_list[0].data:
            pad_keys.append("reference_line")
        if "static_objects" in feature_list[0].data:
            pad_keys.append("static_objects")
        if "cost_maps" in feature_list[0].data:
            stack_keys.append("cost_maps")

        if feature_list[0].data_n is not None:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list]
                        + [f.data_p[key][k] for f in feature_list]
                        + [f.data_n[key][k] for f in feature_list],
                        batch_first=True,
                    )
                    for k in feature_list[0].data[key].keys()
                }

            batch_data["data_n_valid_mask"] = torch.Tensor(
                [f.data_n_info["valid_mask"] for f in feature_list]
            ).bool()
            batch_data["data_n_type"] = torch.Tensor(
                [f.data_n_info["type"] for f in feature_list]
            ).long()

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list]
                    + [f.data_p[key] for f in feature_list]
                    + [f.data_n[key] for f in feature_list],
                    dim=0,
                )
        elif feature_list[0].data_p is not None:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list]
                        + [f.data_p[key][k] for f in feature_list],
                        batch_first=True,
                    )
                    for k in feature_list[0].data[key].keys()
                }

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list]
                    + [f.data_p[key] for f in feature_list],
                    dim=0,
                )
        else:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list], batch_first=True
                    )
                    for k in feature_list[0].data[key].keys()
                }

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list], dim=0
                )

        return PlutoFeature(data=batch_data)

    def to_feature_tensor(self) -> PlutoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_tensor(v)

        if self.data_p is not None:
            new_data_p = {}
            for k, v in self.data_p.items():
                new_data_p[k] = to_tensor(v)
        else:
            new_data_p = None

        if self.data_n is not None:
            new_data_n = {}
            new_data_n_info = {}
            for k, v in self.data_n.items():
                new_data_n[k] = to_tensor(v)
            for k, v in self.data_n_info.items():
                new_data_n_info[k] = to_tensor(v)
        else:
            new_data_n = None
            new_data_n_info = None

        return PlutoFeature(
            data=new_data,
            data_p=new_data_p,
            data_n=new_data_n,
            data_n_info=new_data_n_info,
        )

    def to_numpy(self) -> PlutoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_numpy(v)
        if self.data_p is not None:
            new_data_p = {}
            for k, v in self.data_p.items():
                new_data_p[k] = to_numpy(v)
        else:
            new_data_p = None
        if self.data_n is not None:
            new_data_n = {}
            for k, v in self.data_n.items():
                new_data_n[k] = to_numpy(v)
        else:
            new_data_n = None
        return PlutoFeature(data=new_data, data_p=new_data_p, data_n=new_data_n)

    def to_device(self, device: torch.device) -> PlutoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_device(v, device)
        return PlutoFeature(data=new_data)

    def serialize(self) -> Dict[str, Any]:
        return {"data": self.data}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PlutoFeature:
        return PlutoFeature(data=data["data"])

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError

    @property
    def is_valid(self) -> bool:
        if "reference_line" in self.data:
            return self.data["reference_line"]["valid_mask"].any()
        else:
            return self.data["map"]["point_position"].shape[0] > 0

    @classmethod
    def normalize(
        self, data, first_time=False, radius=None, hist_steps=21
    ) -> PlutoFeature:
        cur_state = data["current_state"]
        center_xy, center_angle = cur_state[:2].copy(), cur_state[2].copy()

        rotate_mat = np.array(
            [
                [np.cos(center_angle), -np.sin(center_angle)],
                [np.sin(center_angle), np.cos(center_angle)],
            ],
            dtype=np.float64,
        )

        data["current_state"][:3] = 0
        data["agent"]["position"] = np.matmul(
            data["agent"]["position"] - center_xy, rotate_mat
        )
        data["agent"]["velocity"] = np.matmul(data["agent"]["velocity"], rotate_mat)
        data["agent"]["heading"] -= center_angle

        data["map"]["point_position"] = np.matmul(
            data["map"]["point_position"] - center_xy, rotate_mat
        )
        data["map"]["point_vector"] = np.matmul(data["map"]["point_vector"], rotate_mat)
        data["map"]["point_orientation"] -= center_angle

        data["map"]["polygon_center"][..., :2] = np.matmul(
            data["map"]["polygon_center"][..., :2] - center_xy, rotate_mat
        )
        data["map"]["polygon_center"][..., 2] -= center_angle
        data["map"]["polygon_position"] = np.matmul(
            data["map"]["polygon_position"] - center_xy, rotate_mat
        )
        data["map"]["polygon_orientation"] -= center_angle

        if "causal" in data:
            if len(data["causal"]["free_path_points"]) > 0:
                data["causal"]["free_path_points"][..., :2] = np.matmul(
                    data["causal"]["free_path_points"][..., :2] - center_xy, rotate_mat
                )
                data["causal"]["free_path_points"][..., 2] -= center_angle
        if "static_objects" in data:
            data["static_objects"]["position"] = np.matmul(
                data["static_objects"]["position"] - center_xy, rotate_mat
            )
            data["static_objects"]["heading"] -= center_angle
        if "route" in data:
            data["route"]["position"] = np.matmul(
                data["route"]["position"] - center_xy, rotate_mat
            )
        if "reference_line" in data:
            data["reference_line"]["position"] = np.matmul(
                data["reference_line"]["position"] - center_xy, rotate_mat
            )
            data["reference_line"]["vector"] = np.matmul(
                data["reference_line"]["vector"], rotate_mat
            )
            data["reference_line"]["orientation"] -= center_angle

        target_position = (
            data["agent"]["position"][:, hist_steps:]
            - data["agent"]["position"][:, hist_steps - 1][:, None]
        )
        target_heading = (
            data["agent"]["heading"][:, hist_steps:]
            - data["agent"]["heading"][:, hist_steps - 1][:, None]
        )
        target = np.concatenate([target_position, target_heading[..., None]], -1)
        target[~data["agent"]["valid_mask"][:, hist_steps:]] = 0
        data["agent"]["target"] = target

        if first_time:
            point_position = data["map"]["point_position"]
            x_max, x_min = radius, -radius
            y_max, y_min = radius, -radius
            valid_mask = (
                (point_position[:, 0, :, 0] < x_max)
                & (point_position[:, 0, :, 0] > x_min)
                & (point_position[:, 0, :, 1] < y_max)
                & (point_position[:, 0, :, 1] > y_min)
            )
            valid_polygon = valid_mask.any(-1)
            data["map"]["valid_mask"] = valid_mask

            for k, v in data["map"].items():
                data["map"][k] = v[valid_polygon]

            if "causal" in data:
                data["causal"]["ego_care_red_light_mask"] = data["causal"][
                    "ego_care_red_light_mask"
                ][valid_polygon]

            data["origin"] = center_xy
            data["angle"] = center_angle

        return PlutoFeature(data=data)
