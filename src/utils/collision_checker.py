import torch
from nuplan.common.actor_state.vehicle_parameters import (
    VehicleParameters,
    get_pacifica_parameters,
)


class CollisionChecker:
    def __init__(
        self,
        vehicle: VehicleParameters = get_pacifica_parameters(),
    ) -> None:
        self._vehicle = vehicle
        self._sdc_half_length = vehicle.length / 2
        self._sdc_half_width = vehicle.width / 2

        self._sdc_normalized_corners = torch.stack(
            [
                torch.tensor([vehicle.length / 2, vehicle.width / 2]),
                torch.tensor([vehicle.length / 2, -vehicle.width / 2]),
                torch.tensor([-vehicle.length / 2, -vehicle.width / 2]),
                torch.tensor([-vehicle.length / 2, vehicle.width / 2]),
            ],
            dim=0,
        )

    def to_device(self, device):
        self._sdc_normalized_corners = self._sdc_normalized_corners.to(device)

    def build_bbox_from_center(self, center, heading, width, length):
        """
        params:
            center: [bs, N, (x, y)]
            heading: [bs, N]
            width: [bs, N]
            length: [bs, N]
        return:
            corners: [bs, 4, (x, y)]
            heading_vec, tanh_vec: [bs, 2]
        """
        cos = torch.cos(heading)
        sin = torch.sin(heading)

        heading_vec = torch.stack([cos, sin], dim=-1) * length.unsqueeze(-1) / 2
        tanh_vec = torch.stack([-sin, cos], dim=-1) * width.unsqueeze(-1) / 2

        corners = torch.stack(
            [
                center + heading_vec + tanh_vec,
                center - heading_vec + tanh_vec,
                center - heading_vec - tanh_vec,
                center + heading_vec - tanh_vec,
            ],
            dim=-2,
        )

        return corners, heading_vec, tanh_vec

    def collision_check(self, ego_state, objects, objects_width, objects_length):
        """performing batch-wise collision check using Separating Axis Theorem
        params:
            ego_states: [bs, (x, y, theta)], center of the ego
            objects: [bs, N, (x, y, theta)], center of the objects
        returns:
            is_collided: [bs, N]
        """

        bs, N = objects.shape[:2]

        # rotate object to ego's local frame
        cos, sin = torch.cos(ego_state[:, 2]), torch.sin(ego_state[:, 2])
        rotate_mat = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(bs, 2, 2)

        rotated_objects = objects.clone()
        rotated_objects[..., :2] = torch.matmul(
            rotated_objects[..., :2] - ego_state[:, :2].unsqueeze(1), rotate_mat
        )
        rotated_objects[..., 2] -= ego_state[..., 2].unsqueeze(1)

        # [bs, N, 4, 2], [bs, N, 2], [bs, N, 2]
        object_corners, axis1, axis2 = self.build_bbox_from_center(
            rotated_objects[..., :2],
            rotated_objects[..., 2],
            objects_width,
            objects_length,
        )

        ego_corners = self._sdc_normalized_corners.reshape(1, 1, 4, 2).repeat(
            bs, N, 1, 1
        )  # [bs, N, 4, 2]

        all_corners = torch.concat(
            [object_corners, ego_corners], dim=-2
        )  # [bs, N, 8, 2]

        x_projection = object_corners[..., 0]
        y_projection = object_corners[..., 1]
        axis1_projection = torch.matmul(all_corners, axis1.unsqueeze(-1)).squeeze(-1)
        axis2_projection = torch.matmul(all_corners, axis2.unsqueeze(-1)).squeeze(-1)

        x_separated = (x_projection.max(-1)[0] < -self._sdc_half_length) | (
            x_projection.min(-1)[0] > self._sdc_half_length
        )
        y_separated = (y_projection.max(-1)[0] < -self._sdc_half_width) | (
            y_projection.min(-1)[0] > self._sdc_half_width
        )
        axis1_separated = (
            axis1_projection[..., :4].max(-1)[0] < axis1_projection[..., 4:].min(-1)[0]
        ) | (
            axis1_projection[..., :4].min(-1)[0] > axis1_projection[..., 4:].max(-1)[0]
        )
        axis2_separated = (
            axis2_projection[..., :4].max(-1)[0] < axis2_projection[..., 4:].min(-1)[0]
        ) | (
            axis2_projection[..., :4].min(-1)[0] > axis2_projection[..., 4:].max(-1)[0]
        )

        collision = ~(x_separated | y_separated | axis1_separated | axis2_separated)

        return collision
