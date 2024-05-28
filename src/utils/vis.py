import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Polygon, Rectangle


def get_polyline_arc_length(xy: np.ndarray) -> np.ndarray:
    """Get the arc length of each point in a polyline"""
    diff = xy[1:] - xy[:-1]
    displacement = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    arc_length = np.cumsum(displacement)
    return np.concatenate((np.zeros(1), arc_length), axis=0)


def interpolate_centerline(xy: np.ndarray, n_points: int):
    arc_length = get_polyline_arc_length(xy)
    steps = np.linspace(0, arc_length[-1], n_points)
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location: np.ndarray,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
    alpha=1.0,
    label=None,
    zorder=50,
    fill=True,
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        xy=(pivot_x, pivot_y),
        width=bbox_length,
        height=bbox_width,
        angle=np.degrees(heading),
        fc=color if fill else "none",
        ec=color,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(vehicle_bounding_box)

    if bbox_length > 1.0:
        direction = (
            0.25 * bbox_size[0] * np.array([math.cos(heading), math.sin(heading)])
        )
        ax.arrow(
            cur_location[0],
            cur_location[1],
            direction[0],
            direction[1],
            color="white",
            zorder=zorder + 1,
            head_width=0.5,
        )


def plot_box(
    ax: plt.Axes,
    cur_location: np.ndarray,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
    alpha=1.0,
    label=None,
    zorder=50,
    fill=True,
    **kwargs,
) -> None:
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        xy=(pivot_x, pivot_y),
        width=bbox_length,
        height=bbox_width,
        angle=np.degrees(heading),
        fc=color if fill else "none",
        ec="dimgrey",
        alpha=alpha,
        label=label,
        zorder=zorder,
        **kwargs,
    )
    ax.add_patch(vehicle_bounding_box)


def plot_polygon(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    alpha=1.0,
    zorder=50,
    label=None,
) -> None:
    ax.add_patch(
        Polygon(
            np.stack([x, y], axis=1),
            closed=True,
            fc=color,
            ec="dimgrey",
            alpha=alpha,
            zorder=zorder,
            label=label,
        )
    )


def plot_polyline(
    ax,
    polylines: List[np.ndarray],
    cmap="spring",
    linewidth=3,
    arrow: bool = True,
    reverse: bool = False,
    alpha=0.5,
    zorder=100,
    color_change: bool = True,
    color=None,
    linestyle="-",
    label=None,
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    if isinstance(color, str):
        color = [color] * len(polylines)

    for i, polyline in enumerate(polylines):
        inter_poly = interpolate_centerline(polyline, 50)
        # inter_poly = polyline[...,:2]

        if arrow:
            point = inter_poly[-1]
            diff = inter_poly[-1] - inter_poly[-2]
            diff = diff / np.linalg.norm(diff)
            if color_change:
                c = plt.cm.get_cmap(cmap)(0)
            else:
                c = color[i]
            arrow = ax.quiver(
                point[0],
                point[1],
                diff[0],
                diff[1],
                alpha=alpha,
                scale_units="xy",
                scale=0.25,
                minlength=0.5,
                zorder=zorder - 1,
                color=c,
            )

        if color_change:
            arc = get_polyline_arc_length(inter_poly)
            polyline = inter_poly.reshape(-1, 1, 2)
            segment = np.concatenate([polyline[:-1], polyline[1:]], axis=1)
            norm = plt.Normalize(arc.min(), arc.max())
            lc = LineCollection(
                segment, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha, label=label
            )
            lc.set_array(arc if not reverse else arc[::-1])
            lc.set_linewidth(linewidth)
            ax.add_collection(lc)
        else:
            ax.plot(
                inter_poly[:, 0],
                inter_poly[:, 1],
                color=color[i],
                linewidth=linewidth,
                zorder=zorder,
                alpha=alpha,
                linestyle=linestyle,
                label=label,
            )


def plot_direction(ax, anchors, dir_vecs, zorder=1):
    for anchor, dir_vec in zip(anchors, dir_vecs):
        if np.linalg.norm(dir_vec) == 0:
            continue
        vec = dir_vec / np.linalg.norm(dir_vec)
        ax.arrow(
            anchor[0],
            anchor[1],
            vec[0],
            vec[1],
            color="black",
            zorder=zorder,
            head_width=0.2,
            head_length=0.2,
        )


def plot_trajectory_with_angle(ax, traj):
    if traj.shape[-1] > 3:
        angle_phase_num = traj.shape[-1] - 2
        phase = 2 * np.pi * np.arange(angle_phase_num) / angle_phase_num
        xn = traj[..., -3:]  # (N, 3)
        angles = -np.arctan2(
            np.sum(np.sin(phase) * xn, axis=-1), np.sum(np.cos(phase) * xn, axis=-1)
        )
    else:
        angles = traj[..., -1]

    ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=2)
    for p, angle in zip(traj, angles):
        ax.arrow(
            p[0],
            p[1],
            np.cos(angle) * 0.5,
            np.sin(angle) * 0.5,
            color="black",
            zorder=1,
            head_width=0.3,
            head_length=0.2,
        )
    ax.axis("equal")


def plot_crosswalk(ax, edge1, edge2):
    polygon = np.concatenate([edge1, edge2[::-1]])
    ax.add_patch(
        Polygon(
            polygon, closed=True, fc="k", alpha=0.3, hatch="///", ec="w", linewidth=2
        )
    )


def plot_sdc(
    ax,
    center,
    heading,
    width,
    length,
    steer=0.0,
    color="pink",
    fill=True,
    wheel=True,
    **kwargs,
):
    vec_heading = np.array([np.cos(heading), np.sin(heading)])
    vec_tan = np.array([np.sin(heading), -np.cos(heading)])

    front_left_wheel = center + 1.419 * vec_heading + 0.35 * width * vec_tan
    front_right_wheel = center + 1.419 * vec_heading - 0.35 * width * vec_tan
    wheel_heading = heading + steer
    wheel_size = (0.8, 0.3)

    plot_box(
        ax, center, heading, color=color, fill=fill, bbox_size=(length, width), **kwargs
    )

    if wheel:
        plot_box(
            ax,
            front_left_wheel,
            wheel_heading,
            color="k",
            fill=True,
            bbox_size=wheel_size,
            **kwargs,
        )
        plot_box(
            ax,
            front_right_wheel,
            wheel_heading,
            color="k",
            fill=True,
            bbox_size=wheel_size,
            **kwargs,
        )


def plot_lane_area(ax, left_bound, right_bound, fc="silver", alpha=1.0, ec=None):
    polygon = np.concatenate([left_bound, right_bound[::-1]])
    ax.add_patch(
        Polygon(polygon, closed=True, fc=fc, alpha=alpha, ec=None, linewidth=2)
    )


def plot_cov_ellipse(logstd, rho, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    # compute covariance matrix from logstd and rho
    std = np.exp(logstd)
    cov = np.array(
        [[std[0] ** 2, rho * std[0] * std[1]], [rho * std[0] * std[1], std[1] ** 2]]
    )

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip
