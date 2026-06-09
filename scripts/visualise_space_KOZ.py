import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from hypercrl.envs.space_KOZ import SatDynEnv, quaternion_multiply

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

env = SatDynEnv()
obs, _ = env.reset()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_transform(q_wxyz):
    """Convert a [w, x, y, z] quaternion to a 4x4 homogeneous transform matrix.

    space_KOZ stores quaternions as [w, x, y, z], but scipy expects [x, y, z, w],
    so we reorder before calling from_quat.
    """
    r = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    mat4 = np.eye(4)
    mat4[:3, :3] = r.as_matrix()   # top-left 3x3 block = rotation, no translation
    return mat4


def build_scene(env):
    """Build all PyVista meshes for one simulation step.

    Returns four meshes:
        sat              – satellite body box (rotated to current attitude)
        boresight_arrow  – instrument boresight direction in inertial frame (green)
        koz_cone         – Keep-Out Zone forbidden cone (red, semi-transparent)
        target_arrow     – desired boresight direction at target attitude (yellow)
    """

    # Absolute quaternion = desired attitude * attitude error.
    # env.state[:4] is the error quaternion q_e; multiplying by q_desired gives
    # the satellite's actual orientation in the inertial frame.
    q_abs = quaternion_multiply(env.q_desired_array, env.state[:4])
    T = quat_to_transform(q_abs)   # 4x4 rotation matrix for this orientation

    # --- Satellite body -------------------------------------------------
    # A rectangular box sized roughly like a small satellite [m].
    # We transform it by T so it matches the current attitude.
    sat = pv.Box(bounds=(-0.3, 0.3, -0.2, 0.2, -0.15, 0.15))
    sat.transform(T)

    # --- Boresight arrow ------------------------------------------------
    # boresight_vector_in_b is defined in the body frame.
    # Rotating it by the upper-left 3x3 of T gives the same direction
    # expressed in the inertial frame, which is what we draw in the scene.
    R = T[:3, :3]
    boresight_i = R @ env.f_zone.boresight_vector_in_b
    boresight_arrow = pv.Arrow(start=[0, 0, 0], direction=boresight_i, scale=0.6)

    # --- KOZ cone -------------------------------------------------------
    # avoid_vector_in_i is the axis of the forbidden zone in the inertial frame.
    # half_angle [rad] is the half-opening angle of the cone.
    # We shift the cone center along its axis so the tip sits at the origin,
    # making it visually clear that the satellite body is the reference point.
    koz_cone = pv.Cone(
        center=env.f_zone.avoid_vector_in_i * 0.4,  # shift tip to origin
        direction=env.f_zone.avoid_vector_in_i,
        angle=np.degrees(env.f_zone.half_angle),
        height=0.8,
        resolution=60,                               # smoothness of the cone rim
    )

    # --- Target arrow ---------------------------------------------------
    # The desired attitude is q_desired = [1, 0, 0, 0] (identity rotation),
    # so the target boresight in the inertial frame equals the boresight in
    # the body frame — no rotation needed.
    target_arrow = pv.Arrow(start=[0, 0, 0],
                            direction=env.f_zone.boresight_vector_in_b,
                            scale=0.6)

    return sat, boresight_arrow, koz_cone, target_arrow

# ---------------------------------------------------------------------------
# Static plot — inspect the initial state before running any animation
# ---------------------------------------------------------------------------

sat, boresight_arrow, koz_cone, target_arrow = build_scene(env)

pl = pv.Plotter()
pl.add_mesh(sat,             color='lightblue', label='Satellite')
pl.add_mesh(boresight_arrow, color='green',     label='Boresight')
pl.add_mesh(koz_cone,        color='red',       opacity=0.3, label='KOZ')
pl.add_mesh(target_arrow,    color='yellow',    label='Target')
pl.add_legend()
pl.add_axes()   # shows X/Y/Z orientation in the corner
pl.show()       # blocks until the window is closed

# ---------------------------------------------------------------------------
# Animation — step through the sim and save a GIF
# Remove pl.open_gif / pl.write_frame lines for a live interactive window.
# ---------------------------------------------------------------------------

pl = pv.Plotter()
pl.open_gif("space_koz.gif")

for _ in range(200):
    # Random action for demonstration; replace with a trained policy if available.
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)

    if done:
        obs, _ = env.reset()

    # Rebuild meshes from the updated env state each step.
    sat, boresight_arrow, koz_cone, target_arrow = build_scene(env)

    pl.clear()
    pl.add_mesh(sat,             color='lightblue', label='Satellite')
    pl.add_mesh(boresight_arrow, color='green',     label='Boresight')
    pl.add_mesh(koz_cone,        color='red',       opacity=0.3, label='KOZ')
    pl.add_mesh(target_arrow,    color='yellow',    label='Target')
    pl.add_axes()
    pl.write_frame()   # write current view as one GIF frame

pl.close()
