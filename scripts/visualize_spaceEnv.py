import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from hypercrl.envs.space_KOZ import SatDynEnv, quaternion_multiply

env = SatDynEnv()
obs, _ = env.reset()


def quat_to_transform(q_wxyz):
    # scipy expects [x,y,z,w], env stores [w,x,y,z]
    r = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    mat4 = np.eye(4)
    mat4[:3, :3] = r.as_matrix()
    return mat4


def build_dynamic(env):
    """Return only the two meshes that change with the satellite attitude."""
    q_abs = quaternion_multiply(env.q_desired_array, env.state[:4])
    T = quat_to_transform(q_abs)

    body    = pv.Box(bounds=(-0.06,  0.06, -0.06,  0.06, -0.09,  0.09))
    panel_l = pv.Box(bounds=(-0.08,  0.08, -0.28, -0.07, -0.005, 0.005))
    panel_r = pv.Box(bounds=(-0.08,  0.08,  0.07,  0.28, -0.005, 0.005))
    sat = pv.merge([body, panel_l, panel_r])
    sat.transform(T, inplace=True)

    boresight_i = T[:3, :3] @ env.f_zone.boresight_vector_in_b
    boresight_arrow = pv.Arrow(start=[0, 0, 0], direction=boresight_i, scale=0.2)

    return sat, boresight_arrow


# --- static scene elements (task geometry, built once) ---
koz_cone = pv.Cone(
    center=env.f_zone.avoid_vector_in_i * 0.45,
    direction=-env.f_zone.avoid_vector_in_i,
    angle=np.degrees(env.f_zone.half_angle),
    height=0.9,
    resolution=80,
)
forbidden = pv.PlatonicSolid('dodecahedron')
forbidden.scale([0.1, 0.1, 0.1], inplace=True)
forbidden.translate(env.f_zone.avoid_vector_in_i * 0.9, inplace=True)

T_desired = quat_to_transform(env.q_desired_array)
goal_dir  = T_desired[:3, :3] @ env.f_zone.boresight_vector_in_b
goal_arrow = pv.Arrow(start=[0, 0, 0], direction=goal_dir, scale=0.3)

# --- dynamic meshes (placeholder, updated every frame) ---
sat_mesh, bore_mesh = build_dynamic(env)

pl = pv.Plotter()
pl.add_mesh(sat_mesh,  color='silver', label='Satellite')
pl.add_mesh(bore_mesh, color='green',  label='Boresight (current)')
pl.add_mesh(koz_cone,  color='red',    opacity=0.6, label='KOZ')
pl.add_mesh(forbidden, color='black',  label='Forbidden object')
pl.add_mesh(goal_arrow, color='yellow', label='Goal')
pl.add_legend()
pl.add_axes()


def step_and_render():
    global obs
    action = env.action_space.sample()
    obs, _, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()

    new_sat, new_bore = build_dynamic(env)
    sat_mesh.overwrite(new_sat)
    bore_mesh.overwrite(new_bore)
    pl.render()


# ~20 fps; increase max_steps or set to 0 for unlimited
pl.add_timer_event(max_steps=1000, duration=50, callback=step_and_render)
pl.show(title="SatDynEnv live render")
