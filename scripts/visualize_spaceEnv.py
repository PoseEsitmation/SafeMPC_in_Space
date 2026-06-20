import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from vtkmodules.vtkRenderingCore import vtkTextActor
from hypercrl.envs.space_KOZ import SatDynEnv, quaternion_multiply

# Create the environment and get the first observation
env = SatDynEnv()
obs, _ = env.reset()


def quat_to_transform(q_wxyz):
    # Convert quaternion [w,x,y,z] to a 4x4 rotation matrix.
    # scipy expects [x,y,z,w], so we reorder before passing.
    r = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    mat4 = np.eye(4)
    mat4[:3, :3] = r.as_matrix()
    return mat4


def build_dynamic(env):
    # Build the satellite and boresight arrow for the current attitude.
    # These change every step as the satellite rotates.
    q_abs = quaternion_multiply(env.q_desired_array, env.state[:4])
    T = quat_to_transform(q_abs)

    # Satellite body (cube) + left and right solar panels, all in body frame
    body    = pv.Box(bounds=(-0.06,  0.06, -0.06,  0.06, -0.09,  0.09))
    panel_l = pv.Box(bounds=(-0.08,  0.08, -0.28, -0.07, -0.005, 0.005))
    panel_r = pv.Box(bounds=(-0.08,  0.08,  0.07,  0.28, -0.005, 0.005))
    sat = pv.merge([body, panel_l, panel_r])
    sat.transform(T, inplace=True)  # rotate into inertial frame

    # Boresight: the instrument axis rotated into the inertial frame
    boresight_i = T[:3, :3] @ env.f_zone.boresight_vector_in_b
    boresight_arrow = pv.Arrow(start=[0, 0, 0], direction=boresight_i, scale=0.2)

    return sat, boresight_arrow


def build_static(env):
    # Build the scene elements that depend on the KOZ.
    # These only change when the episode resets and a new KOZ is sampled.

    # Red cone: the forbidden angular region the boresight must stay outside
    koz_cone = pv.Cone(
        center=env.f_zone.avoid_vector_in_i * 0.45,
        direction=-env.f_zone.avoid_vector_in_i,
        angle=np.degrees(env.f_zone.half_angle),
        height=0.9,
        resolution=80,
    )
    # Black dodecahedron: marks the tip of the KOZ (the forbidden direction)
    forbidden = pv.PlatonicSolid('dodecahedron')
    forbidden.scale([0.1, 0.1, 0.1], inplace=True)
    forbidden.translate(env.f_zone.avoid_vector_in_i * 0.9, inplace=True)

    # Yellow arrow: where the boresight needs to point to solve the task
    T_desired = quat_to_transform(env.q_desired_array)
    goal_dir  = T_desired[:3, :3] @ env.f_zone.boresight_vector_in_b
    goal_arrow = pv.Arrow(start=[0, 0, 0], direction=goal_dir, scale=0.3)

    return koz_cone, forbidden, goal_arrow


# --- Empty mesh placeholders — the plotter holds references to these objects.
# We update their data in-place each frame instead of removing and re-adding actors.
# allow_empty_mesh suppresses PyVista warnings before the first frame is filled.
pv.global_theme.allow_empty_mesh = True
sat_mesh  = pv.PolyData()   # satellite body + panels (updated every step)
bore_mesh = pv.PolyData()   # current boresight arrow (updated every step)
koz_mesh  = pv.PolyData()   # KOZ cone (updated on episode reset)
forb_mesh = pv.PolyData()   # forbidden object (updated on episode reset)
goal_mesh = pv.PolyData()   # goal arrow (updated on episode reset)

pl = pv.Plotter()
pl.add_mesh(sat_mesh,  color='silver', label='Satellite')
pl.add_mesh(bore_mesh, color='green',  label='Boresight (current)')
pl.add_mesh(koz_mesh,  color='red',    opacity=0.6, label='KOZ')
pl.add_mesh(forb_mesh, color='black',  label='Forbidden object')
pl.add_mesh(goal_mesh, color='yellow', label='Goal')
pl.add_legend()
pl.add_axes()

# Fill placeholders with the initial episode geometry before opening the window
new_sat, new_bore = build_dynamic(env)
sat_mesh.copy_from(new_sat)
bore_mesh.copy_from(new_bore)

new_koz, new_forb, new_goal = build_static(env)
koz_mesh.copy_from(new_koz)
forb_mesh.copy_from(new_forb)
goal_mesh.copy_from(new_goal)

# Text overlay: shows theta and theta_margin live in the top-left corner.
# vtkTextActor is used directly because PyVista's add_text wrapper does not
# reliably support in-place text updates via SetInput().
info_actor = vtkTextActor()
info_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
info_actor.SetPosition(0.01, 0.80)
info_actor.GetTextProperty().SetFontSize(16)
info_actor.GetTextProperty().SetColor(0, 0, 0)  # black
info_actor.GetTextProperty().SetBold(True)


def step_and_render():
    global obs
    # Apply a random action (no policy — just for visualization purposes)
    action = env.action_space.sample()
    obs, _, done, _, _ = env.step(action)

    # When the episode ends, reset and rebuild the KOZ scene geometry
    if done:
        obs, _ = env.reset()
        new_koz, new_forb, new_goal = build_static(env)
        koz_mesh.copy_from(new_koz)
        forb_mesh.copy_from(new_forb)
        goal_mesh.copy_from(new_goal)

    # Update satellite attitude and boresight for this step
    new_sat, new_bore = build_dynamic(env)
    sat_mesh.copy_from(new_sat)
    bore_mesh.copy_from(new_bore)

    # Update the text overlay with the latest angles
    theta_deg        = env.state[8] * (180 / np.pi)
    theta_margin_deg = env.state[7] * (180 / np.pi)
    info_actor.SetInput(
        f"theta:         {theta_deg:.1f} deg\n"
        f"theta_margin:  {theta_margin_deg:.1f} deg"
    )

    pl.render()


# Register the text actor after show() opens the window, then run the loop.
# pl.update(50) processes window events and waits ~50ms per frame (~20 fps).
pl.renderer.AddActor2D(info_actor)
pl.show(auto_close=False, interactive_update=True, title="SatDynEnv live render")

while True:
    step_and_render()
    pl.update(50)
