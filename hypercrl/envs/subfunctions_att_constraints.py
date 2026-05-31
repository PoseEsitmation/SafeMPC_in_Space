import numpy as np

def get_camera_angles(vector):
    # Calculate the required camera angles to make the vector point toward the viewer
    x, y, z = vector
    # Azimuth angle (in-plane rotation)
    azim = np.degrees(np.arctan2(y, x))
    # Elevation angle (out-of-plane tilt)
    elev = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return elev, azim

def calM(x_i, y_b, alpha):
    """
    Parameters:
        x_i: 3D array, unit vector to void or to point towards in inertial frame
        y_b: 3D array, unit vector of instrument bore-sight in body frame
        alpha: float, in [radian], half angle of the forbidden or mandatory zone
    Returns:
        M : 4x4 array, matrix for constraint consideration
    """

    M11 = np.array([np.dot(x_i, y_b) - np.cos(alpha)])

    cross_yx = np.cross(y_b,x_i)
    M12 = cross_yx.reshape(1,3)
    M21 = cross_yx.reshape(3,1)
    M22 = np.outer(x_i,y_b) + np.outer(y_b,x_i) - (np.dot(x_i,y_b) + np.cos(alpha)) * np.eye(3)

    M = np.block([[M11, M12],[M21, M22]])
    return M

def generate_F_zone(angle_lower_bound, angle_upper_bound):
    """
    Generating random forbidden zones with
        n_f_ini: pointing direction in inertial frame
        theta_f: [deg], half angle of F-zone
    """
    n_f_ini = np.random.randn(3)
    n_f_ini /= np.linalg.norm(n_f_ini)

    theta_f = np.random.uniform(angle_lower_bound,angle_upper_bound) # [deg]
    return n_f_ini, theta_f


class KeepOutZone:
    def __init__(self, boresight_vector_in_b, avoid_vector_in_i, half_angle):
        self.boresight_vector_in_b = boresight_vector_in_b    # unit vector (3x1), in body frame
        self.avoid_vector_in_i = avoid_vector_in_i            # unit vector (3x1), in inertial frame
        self.half_angle = half_angle                # [rad]

    def get_Mf(self):
        M11 = np.array([np.dot(self.avoid_vector_in_i, self.boresight_vector_in_b) - np.cos(self.half_angle)])

        cross_term = np.cross(self.boresight_vector_in_b, self.avoid_vector_in_i)
        M12 = cross_term.reshape(1,3)
        M21 = cross_term.reshape(3,1)
        M22 = (
            np.outer(self.avoid_vector_in_i, self.boresight_vector_in_b)
            + np.outer(self.boresight_vector_in_b, self.avoid_vector_in_i)
            - (np.dot(self.avoid_vector_in_i, self.boresight_vector_in_b) + np.cos(self.half_angle)) * np.eye(3)
        )
        return np.block([[M11, M12],[M21, M22]])

def random_unit_quat_with_angle_bound_lower_upper(lower_bound, upper_bound, seed=None):
    """
    Generate a random unit quaternion uniformly on S^3 with different bound_lim for angle \theta [deg].
        - lower_bound and upper_bound in [deg]
    """
    if seed is not None:
        np.random.seed(seed)
    e = np.random.randn(3)
    e /= np.linalg.norm(e)

    theta = np.random.uniform(lower_bound, upper_bound) * np.pi / 180    # [rad]

    q = np.array([np.cos(theta/2), e[0] * np.sin(theta/2), e[1] * np.sin(theta/2), e[2] * np.sin(theta/2)])

    if q[0] < 0:    # ensuring a positive scalar element
        q = -q

    return q

def random_angular_rate(rate_bound=0.0 * np.pi /180, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low=-rate_bound, high=rate_bound, size=3)      # [rad/s]

def quaternion_multiply(q1,q2):
    """Multiply two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=np.float32)

def quaternion_conj(q):
    """Conjugate of quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def generate_avoid_vector_in_i_for_1Fzone_phase1(boresight_vector_in_i_initial, boresight_vector_in_i_desired, vector_rotation_angle1_ratio, vector_rotation_angle2):
    """
    :param vector_rotation_angle2:  in [deg]
    """
    e1 = np.cross(boresight_vector_in_i_initial, boresight_vector_in_i_desired)     # normal vector to the vector rotation plane
    e1 /= np.linalg.norm(e1)    # ensure unit vector

    angle1_total = np.acos(np.inner(boresight_vector_in_i_initial, boresight_vector_in_i_desired))      # [rad], angle between initial and desired pointing

    theta1 = -1.0 * vector_rotation_angle1_ratio * angle1_total     # [rad]
    q1 = np.array(
        [np.cos(theta1 / 2), e1[0] * np.sin(theta1 / 2), e1[1] * np.sin(theta1 / 2), e1[2] * np.sin(theta1 / 2)],
        np.float32)
    q1 /= np.linalg.norm(q1)

    boresight_vector_in_i_initial_quat  = np.concatenate(([0.0], boresight_vector_in_i_initial))
    avoid_vector_in_i_projection_quat = quaternion_multiply(quaternion_multiply(quaternion_conj(q1), boresight_vector_in_i_initial_quat),q1)
    avoid_vector_in_i_projection_quat /= np.linalg.norm(avoid_vector_in_i_projection_quat)    # ensure unit quaternion

    e2 = np.cross(avoid_vector_in_i_projection_quat[1:4],e1)
    e2/=np.linalg.norm(e2)  # ensure unit vector

    theta2 = -1.0 * vector_rotation_angle2 * np.pi / 180
    q2 = np.array(
        [np.cos(theta2 / 2), e2[0] * np.sin(theta2 / 2), e2[1] * np.sin(theta2 / 2), e2[2] * np.sin(theta2 / 2)],
        np.float32)
    q2 /= np.linalg.norm(q2)

    avoid_vector_in_i_quat = quaternion_multiply(quaternion_multiply(quaternion_conj(q2), avoid_vector_in_i_projection_quat),q2)
    avoid_vector_in_i_quat/=np.linalg.norm(avoid_vector_in_i_quat)      # ensure unit quaternion

    theta_1 = np.acos(np.inner(boresight_vector_in_i_initial, avoid_vector_in_i_quat[1:4])) * 180/np.pi
    theta_2 = np.acos(np.inner(boresight_vector_in_i_desired, avoid_vector_in_i_quat[1:4])) * 180/np.pi
    min_theta = min(theta_1, theta_2)  # [deg]

    if min_theta < (15.0 + 2.0):
        half_angle_high = 0.0         # Half_angle of F-zone is set as zero when the angle is too small
    else:
        half_angle_high = min_theta - 2.0    # [deg], margin of 2 deg

    return avoid_vector_in_i_quat[1:4], half_angle_high

"""
def generate_avoid_vector_in_i_for_1Fzone_old(boresight_vector_in_i_initial, boresight_vector_in_i_desired, vector_rotation_angle1, vector_rotation_angle2):

    # lower1, upper1, lower2, upper2:  in [deg]

    e1 = np.cross(boresight_vector_in_i_initial, boresight_vector_in_i_desired)     # normal vector to the vector rotation plane
    e1 /= np.linalg.norm(e1)    # ensure unit vector
    #print(f"normal vector e1: {e1}")

    theta1 = -1.0 * vector_rotation_angle1 * np.pi / 180
    q1 = np.array(
        [np.cos(theta1 / 2), e1[0] * np.sin(theta1 / 2), e1[1] * np.sin(theta1 / 2), e1[2] * np.sin(theta1 / 2)],
        np.float32)
    q1 /= np.linalg.norm(q1)
    #print(f"norm of q1:{np.linalg.norm(q1)}")

    boresight_vector_in_i_initial_quat  = np.concatenate(([0.0], boresight_vector_in_i_initial))
    avoid_vector_in_i_projection_quat = quaternion_multiply(quaternion_multiply(quaternion_conj(q1), boresight_vector_in_i_initial_quat),q1)
    avoid_vector_in_i_projection_quat /= np.linalg.norm(avoid_vector_in_i_projection_quat)    # ensure unit quaternion
    #print(f"avoid_vector_in_i_projection: {avoid_vector_in_i_projection_quat[1:4]}")

    e2 = np.cross(avoid_vector_in_i_projection_quat[1:4],e1)
    e2/=np.linalg.norm(e2)  # ensure unit vector
    #print(f"normal vector e2: {e2}")

    theta2 = -1.0 * vector_rotation_angle2 * np.pi / 180
    q2 = np.array(
        [np.cos(theta2 / 2), e2[0] * np.sin(theta2 / 2), e2[1] * np.sin(theta2 / 2), e2[2] * np.sin(theta2 / 2)],
        np.float32)
    q2 /= np.linalg.norm(q2)
    #print(f"norm of q2:{np.linalg.norm(q2)}")

    avoid_vector_in_i_quat = quaternion_multiply(quaternion_multiply(quaternion_conj(q2), avoid_vector_in_i_projection_quat),q2)
    avoid_vector_in_i_quat/=np.linalg.norm(avoid_vector_in_i_quat)      # ensure unit quaternion
    #print(f"norm: {np.linalg.norm(avoid_vector_in_i_quat)}")
    return  avoid_vector_in_i_quat[1:4]
"""

"""
def generate_avoid_vector_in_i_for_1Fzone_phase1_old(boresight_vector_in_b, boresight_vector_in_i_initial, boresight_vector_in_i_desired, q_initial, q_e_initial, vector_rotation_angle1_alpha_low, vector_rotation_angle1_alpha_high):
    # The F-zone is located directly on the rotation path
    q_e_initial_conj = quaternion_conj(q_e_initial)     # q_conj is used since qe_initial is defined w.r.t. desired attitude. Whereas the following quaternion multiplication is w.r.t. the initial attitude.
    beta = np.acos(q_e_initial_conj[0]) * 2.0     # [rad]

    vector_n = q_e_initial_conj[1:4] / np.linalg.norm(q_e_initial_conj[1:4])
    alpha = np.random.uniform(low=vector_rotation_angle1_alpha_low, high=vector_rotation_angle1_alpha_high)
    delta_q_temp = np.concatenate(([np.cos(alpha * beta / 2)], vector_n * np.sin(alpha * beta / 2)))       # Exponential map
    q_temp = quaternion_multiply(q_initial, delta_q_temp)

    boresight_vector_in_b_quat = np.concatenate(([0.0], boresight_vector_in_b))
    avoid_vector_in_i_proj_quat = quaternion_multiply(q_temp, quaternion_multiply(boresight_vector_in_b_quat,
                                                                             quaternion_conj(q_temp)))
    avoid_vector_in_i_proj_quat /= np.linalg.norm(avoid_vector_in_i_proj_quat)          # ensure unit quaternion

    theta_1 = np.acos(np.inner(boresight_vector_in_i_initial, avoid_vector_in_i_proj_quat[1:4])) * 180/np.pi
    theta_2 = np.acos(np.inner(boresight_vector_in_i_desired, avoid_vector_in_i_proj_quat[1:4])) * 180/np.pi
    half_angle_high = min(theta_1, theta_2)  # [deg]

    return avoid_vector_in_i_proj_quat[1:4], half_angle_high, alpha
"""
def generate_avoid_vector_in_i_for_1Fzone_phase1_v2(boresight_vector_in_b, boresight_vector_in_i_initial, boresight_vector_in_i_desired, q_initial, q_e_initial, vector_rotation_angle1_ratio, vector_rotation_angle2):
    """
    The avoid vector is determined based on the exponential map from the initial attitude to the desired one.
    """
    # The F-zone is located directly on the rotation path
    q_e_initial_conj = quaternion_conj(q_e_initial)     # q_conj is used since qe_initial is defined w.r.t. desired attitude. Whereas the following quaternion multiplication is w.r.t. the initial attitude.
    beta = np.acos(q_e_initial_conj[0]) * 2.0           # [rad]

    vector_n = q_e_initial_conj[1:4] / np.linalg.norm(q_e_initial_conj[1:4])
    delta_q_temp = np.concatenate(([np.cos(vector_rotation_angle1_ratio * beta / 2)], vector_n * np.sin(vector_rotation_angle1_ratio * beta / 2)))       # Exponential map
    q_temp = quaternion_multiply(q_initial, delta_q_temp)

    boresight_vector_in_b_quat = np.concatenate(([0.0], boresight_vector_in_b))
    avoid_vector_in_i_proj_quat = quaternion_multiply(q_temp, quaternion_multiply(boresight_vector_in_b_quat,
                                                                             quaternion_conj(q_temp)))
    avoid_vector_in_i_proj_quat /= np.linalg.norm(avoid_vector_in_i_proj_quat)          # ensure unit quaternion

    e1 = np.cross(boresight_vector_in_i_initial, avoid_vector_in_i_proj_quat[1:4])
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(avoid_vector_in_i_proj_quat[1:4],e1)
    e2 /= np.linalg.norm(e2)

    theta2 = -1.0 * vector_rotation_angle2 * np.pi / 180
    q2 = np.array(
        [np.cos(theta2 / 2), e2[0] * np.sin(theta2 / 2), e2[1] * np.sin(theta2 / 2), e2[2] * np.sin(theta2 / 2)],
        np.float32)
    q2 /= np.linalg.norm(q2)

    avoid_vector_in_i_quat = quaternion_multiply(quaternion_multiply(quaternion_conj(q2), avoid_vector_in_i_proj_quat),q2)
    avoid_vector_in_i_quat/=np.linalg.norm(avoid_vector_in_i_quat)      # ensure unit quaternion

    theta_1 = np.acos(np.inner(boresight_vector_in_i_initial, avoid_vector_in_i_quat[1:4])) * 180/np.pi
    theta_2 = np.acos(np.inner(boresight_vector_in_i_desired, avoid_vector_in_i_quat[1:4])) * 180/np.pi
    min_theta = min(theta_1, theta_2)  # [deg]

    if min_theta < (15.0 + 2.0):
        half_angle_high = 0.0         # Half_angle of F-zone is set as zero when the angle is too small
    else:
        half_angle_high = min_theta - 2.0    # [deg], margin of 2 deg

    return avoid_vector_in_i_quat[1:4], half_angle_high