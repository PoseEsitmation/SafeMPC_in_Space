import numpy as np

def get_camera_angles(vector):
 
    x, y, z = vector
    azim = np.degrees(np.arctan2(y, x))
    elev = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return elev, azim

def calM(x_i, y_b, alpha):

    M11 = np.array([np.dot(x_i, y_b) - np.cos(alpha)])

    cross_yx = np.cross(y_b,x_i)
    M12 = cross_yx.reshape(1,3)
    M21 = cross_yx.reshape(3,1)
    M22 = np.outer(x_i,y_b) + np.outer(y_b,x_i) - (np.dot(x_i,y_b) + np.cos(alpha)) * np.eye(3)

    M = np.block([[M11, M12],[M21, M22]])
    return M

def generate_F_zone(angle_lower_bound, angle_upper_bound):

    n_f_ini = np.random.randn(3)
    n_f_ini /= np.linalg.norm(n_f_ini)

    theta_f = np.random.uniform(angle_lower_bound,angle_upper_bound) 
    return n_f_ini, theta_f


class KeepOutZone:
    def __init__(self, boresight_vector_in_b, avoid_vector_in_i, half_angle):
        self.boresight_vector_in_b = boresight_vector_in_b    
        self.avoid_vector_in_i = avoid_vector_in_i            
        self.half_angle = half_angle                

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

    if seed is not None:
        np.random.seed(seed)
    e = np.random.randn(3)
    e /= np.linalg.norm(e)

    theta = np.random.uniform(lower_bound, upper_bound) * np.pi / 180    

    q = np.array([np.cos(theta/2), e[0] * np.sin(theta/2), e[1] * np.sin(theta/2), e[2] * np.sin(theta/2)])

    if q[0] < 0:    
        q = -q

    return q

def random_angular_rate(rate_bound=0.0 * np.pi /180, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low=-rate_bound, high=rate_bound, size=3)      

def quaternion_multiply(q1,q2):

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=np.float32)

def quaternion_conj(q):

    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def generate_avoid_vector_in_i_for_1Fzone_phase1(boresight_vector_in_i_initial, boresight_vector_in_i_desired, vector_rotation_angle1_ratio, vector_rotation_angle2):

    e1 = np.cross(boresight_vector_in_i_initial, boresight_vector_in_i_desired)     
    e1 /= np.linalg.norm(e1)    

    angle1_total = np.acos(np.inner(boresight_vector_in_i_initial, boresight_vector_in_i_desired))      

    theta1 = -1.0 * vector_rotation_angle1_ratio * angle1_total     
    q1 = np.array(
        [np.cos(theta1 / 2), e1[0] * np.sin(theta1 / 2), e1[1] * np.sin(theta1 / 2), e1[2] * np.sin(theta1 / 2)],
        np.float32)
    q1 /= np.linalg.norm(q1)

    boresight_vector_in_i_initial_quat  = np.concatenate(([0.0], boresight_vector_in_i_initial))
    avoid_vector_in_i_projection_quat = quaternion_multiply(quaternion_multiply(quaternion_conj(q1), boresight_vector_in_i_initial_quat),q1)
    avoid_vector_in_i_projection_quat /= np.linalg.norm(avoid_vector_in_i_projection_quat)    

    e2 = np.cross(avoid_vector_in_i_projection_quat[1:4],e1)
    e2/=np.linalg.norm(e2)  

    theta2 = -1.0 * vector_rotation_angle2 * np.pi / 180
    q2 = np.array(
        [np.cos(theta2 / 2), e2[0] * np.sin(theta2 / 2), e2[1] * np.sin(theta2 / 2), e2[2] * np.sin(theta2 / 2)],
        np.float32)
    q2 /= np.linalg.norm(q2)

    avoid_vector_in_i_quat = quaternion_multiply(quaternion_multiply(quaternion_conj(q2), avoid_vector_in_i_projection_quat),q2)
    avoid_vector_in_i_quat/=np.linalg.norm(avoid_vector_in_i_quat)      

    theta_1 = np.acos(np.inner(boresight_vector_in_i_initial, avoid_vector_in_i_quat[1:4])) * 180/np.pi
    theta_2 = np.acos(np.inner(boresight_vector_in_i_desired, avoid_vector_in_i_quat[1:4])) * 180/np.pi
    min_theta = min(theta_1, theta_2)  

    if min_theta < (15.0 + 2.0):
        half_angle_high = 0.0         
    else:
        half_angle_high = min_theta - 2.0    

    return avoid_vector_in_i_quat[1:4], half_angle_high

def generate_avoid_vector_in_i_for_1Fzone_phase1_v2(boresight_vector_in_b, boresight_vector_in_i_initial, boresight_vector_in_i_desired, q_initial, q_e_initial, vector_rotation_angle1_ratio, vector_rotation_angle2):
   
    # The F-zone is located directly on the rotation path
    q_e_initial_conj = quaternion_conj(q_e_initial)     
    beta = np.acos(q_e_initial_conj[0]) * 2.0           

    vector_n = q_e_initial_conj[1:4] / np.linalg.norm(q_e_initial_conj[1:4])
    delta_q_temp = np.concatenate(([np.cos(vector_rotation_angle1_ratio * beta / 2)], vector_n * np.sin(vector_rotation_angle1_ratio * beta / 2)))       
    q_temp = quaternion_multiply(q_initial, delta_q_temp)

    boresight_vector_in_b_quat = np.concatenate(([0.0], boresight_vector_in_b))
    avoid_vector_in_i_proj_quat = quaternion_multiply(q_temp, quaternion_multiply(boresight_vector_in_b_quat,
                                                                             quaternion_conj(q_temp)))
    avoid_vector_in_i_proj_quat /= np.linalg.norm(avoid_vector_in_i_proj_quat)          

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
    avoid_vector_in_i_quat/=np.linalg.norm(avoid_vector_in_i_quat)     

    theta_1 = np.acos(np.inner(boresight_vector_in_i_initial, avoid_vector_in_i_quat[1:4])) * 180/np.pi
    theta_2 = np.acos(np.inner(boresight_vector_in_i_desired, avoid_vector_in_i_quat[1:4])) * 180/np.pi
    min_theta = min(theta_1, theta_2)  

    if min_theta < (15.0 + 2.0):
        half_angle_high = 0.0         
    else:
        half_angle_high = min_theta - 2.0    

    return avoid_vector_in_i_quat[1:4], half_angle_high