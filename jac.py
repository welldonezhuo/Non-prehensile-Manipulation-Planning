import copy
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc


class JacSolver(object):
    """
    The Jacobian solver for the 7-DoF Franka Panda robot.
    """

    def __init__(self):
        self.bullet_client = bc.BulletClient(connection_mode=p.DIRECT)
        self.bullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    def forward_kinematics(self, joint_values):
        """
        Calculate the Forward Kinematics of the robot given joint angle values.
        args: joint_values: The joint angle values of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:       pos: The position of the end-effector.
                            Type: numpy.ndarray [x, y, z]
                      quat: The orientation of the end-effector represented by quaternion.
                            Type: numpy.ndarray [x, y, z, w]
        """
        for j in range(7):
            self.bullet_client.resetJointState(self.panda, j, joint_values[j])
        ee_state = self.bullet_client.getLinkState(self.panda, linkIndex=11)
        pos, quat = np.array(ee_state[4]), np.array(ee_state[5])
        return pos, quat

    def get_jacobian_matrix(self, joint_values):
        """
        Numerically calculate the Jacobian matrix based on joint angles.
        args: joint_values: The joint angles of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:         J: The calculated Jacobian matrix.
                            Type: numpy.ndarray of shape (6, 7)
        """
        ########## TODO ##########
        J = np.zeros(shape=(6, 7))

        dq = 1e-3

        pos0, quat0 = self.forward_kinematics(joint_values)
        R0 = np.array(self.bullet_client.getMatrixFromQuaternion(
            quat0)).reshape(3, 3)

        for i in range(7):
            joint_values_perturbed = copy.deepcopy(joint_values)
            joint_values_perturbed[i] += dq

            pos1, quat1 = self.forward_kinematics(joint_values_perturbed)
            R1 = np.array(self.bullet_client.getMatrixFromQuaternion(
                quat1)).reshape(3, 3)

            # linear part
            J[0:3, i] = (pos1 - pos0) / dq

            # angular part
            R_rel = R1 @ R0.T
            w_hat = (R_rel - R_rel.T) / (2.0 * dq)
            J[3:6, i] = np.array([
                w_hat[2, 1],
                w_hat[0, 2],
                w_hat[1, 0]
            ])
        ##########################
        return J