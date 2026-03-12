import numpy as np
import jac


class Goal(object):
    """
    A trivial goal that is always satisfied.
    """

    def __init__(self):
        pass

    def is_satisfied(self, state):
        """
        Determine if the query state satisfies this goal or not.
        """
        return True


class RelocateGoal(Goal):
    """
    The goal for relocating tasks.
    (i.e., pushing the target object into a circular goal region.)
    """

    def __init__(self, x_g=0.2, y_g=-0.2, r_g=0.1):
        """
        args: x_g: The x-coordinate of the center of the goal region.
              y_g: The y-coordinate of the center of the goal region.
              r_g: The radius of the goal region.
        """
        super(RelocateGoal, self).__init__()
        self.x_g, self.y_g, self.r_g = x_g, y_g, r_g

    def is_satisfied(self, state):
        """
        Check if the state satisfies the RelocateGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        """
        stateVec = state["stateVec"]
        # position of the target object
        x_tgt, y_tgt = stateVec[7], stateVec[8]
        if np.linalg.norm([x_tgt - self.x_g, y_tgt - self.y_g]) < self.r_g:
            return True
        else:
            return False


class GraspGoal(Goal):
    """
    The goal for grasping tasks.
    (i.e., approaching the end-effector to a pose that can grasp the target object.)
    """

    def __init__(self):
        super(GraspGoal, self).__init__()
        self.jac_solver = jac.JacSolver()  # the jacobian solver

    def is_satisfied(self, state):
        """
        Check if the state satisfies the GraspGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        returns: True or False.
        """
        ########## TODO ##########

        # Extract robot joint values (first 7 elements correspond to Panda joints)
        joint_values = state["stateVec"][0:7]

        # Extract target cube pose on the table (x, y position and orientation)
        x_tgt, y_tgt, theta_tgt = state["stateVec"][7:10]

        # Compute end-effector pose using forward kinematics
        # pos_ee: position of the end-effector
        # quat_ee: orientation of the end-effector (quaternion)
        pos_ee, quat_ee = self.jac_solver.forward_kinematics(joint_values)

        # Convert end-effector position from robot base frame to workspace frame
        # (robot base is offset by [-0.4, -0.2] on the table)
        x_ee = pos_ee[0] - 0.4
        y_ee = pos_ee[1] - 0.2

        # Extract the yaw angle (rotation around z-axis) of the end-effector
        theta_ee = self.jac_solver.bullet_client.getEulerFromQuaternion(quat_ee)[
            2]

        # Define two orthogonal directions on the table plane
        # ee_normal: direction perpendicular to the gripper plane
        # ee_tangent: direction along the gripper opening
        ee_normal = np.array([np.cos(theta_ee), np.sin(theta_ee)])
        ee_tangent = np.array([-np.sin(theta_ee), np.cos(theta_ee)])

        # Vector from the end-effector center to the cube center
        delta = np.array([x_tgt - x_ee, y_tgt - y_ee])

        # d1: perpendicular distance from cube center to the gripper plane
        # Ensures the cube is close enough to the gripper surface
        d1 = abs(np.dot(delta, ee_normal))

        # d2: distance along the gripper opening direction
        # Ensures the cube lies between the two gripper fingers
        d2 = abs(np.dot(delta, ee_tangent))

        # gamma: orientation difference between end-effector and cube
        # Since the cube has four symmetric sides, we check orientation modulo 90 degrees
        gamma = min(
            abs((theta_ee - (theta_tgt + k*np.pi/2) + np.pi) % (2*np.pi) - np.pi)
            for k in range(4)
        )

        # A grasp is feasible only if all geometric conditions are satisfied
        return (d1 < 0.01) and (d2 < 0.02) and (gamma < 0.2)
        ##########################
