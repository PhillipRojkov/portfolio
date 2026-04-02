from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

import numpy as np

START_MOTOR_ANGLES = np.array([0, 0, 0.78, -1.57, 0, 0, 0])
HOME_POSE = np.array([0.3, 0, 0.2, 0, 0, 0])
LEFT_OFFSET_Y = 0.4

class ArmKinematics:
    def __init__(self, arm_name, origin_offset=[0, 0, 0]):
        self.arm_name = arm_name

        self.arm_chain = Chain(name=self.arm_name, links=[
            OriginLink(),
            URDFLink(
              name="yaw",
              bounds=(-3*np.pi/4, 3*np.pi/4),
              origin_translation=origin_offset,
              origin_orientation=[0, 0, 0],
              rotation=[0, 0, 1],
            ),
            URDFLink(
              name="shoulder",
              bounds=(0, np.pi),
              origin_translation=[0, 0, 0.138],
              origin_orientation=[0, -np.pi, 0],
              rotation=[0, 1, 0],
            ),
            URDFLink(
              name="elbow",
              bounds=(-np.pi, 0),
              origin_translation=[0.386, 0, 0],
              origin_orientation=[0.0087, 0, 0], # ~0.5 degree of twist in both arms between shoulder and elbow
              rotation=[0, -1, 0],
            ),
            URDFLink(
              name="wrist_rot",
              bounds=(-3*np.pi / 4, 3*np.pi / 4),
              origin_translation=[0.408, 0, 0.0253],
              origin_orientation=[0, 0, 0],
              rotation=[1, 0, 0],
            ),
            URDFLink(
              name="wrist_azimuth",
              bounds=(-np.pi / 2, np.pi / 2),
              origin_translation=[0, 0.0417, 0],
              origin_orientation=[0, 0, 0],
              rotation=[0, 1, 0],
            ),
            URDFLink(
              name="wrist_yaw",
              bounds=(-np.pi / 2, np.pi / 2),
              origin_translation=[0.066, 0, 0],
              origin_orientation=[0, 0, 0],
              rotation=[1, 0, 0],
            ),
                ], active_links_mask=[False, True, True, True, True, True, True])

    def inverseKinematics(self, position, orientation, start_angles=START_MOTOR_ANGLES):
        ik = self.arm_chain.inverse_kinematics(
                target_position=position,
                target_orientation=orientation,
                orientation_mode='all',
                initial_position=start_angles,
                )
        return ik

    def forwardKinematics(self, angles):
        mat = self.arm_chain.forward_kinematics(angles)
        position = mat[:3, 3]
        orientation = mat[:3, :3] 
        return position, orientation

    def computeJacobian(self, joint_angles):
        """
        Compute 6xN geometric Jacobian for an ikpy Chain.

        Parameters:
            chain : ikpy.chain.Chain
            angles     : joint angles array (including origin link)

        Returns:
            J : 6xN Jacobian
        """
        # Get transforms of every link frame
        transforms = self.arm_chain.forward_kinematics(joint_angles, full_kinematics=True)

        n_joints = len(self.arm_chain.links) - 1  # exclude OriginLink
        J = np.zeros((6, n_joints))

        # End-effector position
        T_ee = transforms[-1]
        p_ee = T_ee[:3, 3]

        for i in range(1, len(self.arm_chain.links)):
            link = self.arm_chain.links[i]
            T_i = transforms[i]

            # Joint origin in world frame
            p_i = T_i[:3, 3]

            # Joint axis in local frame (from URDFLink.rotation)
            axis_local = np.array(link.rotation)

            # Transform axis into world frame
            R_i = T_i[:3, :3]
            z_i = R_i @ axis_local

            # Linear velocity component
            Jv = np.cross(z_i, p_ee - p_i)

            # Angular velocity component
            Jw = z_i

            J[:3, i-1] = Jv
            J[3:, i-1] = Jw

        return J

def orientationFromFixedAxes(angles):
    """Return a 3x3 rotation matrix from x, y, z (roll, pitch, yaw) fixed axes rotation angles"""
    x = angles[0]
    y = angles[1]
    z = angles[2]
    return [[np.cos(z)*np.cos(y), -np.sin(z)*np.cos(x) + np.cos(z)*np.sin(y)*np.sin(x), np.sin(z)*np.sin(x) + np.cos(z)*np.sin(y)*np.cos(x)],
            [np.sin(z)*np.cos(y), np.cos(z)*np.cos(x) + np.sin(z)*np.sin(y)*np.sin(x), -np.cos(z)*np.sin(x) + np.sin(z)*np.sin(y)*np.cos(x)],
            [-np.sin(y), np.cos(y)*np.sin(x), np.cos(y)*np.cos(x)]]

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class Arm:
    def __init__(self, arm_name=None, origin_offset=[0, 0, 0]):
        self.arm_kinematics = ArmKinematics(arm_name, origin_offset)
        self.INITIAL_POSE = np.add(HOME_POSE, np.array(origin_offset + [0, 0, 0]))
        self.reset_pose()

    def update_from_pose(self, pose, gripper_closed=True):
        self.pose = pose
        self.gripper_closed = gripper_closed
        self.motor_angles = self.arm_kinematics.inverseKinematics(pose[:3], orientationFromFixedAxes(pose[3:]), self.motor_angles)
        return self.motor_angles

    def update_from_positions_angles(self, position, angles, gripper_closed=True):
        self.pose[:3] = position
        self.pose[3:] = angles
        self.gripper_closed = gripper_closed
        self.motor_angles = self.arm_kinematics.inverseKinematics(position, orientationFromFixedAxes(angles), self.motor_angles)
        return self.motor_angles

    def reset_pose(self):
        self.pose = self.INITIAL_POSE
        self.gripper_closed = True
        self.motor_angles = self.arm_kinematics.inverseKinematics(self.pose[:3], orientationFromFixedAxes(self.pose[3:]))
    
