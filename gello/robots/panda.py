import time
from typing import Dict

import numpy as np

from gello.robots.robot import Robot

MAX_OPEN = 0.09


class PandaRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "100.97.47.74"):
        from polymetis import GripperInterface, RobotInterface

        self.robot = RobotInterface(
            ip_address=robot_ip, enforce_version=False
        )
        self.gripper = GripperInterface(
            ip_address=robot_ip
        )
        self.robot.go_home()
        self.robot.start_joint_impedance()
        self.gripper.goto(width=MAX_OPEN, speed=255, force=255)
        self.gripper_closed = False
        time.sleep(1)

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        return 8

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.robot.get_joint_positions()
        gripper_pos = self.gripper.get_state()
        pos = np.append(robot_joints, gripper_pos.width / MAX_OPEN)
        # print(robot_joints)
        # print(pos)
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        import torch

        self.robot.update_desired_joint_positions(torch.tensor(joint_state[:-1]))
        gripper_closed = joint_state[-1] > 0.25
        if gripper_closed and not self.gripper_closed:
            self.gripper_closed = True
            self.gripper.grasp(speed=0.1, force=1.0)
        elif not gripper_closed and self.gripper_closed:
            self.gripper_closed = False
            # self.gripper.stop()
            self.gripper.goto(width=MAX_OPEN, speed=1, force=1)
        return

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        velocities = self.robot.get_joint_velocities().numpy()

        pos, quat = self.robot.get_ee_pose()
        pos_quat = np.concatenate([pos, quat])
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": velocities,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }
    


def main():
    robot = PandaRobot()
    current_joints = robot.get_joint_state()
    # move a small delta 0.1 rad
    move_joints = current_joints + 0.05
    # make last joint (gripper) closed
    move_joints[-1] = 0.5
    time.sleep(1)
    m = 0.09
    robot.gripper.goto(1 * m, speed=255, force=255)
    time.sleep(1)
    robot.gripper.goto(1.05 * m, speed=255, force=255)
    time.sleep(1)
    robot.gripper.goto(1.1 * m, speed=255, force=255)
    time.sleep(1)


if __name__ == "__main__":
    main()
