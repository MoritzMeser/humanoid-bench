import os

import numpy as np
import mujoco
import gymnasium as gym
from dm_control.utils import rewards
from gymnasium.spaces import Box

from humanoid_bench.tasks import Task
from humanoid_bench.mjx.flax_to_torch import TorchModel, TorchPolicy

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65


class Push(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 0.7 0 1 1 0 0 0"
    }
    dof = 7
    max_episode_steps = 500
    camera_name = "cam_tabletop"
    # Below args are only used for reaching-based hierarchical control
    htarget_low = np.array([0, -1, 0.8])
    htarget_high = np.array([2.0, 1, 1.2])

    success_bar = 700

    def __init__(
            self,
            robot=None,
            env=None,
            **kwargs,
    ):
        super().__init__(robot, env, **kwargs)

        if env is None:
            return

        self.reward_dict = {
            "hand_dist": 0.1,
            "target_dist": 1,
            "success": 1000,
            "terminate": True,
        }

        self.goal = np.array([1.0, 0.0, 1.0])

        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + 12,),
            dtype=np.float64,
        )

    def get_obs(self):
        position = self._env.data.qpos.flat.copy()[: self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[: self.robot.dof - 1]
        left_hand = self.robot.left_hand_position()
        target = self.goal.copy()
        box = self._env.data.qpos.flat.copy()[-7:-4]
        dofadr = self._env.named.model.body_dofadr["object"]
        box_vel = self._env.data.qvel.flat.copy()[dofadr: dofadr + 3]

        return np.concatenate((position, velocity, left_hand, target, box, box_vel))

    def goal_dist(self):
        box = self._env.data.qpos.flat.copy()[-7:-4]
        return np.sqrt(np.square(box - self.goal).sum())

    def modified_reward(self):
        reward = 0
        info = {}

        # stand height
        stand_height = self.robot.head_height()
        stand_height_reward = rewards.tolerance(stand_height,
                                                bounds=(_STAND_HEIGHT, float("inf")),
                                                margin=_STAND_HEIGHT / 4)
        reward += stand_height_reward
        info["stand_height_reward"] = stand_height_reward

        # pelvis / feet
        foot_left_height = self.robot.left_foot_height()
        foot_right_height = self.robot.right_foot_height()
        com = self.robot.center_of_mass_position()[2]
        pelvis_feet = com - (foot_left_height + foot_right_height) / 2
        pelvis_feet_reward = rewards.tolerance(pelvis_feet, bounds=(0.8, float("inf")))
        reward += pelvis_feet_reward
        info["pelvis_feet_reward"] = pelvis_feet_reward

        # balance
        com_pos = self.robot.center_of_mass_position()
        com_vel = self.robot.center_of_mass_velocity()
        capture_point = com_pos + 0.3 * com_vel

        foot_right = self.robot.right_foot_position()
        foot_left = self.robot.left_foot_position()
        axis = foot_right - foot_left
        axis[2] = 0
        axis /= np.linalg.norm(axis)
        length = 0.5 * np.linalg.norm(axis) - 0.05

        center = (foot_right + foot_left) / 2
        vector = capture_point - center

        t = np.dot(vector, axis)
        t = np.clip(t, -length, length)
        vec = axis * t
        pcp = center + vec
        pcp[2] = 0

        capture_point = capture_point[:2] - pcp[:2]
        capture_point = np.linalg.norm(capture_point)
        balance_reward = rewards.tolerance(capture_point, bounds=(0, 0.0), margin=0.5)
        reward += balance_reward
        info["balance_reward"] = balance_reward

        # torso upright
        torso_upright = self.robot.torso_upright()
        torso_upright_reward = rewards.tolerance(torso_upright, bounds=(0.9, float("inf")), margin=1.9)
        reward += torso_upright_reward
        info["torso_upright_reward"] = torso_upright_reward

        # pelvis upright
        pelvis_upright = self.robot.pelvis_upright()
        pelvis_upright_reward = rewards.tolerance(pelvis_upright, bounds=(0.9, float("inf")), margin=1.9)
        reward += pelvis_upright_reward
        info["pelvis_upright_reward"] = pelvis_upright_reward

        # foot upright
        left_foot_upright = self.robot.left_foot_upright()
        left_foot_upright_reward = rewards.tolerance(left_foot_upright, bounds=(0.9, float("inf")), margin=1.9)
        right_foot_upright = self.robot.right_foot_upright()
        right_foot_upright_reward = rewards.tolerance(right_foot_upright, bounds=(0.9, float("inf")), margin=1.9)
        foot_upright_reward = (left_foot_upright_reward + right_foot_upright_reward) / 2
        reward += foot_upright_reward
        info["foot_upright_reward"] = foot_upright_reward

        # posture
        robot_posture = self.robot.joint_angles()
        robot_posture = np.sum(robot_posture ** 2)
        robot_posture_reward = rewards.tolerance(robot_posture, bounds=(-0.1, 0.1), margin=1)
        reward += robot_posture_reward

        # walking speed
        body_velocity = self.robot.body_velocity()[0]
        walking_reward = rewards.tolerance(body_velocity,
                                           bounds=(0.0, 0.0),
                                           margin=0.1)
        reward += walking_reward
        info["walking_reward"] = walking_reward

        # face direction
        facing_direction = self.robot.facing_direction()
        facing_direction_reward = rewards.tolerance(facing_direction, bounds=(0.9, float("inf")), margin=1.9)
        reward += facing_direction_reward
        info["facing_direction_reward"] = facing_direction_reward

        # com velocity
        x_vel, y_vel, z_vel = self.robot.center_of_mass_velocity()
        x_vel_reward = rewards.tolerance(x_vel, bounds=(-0.0, 0.0), margin=0.1)
        y_vel_reward = rewards.tolerance(y_vel, bounds=(-0.0, 0.0), margin=0.1)
        com_vel_reward = (x_vel_reward + y_vel_reward) / 2
        reward += com_vel_reward
        info["com_vel_reward"] = com_vel_reward

        # com position
        x_pos, y_pos, z_pos = self.robot.center_of_mass_position()
        x_pos_reward = rewards.tolerance(x_pos, bounds=(0.0, 0.0), margin=0.1)
        y_pos_reward = rewards.tolerance(y_pos, bounds=(0.0, 0.0), margin=0.1)
        com_pos_reward = (x_pos_reward + y_pos_reward) / 2
        reward += com_pos_reward
        info["com_pos_reward"] = com_pos_reward

        # control
        control = self.robot.control()
        control = np.sum(control ** 2)
        control_reward = rewards.tolerance(control, margin=10)
        reward += control_reward
        info["control_reward"] = control_reward

        # box distance
        box = self._env.data.qpos.flat.copy()[-7:-4]
        box_dist = np.sqrt(np.square(box - self.goal).sum())
        box_dist_reward = rewards.tolerance(box_dist, bounds=(0.0, 0.0), margin=0.1)
        reward += box_dist_reward
        info["box_dist_reward"] = box_dist_reward

        # left hand distance
        left_hand = self.robot.left_hand_position()
        hand_dist = np.sqrt(np.square(left_hand - box).sum())
        hand_dist_reward = rewards.tolerance(hand_dist, bounds=(0.0, 0.0), margin=0.1)
        reward += hand_dist_reward
        info["hand_dist_reward"] = hand_dist_reward

        # right hand distance
        right_hand = self.robot.right_hand_position()
        hand_dist = np.sqrt(np.square(right_hand - box).sum())
        hand_dist_reward = rewards.tolerance(hand_dist, bounds=(0.0, 0.0), margin=0.1)
        reward += hand_dist_reward
        info["hand_dist_reward"] = hand_dist_reward

        return reward, info

    def original_reward(self):
        goal_dist = self.goal_dist()
        penalty_dist = self.reward_dict["target_dist"] * goal_dist
        reward_success = self.reward_dict["success"] if goal_dist < 0.05 else 0

        left_hand = self.robot.left_hand_position()
        # box = self._env.data.qpos.flat.copy()[-7:-4]
        box = self._env.named.data.qpos["free_object"][:3]

        hand_dist = np.sqrt(np.square(left_hand - box).sum())
        hand_penalty = self.reward_dict["hand_dist"] * hand_dist

        reward = -hand_penalty - penalty_dist + reward_success
        info = {
            "target_dist": goal_dist,
            "hand_dist": hand_dist,
            "reward_success": reward_success,
            "success": reward_success > 0,
        }
        return reward, info

    def get_reward(self):
        modified_reward, modified_info = self.modified_reward()
        original_reward, original_info = self.original_reward()

        # Create modified and original keys using dictionary comprehension
        modified_info = {f"modified_{key}": value for key, value in modified_info.items()}
        original_info = {f"original_{key}": value for key, value in original_info.items()}

        # Merge the dictionaries
        combined_info = {"modified_reward": modified_reward,
                         "original_reward": original_reward,
                         **modified_info, **original_info}

        return modified_reward, combined_info

    def get_terminated(self):
        if self.reward_dict["terminate"]:
            terminated = self.goal_dist() < 0.05
        else:
            terminated = False

        return terminated, {}

    def reset_model(self):
        self.goal[0] = np.random.uniform(0.7, 1.0)
        self.goal[1] = np.random.uniform(-0.5, 0.5)

        return self.get_obs()

    def render(self):
        found = False
        for i in range(len(self._env.viewer._markers)):
            if self._env.viewer._markers[i]["objid"] == 789:
                self._env.viewer._markers[i]["pos"] = self.goal
                found = True
                break

        if not found:
            self._env.viewer.add_marker(
                pos=self.goal,
                size=0.05,
                objid=789,
                rgba=(0.8, 0.28, 0.28, 1.0),
                label="",
            )

        return self._env.mujoco_renderer.render(
            self._env.render_mode, self._env.camera_id, self._env.camera_name
        )
