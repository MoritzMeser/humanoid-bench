import os

import numpy as np
import mujoco
import gymnasium as gym
import wandb
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65
_CRAWL_HEIGHT = 0.8

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 5


class Walk(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0"
    }
    _move_speed = _WALK_SPEED
    htarget_low = np.array([-1.0, -1.0, 0.8])
    htarget_high = np.array([1000.0, 1.0, 2.0])
    success_bar = 700

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT, _CRAWL_HEIGHT
            _STAND_HEIGHT = 1.28
            _CRAWL_HEIGHT = 0.6

    @property
    def observation_space(self):
        return Box(
            low=-np.inf, high=np.inf, shape=(self.robot.dof * 2 - 1,), dtype=np.float64
        )

    def modified_reward(self):
        reward = 0

        # stand height
        stand_height = self.robot.head_height()
        stand_height_reward = rewards.tolerance(stand_height,
                                                bounds=(_STAND_HEIGHT, float("inf")),
                                                margin=_STAND_HEIGHT / 4)
        reward += stand_height_reward

        # pelvis / feet
        foot_left_height = self.robot.left_foot_height()
        foot_right_height = self.robot.right_foot_height()
        com = self.robot.center_of_mass_position()[2]
        pelvis_feet = com - (foot_left_height + foot_right_height) / 2
        pelvis_feet_reward = rewards.tolerance(pelvis_feet, bounds=(0.8, float("inf")))
        reward += pelvis_feet_reward

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

        # torso upright
        torso_upright = self.robot.torso_upright()
        torso_upright_reward = rewards.tolerance(torso_upright, bounds=(0.9, float("inf")), margin=1.9)
        reward += torso_upright_reward

        # pelvis upright
        pelvis_upright = self.robot.pelvis_upright()
        pelvis_upright_reward = rewards.tolerance(pelvis_upright, bounds=(0.9, float("inf")), margin=1.9)
        reward += pelvis_upright_reward

        # foot upright
        left_foot_upright = self.robot.left_foot_upright()
        left_foot_upright_reward = rewards.tolerance(left_foot_upright, bounds=(0.9, float("inf")), margin=1.9)
        right_foot_upright = self.robot.right_foot_upright()
        right_foot_upright_reward = rewards.tolerance(right_foot_upright, bounds=(0.9, float("inf")), margin=1.9)
        foot_upright_reward = (left_foot_upright_reward + right_foot_upright_reward) / 2
        reward += foot_upright_reward

        # posture
        robot_posture = self.robot.joint_angles()
        robot_posture = np.sum(robot_posture ** 2)
        robot_posture_reward = rewards.tolerance(robot_posture, bounds=(-0.1, 0.1), margin=1)
        reward += robot_posture_reward

        # walking speed
        com_velocity = self.robot.center_of_mass_velocity()[0]
        # body_velocity = self.robot.body_velocity()[0]
        walking_reward = rewards.tolerance(com_velocity,
                                           bounds=(self._move_speed, float("inf")),
                                           margin=self._move_speed)
        reward += walking_reward

        # face direction
        facing_direction = self.robot.facing_direction()
        facing_direction_reward = rewards.tolerance(facing_direction, bounds=(0.9, float("inf")), margin=1.9)
        reward += facing_direction_reward

        # control
        control = self.robot.control()
        control = np.sum(control ** 2)
        control_reward = rewards.tolerance(control, margin=10)
        reward += control_reward

        reward /= 10.0
        return reward, {
            "stand_height_reward": stand_height_reward,
            "pelvis_feet_reward": pelvis_feet_reward,
            "balance_reward": balance_reward,
            "torso_upright_reward": torso_upright_reward,
            "pelvis_upright_reward": pelvis_upright_reward,
            "foot_upright_reward": foot_upright_reward,
            "robot_posture_reward": robot_posture_reward,
            "walking_reward": walking_reward,
            "facing_direction_reward": facing_direction_reward,
            "control_reward": control_reward}

    def original_reward(self):
        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5
        if self._move_speed == 0:
            horizontal_velocity = self.robot.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move, {
                "small_control": small_control,
                "stand_reward": stand_reward,
                "dont_move": dont_move,
                "standing": standing,
                "upright": upright,
            }
        else:
            com_velocity = self.robot.center_of_mass_velocity()[0]
            move = rewards.tolerance(
                com_velocity,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid="linear",
            )
            move = (5 * move + 1) / 6
            reward = small_control * stand_reward * move
            return reward, {
                "stand_reward": stand_reward,
                "small_control": small_control,
                "move": move,
                "standing": standing,
                "upright": upright,
            }

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
        return self._env.data.qpos[2] < 0.2, {}


class Stand(Walk):
    _move_speed = 0
    success_bar = 800


class Run(Walk):
    _move_speed = _RUN_SPEED


class Crawl(Walk):
    def get_reward(self):
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        com_velocity = self.robot.center_of_mass_velocity()[0]
        move = rewards.tolerance(
            com_velocity,
            bounds=(1, float("inf")),
            margin=1,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6

        crawling_head = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_CRAWL_HEIGHT - 0.2, _CRAWL_HEIGHT + 0.2),
            margin=1,
        )

        crawling = rewards.tolerance(
            self._env.named.data.site_xpos["imu", "z"],
            bounds=(_CRAWL_HEIGHT - 0.2, _CRAWL_HEIGHT + 0.2),
            margin=1,
        )

        reward_xquat = rewards.tolerance(
            np.linalg.norm(
                self._env.data.body("pelvis").xquat - np.array([0.75, 0, 0.65, 0])
            ),
            margin=1,
        )

        in_tunnel = rewards.tolerance(
            self._env.named.data.site_xpos["imu", "y"],
            bounds=(-1, 1),
            margin=0,
        )

        reward = (
                         0.1 * small_control
                         + 0.25 * min(crawling, crawling_head)
                         + 0.4 * move
                         + 0.25 * reward_xquat
                 ) * in_tunnel
        return reward, {
            "crawling": crawling,
            "crawling_head": crawling_head,
            "small_control": small_control,
            "move": move,
            "in_tunnel": in_tunnel,
        }

    def get_terminated(self):
        return False, {}


class ClimbingUpwards(Walk):
    def get_reward(self):
        standing = rewards.tolerance(
            self.robot.head_height() - self.robot.left_foot_height(),
            bounds=(1.2, float("inf")),
            margin=0.45,
        ) * rewards.tolerance(
            self.robot.head_height() - self.robot.right_foot_height(),
            bounds=(1.2, float("inf")),
            margin=0.45,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.5, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        com_velocity = self.robot.center_of_mass_velocity()[0]
        move = rewards.tolerance(
            com_velocity,
            bounds=(_WALK_SPEED, float("inf")),
            margin=_WALK_SPEED,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6
        return stand_reward * small_control * move, {  # small_control *
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "standing": standing,
            "upright": upright,
        }

    def get_terminated(self):
        return self.robot.torso_upright() < 0.1, {}


class Stair(ClimbingUpwards):
    pass


class Slide(ClimbingUpwards):
    pass


class Hurdle(Walk):
    _move_speed = _RUN_SPEED
    camera_name = "cam_hurdle"

    def get_reward(self):
        self.wall_collision_ids = [
            self._env.named.data.geom_xpos.axes.row.names.index(wall_name)
            for wall_name in [
                "left_barrier_collision",
                "right_barrier_collision",
                "behind_barrier_collision",
            ]
        ]

        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.8, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5
        com_velocity = self.robot.center_of_mass_velocity()[0]
        move = rewards.tolerance(
            com_velocity,
            bounds=(self._move_speed, float("inf")),
            margin=self._move_speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6
        wall_collision_discount = 1

        for pair in self._env.data.contact.geom:
            if any(
                    [
                        wall_collision_id in pair
                        for wall_collision_id in self.wall_collision_ids
                    ]
            ):  # for no hand. if for hand, > 155
                wall_collision_discount = 0.1
                # print(pair)
                break

        reward = small_control * stand_reward * move * wall_collision_discount

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "standing": standing,
            "upright": upright,
            "wall_collision_discount": wall_collision_discount,
        }


class Sit(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0"
    }
    dof = 0
    vels = 0
    success_bar = 750

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof + self.vels,),
            dtype=np.float64,
        )

    def get_reward(self):
        sitting = rewards.tolerance(
            self._env.data.qpos[2], bounds=(0.68, 0.72), margin=0.2
        )
        chair_location = self._env.named.data.xpos["chair"]
        on_chair = rewards.tolerance(
            self._env.data.qpos[0] - chair_location[0], bounds=(-0.19, 0.19), margin=0.2
        ) * rewards.tolerance(self._env.data.qpos[1] - chair_location[1], margin=0.1)
        sitting_posture = rewards.tolerance(
            self.robot.head_height() - self._env.named.data.site_xpos["imu", "z"],
            bounds=(0.35, 0.45),
            margin=0.3,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.95, float("inf")),
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )
        sit_reward = (0.5 * sitting + 0.5 * on_chair) * upright * sitting_posture
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        horizontal_velocity = self.robot.center_of_mass_velocity()[[0, 1]]
        dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
        return small_control * sit_reward * dont_move, {
            "small_control": small_control,
            "sit_reward": sit_reward,
            "dont_move": dont_move,
            "sitting": sitting,
            "upright": upright,
            "sitting_posture": sitting_posture,
        }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.5, {}

    @staticmethod
    def euler_to_quat(angles):
        cr, cp, cy = np.cos(angles[0] / 2), np.cos(angles[1] / 2), np.cos(angles[2] / 2)
        sr, sp, sy = np.sin(angles[0] / 2), np.sin(angles[1] / 2), np.sin(angles[2] / 2)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )


class SitHard(Sit):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0"
    }

    dof = 7
    vels = 6

    def reset_model(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()
        position[0] = np.random.uniform(0.2, 0.4)
        position[1] = np.random.uniform(-0.15, 0.15)
        rotation_angle = np.random.uniform(-1.8, 1.8)
        position[3:7] = self.euler_to_quat(np.array([0, 0, rotation_angle]))
        self._env.set_state(position, velocity)
        return super().reset_model()
