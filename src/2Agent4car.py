#!/usr/bin/env python3
"""
notetrainingnote
============================================================
goal:
  - noteDonkeySimnote
  - note: Anote(note)noteBnote(note)
  - notemodel
  - note + note:
  - Anote (note): Blue1, Blue2 - noteBlueTeammodel
  - Bnote (note): Red1, Red2 - noteRedTeammodel

rewardnote:
  - notereward: CTEnote, notereward, note
  - note: note
  - note: note

trainingstage:
  - first2wnote: note(note)
  - note2wnote: note(notereward)

implementnote:
  - notetrainingnote(note)
  - note
  - notefirstnote:
  - DonkeySim note 9091 note
  - noterowsnote:
  python train_quad_team_racing.py --steps 40000 --scene waveshare

============================================================
"""

import os
import sys
import time
import argparse
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

import gym
import gym_donkeycar  # noqa: F401
import numpy as np
import cv2
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from multiprocessing import Process, Manager

# ===== noteunifiednote =====

class _SpeedUnits:
    def __init__(self, speed_max_mps=3.0):
        self.speed_max = float(speed_max_mps)

    def norm(self, v_mps: float) -> float:
        return float(np.clip(v_mps / max(1e-6, self.speed_max), 0.0, 1.0))

# ===== V6 note =====

class YellowLaneEnhancer:
    def __init__(self):
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

    def enhance(self, rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        return mask

class RealLidarProcessor:
    def __init__(self, num_sectors=36, max_distance=3.5, use_hardware=False, port='/dev/ttyUSB0'):
        self.num_sectors = num_sectors
        self.max_distance = max_distance
        self.use_hardware = use_hardware
        self.lidar_hardware = None

        if self.use_hardware:
            try:
                from lidar.ld19 import LD19
                self.lidar_hardware = LD19(port=port)
                print(f"PASS LD19noteLiDARnote ({port})")
            except Exception as e:
                print(f"⚠️ noteLiDARnotefailed: {e}")
                self.lidar_hardware = None

    def process(self, raw_data=None):
        if self.lidar_hardware is not None:
            lidar_raw = self.lidar_hardware.get_sector_distances(self.num_sectors)
        elif raw_data is not None and len(raw_data) > 0:
            lidar_raw = np.array(raw_data, dtype=np.float32)
            if len(lidar_raw)!= self.num_sectors:
                lidar_raw = np.zeros(self.num_sectors, dtype=np.float32)
        else:
            lidar_raw = np.zeros(self.num_sectors, dtype=np.float32)

        lidar_filtered = self._median_filter(lidar_raw, window=3)
        lidar_enhanced = np.where(lidar_filtered < 2.0, lidar_filtered * 0.4, lidar_filtered)
        lidar_normalized = 1.0 - np.clip(lidar_enhanced / self.max_distance, 0.0, 1.0)

        return lidar_normalized.astype(np.float32)

    def _median_filter(self, data, window=3):
        filtered = np.copy(data)
        half_window = window // 2
        for i in range(len(data)):
            indices = [(i + j - half_window) % len(data) for j in range(window)]
            filtered[i] = np.median(data[indices])
        return filtered

    def close(self):
        if self.lidar_hardware is not None:
            self.lidar_hardware.close()

class MotionDetectionWrapper(gym.ObservationWrapper):
    def __init__(self, env, motion_threshold=30):
        super().__init__(env)
        self.last_gray = None
        self.motion_threshold = motion_threshold

        original_space = env.observation_space
        self.observation_space = gym.spaces.Dict({
            'cam': original_space,
            'motion': gym.spaces.Box(low=0, high=255, shape=original_space.shape[:2], dtype=np.uint8)
        })

    def reset(self, **kwargs):
        self.last_gray = None
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def observation(self, obs):
        if isinstance(obs, dict):
            image = obs.get('image', obs.get('cam', obs))
        else:
            image = obs

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.last_gray is not None:
            diff = cv2.absdiff(gray, self.last_gray)
            _, motion = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        else:
            motion = np.zeros_like(gray)

        self.last_gray = gray.copy()
        return {'cam': image, 'motion': motion}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

class V6MultiModalWrapper(gym.Wrapper):
    def __init__(self, env, use_lidar=True, lidar_port='/dev/ttyUSB0', action_noise_std=0.15):
        super().__init__(env)

        self.use_lidar = use_lidar
        self.yellow_enhancer = YellowLaneEnhancer()
        self.lidar_processor = RealLidarProcessor(use_hardware=use_lidar, port=lidar_port)

        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.action_noise_std = action_noise_std

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(6, 120, 160), dtype=np.uint8),
            'lidar': gym.spaces.Box(low=0.0, high=1.0, shape=(36,), dtype=np.float32)
        })

        print(f"PASS V6note")

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._process_observation(obs, None)

    def step(self, action):
        noisy_action = np.array(action, dtype=np.float32)
        noise = np.random.normal(0, self.action_noise_std, size=noisy_action.shape)
        noisy_action = np.clip(noisy_action + noise, -1.0, 1.0)

        obs, reward, done, info = self.env.step(noisy_action)
        sim_lidar = info.get("lidar", None)
        return self._process_observation(obs, sim_lidar), reward, done, info

    def _process_observation(self, obs, lidar_data):
        if isinstance(obs, dict):
            image = obs.get('image', obs.get('cam', obs))
            motion = obs.get('motion', np.zeros((120, 160), dtype=np.uint8))
        else:
            image = obs
            motion = np.zeros((120, 160), dtype=np.uint8)

        image = cv2.resize(image, (160, 120))

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        rgb = np.transpose(rgb_enhanced, (2, 0, 1)).astype(np.float32) / 255.0
        yellow_mask = self.yellow_enhancer.enhance(image)
        yellow_mask = yellow_mask.astype(np.float32) / 255.0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        motion_resized = cv2.resize(motion, (160, 120))
        motion_normalized = motion_resized.astype(np.float32) / 255.0

        image_6ch = np.concatenate([
            rgb, yellow_mask[np.newaxis,:,:],
            edges[np.newaxis,:,:], motion_normalized[np.newaxis,:,:]
        ], axis=0)
        image_chw = (image_6ch * 255).astype(np.uint8)

        lidar = self.lidar_processor.process(raw_data=lidar_data)

        return {'image': image_chw, 'lidar': lidar}

    def close(self):
        self.lidar_processor.close()
        super().close()

class MultiModalCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=320):
        super().__init__(observation_space, features_dim)

        self.image_cnn = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_image = torch.zeros(1, 6, 120, 160)
            cnn_output_size = self.image_cnn(dummy_image).shape[1]

        self.lidar_mlp = nn.Sequential(
            nn.Linear(36, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_output_size + 32, features_dim), nn.ReLU()
        )

    def forward(self, observations):
        image = observations['image'].float() / 255.0
        lidar = observations['lidar'].float()

        image_features = self.image_cnn(image)
        lidar_features = self.lidar_mlp(lidar)

        combined = torch.cat([image_features, lidar_features], dim=1)
        return self.fusion(combined)

# ===== noterewardnote =====

class TeamRewardWrapper(gym.Wrapper):
    """
    noterewardnote: note + note

    rewardnote:
    - notereward: CTEnote, notereward, note
    - note: notereward
    - note: notereward
    """

    def __init__(self, env, team_name, car_id, shared_state, team_coop_coeff=0.1, team_compete_coeff=1.0):
        super().__init__(env)
        self.team_name = team_name  # 'blue' or 'red'
        self.car_id = car_id  # 1 or 2
        self.shared_state = shared_state
        self.team_coop_coeff = team_coop_coeff
        self.team_compete_coeff = team_compete_coeff

        # noterewardnote
        self.survival_reward_per_step = 0.5
        self.cte_penalty_threshold = 3.0
        self.cte_penalty = -0.2
        self.collision_penalty = -50.0

        # note
        self.last_lap_count = 0
        self.my_lap_times = []
        self.teammate_lap_times = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # notereward
        shaped_reward = self.survival_reward_per_step

        # CTEnote
        cte = abs(info.get('cte', 0.0))
        if cte > self.cte_penalty_threshold:
            shaped_reward += self.cte_penalty

        # note
        hit = info.get('hit', None)
        collision = bool(info.get('collision', False)) or (hit is not None and hit!= "" and hit!= "none")
        if collision:
            shaped_reward += self.collision_penalty
            info['collision_penalty'] = True

        # noterewardnote
        lap_count = int(info.get('lap_count', info.get('lap', 0)))
        last_lap_time = float(info.get('last_lap_time', 0.0))

        team_reward = 0.0

        if lap_count > self.last_lap_count and last_lap_time > 0.0:
            self.my_lap_times.append(last_lap_time)

            # note
            team_key = f"{self.team_name}_car{self.car_id}_lap_time"
            self.shared_state[team_key] = last_lap_time
            self.shared_state[f"{self.team_name}_car{self.car_id}_lap_count"] = lap_count

            # notereward: notereward
            teammate_id = 2 if self.car_id == 1 else 1
            teammate_key = f"{self.team_name}_car{teammate_id}_lap_time"
            if teammate_key in self.shared_state:
                teammate_time = self.shared_state[teammate_key]
                # note, notereward
                if teammate_time < last_lap_time:
                    team_reward += self.team_coop_coeff
                    info['team_coop_reward'] = self.team_coop_coeff

            # notereward(notestage)
            if self.shared_state.get('phase', 'learning') == 'competition':
                opponent_team = 'red' if self.team_name == 'blue' else 'blue'

                # computenoteaverage time
                my_team_times = []
                for cid in [1, 2]:
                    tkey = f"{self.team_name}_car{cid}_lap_time"
                    if tkey in self.shared_state:
                        my_team_times.append(self.shared_state[tkey])

                # computenoteaverage time
                opponent_team_times = []
                for cid in [1, 2]:
                    tkey = f"{opponent_team}_car{cid}_lap_time"
                    if tkey in self.shared_state:
                        opponent_team_times.append(self.shared_state[tkey])

                if my_team_times and opponent_team_times:
                    my_avg_time = np.mean(my_team_times)
                    opponent_avg_time = np.mean(opponent_team_times)

                    # noteaverage timenote, notereward
                    if my_avg_time < opponent_avg_time:
                        compete_reward = self.team_compete_coeff * (opponent_avg_time - my_avg_time) / opponent_avg_time
                        team_reward += compete_reward
                        info['team_compete_reward'] = compete_reward
                    # note, note
                    else:
                        compete_penalty = -self.team_compete_coeff * (my_avg_time - opponent_avg_time) / my_avg_time
                        team_reward += compete_penalty
                        info['team_compete_penalty'] = compete_penalty

            self.last_lap_count = lap_count

        info['shaped_reward'] = shaped_reward
        info['team_reward'] = team_reward

        return obs, shaped_reward + team_reward, done, info

# ===== notedetectionnote =====

class StagnationResetWrapper(gym.Wrapper):
    def __init__(self, env, max_stagnant_steps: int = 25, speed_threshold: float = 0.05):
        super().__init__(env)
        self.max_stagnant_steps = max_stagnant_steps
        self.speed_threshold = speed_threshold
        self.stagnant_count = 0
        self.last_speed = 0.0

    def reset(self, **kwargs):
        self.stagnant_count = 0
        self.last_speed = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        current_speed = info.get('speed', 0.0)

        if abs(current_speed) < self.speed_threshold:
            self.stagnant_count += 1
        else:
            self.stagnant_count = 0

        if self.stagnant_count >= self.max_stagnant_steps:
            done = True
            reward -= 10.0
            info['stagnation_reset'] = True
            info['stagnant_steps'] = self.stagnant_count

        self.last_speed = current_speed
        info['stagnant_count'] = self.stagnant_count

        return obs, reward, done, info

# ===== note =====

class TeamSaveCallback(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str, team_name: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.team_name = team_name
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            ckpt_path = os.path.join(self.save_dir, "checkpoints", f"ckpt_{self.num_timesteps}.zip")
            self.model.save(ckpt_path)
            if self.verbose:
                print(f"[{self.team_name}] saved savenote: {ckpt_path}")
        return True

class TeamPhaseCallback(BaseCallback):
    def __init__(self, phase_change_step=20000, shared_state=None, team_name="", verbose: int = 1):
        super().__init__(verbose)
        self.phase_change_step = phase_change_step
        self.shared_state = shared_state
        self.team_name = team_name
        self.current_phase = 'learning'

    def _on_step(self) -> bool:
        if self.n_calls >= self.phase_change_step and self.current_phase == 'learning':
            self.current_phase = 'competition'
            if self.verbose:
                print(f"🔄 {self.team_name} trainingstagenote: note(note: {self.n_calls})")
            if self.shared_state:
                self.shared_state['phase'] = 'competition'
        elif self.n_calls < self.phase_change_step:
            if self.shared_state:
                self.shared_state['phase'] = 'learning'
        return True

# ===== note =====

def build_team_env(team_name: str, car_id: int, port: int, scene: str, max_cte: float,
                   shared_state: dict, use_lidar: bool = False, lidar_port: str = '/dev/ttyUSB0',
                   action_noise_std: float = 0.15):

    # noteconfiguration: note vs note
    if team_name == 'blue':
        body_rgb = (0, 0, 255) if car_id == 1 else (0, 100, 255)  # note vs note
        racer_name = f"Blue{car_id}"
    else:  # red team
        body_rgb = (255, 0, 0) if car_id == 1 else (255, 100, 100)  # note vs note
        racer_name = f"Red{car_id}"

    conf = {
        "host": "127.0.0.1",
        "port": port,
        "body_style": "donkey",
        "body_rgb": body_rgb,
        "car_name": racer_name,
        "racer_name": racer_name,
        "bio": f"Team {team_name.upper()} Car {car_id} - Group Racing",
        "country": "CN",
        "guid": f"team-{team_name}-{car_id}-{int(time.time())}",
        "max_cte": max_cte,
        "level": scene,
        "font_size": 50,
        "lidar_config": {
            "deg_per_sweep_inc": 10.0,
            "deg_ang_down": 0.0,
            "deg_ang_delta": -1.0,
            "num_sweeps_levels": 1,
            "max_range": 3.5,
            "noise": 0.05,
            "offset_x": 0.0,
            "offset_y": 0.3,
            "offset_z": 0.2,
            "rot_x": 0.0,
        }
    }

    env = gym.make(
        "donkey-waveshare-v0" if scene == "waveshare"
        else "donkey-circuit-launch-track-v0" if scene == "circuit_launch"
        else "donkey-generated-track-v0",
        conf=conf
    )
    env = MotionDetectionWrapper(env, motion_threshold=30)
    env = V6MultiModalWrapper(env, use_lidar=use_lidar, lidar_port=lidar_port, action_noise_std=action_noise_std)

    # noterewardnote
    env = TeamRewardWrapper(env, team_name, car_id, shared_state)

    # notedetectionnote
    env = StagnationResetWrapper(env, max_stagnant_steps=25, speed_threshold=0.05)

    env = Monitor(env, filename=None, allow_early_resets=True)

    return env

def create_policy_kwargs():
    return dict(
        features_extractor_class=MultiModalCNN,
        features_extractor_kwargs=dict(features_dim=320),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )

# ===== notetrainingnote =====

def train_team(team_name: str, total_steps: int, save_dir: str, port: int, scene: str,
               max_cte: float, lr: float, save_freq: int, shared_state: dict,
               use_lidar: bool = False, lidar_port: str = '/dev/ttyUSB0',
               action_noise_std: float = 0.15):
    """trainingnote(notemodel)"""
    try:
        print(f"\n{'='*60}\n🚗🚗 note {team_name.upper()} notetraining\n{'='*60}")

        # note
        env1 = build_team_env(team_name, 1, port, scene, max_cte, shared_state,
                             use_lidar, lidar_port, action_noise_std)
        env2 = build_team_env(team_name, 2, port, scene, max_cte, shared_state,
                             use_lidar, lidar_port, action_noise_std)

        # note, noterowstraining
        env = DummyVecEnv([lambda: env1, lambda: env2])

        policy_kwargs = create_policy_kwargs()

        model = PPO(
            policy="MultiInputPolicy",
            env=env,  # note
            learning_rate=lr,
            n_steps=2048,  # note2048note, note4096note
            batch_size=128,  # notebatch_sizenotedata
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=os.path.join(save_dir, "tensorboard"),
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

        callbacks = [
            TeamSaveCallback(save_freq=save_freq, save_dir=save_dir, team_name=team_name.upper(), verbose=1),
            TeamPhaseCallback(phase_change_step=500000, shared_state=shared_state, team_name=team_name.upper(), verbose=1)
        ]

        print(f"🔥 {team_name.upper()} notetraining(noterows)...")

        # trainingmodel(note)
        model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)

        # savenotemodel
        final_path = os.path.join(save_dir, "final_model.zip")
        model.save(final_path)
        print(f"PASS {team_name.upper()} notetraining, modelsavenote: {final_path}")

        env.close()

    except Exception as e:
        print(f"FAIL {team_name.upper()} notetrainingfailed: {e}")
        traceback.print_exc()

def launch_quad_team_training(
    steps: int,
    port: int,
    scene: str,
    save_root: str,
    lr_blue: float,
    lr_red: float,
    max_cte: float,
    save_freq: int,
    use_lidar: bool = False,
    lidar_port: str = '/dev/ttyUSB0',
    action_noise_std: float = 0.15,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = f"{save_root}_{timestamp}"
    blue_dir = os.path.join(root_dir, "blue_team")
    red_dir = os.path.join(root_dir, "red_team")
    os.makedirs(blue_dir, exist_ok=True)
    os.makedirs(red_dir, exist_ok=True)

    print(f"\n📁 modelnotedirectory: {root_dir}")
    print(f"  ↳ Blue Team: {blue_dir}")
    print(f"  ↳ Red Team: {red_dir}")
    print(f"  ↳ trainingstage: first50wnote, note50wnote")
    print(f"  ↳ note2notemodel")

    manager = Manager()
    shared_state = manager.dict()
    shared_state['phase'] = 'learning'

    # notetrainingnote
    p_blue = Process(
        target=train_team,
        args=('blue', steps, blue_dir, port, scene, max_cte, lr_blue, save_freq,
              shared_state, use_lidar, lidar_port, action_noise_std),
        daemon=False,
    )
    p_red = Process(
        target=train_team,
        args=('red', steps, red_dir, port, scene, max_cte, lr_red, save_freq,
              shared_state, use_lidar, lidar_port, action_noise_std),
        daemon=False,
    )

    print("\n🚦 notetraining...")
    p_red.start()   # note
    time.sleep(10)  # note
    p_blue.start()  # note

    print("\n⏳ notetrainingnote...")
    try:
        p_blue.join()
        p_red.join()
    except KeyboardInterrupt:
        print("\n⚠️ notetraining...")
        for p in (p_blue, p_red):
            if p.is_alive():
                p.terminate()

    print("\nPASS notetrainingnote")

# ===== CLI =====

def parse_args():
    parser = argparse.ArgumentParser(description="notetraining")
    parser.add_argument("--steps", type=int, default=1000000, help="notetrainingnote")
    parser.add_argument("--port", type=int, default=9091, help="DonkeySimnote")
    parser.add_argument("--scene", type=str, default="waveshare", choices=["generated_track", "waveshare", "circuit_launch"], help="note/note")
    parser.add_argument("--save-root", type=str, default="models/quad_team_racing", help="savenotedirectoryfirstnote")
    parser.add_argument("--lr-blue", type=float, default=3e-4, help="note")
    parser.add_argument("--lr-red", type=float, default=3e-4, help="note")
    parser.add_argument("--max-cte", type=float, default=8.0, help="note")
    parser.add_argument("--save-freq", type=int, default=10000, help="notesavenote")
    parser.add_argument("--use-lidar", action="store_true", help="noteLiDAR")
    parser.add_argument("--lidar-port", type=str, default='/dev/ttyUSB0', help="noteLiDARnote")
    parser.add_argument("--action-noise", type=float, default=0.15, help="note")
    return parser.parse_args()

def main():
    args = parse_args()
    print("\n" + "="*70)
    print("🚗🚗🚗🚗 notetraining - note + note")
    print("="*70)
    print("⚠️ note DonkeySim note")
    print(f"\nmetrics trainingconfiguration:")
    print(f"   first50wnote: note(note + note)")
    print(f"   note50wnote: note(notereward)")
    print(f"   Blue Team: Blue1 + Blue2 (notemodel)")
    print(f"   Red Team: Red1 + Red2 (notemodel)")
    print(f"   rewardnote: notereward + note + note")
    print("="*70 + "\n")

    launch_quad_team_training(
        steps=args.steps,
        port=args.port,
        scene=args.scene,
        save_root=args.save_root,
        lr_blue=args.lr_blue,
        lr_red=args.lr_red,
        max_cte=args.max_cte,
        save_freq=args.save_freq,
        use_lidar=args.use_lidar,
        lidar_port=args.lidar_port,
        action_noise_std=args.action_noise,
    )

if __name__ == "__main__":
    main()
