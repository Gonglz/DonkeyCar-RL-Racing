#!/usr/bin/env python3
"""
🚗 V9 notetraining - noteV8note + note + note + NPCobstaclenote(notewavesharenote):
1. PASS note: noteenv + noteNPC + notequery nodes (stable)
2. PASS note: note set_position + node_position API notetracknote
3. PASS NPCobstaclenote: note2noteTCPnote, Pure Pursuitnote
4. PASS note+notereward: notetracknote(fine_track index)note, note
5. PASS note:
   - stage1: noteNPC, Learnernote
   - stage2: NPCnote(staticobstacle), notetracknote
   - stage3: NPCnote(0.12->0.20), note
   - stage4: NPCnoteLearnernote(NPCnote=Learnernote90%)

noteV8:
- 8note: RGB(3) + DiffRGB(3) + YellowMask(1) + Edges(1)
- HWC -> CHW -> note (RGB/Mask/Edges->[0,1], DiffRGB->[-1,1])
- ActionSafetyWrapper (note: slew-rate + LPF)
- ThrottleControlWrapper (note)
- noterewardnote (noteV8: noteCTE, note/notedetection, note, +notereward)

noteDonkeySim API:
- set_position: note = telemetrynote x 8 (COORD_SCALE=8), note0.0000m
- node_position: noteset_positionnote+note
- telemetrynote activeNode/totalNodes note
- noteTCPnote
- exit_scene + load_scene noterowsnote(notestable, noteenv)

NPCnote: Pure Pursuit, 30+notestable, note2.1, pathnote<0.1m

note:
  # note
  /home/glz/Car/DonkeySimLinux/donkey_sim.x86_64

  # noterowstraining
  python ppo_waveshare_v9_overtake.py --total-steps 600000

  # note
  python ppo_waveshare_v9_overtake.py --scenes generated_track,waveshare

  # note+note(wavesharenote)
  python ppo_waveshare_v9_overtake.py --scenes waveshare --num-npc 1
"""

import os
import sys
import time
import math
import json
import random
import threading
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import gym
import gym_donkeycar
import numpy as np
import cv2
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ==================== 1. DonkeySim note ====================
class SimExtendedAPI:
    """
    note DonkeySim note API:
    - set_position: note
    - node_position: notetrackpathnote: set_position/node_position note = telemetrynote x COORD_SCALE
    """

    COORD_SCALE = 8.0

    @staticmethod
    def send_set_position(handler, pos_x, pos_y, pos_z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """note (node_positionnote, = telemetry * 8)"""
        msg = {
            "msg_type": "set_position",
            "pos_x": str(pos_x),
            "pos_y": str(pos_y),
            "pos_z": str(pos_z),
            "Qx": str(qx),
            "Qy": str(qy),
            "Qz": str(qz),
            "Qw": str(qw),
        }
        handler.blocking_send(msg)
        time.sleep(0.05)

    @staticmethod
    def send_set_position_from_telemetry(handler, tel_x, tel_y, tel_z, qx=0, qy=0, qz=0, qw=1):
        """note telemetry note(note * COORD_SCALE note)"""
        S = SimExtendedAPI.COORD_SCALE
        SimExtendedAPI.send_set_position(handler, tel_x*S, tel_y*S, tel_z*S, qx, qy, qz, qw)

    @staticmethod
    def send_node_position_request(handler, index):
        """notetrackpathnote"""
        msg = {
            "msg_type": "node_position",
            "index": str(index),
        }
        handler.blocking_send(msg)

    @staticmethod
    def yaw_to_quaternion(yaw_degrees):
        """note(note)note (Unity Ynote)"""
        yaw_rad = math.radians(yaw_degrees)
        qx = 0.0
        qy = math.sin(yaw_rad / 2.0)
        qz = 0.0
        qw = math.cos(yaw_rad / 2.0)
        return qx, qy, qz, qw


# ==================== 2. tracknote ====================
class TrackNodeCache:
    """notetrackpathnote"""

    def __init__(self):
        self.nodes = {}        # {scene_name: [(x, y, z, qx, qy, qz, qw),...]}
        self.total_nodes = {}  # {scene_name: int}
        self.fine_track = {}   # {scene_name: [(x, z),...]} telemetrynote

        self._presets = {
            'waveshare':       {'total': 24},
            'generated_track': {'total': 108},
            'warehouse':       {'total': 50},
            'mini_monaco':     {'total': 50},
            'mountain_track':  {'total': 50},
        }

    def query_nodes(self, handler, scene_name, total_nodes=None):
        """
        notepathnote (node_positionnote = telemetry * 8)

        note: fns['node_position'] note finally note,
        note, finally noterowsnote.
        note: note handler.fns['node_position'], note.
        """
        if total_nodes is None:
            total_nodes = self._presets.get(scene_name, {}).get('total', 50)

        nodes = []
        print(f"📍 note {scene_name} notepathnote (note{total_nodes}note)...")

        received_nodes = {}

        def on_node_position(message):
            idx_raw = message.get('index', -1)
            try:
                idx = int(idx_raw)
            except (TypeError, ValueError):
                return
            if idx >= 0:
                received_nodes[idx] = {
                    'x': float(message.get('pos_x', 0)),
                    'y': float(message.get('pos_y', 0)),
                    'z': float(message.get('pos_z', 0)),
                    'qx': float(message.get('Qx', message.get('qx', 0))),
                    'qy': float(message.get('Qy', message.get('qy', 0))),
                    'qz': float(message.get('Qz', message.get('qz', 0))),
                    'qw': float(message.get('Qw', message.get('qw', 1))),
                }

        # note(note finally note), note
        handler.fns['node_position'] = on_node_position

        # note(note50ms)
        for i in range(total_nodes):
            SimExtendedAPI.send_node_position_request(handler, i)
            time.sleep(0.05)

        # note(waveshare 24note1.2s, generated_track 108note5s)
        wait_time = max(2.0, total_nodes * 0.06)
        print(f"   note {wait_time:.1f}s...")
        time.sleep(wait_time)

        # note
        if len(received_nodes) > 0:
            max_idx = max(received_nodes.keys())
            missing = [i for i in range(max_idx + 1) if i not in received_nodes]
            if missing:
                print(f"   note: {missing}")
                for i in missing:
                    SimExtendedAPI.send_node_position_request(handler, i)
                    time.sleep(0.12)
                time.sleep(2.0)

        # noteresult
        if len(received_nodes) > 0:
            for i in range(max(received_nodes.keys()) + 1):
                if i in received_nodes:
                    n = received_nodes[i]
                    nodes.append((n['x'], n['y'], n['z'], n['qx'], n['qy'], n['qz'], n['qw']))

            self.nodes[scene_name] = nodes
            self.total_nodes[scene_name] = len(nodes)
            print(f"PASS succeedednote {len(nodes)} note")

            self._build_fine_track(scene_name)

            S = SimExtendedAPI.COORD_SCALE
            xs = [n[0]/S for n in nodes]
            zs = [n[2]/S for n in nodes]
            print(f"   tracknote(telemetry): X=[{min(xs):.1f}, {max(xs):.1f}] Z=[{min(zs):.1f}, {max(zs):.1f}]")
            total_length = 0
            for i in range(len(nodes)):
                j = (i+1) % len(nodes)
                dx = (nodes[j][0]-nodes[i][0]) / S
                dz = (nodes[j][2]-nodes[i][2]) / S
                total_length += math.sqrt(dx**2 + dz**2)
            print(f"   tracknote(note): {total_length:.1f}m")
        else:
            print(f"⚠️ notedata, note")

        return nodes

    def _build_fine_track(self, scene_name, interp_factor=10):
        """notetracknote, notepathnote (telemetrynote)"""
        nodes = self.nodes.get(scene_name, [])
        if not nodes:
            return

        S = SimExtendedAPI.COORD_SCALE
        coarse = [(n[0]/S, n[2]/S) for n in nodes]

        fine = []
        n = len(coarse)
        for i in range(n):
            x0, z0 = coarse[i]
            x1, z1 = coarse[(i+1) % n]
            for j in range(interp_factor):
                t = j / interp_factor
                fine.append((x0 + t*(x1-x0), z0 + t*(z1-z0)))

        self.fine_track[scene_name] = fine
        print(f"   notetrack: {len(coarse)}note -> {len(fine)}note")

    def get_random_position(self, scene_name, exclude_range=None, exclude_set=None):
        """notetracknote (nodenote)

        exclude_set: note(noteCTEnote)
        """
        if scene_name not in self.nodes or len(self.nodes[scene_name]) == 0:
            return None

        nodes = self.nodes[scene_name]
        available_indices = list(range(len(nodes)))

        if exclude_range is not None:
            start, end = exclude_range
            available_indices = [i for i in available_indices if i < start or i > end]

        if exclude_set is not None:
            available_indices = [i for i in available_indices if i not in exclude_set]

        if not available_indices:
            available_indices = list(range(len(nodes)))

        idx = random.choice(available_indices)
        return nodes[idx], idx

    def get_position_ahead(self, scene_name, current_idx, offset=5):
        """notecurrentnotefirstnote offset note"""
        if scene_name not in self.nodes or len(self.nodes[scene_name]) == 0:
            return None, -1
        nodes = self.nodes[scene_name]
        target_idx = (current_idx + offset) % len(nodes)
        return nodes[target_idx], target_idx

    def find_nearest_node(self, scene_name, tel_x, tel_z):
        """note (inputnotetelemetrynote)"""
        if scene_name not in self.nodes:
            return 0
        S = SimExtendedAPI.COORD_SCALE
        nodes = self.nodes[scene_name]
        min_dist = float('inf')
        min_idx = 0
        for i, node in enumerate(nodes):
            dist = (node[0]/S - tel_x)**2 + (node[2]/S - tel_z)**2
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_idx

    def find_nearest_fine_track(self, scene_name, tel_x, tel_z):
        """notetracknote (telemetrynote)"""
        fine = self.fine_track.get(scene_name, [])
        if not fine:
            return 0, 999.0
        min_d = float('inf')
        best = 0
        for i, (fx, fz) in enumerate(fine):
            d = (tel_x - fx)**2 + (tel_z - fz)**2
            if d < min_d:
                min_d = d
                best = i
        return best, math.sqrt(min_d)

    def progress_diff(self, scene_name, idx_a, idx_b):
        """
        compute A note B notetracknote(note)

        note = A note B firstnote= A note B notefine_track note, note [-total/2, total/2] note
        """
        fine = self.fine_track.get(scene_name, [])
        if not fine:
            return 0
        total = len(fine)
        diff = (idx_a - idx_b) % total
        if diff > total // 2:
            diff -= total
        return diff


# ==================== 3. NPCcontrolnote ====================
class NPCController:
    """
    NPCobstaclenotecontrolnoteTCPnotecontrolNPCnote.
    NPCnote Pure Pursuit pathnotetrackrowsnote.
    """

    def __init__(self, npc_id, host='127.0.0.1', port=9091, scene='waveshare',
                 track_cache=None):
        self.npc_id = npc_id
        self.host = host
        self.port = port
        self.scene = scene
        self.track_cache = track_cache
        self.connected = False
        self.handler = None
        self.client = None

        # NPCrowsnote
        self.mode = 'static'  # 'static' | 'slow' | 'random' | 'chaos'
        self.base_throttle = 0.35
        self.running = False
        self._thread = None
        # random/chaos note
        self._rand_next_change_step = 0
        self._rand_steer_bias = 0.0
        self._rand_target_steer_bias = 0.0
        self._rand_throttle_cmd = 0.0
        self._rand_target_throttle = 0.0
        self._rand_reverse_steps_left = 0
        self._rand_reverse_cooldown = 0
        self._rand_hard_brake_steps_left = 0
        # note(noteNPC-NPCnote)
        self._force_brake_steps_left = 0
        # dynamicnote(sim speed).None note.
        self.speed_cap_sim = None

        # NPCnote (telemetrynote)
        self.tel_x = 0.0
        self.tel_y = 0.0
        self.tel_z = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.active_node = 0

        # NPC notefine_tracknote
        self.fine_track_idx = 0

        # telemetrynote
        self._raw_msgs = []

    def connect(self, body_rgb=(255, 100, 100)):
        """noteNPCnote"""
        try:
            from gym_donkeycar.core.sim_client import SimClient
            from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimHandler

            conf = {
                'host': self.host,
                'port': self.port,
                'log_level': 20,
                'max_cte': 100.0,
                'cam_resolution': (120, 160, 3),
                'cam_encode': 'JPG',
                'level': self.scene,
                'body_style': 'donkey',
                'body_rgb': body_rgb,
                'car_name': f'NPC_{self.npc_id}',
                'racer_name': f'npc{self.npc_id}',
                'country': 'US',
                'bio': '',
                'guid': f'npc_{self.npc_id}_{int(time.time())}',
                'font_size': 50,
            }

            self.handler = DonkeyUnitySimHandler(conf=conf)
            self.client = SimClient((self.host, self.port), self.handler)

            # notetelemetrynoteactiveNode
            orig_telem = self.handler.on_telemetry
            def hook_telem(msg):
                self._raw_msgs.append(msg)
                if len(self._raw_msgs) > 10:
                    self._raw_msgs.pop(0)
                orig_telem(msg)
            self.handler.fns['telemetry'] = hook_telem

            # note
            timeout = 15
            start = time.time()
            while not self.handler.loaded and time.time() - start < timeout:
                time.sleep(0.5)

            if self.handler.loaded:
                self.connected = True
                # NPCnote, notegymnoteover
                self.handler.determine_episode_over = lambda: None
                self.handler.over = False
                print(f"PASS NPC_{self.npc_id} note (note: {body_rgb})")
            else:
                print(f"⚠️ NPC_{self.npc_id} note")

            return self.connected
        except Exception as e:
            print(f"FAIL NPC_{self.npc_id} notefailed: {e}")
            return False

    def set_position_node_coords(self, x, y, z, qx=0, qy=0, qz=0, qw=1):
        """noteNPCnote (node_positionnote), notetelemetrynote"""
        if not self.connected:
            return
        SimExtendedAPI.send_set_position(self.handler, x, y, z, qx, qy, qz, qw)
        # notetelemetrynote
        S = SimExtendedAPI.COORD_SCALE
        self.tel_x = x / S
        self.tel_y = y / S
        self.tel_z = z / S

    def set_mode(self, mode, throttle=0.35):
        """noteNPCrowsnote"""
        m = str(mode).strip().lower()
        if m == 'slow_policy':
            m = 'slow'
        if m in ('random_reverse', 'chaos_reverse'):
            m = 'chaos'
        if m not in ('static', 'wobble', 'slow', 'random', 'chaos'):
            m = 'static'
        self.mode = m
        self.base_throttle = float(throttle)
        self._reset_random_state()

    def _reset_random_state(self):
        self._rand_next_change_step = 0
        self._rand_steer_bias = 0.0
        self._rand_target_steer_bias = 0.0
        self._rand_throttle_cmd = float(self.base_throttle)
        self._rand_target_throttle = float(self.base_throttle)
        self._rand_reverse_steps_left = 0
        self._rand_reverse_cooldown = 0
        self._rand_hard_brake_steps_left = 0

    def set_speed_cap(self, cap_sim=None):
        """noteNPCnote(m/s, telemetry speed)."""
        if cap_sim is None:
            self.speed_cap_sim = None
            return
        try:
            v = float(cap_sim)
        except Exception:
            self.speed_cap_sim = None
            return
        if not np.isfinite(v) or v <= 0.0:
            self.speed_cap_sim = None
            return
        self.speed_cap_sim = float(v)

    def set_emergency_brake(self, steps=6):
        """note.steps note drive_loop tick note."""
        try:
            n = int(steps)
        except Exception:
            n = 0
        if n <= 0:
            return
        self._force_brake_steps_left = max(int(self._force_brake_steps_left), n)

    def _apply_speed_cap(self, throttle, brake):
        """notedynamicnote, note/note."""
        cap = self.speed_cap_sim
        if cap is None:
            return float(throttle), float(brake)
        v_now = float(abs(self.speed))
        cap = max(0.05, float(cap))
        thr = float(throttle)
        brk = float(brake)

        if v_now > cap + 0.25:
            # note: note
            return min(thr, -0.18), max(brk, 0.0)
        if v_now > cap + 0.10:
            # note: note
            return min(thr, 0.0), max(brk, 1.0)
        if thr > 0.0 and v_now >= cap:
            # note: note
            return 0.0, max(brk, 1.0)
        return thr, brk

    def start_driving(self):
        """noteNPCnote (Pure Pursuit)"""
        if not self.connected or self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._drive_loop, daemon=True)
        self._thread.start()
        print(f"🚗 NPC_{self.npc_id} noterowsnote (note: {self.mode}, throttle: {self.base_throttle:.2f})")

    def stop_driving(self):
        """noteNPCnote"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self.connected:
            self.handler.send_control(0, 0, 1.0)

    def _pure_pursuit_steer(self, lookahead=0.5):
        """Pure Pursuit pathnote"""
        if not self.track_cache:
            return 0.0

        fine = self.track_cache.fine_track.get(self.scene, [])
        if not fine:
            cte = self.handler.cte
            return max(-1, min(1, -cte * 1.0))

        car_x, car_z = self.handler.x, self.handler.z
        car_yaw = self.handler.yaw

        nearest, _ = self.track_cache.find_nearest_fine_track(self.scene, car_x, car_z)

        total = len(fine)
        target_idx = nearest
        for offset in range(1, total):
            idx = (nearest + offset) % total
            tx, tz = fine[idx]
            d = math.sqrt((car_x - tx)**2 + (car_z - tz)**2)
            if d >= lookahead:
                target_idx = idx
                break

        tx, tz = fine[target_idx]
        dx = tx - car_x
        dz = tz - car_z

        target_angle = math.atan2(dx, dz) * 180.0 / math.pi
        angle_diff = target_angle - car_yaw
        while angle_diff > 180: angle_diff -= 360
        while angle_diff < -180: angle_diff += 360

        steer = angle_diff / 25.0
        return max(-1.0, min(1.0, steer))

    def _drive_loop(self):
        """NPCnote (Pure Pursuit)"""
        step = 0
        while self.running and self.connected:
            try:
                if self._force_brake_steps_left > 0:
                    self._force_brake_steps_left -= 1
                    self.handler.send_control(0, 0, 1.0)
                    time.sleep(0.05)
                    # noteNPCnote (telemetrynote from handler)
                    self.tel_x = self.handler.x
                    self.tel_y = self.handler.y
                    self.tel_z = self.handler.z
                    self.yaw = self.handler.yaw
                    self.speed = self.handler.speed
                    if self.track_cache:
                        self.fine_track_idx, _ = self.track_cache.find_nearest_fine_track(
                            self.scene, self.tel_x, self.tel_z)
                    if self._raw_msgs:
                        self.active_node = self._raw_msgs[-1].get('activeNode', self.active_node)
                    step += 1
                    continue

                if self.mode == 'static':
                    # noteNPCnote
                    self.handler.send_control(0, 0, 1.0)
                    time.sleep(0.2)
                elif self.mode == 'wobble':
                    # notefirstnote, notedynamicobstaclenote
                    phase = (step % 80) / 80.0  # 0~1 note 4 note
                    angle = phase * 2 * math.pi
                    steer = 0.25 * math.sin(angle)          # note
                    throttle = 0.08 * math.sin(angle * 2)   # firstnote(note)
                    brake = 1.0 if abs(throttle) < 0.02 else 0.0
                    self.handler.send_control(steer, throttle, brake)
                    time.sleep(0.05)
                elif self.mode == 'slow':
                    steer = self._pure_pursuit_steer(lookahead=0.5)
                    throttle = float(self.base_throttle)
                    brake = 0.0
                    throttle, brake = self._apply_speed_cap(throttle, brake)
                    self.handler.send_control(steer, throttle, brake)
                    time.sleep(0.05)
                elif self.mode == 'random':
                    # forward-only note+note
                    steer_pp = self._pure_pursuit_steer(lookahead=0.55)
                    if step >= self._rand_next_change_step:
                        self._rand_next_change_step = step + random.randint(16, 40)
                        self._rand_target_steer_bias = random.uniform(-0.18, 0.18)
                        # note, note
                        r = random.random()
                        if r < 0.62:
                            lo = 0.02
                            hi = max(0.08, self.base_throttle * 0.82)
                        elif r < 0.90:
                            lo = max(0.06, self.base_throttle * 0.65)
                            hi = max(lo, self.base_throttle * 1.00)
                        else:
                            lo = max(0.08, self.base_throttle * 0.95)
                            hi = min(0.55, self.base_throttle * 1.12)
                        if hi < lo:
                            hi = lo
                        self._rand_target_throttle = random.uniform(lo, hi)

                    self._rand_steer_bias = 0.82 * self._rand_steer_bias + 0.18 * self._rand_target_steer_bias
                    self._rand_throttle_cmd = 0.85 * self._rand_throttle_cmd + 0.15 * self._rand_target_throttle
                    steer = max(-1.0, min(1.0, steer_pp + self._rand_steer_bias))
                    throttle = max(0.0, min(0.65, self._rand_throttle_cmd))
                    brake = 1.0 if abs(throttle) < 0.02 else 0.0
                    throttle, brake = self._apply_speed_cap(throttle, brake)
                    self.handler.send_control(steer, throttle, brake)
                    time.sleep(0.05)
                elif self.mode == 'chaos':
                    # chaos: note + note + note
                    steer_pp = self._pure_pursuit_steer(lookahead=0.45)
                    if step >= self._rand_next_change_step:
                        self._rand_next_change_step = step + random.randint(12, 30)
                        self._rand_target_steer_bias = random.uniform(-0.32, 0.32)
                        # note: note, note, note
                        r = random.random()
                        if r < 0.60:
                            lo = 0.00
                            hi = max(0.10, self.base_throttle * 0.78)
                        elif r < 0.85:
                            lo = max(0.05, self.base_throttle * 0.60)
                            hi = max(lo, self.base_throttle * 0.95)
                        elif r < 0.95:
                            lo = max(0.08, self.base_throttle * 0.95)
                            hi = min(0.65, self.base_throttle * 1.10)
                        else:
                            lo = 0.0
                            hi = 0.03
                            self._rand_hard_brake_steps_left = random.randint(2, 6)
                        if hi < lo:
                            hi = lo
                        self._rand_target_throttle = random.uniform(lo, hi)

                    self._rand_steer_bias = 0.78 * self._rand_steer_bias + 0.22 * self._rand_target_steer_bias
                    self._rand_throttle_cmd = 0.80 * self._rand_throttle_cmd + 0.20 * self._rand_target_throttle

                    brake = 0.0
                    if self._rand_reverse_steps_left > 0:
                        self._rand_reverse_steps_left -= 1
                        throttle = -random.uniform(0.10, 0.22)
                        if self._rand_reverse_steps_left == 0:
                            self._rand_reverse_cooldown = random.randint(90, 220)
                    elif self._rand_hard_brake_steps_left > 0:
                        self._rand_hard_brake_steps_left -= 1
                        throttle = 0.0
                        brake = 1.0
                    else:
                        if self._rand_reverse_cooldown > 0:
                            self._rand_reverse_cooldown -= 1
                        elif abs(float(self.speed)) < 0.9 and random.random() < 0.004:
                            # note
                            self._rand_reverse_steps_left = random.randint(4, 10)
                        throttle = self._rand_throttle_cmd

                    steer = steer_pp + self._rand_steer_bias
                    if self._rand_reverse_steps_left > 0:
                        steer += random.uniform(-0.20, 0.20)
                    steer = max(-1.0, min(1.0, steer))
                    throttle = max(-0.35, min(0.65, throttle))
                    if abs(throttle) < 0.02:
                        brake = max(brake, 1.0)
                    throttle, brake = self._apply_speed_cap(throttle, brake)
                    self.handler.send_control(steer, throttle, brake)
                    time.sleep(0.05)
                else:
                    time.sleep(0.1)
                    continue

                # noteNPCnote (telemetrynote from handler)
                self.tel_x = self.handler.x
                self.tel_y = self.handler.y
                self.tel_z = self.handler.z
                self.yaw = self.handler.yaw
                self.speed = self.handler.speed

                # notefine_tracknote
                if self.track_cache:
                    self.fine_track_idx, _ = self.track_cache.find_nearest_fine_track(
                        self.scene, self.tel_x, self.tel_z)

                # noteactiveNode
                if self._raw_msgs:
                    self.active_node = self._raw_msgs[-1].get('activeNode', self.active_node)

                step += 1
            except Exception:
                time.sleep(0.1)

    def get_telemetry_position(self):
        """noteNPCcurrentnote (telemetrynote)"""
        return (self.tel_x, self.tel_y, self.tel_z)

    def close(self):
        """noteNPCnote"""
        self.stop_driving()
        if self.client:
            try:
                self.client.stop()
            except:
                pass
        self.connected = False
        print(f"🔒 NPC_{self.npc_id} note")


# ==================== 4. note ====================
class MultiMapManager:
    """
    note: note env + note NPC + note query nodes
    """

    MAP_CONFIGS = {
        'waveshare': {
            'env_id': 'donkey-waveshare-v0',
            'level': 'waveshare',
            'max_cte': 5.0,
            'description': 'notetrack(24note, 66m)',
            'difficulty': 1,
            'estimated_nodes': 24,
        },
        'generated_track': {
            'env_id': 'donkey-generated-track-v0',
            'level': 'generated_track',
            'max_cte': 8.0,
            'description': 'notegeneratetrack(108note, 215m)',
            'difficulty': 2,
            'estimated_nodes': 108,
        },
        'warehouse': {
            'env_id': 'donkey-warehouse-v0',
            'level': 'warehouse',
            'max_cte': 8.0,
            'description': 'note(note)',
            'difficulty': 2,
            'estimated_nodes': 30,
        },
        'mini_monaco': {
            'env_id': 'donkey-minimonaco-track-v0',
            'level': 'mini_monaco',
            'max_cte': 6.0,
            'description': 'notetrack(note)',
            'difficulty': 3,
            'estimated_nodes': 35,
        },
        'mountain_track': {
            'env_id': 'donkey-mountain-track-v0',
            'level': 'mountain_track',
            'max_cte': 8.0,
            'description': 'notetrack(note)',
            'difficulty': 3,
            'estimated_nodes': 40,
        },
    }

    def __init__(self, scene_names, switch_interval=50000):
        self.scenes = [s for s in scene_names if s in self.MAP_CONFIGS]
        if not self.scenes:
            self.scenes = ['waveshare']
        self.switch_interval = switch_interval
        self.current_scene_idx = 0
        self.steps_on_current = 0

        print(f"🗺️ note:")
        for s in self.scenes:
            cfg = self.MAP_CONFIGS[s]
            print(f"   PASS {s}: {cfg['description']}")
        print(f"   note: note {switch_interval:,} note")

    @property
    def current_scene(self):
        return self.scenes[self.current_scene_idx]

    @property
    def current_config(self):
        return self.MAP_CONFIGS[self.current_scene]

    def should_switch(self, global_step):
        """note"""
        if len(self.scenes) <= 1:
            return False
        self.steps_on_current += 1
        if self.steps_on_current >= self.switch_interval:
            self.steps_on_current = 0
            return True
        return False

    def next_scene(self):
        """note"""
        old = self.current_scene
        self.current_scene_idx = (self.current_scene_idx + 1) % len(self.scenes)
        self.steps_on_current = 0
        new = self.current_scene
        print(f"\n🗺️ note: {old} -> {new}")
        return new


# ==================== 5. note (noteV8) ====================
class YellowLaneEnhancer:
    """note (noteV8, noteDRnoteCLAHE)"""

    def __init__(self, enable_dr=False, dr_prob=0.6):
        self.yellow_lower = np.array([15, 60, 60])
        self.yellow_upper = np.array([40, 255, 255])
        self.enable_dr = enable_dr
        self.dr_prob = dr_prob

    def _dr_brightness_contrast(self, rgb):
        rgb = rgb.astype(np.float32)
        if random.random() < 0.5:
            b = random.uniform(-30, 30)
            rgb = np.clip(rgb + b, 0, 255)
        if random.random() < 0.5:
            c = random.uniform(0.75, 1.25)
            rgb = np.clip(rgb * c, 0, 255)
        return rgb.astype(np.uint8)

    def _dr_blur(self, rgb):
        if random.random() < 0.3:
            k = random.choice([1, 3, 5])
            if k > 1:
                rgb = cv2.GaussianBlur(rgb, (k, k), 0)
        return rgb

    def _dr_noise(self, rgb):
        if random.random() < 0.25:
            sigma = random.uniform(3, 12)
            noise = np.random.normal(0, sigma, rgb.shape)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return rgb

    def _dr_hsv(self, rgb):
        if random.random() < 0.4:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:, 0] = np.clip(hsv[:,:, 0] + random.uniform(-8, 8), 0, 179)
            hsv[:,:, 1] = np.clip(hsv[:,:, 1] * random.uniform(0.8, 1.2), 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb

    def enhance(self, rgb, apply_dr=False):
        """
        note (rgb_processed, yellow_mask, edges)
        DRnoteRGB, mask/edgesnotecurrentRGBnotecompute
        """
        if apply_dr and self.enable_dr and random.random() < self.dr_prob:
            rgb = self._dr_brightness_contrast(rgb)
            rgb = self._dr_blur(rgb)
            rgb = self._dr_noise(rgb)
            rgb = self._dr_hsv(rgb)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        edges = cv2.Canny(gray_enhanced, 40, 120)

        return rgb, yellow_mask, edges


# ==================== 6. V9 notetrainingnote ====================
class OvertakeTrainingWrapper(gym.Wrapper):
    """
    V9 notetrainingnote (noteV8 8note)

    note:
    1. RGB(3) + DiffRGB(3) + YellowMask(1) + Edges(1) = 8note
    2. HWC -> CHW -> note float32

    note:
    - stage1noteNPC: resetnotestagenoteNPC
    - notetracknote(fine_track index)note
    - NPCnoteunifiednotetelemetrynotecompute
    """

    def __init__(self, env,
                 npc_controllers=None,
                 track_cache=None,
                 scene_name='waveshare',
                 use_random_start=True,
                 max_episode_steps=600,
                 curriculum_stage_ref=None,
                 target_size=(120, 160),
                 enable_dr=True,
                 max_throttle=0.3,
                 delta_max=0.10,
                 enable_lpf=True,
                 beta=0.6,
                 w_d=0.25, w_dd=0.08, w_sat=0.15):
        super().__init__(env)

        self.npc_controllers = npc_controllers or []
        self.track_cache = track_cache
        self.scene_name = scene_name
        self.use_random_start = use_random_start
        self.max_episode_steps = max_episode_steps
        self.curriculum_stage_ref = curriculum_stage_ref or {'stage': 1, 'npc_active': False}
        self.target_size = target_size
        self.max_throttle = max_throttle

        # note (noteV8)
        self.enhancer = YellowLaneEnhancer(enable_dr=enable_dr, dr_prob=0.6)
        self.prev_rgb = None

        # ActionSafety (noteV8)
        self.delta_max = delta_max
        self.enable_lpf = enable_lpf
        self.beta = beta
        self.steer_prev_limited = 0.0
        self.steer_prev_exec = 0.0
        self.delta_steer_prev = 0.0

        # note
        self.w_d = w_d
        self.w_dd = w_dd
        self.w_sat = w_sat

        # note
        self.episode_step = 0
        self.last_active_node = 0
        self.nodes_passed = 0
        self.speed_history = deque(maxlen=50)
        self.stuck_counter = 0
        self.offtrack_counter = 0

        # note (tracknote)
        self.learner_fine_idx = 0
        self.overtake_count = 0
        self._npc_overtake_state = {}

        # noteCTE (noteV8)
        self.max_cte_initial = 5.0
        self.current_max_cte = 5.0
        self.max_cte_min = 2.0
        self.recent_episodes = deque(maxlen=30)
        self.total_train_steps = 0
        self.total_episodes = 0
        self.last_cte_adjust_step = 0

        self.episode_stats = {
            'cte_sum': 0, 'steps': 0, 'max_speed': 0,
            'collision': False, 'total_reward': 0,
        }

        # note: 8note CHW float32
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(8, target_size[0], target_size[1]),
            dtype=np.float32
        )

        print(f"PASS V9 notetrainingnote (note: {scene_name})")
        print(f"   note: 8note CHW float32 (noteV8)")
        print(f"   NPCnote: {len(self.npc_controllers)}")
        print(f"   note: {'note' if use_random_start else 'note'}")

        # === note: note gym-donkeycar note ===
        # gym note determine_episode_over() note telemetry note/CTE->over=True->done=True
        # note wrapper note, note
        # notefunction, noteunifiednote wrapper.step() control
        try:
            handler = self.env.viewer.handler
            handler.determine_episode_over = lambda: None
            print(f"   PASS notegymnote(notewrapperunifiedcontrol)")
        except Exception as e:
            print(f"   ⚠️ notegymnotefunction: {e}")

    def reset(self, **kwargs):
        # note
        if self.episode_stats['steps'] > 0:
            avg_cte = self.episode_stats['cte_sum'] / max(1, self.episode_stats['steps'])
            term_reason = self.episode_stats.get('termination_reason', 'max_steps')
            self.recent_episodes.append({
                'avg_cte': avg_cte,
                'collision': self.episode_stats['collision'],
                'steps': self.episode_stats['steps'],
                'reward': self.episode_stats['total_reward'],
            })
            # noteCTEnote (noteV8)
            self._evaluate_and_adjust_cte()

            self.total_episodes += 1

            # === note ===
            stage = self.curriculum_stage_ref.get('stage', 1)
            stage_names = {1: 'S1note', 2: 'S2note', 3: 'S3note', 4: 'S4note'}
            print(f"[{stage_names.get(stage,'?')}] note{self.total_train_steps:,} | "
                  f"note{self.total_episodes:,}: "
                  f"{self.episode_stats['steps']}note "
                  f"R={self.episode_stats['total_reward']:.1f} "
                  f"CTE={avg_cte:.2f} "
                  f"v_max={self.episode_stats['max_speed']:.1f} "
                  f"note={term_reason}")

            # note20note
            if self.total_episodes % 20 == 0:
                last20 = list(self.recent_episodes)[-20:]
                avg_r = np.mean([e['reward'] for e in last20])
                avg_s = np.mean([e['steps'] for e in last20])
                avg_c = np.mean([e['collision'] for e in last20])
                avg_ct = np.mean([e['avg_cte'] for e in last20])
                print(f"  metrics note20note: noteR={avg_r:.1f} note={avg_s:.0f} "
                      f"note={avg_c:.0%} CTE={avg_ct:.2f} | CTEnote={self.current_max_cte:.2f}")

        # ===== note1note: env note reset(notedefaultnote)=====
        obs = self.env.reset(**kwargs)
        time.sleep(0.2)  # notereset, note, notestable

        # === notestagenote ===
        npc_active = self.curriculum_stage_ref.get('npc_active', False)
        npc_costart = self.curriculum_stage_ref.get('npc_costart', False)
        npc_random_pos = self.curriculum_stage_ref.get('npc_random_pos', False)
        current_stage = self.curriculum_stage_ref.get('stage', 1)

        if npc_costart:
            # stage4: Learnernote, NPCnote
            self.last_active_node = 0
            self._place_npcs_at_start()

        elif self.use_random_start and self.track_cache:
            # stage1/2/3: Learnernotestart, NPCnotestagenote
            # _randomize_start_position noteteleport+noteCTE(notestep+sleep)
            self._randomize_start_position()

            if npc_active and npc_random_pos:
                # stage2/3: NPCnotecurrentLearnernote(last_active_node)note
                self._randomize_npc_positions()
                time.sleep(0.1)  # noteNPCnotestable
            elif not npc_active:
                # stage1: noteNPCnoteNPCnote
                self._park_npcs_offtrack_lazy()
        else:
            # note
            self.last_active_node = 0
            if not npc_active:
                self._park_npcs_offtrack_lazy()

        # ===== note2note: notestablenote, noteobs =====
        time.sleep(0.15)
        # note env.step(action=0) noteobsnotetelemetry
        # action=0 note, note, note, note
        try:
            obs, _, _, _ = self.env.step(np.array([0.0, 0.0], dtype=np.float32))
        except Exception as e:
            print(f"⚠️ resetnotestepnote: {e}")

        # note
        self.episode_step = 0
        self.nodes_passed = 0
        self.speed_history.clear()
        self.stuck_counter = 0
        self.offtrack_counter = 0
        self.learner_fine_idx = 0
        self._npc_overtake_state = {}

        # noteActionSafety
        self.steer_prev_limited = 0.0
        self.steer_prev_exec = 0.0
        self.delta_steer_prev = 0.0

        # note
        self.prev_rgb = None

        self.episode_stats = {
            'cte_sum': 0, 'steps': 0, 'max_speed': 0,
            'collision': False, 'total_reward': 0,
            'termination_reason': 'max_steps',
        }

        return self._process_observation(obs)

    def _evaluate_and_adjust_cte(self):
        """noteCTEnote (noteV8)"""
        if len(self.recent_episodes) < 10:
            return
        if self.total_train_steps - self.last_cte_adjust_step < 10000:
            return

        avg_cte = np.mean([ep['avg_cte'] for ep in self.recent_episodes])

        target_cte = self.max_cte_initial
        if self.total_train_steps >= 150000:
            target_cte = 1.0
        elif self.total_train_steps >= 100000:
            target_cte = 2.0
        elif self.total_train_steps >= 50000:
            target_cte = 3.5

        can_tighten = False
        if avg_cte < 0.6 * self.current_max_cte:
            can_tighten = True
        elif target_cte < self.current_max_cte:
            can_tighten = True

        if can_tighten and self.current_max_cte > self.max_cte_min:
            old_cte = self.current_max_cte
            new_cte = max(target_cte, self.current_max_cte * 0.8)
            self.current_max_cte = max(self.max_cte_min, new_cte)
            self.last_cte_adjust_step = self.total_train_steps
            print(f"\n🎯 CTEnote: {old_cte:.2f} -> {self.current_max_cte:.2f} (note={self.total_train_steps:,})")

    def _randomize_start_position(self):
        """notetracknote(noteenv.reset()noteset_positionnote)

        teleportnoteCTE, noteCTEnote(descriptionnote/notetracknote)note, note
        """
        MAX_RETRIES = 4
        CTE_THRESHOLD = 6.0  # teleportnoteCTEnote

        # note(note, noteresetnote)
        if not hasattr(self, '_bad_nodes'):
            self._bad_nodes = set()

        tried_indices = set()
        handler = self.env.viewer.handler
        last_result = None

        for attempt in range(MAX_RETRIES):
            try:
                result = self.track_cache.get_random_position(
                    self.scene_name, exclude_set=self._bad_nodes | tried_indices)
                if result is None:
                    # note, note
                    result = self.track_cache.get_random_position(
                        self.scene_name, exclude_set=tried_indices)
                if result is None:
                    break
                node, idx = result
                last_result = result
                tried_indices.add(idx)

                # teleport
                SimExtendedAPI.send_set_position(handler,
                    node[0], node[1], node[2], node[3], node[4], node[5], node[6])
                time.sleep(0.15)

                # noteteleportnoteCTE
                try:
                    _, _, _, step_info = self.env.step(np.array([0.0, 0.0], dtype=np.float32))
                    cte_after = abs(step_info.get('cte', 0))
                    # notegymnoteovernote
                    try:
                        handler.over = False
                    except Exception:
                        pass
                except Exception:
                    cte_after = 0  # note

                if cte_after < CTE_THRESHOLD:
                    self.last_active_node = idx
                    return  # succeeded
                else:
                    # note
                    self._bad_nodes.add(idx)
                    print(f"  ⚠️ teleportnotenode{idx} CTE={cte_after:.1f}>note{CTE_THRESHOLD}, "
                          f"note(note{len(self._bad_nodes)}note), note({attempt+1}/{MAX_RETRIES})")

            except Exception as e:
                print(f"⚠️ _randomize_start_position note: {e}")
                break

        # notefailed, note
        if last_result is not None:
            _, idx = last_result
            self.last_active_node = idx
            print(f"  ⚠️ teleportnotefailed, notenode{idx} (CTEnote)")

    def _randomize_npc_positions(self):
        """noteNPCnoteLearnerfirstnote (stage2/3note)"""
        if not self.npc_controllers or not self.track_cache:
            return
        try:
            nodes = self.track_cache.nodes.get(self.scene_name, [])
            if len(nodes) < 3:
                print(f"⚠️ note({len(nodes)}note), noteNPCnote")
                return

            # note: notecurrentLearnernotenodenote(resetnote)
            learner_node = self.last_active_node
            num_nodes = len(nodes)
            current_stage = self.curriculum_stage_ref.get('stage', 2)

            for i, npc in enumerate(self.npc_controllers):
                if not npc.connected:
                    continue

                # noteNPCnote(noteresetnote)
                if not npc.running:
                    npc.start_driving()

                # NPCnoteLearnerfirstnote5-10note(note), noteNPCnote3note
                offset = random.randint(5, min(10, max(5, num_nodes // 3)))
                npc_node_idx = (learner_node + offset + i * 3) % num_nodes
                node = nodes[npc_node_idx]

                # Debug log
                if i == 0:  # noteNPCnote, note
                    print(f"  🚗 [NPCnote] Learner=node{learner_node} -> NPC=node{npc_node_idx} (offset={offset})")

                npc.set_position_node_coords(
                    node[0], node[1], node[2], node[3], node[4], node[5], node[6])
                time.sleep(0.2)  # noteNPC teleportnote

                # stage2: teleportnote, note_drive_loopnote
                if current_stage == 2:
                    npc.handler.send_control(0, 0, 1.0)  # steer=0, throttle=0, brake=1
                    time.sleep(0.1)
                    npc.handler.send_control(0, 0, 1.0)  # note

                # noteNPC fine_tracknote
                npc.fine_track_idx, _ = self.track_cache.find_nearest_fine_track(
                    self.scene_name, npc.tel_x, npc.tel_z)
        except Exception as e:
            print(f"⚠️ _randomize_npc_positions note: {e}")
            import traceback
            traceback.print_exc()

    def _place_npcs_at_start(self):
        """
        stage4note: noteNPCnote, noteLearnernote

        NPCnoteLearnernote1-2note(note, note)
        NPCnotePure Pursuitnote
        """
        if not self.npc_controllers or not self.track_cache:
            return
        try:
            nodes = self.track_cache.nodes.get(self.scene_name, [])
            if len(nodes) < 3:
                return
            num_nodes = len(nodes)
            learner_node = self.last_active_node  # stage4note0(defaultnote)

            for i, npc in enumerate(self.npc_controllers):
                if not npc.connected:
                    continue
                # NPCnoteLearnernote1note(notefirstnote)
                npc_node_idx = (learner_node - 1 - i) % num_nodes
                node = nodes[npc_node_idx]
                npc.set_position_node_coords(
                    node[0], node[1], node[2], node[3], node[4], node[5], node[6])
                time.sleep(0.2)
                npc.fine_track_idx, _ = self.track_cache.find_nearest_fine_track(
                    self.scene_name, npc.tel_x, npc.tel_z)
                # noteNPCnote
                if not npc.running:
                    npc.start_driving()
        except Exception as e:
            print(f"⚠️ _place_npcs_at_start note: {e}")

    def _park_npcs_offtrack(self):
        """noteNPCnotetracknote (note, note)"""
        for npc in self.npc_controllers:
            if npc.connected:
                npc.set_position_node_coords(0, 100.0, 0, 0, 0, 0, 1)
                if npc.running:
                    npc.stop_driving()

    def _park_npcs_offtrack_lazy(self):
        """
        noteNPCnotetracknote (noterowsnote)
        noteNPCnote,
        noteresetnoteteleportnote.
        """
        for npc in self.npc_controllers:
            if not npc.connected:
                continue
            # noteNPCnotetel_ynote(ynote12.5, note100/8=12.5)
            # note
            if npc.running:
                npc.stop_driving()
            if abs(npc.tel_y - 12.5) > 1.0:  # note
                npc.set_position_node_coords(0, 100.0, 0, 0, 0, 0, 1)

    def step(self, action):
        self.episode_step += 1
        self.total_train_steps += 1

        # === ActionSafety: note (noteV8) ===
        steer_raw = float(action[0])
        throttle_raw = float(action[1])

        # Slew-rate limit
        delta = steer_raw - self.steer_prev_limited
        rate_limit_hit = abs(delta) > self.delta_max
        rate_excess_raw = max(0.0, abs(delta) - self.delta_max) / max(self.delta_max, 1e-6)
        if abs(delta) > self.delta_max:
            delta = np.clip(delta, -self.delta_max, self.delta_max)
        steer_limited = self.steer_prev_limited + delta

        if self.enable_lpf:
            steer_exec = (1 - self.beta) * self.steer_prev_exec + self.beta * steer_limited
        else:
            steer_exec = steer_limited
        steer_exec = np.clip(steer_exec, -1.0, 1.0)

        actual_delta = steer_exec - self.steer_prev_exec
        rate_excess_bounded = float(np.tanh(rate_excess_raw))

        prev_delta_steer = self.delta_steer_prev
        self.delta_steer_prev = actual_delta
        self.steer_prev_limited = steer_limited
        self.steer_prev_exec = steer_exec

        # ThrottleControl
        throttle_exec = min(throttle_raw, self.max_throttle)

        safe_action = np.array([steer_exec, throttle_exec], dtype=np.float32)

        # noterowsnote
        obs, base_reward, gym_done, info = self.env.step(safe_action)

        # === note gym note done ===
        # gym note determine_episode_over note
        # noteunifiednote wrapper control
        done = False

        # note gym note over=True, note
        if gym_done:
            try:
                self.env.viewer.handler.over = False
            except Exception:
                pass

        # note
        processed_obs = self._process_observation(obs)

        # === computenotereward ===
        cte = abs(info.get('cte', 0))
        speed = info.get('speed', 0)
        hit = info.get('hit', 'none')
        lap_count = info.get('lap_count', 0)
        self.speed_history.append(speed)
        self.episode_stats['cte_sum'] += cte
        self.episode_stats['max_speed'] = max(self.episode_stats['max_speed'], speed)

        ontrack = float(cte <= self.current_max_cte)
        reward = 0.0

        # notereward
        reward += 0.2

        # notereward
        v_normalized = np.clip(speed, 0.0, 4.0) / 4.0
        reward += ontrack * (1.0 * v_normalized)
        if ontrack and cte < 0.6 * self.current_max_cte and v_normalized > 0.3:
            reward += 0.2

        # CTE
        if cte > self.current_max_cte:
            exceed_ratio = (cte - self.current_max_cte) / max(1e-6, self.current_max_cte)
            reward += -(1.0 + 4.0 * exceed_ratio)
        else:
            reward += 0.2 * (1.0 - cte / max(1e-6, self.current_max_cte))

        # note - teleport notefirst20note(notestable)
        if hit!= 'none' and self.episode_step > 20:
            reward -= 6.0
            self.episode_stats['collision'] = True
            done = True
            info['termination_reason'] = 'collision'
            self.episode_stats['termination_reason'] = 'collision'

        # notereward (note)
        # (gym-donkeycarnoteinfonotelap_count)

        # notedetection - teleportnotefirst20note(note)
        if self.episode_step > 20 and ontrack and speed < 0.3:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > 50:
            done = True
            reward -= 2.0
            info['termination_reason'] = 'stuck'
            self.episode_stats['termination_reason'] = 'stuck'

        # note - teleportnotefirst25note(teleportnoteCTEnotestablenote)
        if self.episode_step > 25 and not ontrack:
            self.offtrack_counter += 1
        else:
            self.offtrack_counter = 0
        if self.offtrack_counter > 25:
            done = True
            reward -= 4.0
            info['termination_reason'] = 'persistent_offtrack'
            self.episode_stats['termination_reason'] = 'persistent_offtrack'

        # note (noteV8)
        abs_delta = abs(actual_delta)
        abs_jerk = abs(actual_delta - prev_delta_steer)
        reward += -self.w_d * abs_delta
        reward += -self.w_dd * abs_jerk
        reward += -self.w_sat * rate_excess_bounded

        # notereward
        active_node = info.get('activeNode', self.last_active_node)
        if isinstance(active_node, str):
            try:
                active_node = int(active_node)
            except:
                active_node = self.last_active_node
        if active_node!= self.last_active_node:
            self.nodes_passed += 1
            reward += 0.2
            self.last_active_node = active_node

        # *** NPCnotereward ***
        npc_active = self.curriculum_stage_ref.get('npc_active', False)
        current_stage = self.curriculum_stage_ref.get('stage', 1)
        if npc_active:
            if current_stage == 2:
                # stage2: notereward(NPCnote, note)
                reward, done = self._compute_avoidance_reward(reward, done, info)
            else:
                # stage3/4: notereward(note+note)
                reward, done = self._compute_overtake_reward(reward, done, info)

        # note
        if self.episode_step >= self.max_episode_steps:
            done = True
            self.episode_stats['termination_reason'] = 'max_steps'

        self.episode_stats['total_reward'] += reward
        self.episode_stats['steps'] += 1

        return processed_obs, reward, done, info

    def _compute_avoidance_reward(self, reward, done, info):
        """
        stage2note: notereward(NPCnote)

        notereward:
        - noteNPCnote
        - noteNPC(note)
        - noteNPCnote(notedetectionnote)
        """
        if not self.npc_controllers:
            return reward, done

        learner_pos = info.get('pos', (0, 0, 0))
        if isinstance(learner_pos, (tuple, list)):
            lx, ly, lz = learner_pos
        else:
            lx, ly, lz = 0, 0, 0

        for npc in self.npc_controllers:
            if not npc.connected:
                continue
            nx, ny, nz = npc.get_telemetry_position()
            dist = math.sqrt((lx - nx)**2 + (lz - nz)**2)

            # noteNPCnotereward
            if dist < 4.0 and len(self.speed_history) > 5:
                avg_speed = np.mean(list(self.speed_history)[-10:])
                if avg_speed < 2.5:
                    reward += 0.2

            # noteNPC(note)
            if 0.8 < dist < 2.0:
                reward += 0.3

        return reward, done

    def _compute_overtake_reward(self, reward, done, info):
        """
        notetracknote(stage3/4note)

        note:
        1. computeLearnernoteNPCnotefine_tracknote
        2. note = learner_idx - npc_idx (note = learnernotefirst)
        3. LearnernoteNPCnote -> NPCfirstnote, noteNnote -> notesucceeded
        """
        if not self.track_cache or not self.npc_controllers:
            return reward, done

        # Learnernotetelemetrynote
        learner_pos = info.get('pos', (0, 0, 0))
        if isinstance(learner_pos, (tuple, list)):
            lx, ly, lz = learner_pos
        else:
            lx, ly, lz = 0, 0, 0

        self.learner_fine_idx, _ = self.track_cache.find_nearest_fine_track(
            self.scene_name, lx, lz)

        for npc in self.npc_controllers:
            if not npc.connected:
                continue

            npc_id = npc.npc_id
            if npc_id not in self._npc_overtake_state:
                self._npc_overtake_state[npc_id] = {
                    'behind_count': 0,
                    'ahead_count': 0,
                    'was_behind': False,
                    'overtake_cooldown': 0,
                }
            state = self._npc_overtake_state[npc_id]

            # note
            if state['overtake_cooldown'] > 0:
                state['overtake_cooldown'] -= 1
                continue

            npc_fine_idx = npc.fine_track_idx

            # tracknote (note = Learnernotefirst)
            progress_diff = self.track_cache.progress_diff(
                self.scene_name, self.learner_fine_idx, npc_fine_idx)

            # NPCnote (telemetrynote, unifiednote)
            nx, ny, nz = npc.get_telemetry_position()
            dist = math.sqrt((lx - nx)**2 + (lz - nz)**2)

            # noteNPCnotereward
            if dist < 4.0 and len(self.speed_history) > 5:
                avg_speed = np.mean(list(self.speed_history)[-10:])
                if avg_speed < 2.5:
                    reward += 0.2

            # note
            if 0.8 < dist < 2.0:
                reward += 0.3

            # note (waveshare 240 fine_tracknote, note)
            # note: LearnernoteNPCnote20notefinenote(note2notenodenote)
            #        noteNPCfirstnote20notefinenote, note15note
            BEHIND_THRESHOLD = -20    # noteNPCnote
            AHEAD_THRESHOLD = 20      # noteNPCfirstnote
            BEHIND_CONFIRM = 10       # note10note"note"
            AHEAD_CONFIRM = 15        # note15note"notesucceeded"

            if progress_diff < BEHIND_THRESHOLD:
                state['behind_count'] += 1
                state['ahead_count'] = 0
                if state['behind_count'] >= BEHIND_CONFIRM:
                    state['was_behind'] = True
            elif progress_diff > AHEAD_THRESHOLD:
                if state['was_behind']:
                    state['ahead_count'] += 1
                    if state['ahead_count'] >= AHEAD_CONFIRM:
                        self.overtake_count += 1
                        reward += 10.0
                        state['was_behind'] = False
                        state['behind_count'] = 0
                        state['ahead_count'] = 0
                        state['overtake_cooldown'] = 150  # note150note
                        info['overtake_success'] = True
                        info['total_overtakes'] = self.overtake_count
                        print(f"🏎️ notesucceeded! note{self.overtake_count}note "
                              f"(note: {self.episode_step}, note: {progress_diff})")
            else:
                state['ahead_count'] = 0

        return reward, done

    def _process_observation(self, obs):
        """
        note -> 8note CHW float32 (noteV8)

        note0-2: RGB [0,1]
        note3-5: DiffRGB [-1,1]
        note6:   YellowMask [0,1]
        note7:   Edges [0,1]
        """
        if isinstance(obs, dict):
            image = obs.get('image', obs.get('cam', obs))
        else:
            image = obs

        image = cv2.resize(image, (self.target_size[1], self.target_size[0]),
                           interpolation=cv2.INTER_LINEAR)

        rgb, yellow_mask, edges = self.enhancer.enhance(image, apply_dr=True)

        # note
        if self.prev_rgb is not None:
            diff_rgb = rgb.astype(np.float32) - self.prev_rgb.astype(np.float32)
            diff_rgb = np.clip(diff_rgb, -255, 255)
        else:
            diff_rgb = np.zeros_like(rgb, dtype=np.float32)

        self.prev_rgb = rgb.copy()

        # CHW + note
        rgb_chw = np.transpose(rgb.astype(np.float32), (2, 0, 1)) / 255.0
        diff_chw = np.transpose(diff_rgb, (2, 0, 1)) / 255.0
        mask_chw = yellow_mask.astype(np.float32)[np.newaxis,:,:] / 255.0
        edges_chw = edges.astype(np.float32)[np.newaxis,:,:] / 255.0

        obs_8ch = np.concatenate([rgb_chw, diff_chw, mask_chw, edges_chw], axis=0)
        return obs_8ch.astype(np.float32)

    def close(self):
        super().close()


# ==================== 7. CNN (noteV8, dynamicnote) ====================
class LightweightCNN(BaseFeaturesExtractor):
    """
    noteCNN - noteV8
    noteobservation_space.shape[0]notereadinputnote: N_ch -> 32 -> 64 -> 64 -> features_dim
    """

    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        print(f"\n🧠 V9 CNNnote (noteV8):")
        print(f"   input: {n_input_channels}note x {observation_space.shape[1]}x{observation_space.shape[2]}")

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"   note: {n_flatten}")
        print(f"   note: {features_dim}")
        print(f"   note: {total_params:,}")

    def forward(self, observations):
        return self.linear(self.cnn(observations))


# ==================== 8. note ====================
class OvertakeCurriculumCallback(BaseCallback):
    """
    notetrainingnote(note)

    stagenote:
    1. 0-100k:   note - NPCnotetrack, Learnernote
    2. 100k-200k: staticobstacle - NPCnote(throttle=0), notetracknoteobstaclenote
                             Learnernote, noteobstacle
    3. 200k-400k: noteNPC - NPCnote(throttlenote0.12note0.20)
                             NPCnote, Learnernote+note
    4. 400k-600k: note - NPCnoteLearnernote(throttlenoteLearnernotemax_throttle)
                             noteNPC, note
    """

    def __init__(self, npc_controllers, map_manager, curriculum_stage_ref,
                 start_step_offset=0, learner_max_throttle=0.3, verbose=1):
        super().__init__(verbose)
        self.npc_controllers = npc_controllers
        self.map_manager = map_manager
        self.curriculum_stage_ref = curriculum_stage_ref
        self.start_step_offset = start_step_offset  # note
        self.learner_max_throttle = learner_max_throttle
        self.last_stage = 0
        self._first_num_timesteps = None  # model.learnnotenum_timesteps

    def _on_step(self):
        # notelearn()notenum_timesteps, note"noterowsnote"
        if self._first_num_timesteps is None:
            self._first_num_timesteps = self.num_timesteps
        # noterunnote = num_timesteps - note
        steps_in_run = self.num_timesteps - self._first_num_timesteps
        # note = noterunnote + note
        step = steps_in_run + self.start_step_offset

        if step < 100000:
            stage = 1
            npc_mode = 'static'
            npc_throttle = 0.0
            npc_active = False
            npc_random_pos = False    # note
            npc_costart = False       # note
        elif step < 200000:
            stage = 2
            npc_mode = 'static'       # NPCnote, note
            npc_throttle = 0.0
            npc_active = True
            npc_random_pos = True     # notetracknote
            npc_costart = False
        elif step < 400000:
            stage = 3
            npc_mode = 'slow'
            progress = (step - 200000) / 200000   # 0->1
            npc_throttle = 0.12 + 0.08 * progress  # 0.12->0.20 note
            npc_active = True
            npc_random_pos = True     # note
            npc_costart = False
        else:
            stage = 4
            npc_mode = 'slow'         # note, noteslownotePure Pursuitstablerowsnote
            # NPCnote = Learnernote90%(note)
            npc_throttle = self.learner_max_throttle * 0.90
            npc_active = True
            npc_random_pos = False    # note
            npc_costart = True        # noteLearnernote

        self.curriculum_stage_ref['stage'] = stage
        self.curriculum_stage_ref['npc_active'] = npc_active
        self.curriculum_stage_ref['npc_random_pos'] = npc_random_pos
        self.curriculum_stage_ref['npc_costart'] = npc_costart

        if stage!= self.last_stage:
            print(f"\n{'='*70}")
            stage_names = {
                1: 'stage1 - note(NPCnotetrack, Learnernote)',
                2: 'stage2 - staticobstacle(NPCnote, notetracknote)',
                3: 'stage3 - noteNPC(NPCnote, note)',
                4: 'stage4 - note(NPCnote, noteLearner)',
            }
            print(f"🎓 note: {stage_names[stage]}")
            print(f"   note: {step:,}")
            print(f"   NPCnote: {npc_mode} | throttle: {npc_throttle:.2f} | note: {npc_active}")
            print(f"   NPCnote: {npc_random_pos} | note: {npc_costart}")
            print(f"{'='*70}\n")

            for npc in self.npc_controllers:
                if npc_active and npc.connected:
                    npc.set_mode(npc_mode, npc_throttle)
                    # noteNPCnotestagenote:
                    # - stage2 static: note(0,0,brake=1)noteNPCnote
                    # - stage3/4: Pure Pursuitnote
                    if not npc.running:
                        npc.start_driving()
                elif not npc_active and npc.connected:
                    if npc.running:
                        npc.stop_driving()

            self.last_stage = stage

        # stage3dynamicnoteNPCnote(note)
        if stage == 3:
            for npc in self.npc_controllers:
                if npc.connected:
                    npc.base_throttle = npc_throttle

        if step % 10000 == 0 and step > 0:
            stage_names = {1: 'note', 2: 'staticobstacle', 3: 'noteNPC', 4: 'note'}
            npc_info = ""
            for npc in self.npc_controllers:
                if npc.connected:
                    npc_info += (f" NPC_{npc.npc_id}(note={npc.mode},thr={npc.base_throttle:.2f},"
                                 f"noterows={npc.running},spd={npc.speed:.1f})")
            print(f"metrics note: {step:,} | {stage_names.get(stage, '?')} | "
                  f"NPC throttle: {npc_throttle:.2f} | note: {self.map_manager.current_scene}"
                  f"{npc_info}")

        return True


# ==================== 9. note ====================
class MapSwitchCallback(BaseCallback):
    """
    noteFalsenotecurrentlearn(), notetrainingnoterowsnote
    """

    def __init__(self, map_manager, switch_flag_ref, verbose=1):
        super().__init__(verbose)
        self.map_manager = map_manager
        self.switch_flag_ref = switch_flag_ref

    def _on_step(self):
        if self.map_manager.should_switch(self.num_timesteps):
            self.switch_flag_ref['should_switch'] = True
            return False
        return True


# ==================== 10. savenote ====================
class AutoSaveCallback(BaseCallback):
    """notesavemodel +.pth"""

    def __init__(self, save_dir, save_freq=50000, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            path = os.path.join(self.save_dir, f"v9_overtake_{self.num_timesteps}")
            self.model.save(path)
            pth_path = path + "_policy.pth"
            torch.save(self.model.policy.state_dict(), pth_path)
            print(f"saved modelnotesave: {path}.zip + {pth_path}")
        return True


class BestModelCallback(BaseCallback):
    """savenotemodel"""

    def __init__(self, save_path, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                mean_len = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])

                if self.verbose > 0:
                    print(f"\nmetrics {self.num_timesteps}note | notereward:{mean_reward:.1f} | note:{mean_len:.0f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    best_model_path = os.path.join(self.save_path, "best_model")
                    self.model.save(best_model_path)
                    best_pth_path = os.path.join(self.save_path, "best_model_policy.pth")
                    torch.save(self.model.policy.state_dict(), best_pth_path)
                    if self.verbose > 0:
                        print(f"⭐ notemodel! notereward: {mean_reward:.2f}")
        return True


# ==================== 11. note/note ====================
def create_env_and_npcs(map_manager, track_cache, args, curriculum_stage_ref,
                        npc_controllers_old=None):
    """
    note(note)note + NPC + note.note:
    1. noteNPC
    2. noteenv
    3. note
    4. noteNPC
    5. note
    """
    current_scene = map_manager.current_scene
    current_config = map_manager.current_config

    # noteNPC
    if npc_controllers_old:
        for npc in npc_controllers_old:
            npc.close()
        time.sleep(1)

    # note
    conf = {
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 255),
        "car_name": "V9_Learner",
        "racer_name": "learner",
        "country": "US",
        "bio": "",
        "guid": f"learner_{int(time.time())}",
        "font_size": 50,
        "max_cte": current_config['max_cte'],
        "cam_resolution": (120, 160, 3),
        "cam_encode": "JPG",
        "log_level": 20,
    }

    if args.exe_path:
        conf["exe_path"] = args.exe_path

    print(f"\n🚗 noteLearnernote (note: {current_scene})...")
    env = gym.make(current_config['env_id'], conf=conf)
    time.sleep(2)

    # notetracknote
    try:
        handler = env.viewer.handler
        total_nodes_hint = current_config.get('estimated_nodes', 26)
        track_cache.query_nodes(handler, current_scene, total_nodes=total_nodes_hint)
    except Exception as e:
        print(f"⚠️ notefailed: {e}")

    # noteNPC
    npc_controllers = []
    npc_colors = [(255, 100, 100), (100, 255, 100), (255, 255, 100)]

    if args.num_npc > 0:
        print(f"\n🚗 note {args.num_npc} noteNPCobstaclenote...")
        time.sleep(1)

        for i in range(args.num_npc):
            npc = NPCController(
                npc_id=i+1,
                host='127.0.0.1',
                port=args.port,
                scene=current_scene,
                track_cache=track_cache,
            )
            color = npc_colors[i % len(npc_colors)]
            if npc.connect(body_rgb=color):
                # notestagenote
                init_stage = curriculum_stage_ref.get('stage', 1)
                if init_stage >= 2:
                    npc.set_mode('static', 0.0)
                    # noteNPCnote, noteLearnernotedefaultnote
                    npc.set_position_node_coords(0, 100.0, 0, 0, 0, 0, 1)
                    # note -- stage2 staticnote
                    # note reset() noteNPC teleportnotetracknote, _drive_loopnote
                    # note (0,0,brake=1), NPCnote
                    npc.start_driving()
                else:
                    npc.set_mode('static', 0.0)
                    npc.set_position_node_coords(0, 100.0, 0, 0, 0, 0, 1)
                npc_controllers.append(npc)
            time.sleep(2)

    # note
    wrapped_env = OvertakeTrainingWrapper(
        env,
        npc_controllers=npc_controllers,
        track_cache=track_cache,
        scene_name=current_scene,
        use_random_start=args.random_start,
        max_episode_steps=600,
        curriculum_stage_ref=curriculum_stage_ref,
        enable_dr=True,
        max_throttle=args.max_throttle,
        delta_max=args.delta_max,
        enable_lpf=args.enable_lpf,
        beta=args.beta,
        w_d=args.w_d,
        w_dd=args.w_dd,
        w_sat=args.w_sat,
    )

    wrapped_env = Monitor(wrapped_env, filename=None, allow_early_resets=True)
    vec_env = DummyVecEnv([lambda: wrapped_env])

    return vec_env, npc_controllers, wrapped_env


# ==================== 12. notetrainingfunction ====================
def train(args):
    """main training entry point"""

    scenes = [s.strip() for s in args.scenes.split(',')]

    print("\n" + "="*80)
    print("🏎️ V9 notetraining (noteV8)")
    print("="*80)
    print(f"📋 configuration:")
    print(f"   note: {scenes}")
    print(f"   NPCnote: {args.num_npc}")
    print(f"   note: {args.total_steps:,}")
    print(f"   note: {'note' if args.random_start else 'note'}")
    print(f"   note: {args.port}")
    print(f"   note: {args.max_throttle}")
    print(f"   note: delta_max={args.delta_max}, LPF={'note' if args.enable_lpf else 'note'}(beta={args.beta})")
    print(f"   note: w_d={args.w_d}, w_dd={args.w_dd}, w_sat={args.w_sat}")
    print()

    # 1. note
    map_manager = MultiMapManager(scenes, switch_interval=args.map_switch_interval)
    track_cache = TrackNodeCache()

    # start_stage -> step_offset note
    stage_step_map = {1: 0, 2: 100000, 3: 200000, 4: 400000}
    start_step_offset = stage_step_map.get(args.start_stage, 0)

    # notestagenote
    if args.start_stage >= 2:
        curriculum_stage_ref = {
            'stage': args.start_stage,
            'npc_active': True,
            'npc_random_pos': args.start_stage in (2, 3),
            'npc_costart': args.start_stage == 4,
        }
        print(f"🎓 notestage{args.start_stage}note (step_offset={start_step_offset:,})")
    else:
        curriculum_stage_ref = {
            'stage': 1,
            'npc_active': False,
            'npc_random_pos': False,
            'npc_costart': False,
        }
    switch_flag_ref = {'should_switch': False}

    # 2. note + NPC
    vec_env, npc_controllers, wrapped_env = create_env_and_npcs(
        map_manager, track_cache, args, curriculum_stage_ref)

    # 3. notemodel
    policy_kwargs = dict(
        features_extractor_class=LightweightCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

    save_dir = "models/v9_overtake"
    os.makedirs(save_dir, exist_ok=True)

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"\nbuild notetrainingmodel: {args.pretrained_model}")
        model = PPO.load(
            args.pretrained_model,
            env=vec_env,
            learning_rate=args.lr,
            tensorboard_log="./logs/v9_overtake/",
        )
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./logs/v9_overtake/",
        )

    print(f"\n📋 PPOnote (noteV8):")
    print(f"   note: {model.learning_rate}")
    print(f"   n_steps: {model.n_steps}")
    print(f"   batch_size: {model.batch_size}")
    print(f"   n_epochs: {model.n_epochs}")
    print(f"   gamma: {model.gamma}")
    print(f"   clip_range: {model.clip_range}")
    print(f"   target_kl: {model.target_kl}")
    print(f"   note: CnnPolicy 8note (noteV8)")

    # 4. trainingnote (note)
    total_steps_done = 0
    remaining_steps = args.total_steps

    print(f"\n🚀 notetraining ({args.total_steps:,} note)...\n")

    try:
        while remaining_steps > 0:
            callbacks = [
                OvertakeCurriculumCallback(
                    npc_controllers=npc_controllers,
                    map_manager=map_manager,
                    curriculum_stage_ref=curriculum_stage_ref,
                    start_step_offset=start_step_offset,
                    learner_max_throttle=args.max_throttle,
                ),
                AutoSaveCallback(save_dir=save_dir, save_freq=50000),
                BestModelCallback(save_path=save_dir, check_freq=1000, verbose=1),
            ]

            if len(map_manager.scenes) > 1:
                callbacks.append(MapSwitchCallback(
                    map_manager=map_manager,
                    switch_flag_ref=switch_flag_ref,
                ))

            switch_flag_ref['should_switch'] = False

            if len(map_manager.scenes) > 1:
                steps_this_round = min(remaining_steps, args.map_switch_interval)
            else:
                steps_this_round = remaining_steps

            model.learn(
                total_timesteps=steps_this_round,
                callback=callbacks,
                tb_log_name="v9_overtake",
                reset_num_timesteps=False,
            )

            total_steps_done = model.num_timesteps
            remaining_steps = args.total_steps - total_steps_done

            if switch_flag_ref['should_switch'] and remaining_steps > 0:
                print(f"\n🗺️ note... (note {total_steps_done:,}/{args.total_steps:,} note)")

                new_scene = map_manager.next_scene()
                vec_env.close()
                time.sleep(2)

                vec_env, npc_controllers, wrapped_env = create_env_and_npcs(
                    map_manager, track_cache, args, curriculum_stage_ref,
                    npc_controllers_old=npc_controllers)

                model.set_env(vec_env)
                print(f"PASS note: {new_scene}")

        final_path = os.path.join(save_dir, "v9_overtake_final")
        model.save(final_path)
        final_pth = final_path + "_policy.pth"
        torch.save(model.policy.state_dict(), final_pth)
        print(f"\nPASS trainingnote! modelsave: {final_path}.zip + {final_pth}")

    except KeyboardInterrupt:
        print("\n⚠️ trainingnote")
        interrupted_path = os.path.join(save_dir, f"v9_overtake_interrupted_{total_steps_done}")
        model.save(interrupted_path)
        interrupted_pth = interrupted_path + "_policy.pth"
        torch.save(model.policy.state_dict(), interrupted_pth)
        print(f"saved notemodelnotesave: {interrupted_path}.zip + {interrupted_pth}")

    except Exception as e:
        import traceback
        print(f"\nFAIL trainingnote!")
        print(f"   noteclassnote: {type(e).__name__}")
        print(f"   note: {e}")
        traceback.print_exc()
        try:
            crash_path = os.path.join(save_dir, f"v9_overtake_crash_{total_steps_done}")
            model.save(crash_path)
            print(f"saved notefirstmodelnotesave: {crash_path}.zip")
        except Exception:
            print("⚠️ notesavemodelnotefailednote")

    finally:
        for npc in npc_controllers:
            npc.close()
        vec_env.close()
        print("🔒 note")


# ==================== 13. CLI ====================
def parse_args():
    parser = argparse.ArgumentParser(description='V9 notetraining (noteV8)')

    parser.add_argument('--scenes', type=str, default='waveshare', help='trainingnote, note')
    parser.add_argument('--port', type=int, default=9091, help='note')
    parser.add_argument('--exe-path', type=str, default=None, help='notepath')

    parser.add_argument('--total-steps', type=int, default=600000, help='notetrainingnote')
    parser.add_argument('--lr', type=float, default=3e-4, help='note')
    parser.add_argument('--pretrained-model', type=str, default=None, help='notetrainingmodelpath')
    parser.add_argument('--start-stage', type=int, default=1, choices=[1,2,3,4],
                        help='notestagenote (1=note, 2=staticnote, 3=note, 4=note)')

    parser.add_argument('--num-npc', type=int, default=1, help='NPCobstaclenote (0-3)')

    parser.add_argument('--random-start', action='store_true', default=True, help='note')
    parser.add_argument('--no-random-start', action='store_false', dest='random_start')
    parser.add_argument('--map-switch-interval', type=int, default=50000, help='note')
    parser.add_argument('--max-throttle', type=float, default=0.3, help='note')

    parser.add_argument('--delta-max', type=float, default=0.10, help='noteslew-rate limit')
    parser.add_argument('--enable-lpf', action='store_true', default=True, help='note')
    parser.add_argument('--no-lpf', action='store_false', dest='enable_lpf')
    parser.add_argument('--beta', type=float, default=0.6, help='note')

    parser.add_argument('--w-d', type=float, default=0.25, help='|delta_steer| note')
    parser.add_argument('--w-dd', type=float, default=0.08, help='|jerk| note')
    parser.add_argument('--w-sat', type=float, default=0.15, help='saturation note')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
