#!/usr/bin/env python3
"""
🚗 V9 避障超车训练 - 基于V8观测管道 + 多地图 + 随机起点 + NPC障碍车

核心改进（解决waveshare地图太小的问题）:
1. ✅ 多地图轮换: 切地图时重建env + 重连NPC + 重新query nodes (稳定)
2. ✅ 随机起点: 利用 set_position + node_position API 在赛道不同位置重生
3. ✅ NPC障碍车: 第2个TCP客户端连接，Pure Pursuit自动驾驶
4. ✅ 避障+超车奖励: 用赛道进度差(fine_track index)判定超车，而非距离
5. ✅ 课程学习:
   - 阶段1: 无NPC，Learner随机起点练驾驶
   - 阶段2: NPC不动(静态障碍)，随机出现在赛道上
   - 阶段3: NPC慢慢跑(0.12→0.20)，随机刷新位置
   - 阶段4: NPC和Learner同起点出发比赛(NPC油门=Learner的90%)

观测管道对齐V8:
- 8通道图像: RGB(3) + DiffRGB(3) + YellowMask(1) + Edges(1)
- HWC → CHW → 归一化 (RGB/Mask/Edges→[0,1], DiffRGB→[-1,1])
- ActionSafetyWrapper (舵机保护: slew-rate + LPF)
- ThrottleControlWrapper (油门限制)
- 完善的奖励系统 (对齐V8: 自适应CTE, 僵死/越界检测, 平滑惩罚, +超车奖励)

已验证的DonkeySim API:
- set_position: 坐标 = telemetry坐标 × 8 (COORD_SCALE=8), 精度0.0000m
- node_position: 返回set_position坐标系的节点位置+朝向
- telemetry包含 activeNode/totalNodes 字段
- 多TCP连接创建多辆车
- exit_scene + load_scene 运行时切换地图(不稳定，改用重建env)

NPC驾驶验证: Pure Pursuit, 30+圈稳定, 平均速度2.1, 路径误差<0.1m

用法:
  # 先启动仿真器
  /home/glz/Car/DonkeySimLinux/donkey_sim.x86_64

  # 然后运行训练
  python ppo_waveshare_v9_overtake.py --total-steps 600000

  # 使用更大地图轮换
  python ppo_waveshare_v9_overtake.py --scenes generated_track,waveshare

  # 单地图+随机起点（waveshare专用）
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


# ==================== 1. DonkeySim 扩展协议 ====================
class SimExtendedAPI:
    """
    封装 DonkeySim 的隐藏 API:
    - set_position: 设置车辆位置和朝向
    - node_position: 查询赛道路径节点

    坐标缩放: set_position/node_position 坐标 = telemetry坐标 × COORD_SCALE
    """

    COORD_SCALE = 8.0

    @staticmethod
    def send_set_position(handler, pos_x, pos_y, pos_z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """设置车辆位置 (node_position坐标系, = telemetry * 8)"""
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
        """使用 telemetry 坐标来设置车辆位置（自动 * COORD_SCALE 转换）"""
        S = SimExtendedAPI.COORD_SCALE
        SimExtendedAPI.send_set_position(handler, tel_x*S, tel_y*S, tel_z*S, qx, qy, qz, qw)

    @staticmethod
    def send_node_position_request(handler, index):
        """请求赛道路径节点的位置"""
        msg = {
            "msg_type": "node_position",
            "index": str(index),
        }
        handler.blocking_send(msg)

    @staticmethod
    def yaw_to_quaternion(yaw_degrees):
        """将偏航角(度)转换为四元数 (Unity Y轴旋转)"""
        yaw_rad = math.radians(yaw_degrees)
        qx = 0.0
        qy = math.sin(yaw_rad / 2.0)
        qz = 0.0
        qw = math.cos(yaw_rad / 2.0)
        return qx, qy, qz, qw


# ==================== 2. 赛道节点缓存 ====================
class TrackNodeCache:
    """缓存赛道路径节点坐标"""

    def __init__(self):
        self.nodes = {}        # {scene_name: [(x, y, z, qx, qy, qz, qw), ...]}
        self.total_nodes = {}  # {scene_name: int}
        self.fine_track = {}   # {scene_name: [(x, z), ...]} telemetry坐标系

        self._presets = {
            'waveshare':       {'total': 24},
            'generated_track': {'total': 108},
            'warehouse':       {'total': 50},
            'mini_monaco':     {'total': 50},
            'mountain_track':  {'total': 50},
        }

    def query_nodes(self, handler, scene_name, total_nodes=None):
        """
        从仿真器查询所有路径节点 (node_position坐标系 = telemetry * 8)

        关键修复: fns['node_position'] 注册后不能在 finally 里移除，
        因为回包是异步到达的，finally 执行完后仍有回包在途。
        改为: 永久注册 handler.fns['node_position']，查询完后回包仍能被接收。
        """
        if total_nodes is None:
            total_nodes = self._presets.get(scene_name, {}).get('total', 50)

        nodes = []
        print(f"📍 正在查询 {scene_name} 的路径节点 (最多{total_nodes}个)...")

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

        # 永久注册（不在 finally 里删除），保证异步回包能被接收
        handler.fns['node_position'] = on_node_position

        # 发送所有请求（每个间隔50ms）
        for i in range(total_nodes):
            SimExtendedAPI.send_node_position_request(handler, i)
            time.sleep(0.05)

        # 等待回包（waveshare 24个节点约需1.2s, generated_track 108个约需5s）
        wait_time = max(2.0, total_nodes * 0.06)
        print(f"   等待回包 {wait_time:.1f}s ...")
        time.sleep(wait_time)

        # 重试缺失节点
        if len(received_nodes) > 0:
            max_idx = max(received_nodes.keys())
            missing = [i for i in range(max_idx + 1) if i not in received_nodes]
            if missing:
                print(f"   重试缺失节点: {missing}")
                for i in missing:
                    SimExtendedAPI.send_node_position_request(handler, i)
                    time.sleep(0.12)
                time.sleep(2.0)

        # 整理结果
        if len(received_nodes) > 0:
            for i in range(max(received_nodes.keys()) + 1):
                if i in received_nodes:
                    n = received_nodes[i]
                    nodes.append((n['x'], n['y'], n['z'], n['qx'], n['qy'], n['qz'], n['qw']))

            self.nodes[scene_name] = nodes
            self.total_nodes[scene_name] = len(nodes)
            print(f"✅ 成功获取 {len(nodes)} 个节点")

            self._build_fine_track(scene_name)

            S = SimExtendedAPI.COORD_SCALE
            xs = [n[0]/S for n in nodes]
            zs = [n[2]/S for n in nodes]
            print(f"   赛道范围(telemetry): X=[{min(xs):.1f}, {max(xs):.1f}] Z=[{min(zs):.1f}, {max(zs):.1f}]")
            total_length = 0
            for i in range(len(nodes)):
                j = (i+1) % len(nodes)
                dx = (nodes[j][0]-nodes[i][0]) / S
                dz = (nodes[j][2]-nodes[i][2]) / S
                total_length += math.sqrt(dx**2 + dz**2)
            print(f"   赛道总长(估算): {total_length:.1f}m")
        else:
            print(f"⚠️ 未收到节点数据，请检查仿真器连接")

        return nodes

    def _build_fine_track(self, scene_name, interp_factor=10):
        """线性插值赛道节点，构建密集路径点 (telemetry坐标)"""
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
        print(f"   插值赛道: {len(coarse)}点 -> {len(fine)}点")

    def get_random_position(self, scene_name, exclude_range=None, exclude_set=None):
        """获取赛道上的随机位置 (node坐标系)
        
        exclude_set: 需要排除的节点索引集合（如已知CTE过高的坏节点）
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
        """获取当前位置前方 offset 个节点的位置"""
        if scene_name not in self.nodes or len(self.nodes[scene_name]) == 0:
            return None, -1
        nodes = self.nodes[scene_name]
        target_idx = (current_idx + offset) % len(nodes)
        return nodes[target_idx], target_idx

    def find_nearest_node(self, scene_name, tel_x, tel_z):
        """找到最近的节点索引 (输入为telemetry坐标)"""
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
        """找到插值赛道上最近的点索引 (telemetry坐标)"""
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
        计算 A 相对于 B 的赛道进度差（用于超车判定）

        返回正值 = A 在 B 前方
        返回负值 = A 在 B 后方

        使用 fine_track 的环形索引差，取 [-total/2, total/2] 范围
        """
        fine = self.fine_track.get(scene_name, [])
        if not fine:
            return 0
        total = len(fine)
        diff = (idx_a - idx_b) % total
        if diff > total // 2:
            diff -= total
        return diff


# ==================== 3. NPC控制器 ====================
class NPCController:
    """
    NPC障碍车控制器

    通过独立的TCP连接控制NPC车辆。
    NPC使用 Pure Pursuit 路径追踪算法沿赛道行驶。
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

        # NPC行为参数
        self.mode = 'static'  # 'static' | 'slow' | 'random' | 'chaos'
        self.base_throttle = 0.35
        self.running = False
        self._thread = None
        # random/chaos 驱动内部状态
        self._rand_next_change_step = 0
        self._rand_steer_bias = 0.0
        self._rand_target_steer_bias = 0.0
        self._rand_throttle_cmd = 0.0
        self._rand_target_throttle = 0.0
        self._rand_reverse_steps_left = 0
        self._rand_reverse_cooldown = 0
        self._rand_hard_brake_steps_left = 0
        # 外部守卫可触发短时强制刹车（用于NPC-NPC防碰撞）
        self._force_brake_steps_left = 0
        # 动态速度上限（sim speed）。None 表示不限制。
        self.speed_cap_sim = None

        # NPC位置跟踪 (telemetry坐标)
        self.tel_x = 0.0
        self.tel_y = 0.0
        self.tel_z = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.active_node = 0

        # NPC 在fine_track上的进度索引
        self.fine_track_idx = 0

        # telemetry拦截
        self._raw_msgs = []

    def connect(self, body_rgb=(255, 100, 100)):
        """连接到仿真器并加载NPC车"""
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

            # 拦截telemetry获取activeNode
            orig_telem = self.handler.on_telemetry
            def hook_telem(msg):
                self._raw_msgs.append(msg)
                if len(self._raw_msgs) > 10:
                    self._raw_msgs.pop(0)
                orig_telem(msg)
            self.handler.fns['telemetry'] = hook_telem

            # 等待加载
            timeout = 15
            start = time.time()
            while not self.handler.loaded and time.time() - start < timeout:
                time.sleep(0.5)

            if self.handler.loaded:
                self.connected = True
                # NPC不需要终止判定，禁用gym层的自动over
                self.handler.determine_episode_over = lambda: None
                self.handler.over = False
                print(f"✅ NPC_{self.npc_id} 已连接 (颜色: {body_rgb})")
            else:
                print(f"⚠️ NPC_{self.npc_id} 连接超时")

            return self.connected
        except Exception as e:
            print(f"❌ NPC_{self.npc_id} 连接失败: {e}")
            return False

    def set_position_node_coords(self, x, y, z, qx=0, qy=0, qz=0, qw=1):
        """设置NPC位置 (node_position坐标系)，同时更新telemetry估计"""
        if not self.connected:
            return
        SimExtendedAPI.send_set_position(self.handler, x, y, z, qx, qy, qz, qw)
        # 立刻更新telemetry坐标估计
        S = SimExtendedAPI.COORD_SCALE
        self.tel_x = x / S
        self.tel_y = y / S
        self.tel_z = z / S

    def set_mode(self, mode, throttle=0.35):
        """设置NPC行为模式"""
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
        """设置NPC速度上限（m/s，telemetry speed）。"""
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
        """请求短时强制刹车。steps 按 drive_loop tick 计数。"""
        try:
            n = int(steps)
        except Exception:
            n = 0
        if n <= 0:
            return
        self._force_brake_steps_left = max(int(self._force_brake_steps_left), n)

    def _apply_speed_cap(self, throttle, brake):
        """按动态速度上限抑制油门，必要时触发刹车/轻微反拖。"""
        cap = self.speed_cap_sim
        if cap is None:
            return float(throttle), float(brake)
        v_now = float(abs(self.speed))
        cap = max(0.05, float(cap))
        thr = float(throttle)
        brk = float(brake)

        if v_now > cap + 0.25:
            # 明显超速：优先反拖减速
            return min(thr, -0.18), max(brk, 0.0)
        if v_now > cap + 0.10:
            # 轻微超速：强制刹车
            return min(thr, 0.0), max(brk, 1.0)
        if thr > 0.0 and v_now >= cap:
            # 到达上限：禁止继续加速
            return 0.0, max(brk, 1.0)
        return thr, brk

    def start_driving(self):
        """启动NPC自动驾驶线程 (Pure Pursuit)"""
        if not self.connected or self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._drive_loop, daemon=True)
        self._thread.start()
        print(f"🚗 NPC_{self.npc_id} 开始行驶 (模式: {self.mode}, throttle: {self.base_throttle:.2f})")

    def stop_driving(self):
        """停止NPC驾驶"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self.connected:
            self.handler.send_control(0, 0, 1.0)

    def _pure_pursuit_steer(self, lookahead=0.5):
        """Pure Pursuit 路径追踪算法"""
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
        """NPC驾驶循环 (Pure Pursuit)"""
        step = 0
        while self.running and self.connected:
            try:
                if self._force_brake_steps_left > 0:
                    self._force_brake_steps_left -= 1
                    self.handler.send_control(0, 0, 1.0)
                    time.sleep(0.05)
                    # 更新NPC位置 (telemetry坐标 from handler)
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
                    # 持续发刹车指令确保NPC完全静止不滑动
                    self.handler.send_control(0, 0, 1.0)
                    time.sleep(0.2)
                elif self.mode == 'wobble':
                    # 原地前后左右小幅晃动，制造动态障碍感
                    phase = (step % 80) / 80.0  # 0~1 周期约 4 秒
                    angle = phase * 2 * math.pi
                    steer = 0.25 * math.sin(angle)          # 左右晃
                    throttle = 0.08 * math.sin(angle * 2)   # 前后微动（双倍频率）
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
                    # forward-only 的随机转向+加减速
                    steer_pp = self._pure_pursuit_steer(lookahead=0.55)
                    if step >= self._rand_next_change_step:
                        self._rand_next_change_step = step + random.randint(16, 40)
                        self._rand_target_steer_bias = random.uniform(-0.18, 0.18)
                        # 减速场景偏多，加速场景偏少
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
                    # chaos: 随机转向 + 加减速 + 概率短时倒车
                    steer_pp = self._pure_pursuit_steer(lookahead=0.45)
                    if step >= self._rand_next_change_step:
                        self._rand_next_change_step = step + random.randint(12, 30)
                        self._rand_target_steer_bias = random.uniform(-0.32, 0.32)
                        # 按用户诉求：减速多、加速少，允许急刹
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
                            # 倒车事件少一点
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

                # 更新NPC位置 (telemetry坐标 from handler)
                self.tel_x = self.handler.x
                self.tel_y = self.handler.y
                self.tel_z = self.handler.z
                self.yaw = self.handler.yaw
                self.speed = self.handler.speed

                # 更新fine_track进度
                if self.track_cache:
                    self.fine_track_idx, _ = self.track_cache.find_nearest_fine_track(
                        self.scene, self.tel_x, self.tel_z)

                # 更新activeNode
                if self._raw_msgs:
                    self.active_node = self._raw_msgs[-1].get('activeNode', self.active_node)

                step += 1
            except Exception:
                time.sleep(0.1)

    def get_telemetry_position(self):
        """获取NPC当前位置 (telemetry坐标)"""
        return (self.tel_x, self.tel_y, self.tel_z)

    def close(self):
        """关闭NPC连接"""
        self.stop_driving()
        if self.client:
            try:
                self.client.stop()
            except:
                pass
        self.connected = False
        print(f"🔒 NPC_{self.npc_id} 已关闭")


# ==================== 4. 多地图管理器 ====================
class MultiMapManager:
    """
    多地图轮换管理器

    策略: 切地图时直接重建 env + 重连所有 NPC + 重新 query nodes
    """

    MAP_CONFIGS = {
        'waveshare': {
            'env_id': 'donkey-waveshare-v0',
            'level': 'waveshare',
            'max_cte': 5.0,
            'description': '小型椭圆赛道（24节点, 66m）',
            'difficulty': 1,
            'estimated_nodes': 24,
        },
        'generated_track': {
            'env_id': 'donkey-generated-track-v0',
            'level': 'generated_track',
            'max_cte': 8.0,
            'description': '大型生成赛道（108节点, 215m）',
            'difficulty': 2,
            'estimated_nodes': 108,
        },
        'warehouse': {
            'env_id': 'donkey-warehouse-v0',
            'level': 'warehouse',
            'max_cte': 8.0,
            'description': '仓库环境（宽阔场地）',
            'difficulty': 2,
            'estimated_nodes': 30,
        },
        'mini_monaco': {
            'env_id': 'donkey-minimonaco-track-v0',
            'level': 'mini_monaco',
            'max_cte': 6.0,
            'description': '迷你摩纳哥赛道（急弯多）',
            'difficulty': 3,
            'estimated_nodes': 35,
        },
        'mountain_track': {
            'env_id': 'donkey-mountain-track-v0',
            'level': 'mountain_track',
            'max_cte': 8.0,
            'description': '山地赛道（起伏路面）',
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

        print(f"🗺️ 多地图管理器初始化:")
        for s in self.scenes:
            cfg = self.MAP_CONFIGS[s]
            print(f"   ✅ {s}: {cfg['description']}")
        print(f"   切换间隔: 每 {switch_interval:,} 步")

    @property
    def current_scene(self):
        return self.scenes[self.current_scene_idx]

    @property
    def current_config(self):
        return self.MAP_CONFIGS[self.current_scene]

    def should_switch(self, global_step):
        """检查是否应该切换地图"""
        if len(self.scenes) <= 1:
            return False
        self.steps_on_current += 1
        if self.steps_on_current >= self.switch_interval:
            self.steps_on_current = 0
            return True
        return False

    def next_scene(self):
        """切换到下一个地图"""
        old = self.current_scene
        self.current_scene_idx = (self.current_scene_idx + 1) % len(self.scenes)
        self.steps_on_current = 0
        new = self.current_scene
        print(f"\n🗺️ 地图切换: {old} -> {new}")
        return new


# ==================== 5. 图像预处理 (对齐V8) ====================
class YellowLaneEnhancer:
    """黄色车道线增强 (对齐V8, 含DR和CLAHE)"""

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
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.uniform(-8, 8), 0, 179)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb

    def enhance(self, rgb, apply_dr=False):
        """
        返回 (rgb_processed, yellow_mask, edges)
        DR只作用于RGB，mask/edges每帧从当前RGB重新计算
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


# ==================== 6. V9 超车训练环境包装器 ====================
class OvertakeTrainingWrapper(gym.Wrapper):
    """
    V9 超车训练环境包装器 (观测对齐V8 8通道管道)

    观测管道:
    1. RGB(3) + DiffRGB(3) + YellowMask(1) + Edges(1) = 8通道
    2. HWC -> CHW -> 归一化 float32

    关键修复:
    - 阶段1真正无NPC: reset时根据课程阶段决定是否摆放NPC
    - 超车判定用赛道进度(fine_track index)而不是距离
    - NPC距离用统一的telemetry坐标计算
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

        # 图像处理 (对齐V8)
        self.enhancer = YellowLaneEnhancer(enable_dr=enable_dr, dr_prob=0.6)
        self.prev_rgb = None

        # ActionSafety (对齐V8)
        self.delta_max = delta_max
        self.enable_lpf = enable_lpf
        self.beta = beta
        self.steer_prev_limited = 0.0
        self.steer_prev_exec = 0.0
        self.delta_steer_prev = 0.0

        # 平滑惩罚权重
        self.w_d = w_d
        self.w_dd = w_dd
        self.w_sat = w_sat

        # 状态
        self.episode_step = 0
        self.last_active_node = 0
        self.nodes_passed = 0
        self.speed_history = deque(maxlen=50)
        self.stuck_counter = 0
        self.offtrack_counter = 0

        # 超车判定 (赛道进度差)
        self.learner_fine_idx = 0
        self.overtake_count = 0
        self._npc_overtake_state = {}

        # 自适应CTE (对齐V8)
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

        # 观测空间: 8通道 CHW float32
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(8, target_size[0], target_size[1]),
            dtype=np.float32
        )

        print(f"✅ V9 超车训练环境初始化 (地图: {scene_name})")
        print(f"   观测: 8通道 CHW float32 (对齐V8)")
        print(f"   NPC数量: {len(self.npc_controllers)}")
        print(f"   随机起点: {'是' if use_random_start else '否'}")

        # === 关键: 禁用 gym-donkeycar 层的自动终止 ===
        # gym 的 determine_episode_over() 在每帧 telemetry 时自动判定碰撞/CTE→over=True→done=True
        # 这会和我们 wrapper 的终止逻辑冲突，也没有宽限期
        # 覆盖为空函数，所有终止逻辑统一由 wrapper.step() 控制
        try:
            handler = self.env.viewer.handler
            handler.determine_episode_over = lambda: None
            print(f"   ✅ 已禁用gym层自动终止（由wrapper统一控制）")
        except Exception as e:
            print(f"   ⚠️ 无法覆盖gym层终止函数: {e}")

    def reset(self, **kwargs):
        # 记录上一回合统计
        if self.episode_stats['steps'] > 0:
            avg_cte = self.episode_stats['cte_sum'] / max(1, self.episode_stats['steps'])
            term_reason = self.episode_stats.get('termination_reason', 'max_steps')
            self.recent_episodes.append({
                'avg_cte': avg_cte,
                'collision': self.episode_stats['collision'],
                'steps': self.episode_stats['steps'],
                'reward': self.episode_stats['total_reward'],
            })
            # 自适应CTE调整 (对齐V8)
            self._evaluate_and_adjust_cte()

            self.total_episodes += 1

            # === 每回合都打印摘要 ===
            stage = self.curriculum_stage_ref.get('stage', 1)
            stage_names = {1: 'S1基础', 2: 'S2避障', 3: 'S3慢超', 4: 'S4竞赛'}
            print(f"[{stage_names.get(stage,'?')}] 步{self.total_train_steps:,} | "
                  f"回合{self.total_episodes:,}: "
                  f"{self.episode_stats['steps']}步 "
                  f"R={self.episode_stats['total_reward']:.1f} "
                  f"CTE={avg_cte:.2f} "
                  f"v_max={self.episode_stats['max_speed']:.1f} "
                  f"结束={term_reason}")

            # 每20回合打印汇总
            if self.total_episodes % 20 == 0:
                last20 = list(self.recent_episodes)[-20:]
                avg_r = np.mean([e['reward'] for e in last20])
                avg_s = np.mean([e['steps'] for e in last20])
                avg_c = np.mean([e['collision'] for e in last20])
                avg_ct = np.mean([e['avg_cte'] for e in last20])
                print(f"  📊 近20回合汇总: 平均R={avg_r:.1f} 步={avg_s:.0f} "
                      f"碰撞率={avg_c:.0%} CTE={avg_ct:.2f} | CTE阈值={self.current_max_cte:.2f}")

        # ===== 第1步：env 层 reset（车回到默认起点）=====
        obs = self.env.reset(**kwargs)
        time.sleep(0.2)  # 等仿真器处理reset，车到位，物理稳定

        # === 根据课程阶段决定处理逻辑 ===
        npc_active = self.curriculum_stage_ref.get('npc_active', False)
        npc_costart = self.curriculum_stage_ref.get('npc_costart', False)
        npc_random_pos = self.curriculum_stage_ref.get('npc_random_pos', False)
        current_stage = self.curriculum_stage_ref.get('stage', 1)

        if npc_costart:
            # 阶段4: Learner固定起点，NPC同起点
            self.last_active_node = 0
            self._place_npcs_at_start()
            
        elif self.use_random_start and self.track_cache:
            # 阶段1/2/3: Learner随机start, NPC视阶段处理
            # _randomize_start_position 内部会teleport+验证CTE(含step+sleep)
            self._randomize_start_position()
            
            if npc_active and npc_random_pos:
                # 阶段2/3: NPC基于当前Learner位置(last_active_node)随机放置
                self._randomize_npc_positions()
                time.sleep(0.1)  # 等NPC物理稳定
            elif not npc_active:
                # 阶段1: 无NPC或NPC停在场外
                self._park_npcs_offtrack_lazy()
        else:
            # 无随机起点模式
            self.last_active_node = 0
            if not npc_active:
                self._park_npcs_offtrack_lazy()

        # ===== 第2步：等待所有物理完全稳定后，取一帧干净obs =====
        time.sleep(0.15)
        # 必须调用 env.step(action=0) 来获取新的obs和telemetry
        # action=0 表示零舵机、零油门，不推进车的逻辑，只更新观测
        try:
            obs, _, _, _ = self.env.step(np.array([0.0, 0.0], dtype=np.float32))
        except Exception as e:
            print(f"⚠️ reset中step异常: {e}")

        # 重置状态
        self.episode_step = 0
        self.nodes_passed = 0
        self.speed_history.clear()
        self.stuck_counter = 0
        self.offtrack_counter = 0
        self.learner_fine_idx = 0
        self._npc_overtake_state = {}

        # 重置ActionSafety
        self.steer_prev_limited = 0.0
        self.steer_prev_exec = 0.0
        self.delta_steer_prev = 0.0

        # 重置差分帧
        self.prev_rgb = None

        self.episode_stats = {
            'cte_sum': 0, 'steps': 0, 'max_speed': 0,
            'collision': False, 'total_reward': 0,
            'termination_reason': 'max_steps',
        }

        return self._process_observation(obs)

    def _evaluate_and_adjust_cte(self):
        """自适应CTE收紧 (对齐V8)"""
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
            print(f"\n🎯 CTE阈值收紧: {old_cte:.2f} -> {self.current_max_cte:.2f} (步数={self.total_train_steps:,})")

    def _randomize_start_position(self):
        """随机选择赛道上的起点（在env.reset()之后用set_position覆盖）
        
        teleport后会验证CTE，如果CTE过高（说明节点方向不对/落在赛道外）则重试其他位置
        坏节点会被记录到黑名单，后续不再选择
        """
        MAX_RETRIES = 4
        CTE_THRESHOLD = 6.0  # teleport后CTE超过此值就重试

        # 初始化坏节点黑名单（持久化，不在每次reset时清空）
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
                    # 所有节点都在黑名单或已尝试，放宽限制重试
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

                # 验证teleport后的CTE
                try:
                    _, _, _, step_info = self.env.step(np.array([0.0, 0.0], dtype=np.float32))
                    cte_after = abs(step_info.get('cte', 0))
                    # 清除gym可能设置的over标志
                    try:
                        handler.over = False
                    except Exception:
                        pass
                except Exception:
                    cte_after = 0  # 无法获取就跳过验证

                if cte_after < CTE_THRESHOLD:
                    self.last_active_node = idx
                    return  # 成功
                else:
                    # 记录坏节点
                    self._bad_nodes.add(idx)
                    print(f"  ⚠️ teleport到node{idx} CTE={cte_after:.1f}>阈值{CTE_THRESHOLD}, "
                          f"加入黑名单(共{len(self._bad_nodes)}个坏节点), 重试({attempt+1}/{MAX_RETRIES})")

            except Exception as e:
                print(f"⚠️ _randomize_start_position 异常: {e}")
                break

        # 所有重试都失败，用最后一个位置
        if last_result is not None:
            _, idx = last_result
            self.last_active_node = idx
            print(f"  ⚠️ teleport验证全部失败, 使用node{idx} (CTE可能较高)")

    def _randomize_npc_positions(self):
        """将NPC随机放置在Learner前方 (阶段2/3用)"""
        if not self.npc_controllers or not self.track_cache:
            return
        try:
            nodes = self.track_cache.nodes.get(self.scene_name, [])
            if len(nodes) < 3:
                print(f"⚠️ 节点不足({len(nodes)}个)，跳过NPC位置刷新")
                return
            
            # 关键：使用当前Learner的node位置（reset中已更新）
            learner_node = self.last_active_node
            num_nodes = len(nodes)
            current_stage = self.curriculum_stage_ref.get('stage', 2)

            for i, npc in enumerate(self.npc_controllers):
                if not npc.connected:
                    continue

                # 确保NPC驾驶线程已启动（防止第一次reset时线程还没启动）
                if not npc.running:
                    npc.start_driving()

                # NPC放在Learner前方5-10个节点（留足安全距离），多辆NPC各自错开3个节点
                offset = random.randint(5, min(10, max(5, num_nodes // 3)))
                npc_node_idx = (learner_node + offset + i * 3) % num_nodes
                node = nodes[npc_node_idx]
                
                # Debug log
                if i == 0:  # 只打第一个NPC的日志，避免重复
                    print(f"  🚗 [NPC位置] Learner=node{learner_node} → NPC=node{npc_node_idx} (offset={offset})")
                
                npc.set_position_node_coords(
                    node[0], node[1], node[2], node[3], node[4], node[5], node[6])
                time.sleep(0.2)  # 每辆NPC teleport后等仿真器处理完

                # 阶段2: teleport后立即发刹车，等_drive_loop接管持续刹车
                if current_stage == 2:
                    npc.handler.send_control(0, 0, 1.0)  # steer=0, throttle=0, brake=1
                    time.sleep(0.1)
                    npc.handler.send_control(0, 0, 1.0)  # 发两次确保生效

                # 更新NPC fine_track进度
                npc.fine_track_idx, _ = self.track_cache.find_nearest_fine_track(
                    self.scene_name, npc.tel_x, npc.tel_z)
        except Exception as e:
            print(f"⚠️ _randomize_npc_positions 异常: {e}")
            import traceback
            traceback.print_exc()

    def _place_npcs_at_start(self):
        """
        阶段4专用: 将NPC放在起点附近，和Learner一起出发

        NPC放在Learner后方1-2个节点（同起跑线附近，不挡路）
        NPC此时应已启动Pure Pursuit驾驶线程
        """
        if not self.npc_controllers or not self.track_cache:
            return
        try:
            nodes = self.track_cache.nodes.get(self.scene_name, [])
            if len(nodes) < 3:
                return
            num_nodes = len(nodes)
            learner_node = self.last_active_node  # 阶段4时是0（默认起点）

            for i, npc in enumerate(self.npc_controllers):
                if not npc.connected:
                    continue
                # NPC放在Learner后方1个节点（避免挡在正前方被撞）
                npc_node_idx = (learner_node - 1 - i) % num_nodes
                node = nodes[npc_node_idx]
                npc.set_position_node_coords(
                    node[0], node[1], node[2], node[3], node[4], node[5], node[6])
                time.sleep(0.2)
                npc.fine_track_idx, _ = self.track_cache.find_nearest_fine_track(
                    self.scene_name, npc.tel_x, npc.tel_z)
                # 确保NPC驾驶线程在跑
                if not npc.running:
                    npc.start_driving()
        except Exception as e:
            print(f"⚠️ _place_npcs_at_start 异常: {e}")

    def _park_npcs_offtrack(self):
        """将NPC移到赛道外 (强制，每次都移)"""
        for npc in self.npc_controllers:
            if npc.connected:
                npc.set_position_node_coords(0, 100.0, 0, 0, 0, 0, 1)
                if npc.running:
                    npc.stop_driving()

    def _park_npcs_offtrack_lazy(self):
        """
        将NPC移到赛道外 (懒执行版本)
        只在NPC首次连接后或明确需要时移出一次，
        避免每回合reset都触发teleport造成闪烁。
        """
        for npc in self.npc_controllers:
            if not npc.connected:
                continue
            # 只要NPC的tel_y不在停车位(y应该接近12.5，停在天上是100/8=12.5)
            # 就把它移出去
            if npc.running:
                npc.stop_driving()
            if abs(npc.tel_y - 12.5) > 1.0:  # 还不在停车位
                npc.set_position_node_coords(0, 100.0, 0, 0, 0, 0, 1)

    def step(self, action):
        self.episode_step += 1
        self.total_train_steps += 1

        # === ActionSafety: 舵机保护 (对齐V8) ===
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

        # 执行动作
        obs, base_reward, gym_done, info = self.env.step(safe_action)

        # === 完全忽略 gym 层的 done ===
        # gym 层的 determine_episode_over 已被禁用
        # 所有终止判定统一由 wrapper 控制
        done = False

        # 如果 gym 内部意外 over=True，强制清除
        if gym_done:
            try:
                self.env.viewer.handler.over = False
            except Exception:
                pass

        # 处理观测
        processed_obs = self._process_observation(obs)

        # === 计算综合奖励 ===
        cte = abs(info.get('cte', 0))
        speed = info.get('speed', 0)
        hit = info.get('hit', 'none')
        lap_count = info.get('lap_count', 0)
        self.speed_history.append(speed)
        self.episode_stats['cte_sum'] += cte
        self.episode_stats['max_speed'] = max(self.episode_stats['max_speed'], speed)

        ontrack = float(cte <= self.current_max_cte)
        reward = 0.0

        # 生存奖励
        reward += 0.2

        # 速度奖励
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

        # 碰撞 — teleport 后前20步忽略碰撞（物理引擎需要稳定）
        if hit != 'none' and self.episode_step > 20:
            reward -= 6.0
            self.episode_stats['collision'] = True
            done = True
            info['termination_reason'] = 'collision'
            self.episode_stats['termination_reason'] = 'collision'

        # 圈奖励 (如果有)
        # (gym-donkeycar的info里有lap_count)

        # 僵死检测 — teleport后前20步不判（车需要加速时间）
        if self.episode_step > 20 and ontrack and speed < 0.3:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > 50:
            done = True
            reward -= 2.0
            info['termination_reason'] = 'stuck'
            self.episode_stats['termination_reason'] = 'stuck'

        # 持续越界 — teleport后前25步不判（teleport后CTE需要稳定时间）
        if self.episode_step > 25 and not ontrack:
            self.offtrack_counter += 1
        else:
            self.offtrack_counter = 0
        if self.offtrack_counter > 25:
            done = True
            reward -= 4.0
            info['termination_reason'] = 'persistent_offtrack'
            self.episode_stats['termination_reason'] = 'persistent_offtrack'

        # 平滑惩罚 (对齐V8)
        abs_delta = abs(actual_delta)
        abs_jerk = abs(actual_delta - prev_delta_steer)
        reward += -self.w_d * abs_delta
        reward += -self.w_dd * abs_jerk
        reward += -self.w_sat * rate_excess_bounded

        # 进度奖励
        active_node = info.get('activeNode', self.last_active_node)
        if isinstance(active_node, str):
            try:
                active_node = int(active_node)
            except:
                active_node = self.last_active_node
        if active_node != self.last_active_node:
            self.nodes_passed += 1
            reward += 0.2
            self.last_active_node = active_node

        # *** NPC相关奖励 ***
        npc_active = self.curriculum_stage_ref.get('npc_active', False)
        current_stage = self.curriculum_stage_ref.get('stage', 1)
        if npc_active:
            if current_stage == 2:
                # 阶段2: 只有避障奖励（NPC不动，不需要超车判定）
                reward, done = self._compute_avoidance_reward(reward, done, info)
            else:
                # 阶段3/4: 完整的超车奖励（含避障+超车判定）
                reward, done = self._compute_overtake_reward(reward, done, info)

        # 最大步数限制
        if self.episode_step >= self.max_episode_steps:
            done = True
            self.episode_stats['termination_reason'] = 'max_steps'

        self.episode_stats['total_reward'] += reward
        self.episode_stats['steps'] += 1

        return processed_obs, reward, done, info

    def _compute_avoidance_reward(self, reward, done, info):
        """
        阶段2专用: 纯避障奖励（NPC静止不动）

        只奖励:
        - 接近NPC时减速
        - 安全绕过NPC（近距离通过不撞）
        - 撞到NPC惩罚（已在碰撞检测中处理）
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

            # 接近NPC减速奖励
            if dist < 4.0 and len(self.speed_history) > 5:
                avg_speed = np.mean(list(self.speed_history)[-10:])
                if avg_speed < 2.5:
                    reward += 0.2

            # 安全通过NPC（近距离但没撞）
            if 0.8 < dist < 2.0:
                reward += 0.3

        return reward, done

    def _compute_overtake_reward(self, reward, done, info):
        """
        用赛道进度差判定超车（阶段3/4用）

        逻辑:
        1. 计算Learner和NPC在fine_track上的进度索引
        2. 进度差 = learner_idx - npc_idx (正值 = learner在前)
        3. Learner从NPC后方 -> NPC前方，且持续N步 -> 超车成功
        """
        if not self.track_cache or not self.npc_controllers:
            return reward, done

        # Learner的telemetry位置
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

            # 冷却期
            if state['overtake_cooldown'] > 0:
                state['overtake_cooldown'] -= 1
                continue

            npc_fine_idx = npc.fine_track_idx

            # 赛道进度差 (正 = Learner在前)
            progress_diff = self.track_cache.progress_diff(
                self.scene_name, self.learner_fine_idx, npc_fine_idx)

            # NPC距离 (telemetry坐标, 统一坐标系)
            nx, ny, nz = npc.get_telemetry_position()
            dist = math.sqrt((lx - nx)**2 + (lz - nz)**2)

            # 接近NPC减速奖励
            if dist < 4.0 and len(self.speed_history) > 5:
                avg_speed = np.mean(list(self.speed_history)[-10:])
                if avg_speed < 2.5:
                    reward += 0.2

            # 近距离安全通过
            if 0.8 < dist < 2.0:
                reward += 0.3

            # 超车判定 (waveshare 240 fine_track点, 需要更严格的阈值)
            # 要求: Learner必须先在NPC后方至少20个fine点(约2个node距离)
            #        然后到NPC前方至少20个fine点, 持续15步确认
            BEHIND_THRESHOLD = -20    # 必须真正在NPC后方
            AHEAD_THRESHOLD = 20      # 必须真正超到NPC前方
            BEHIND_CONFIRM = 10       # 在后方持续10步才算"跟车中"
            AHEAD_CONFIRM = 15        # 超过后持续15步才确认"超车成功"

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
                        state['overtake_cooldown'] = 150  # 冷却150步
                        info['overtake_success'] = True
                        info['total_overtakes'] = self.overtake_count
                        print(f"🏎️ 超车成功! 第{self.overtake_count}次 "
                              f"(步数: {self.episode_step}, 进度差: {progress_diff})")
            else:
                state['ahead_count'] = 0

        return reward, done

    def _process_observation(self, obs):
        """
        处理观测 -> 8通道 CHW float32 (对齐V8)

        通道0-2: RGB [0,1]
        通道3-5: DiffRGB [-1,1]
        通道6:   YellowMask [0,1]
        通道7:   Edges [0,1]
        """
        if isinstance(obs, dict):
            image = obs.get('image', obs.get('cam', obs))
        else:
            image = obs

        image = cv2.resize(image, (self.target_size[1], self.target_size[0]),
                           interpolation=cv2.INTER_LINEAR)

        rgb, yellow_mask, edges = self.enhancer.enhance(image, apply_dr=True)

        # 差分帧
        if self.prev_rgb is not None:
            diff_rgb = rgb.astype(np.float32) - self.prev_rgb.astype(np.float32)
            diff_rgb = np.clip(diff_rgb, -255, 255)
        else:
            diff_rgb = np.zeros_like(rgb, dtype=np.float32)

        self.prev_rgb = rgb.copy()

        # CHW + 归一化
        rgb_chw = np.transpose(rgb.astype(np.float32), (2, 0, 1)) / 255.0
        diff_chw = np.transpose(diff_rgb, (2, 0, 1)) / 255.0
        mask_chw = yellow_mask.astype(np.float32)[np.newaxis, :, :] / 255.0
        edges_chw = edges.astype(np.float32)[np.newaxis, :, :] / 255.0

        obs_8ch = np.concatenate([rgb_chw, diff_chw, mask_chw, edges_chw], axis=0)
        return obs_8ch.astype(np.float32)

    def close(self):
        super().close()


# ==================== 7. CNN (对齐V8, 动态通道数) ====================
class LightweightCNN(BaseFeaturesExtractor):
    """
    轻量级CNN - 对齐V8
    从observation_space.shape[0]自动读取输入通道数
    架构: N_ch -> 32 -> 64 -> 64 -> features_dim
    """

    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        print(f"\n🧠 V9 CNN架构 (对齐V8):")
        print(f"   输入: {n_input_channels}通道 x {observation_space.shape[1]}x{observation_space.shape[2]}")

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
        print(f"   展平维度: {n_flatten}")
        print(f"   特征维度: {features_dim}")
        print(f"   总参数: {total_params:,}")

    def forward(self, observations):
        return self.linear(self.cnn(observations))


# ==================== 8. 课程学习回调 ====================
class OvertakeCurriculumCallback(BaseCallback):
    """
    超车训练课程学习（修正版）

    阶段设计:
    1. 0-100k:   基础驾驶 — NPC移出赛道，Learner随机起点练驾驶
    2. 100k-200k: 静态障碍 — NPC不动(throttle=0)，随机出现在赛道上作为障碍物
                             Learner随机起点，学习识别和绕过静止障碍
    3. 200k-400k: 慢速NPC — NPC慢慢跑(throttle从0.12渐增到0.20)
                             NPC随机刷新位置，Learner学习跟车+超车
    4. 400k-600k: 同速竞赛 — NPC和Learner同起点出发(throttle匹配Learner的max_throttle)
                             不再随机刷新NPC，一起跑完整圈比赛
    """

    def __init__(self, npc_controllers, map_manager, curriculum_stage_ref,
                 start_step_offset=0, learner_max_throttle=0.3, verbose=1):
        super().__init__(verbose)
        self.npc_controllers = npc_controllers
        self.map_manager = map_manager
        self.curriculum_stage_ref = curriculum_stage_ref
        self.start_step_offset = start_step_offset  # 虚拟起始步数
        self.learner_max_throttle = learner_max_throttle
        self.last_stage = 0
        self._first_num_timesteps = None  # model.learn开始时的num_timesteps

    def _on_step(self):
        # 记录本次learn()开始时的num_timesteps，用来算"本次运行了多少步"
        if self._first_num_timesteps is None:
            self._first_num_timesteps = self.num_timesteps
        # 本次run的步数 = num_timesteps - 开始时的值
        steps_in_run = self.num_timesteps - self._first_num_timesteps
        # 虚拟步数 = 本次run步数 + 起始偏移
        step = steps_in_run + self.start_step_offset

        if step < 100000:
            stage = 1
            npc_mode = 'static'
            npc_throttle = 0.0
            npc_active = False
            npc_random_pos = False    # 不适用
            npc_costart = False       # 不适用
        elif step < 200000:
            stage = 2
            npc_mode = 'static'       # NPC完全不动，只是路障
            npc_throttle = 0.0
            npc_active = True
            npc_random_pos = True     # 每回合随机放在赛道上
            npc_costart = False
        elif step < 400000:
            stage = 3
            npc_mode = 'slow'
            progress = (step - 200000) / 200000   # 0→1
            npc_throttle = 0.12 + 0.08 * progress  # 0.12→0.20 非常慢
            npc_active = True
            npc_random_pos = True     # 随机刷新位置
            npc_costart = False
        else:
            stage = 4
            npc_mode = 'slow'         # 同速模式，用slow的Pure Pursuit稳定行驶
            # NPC油门 = Learner最大油门的90%（稍慢一点点让超车可能）
            npc_throttle = self.learner_max_throttle * 0.90
            npc_active = True
            npc_random_pos = False    # 不随机刷新
            npc_costart = True        # 和Learner一起从起点出发

        self.curriculum_stage_ref['stage'] = stage
        self.curriculum_stage_ref['npc_active'] = npc_active
        self.curriculum_stage_ref['npc_random_pos'] = npc_random_pos
        self.curriculum_stage_ref['npc_costart'] = npc_costart

        if stage != self.last_stage:
            print(f"\n{'='*70}")
            stage_names = {
                1: '阶段1 - 基础驾驶（NPC移出赛道，Learner随机起点）',
                2: '阶段2 - 静态障碍（NPC不动，随机出现在赛道上）',
                3: '阶段3 - 慢速NPC（NPC慢跑，随机刷新位置）',
                4: '阶段4 - 同速竞赛（NPC同起点出发，油门匹配Learner）',
            }
            print(f"🎓 课程学习: {stage_names[stage]}")
            print(f"   步数: {step:,}")
            print(f"   NPC模式: {npc_mode} | throttle: {npc_throttle:.2f} | 激活: {npc_active}")
            print(f"   NPC随机位置: {npc_random_pos} | 同起点: {npc_costart}")
            print(f"{'='*70}\n")

            for npc in self.npc_controllers:
                if npc_active and npc.connected:
                    npc.set_mode(npc_mode, npc_throttle)
                    # 所有NPC激活阶段都需要驾驶线程:
                    # - 阶段2 static: 持续发(0,0,brake=1)确保NPC不滑动
                    # - 阶段3/4: Pure Pursuit驾驶
                    if not npc.running:
                        npc.start_driving()
                elif not npc_active and npc.connected:
                    if npc.running:
                        npc.stop_driving()

            self.last_stage = stage

        # 阶段3动态更新NPC油门（渐增）
        if stage == 3:
            for npc in self.npc_controllers:
                if npc.connected:
                    npc.base_throttle = npc_throttle

        if step % 10000 == 0 and step > 0:
            stage_names = {1: '基础驾驶', 2: '静态障碍', 3: '慢速NPC', 4: '同速竞赛'}
            npc_info = ""
            for npc in self.npc_controllers:
                if npc.connected:
                    npc_info += (f" NPC_{npc.npc_id}(模式={npc.mode},thr={npc.base_throttle:.2f},"
                                 f"运行={npc.running},spd={npc.speed:.1f})")
            print(f"📊 步数: {step:,} | {stage_names.get(stage, '?')} | "
                  f"NPC throttle: {npc_throttle:.2f} | 地图: {self.map_manager.current_scene}"
                  f"{npc_info}")

        return True


# ==================== 9. 地图切换回调 ====================
class MapSwitchCallback(BaseCallback):
    """
    多地图切换回调
    到达切换步数时返回False停止当前learn()，由外层训练循环执行重建
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


# ==================== 10. 保存回调 ====================
class AutoSaveCallback(BaseCallback):
    """定期保存模型 + .pth"""

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
            print(f"💾 模型已保存: {path}.zip + {pth_path}")
        return True


class BestModelCallback(BaseCallback):
    """保存最佳模型"""

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
                    print(f"\n📊 {self.num_timesteps}步 | 平均奖励:{mean_reward:.1f} | 平均长度:{mean_len:.0f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    best_model_path = os.path.join(self.save_path, "best_model")
                    self.model.save(best_model_path)
                    best_pth_path = os.path.join(self.save_path, "best_model_policy.pth")
                    torch.save(self.model.policy.state_dict(), best_pth_path)
                    if self.verbose > 0:
                        print(f"⭐ 新最佳模型! 平均奖励: {mean_reward:.2f}")
        return True


# ==================== 11. 环境创建/重建 ====================
def create_env_and_npcs(map_manager, track_cache, args, curriculum_stage_ref,
                        npc_controllers_old=None):
    """
    创建（或重建）环境 + NPC + 查询节点

    用于初始化和多地图切换。切换时:
    1. 关闭旧NPC
    2. 创建新env
    3. 查询新地图节点
    4. 创建新NPC
    5. 包装环境
    """
    current_scene = map_manager.current_scene
    current_config = map_manager.current_config

    # 关闭旧NPC
    if npc_controllers_old:
        for npc in npc_controllers_old:
            npc.close()
        time.sleep(1)

    # 创建主环境
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

    print(f"\n🚗 创建Learner环境 (地图: {current_scene})...")
    env = gym.make(current_config['env_id'], conf=conf)
    time.sleep(2)

    # 查询赛道节点
    try:
        handler = env.viewer.handler
        total_nodes_hint = current_config.get('estimated_nodes', 26)
        track_cache.query_nodes(handler, current_scene, total_nodes=total_nodes_hint)
    except Exception as e:
        print(f"⚠️ 节点查询失败: {e}")

    # 创建NPC
    npc_controllers = []
    npc_colors = [(255, 100, 100), (100, 255, 100), (255, 255, 100)]

    if args.num_npc > 0:
        print(f"\n🚗 创建 {args.num_npc} 个NPC障碍车...")
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
                # 根据课程阶段决定初始模式
                init_stage = curriculum_stage_ref.get('stage', 1)
                if init_stage >= 2:
                    npc.set_mode('static', 0.0)
                    # 连接后立即把NPC停到场外，避免和Learner重叠在默认起点
                    npc.set_position_node_coords(0, 100.0, 0, 0, 0, 0, 1)
                    # 立即启动驾驶线程 —— stage2 static模式会持续发刹车
                    # 这样第一次 reset() 把NPC teleport到赛道后，_drive_loop已经在
                    # 持续发 (0,0,brake=1)，NPC不会因物理引擎滑动
                    npc.start_driving()
                else:
                    npc.set_mode('static', 0.0)
                    npc.set_position_node_coords(0, 100.0, 0, 0, 0, 0, 1)
                npc_controllers.append(npc)
            time.sleep(2)

    # 包装环境
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


# ==================== 12. 主训练函数 ====================
def train(args):
    """主训练入口"""

    scenes = [s.strip() for s in args.scenes.split(',')]

    print("\n" + "="*80)
    print("🏎️ V9 避障超车训练 (观测对齐V8)")
    print("="*80)
    print(f"📋 配置:")
    print(f"   地图: {scenes}")
    print(f"   NPC数量: {args.num_npc}")
    print(f"   总步数: {args.total_steps:,}")
    print(f"   随机起点: {'是' if args.random_start else '否'}")
    print(f"   端口: {args.port}")
    print(f"   最大油门: {args.max_throttle}")
    print(f"   舵机保护: delta_max={args.delta_max}, LPF={'启用' if args.enable_lpf else '禁用'}(beta={args.beta})")
    print(f"   平滑惩罚: w_d={args.w_d}, w_dd={args.w_dd}, w_sat={args.w_sat}")
    print()

    # 1. 初始化
    map_manager = MultiMapManager(scenes, switch_interval=args.map_switch_interval)
    track_cache = TrackNodeCache()

    # start_stage → step_offset 映射
    stage_step_map = {1: 0, 2: 100000, 3: 200000, 4: 400000}
    start_step_offset = stage_step_map.get(args.start_stage, 0)

    # 根据起始阶段设置初始课程状态
    if args.start_stage >= 2:
        curriculum_stage_ref = {
            'stage': args.start_stage,
            'npc_active': True,
            'npc_random_pos': args.start_stage in (2, 3),
            'npc_costart': args.start_stage == 4,
        }
        print(f"🎓 直接从阶段{args.start_stage}开始 (step_offset={start_step_offset:,})")
    else:
        curriculum_stage_ref = {
            'stage': 1,
            'npc_active': False,
            'npc_random_pos': False,
            'npc_costart': False,
        }
    switch_flag_ref = {'should_switch': False}

    # 2. 创建环境 + NPC
    vec_env, npc_controllers, wrapped_env = create_env_and_npcs(
        map_manager, track_cache, args, curriculum_stage_ref)

    # 3. 创建模型
    policy_kwargs = dict(
        features_extractor_class=LightweightCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

    save_dir = "models/v9_overtake"
    os.makedirs(save_dir, exist_ok=True)

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"\n📦 加载预训练模型: {args.pretrained_model}")
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

    print(f"\n📋 PPO超参数 (对齐V8):")
    print(f"   学习率: {model.learning_rate}")
    print(f"   n_steps: {model.n_steps}")
    print(f"   batch_size: {model.batch_size}")
    print(f"   n_epochs: {model.n_epochs}")
    print(f"   gamma: {model.gamma}")
    print(f"   clip_range: {model.clip_range}")
    print(f"   target_kl: {model.target_kl}")
    print(f"   观测: CnnPolicy 8通道 (对齐V8)")

    # 4. 训练循环 (支持多地图切换)
    total_steps_done = 0
    remaining_steps = args.total_steps

    print(f"\n🚀 开始训练 ({args.total_steps:,} 步)...\n")

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
                print(f"\n🗺️ 切换地图中... (已完成 {total_steps_done:,}/{args.total_steps:,} 步)")

                new_scene = map_manager.next_scene()
                vec_env.close()
                time.sleep(2)

                vec_env, npc_controllers, wrapped_env = create_env_and_npcs(
                    map_manager, track_cache, args, curriculum_stage_ref,
                    npc_controllers_old=npc_controllers)

                model.set_env(vec_env)
                print(f"✅ 地图切换完成: {new_scene}")

        final_path = os.path.join(save_dir, "v9_overtake_final")
        model.save(final_path)
        final_pth = final_path + "_policy.pth"
        torch.save(model.policy.state_dict(), final_pth)
        print(f"\n✅ 训练完成! 模型保存: {final_path}.zip + {final_pth}")

    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断")
        interrupted_path = os.path.join(save_dir, f"v9_overtake_interrupted_{total_steps_done}")
        model.save(interrupted_path)
        interrupted_pth = interrupted_path + "_policy.pth"
        torch.save(model.policy.state_dict(), interrupted_pth)
        print(f"💾 中断模型已保存: {interrupted_path}.zip + {interrupted_pth}")

    except Exception as e:
        import traceback
        print(f"\n❌ 训练异常崩溃!")
        print(f"   异常类型: {type(e).__name__}")
        print(f"   异常信息: {e}")
        traceback.print_exc()
        try:
            crash_path = os.path.join(save_dir, f"v9_overtake_crash_{total_steps_done}")
            model.save(crash_path)
            print(f"💾 崩溃前模型已保存: {crash_path}.zip")
        except Exception:
            print("⚠️ 崩溃时保存模型也失败了")

    finally:
        for npc in npc_controllers:
            npc.close()
        vec_env.close()
        print("🔒 环境已关闭")


# ==================== 13. CLI ====================
def parse_args():
    parser = argparse.ArgumentParser(description='V9 避障超车训练 (观测对齐V8)')

    parser.add_argument('--scenes', type=str, default='waveshare', help='训练地图，逗号分隔')
    parser.add_argument('--port', type=int, default=9091, help='仿真器端口')
    parser.add_argument('--exe-path', type=str, default=None, help='仿真器路径')

    parser.add_argument('--total-steps', type=int, default=600000, help='总训练步数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--pretrained-model', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--start-stage', type=int, default=1, choices=[1,2,3,4],
                        help='从第几阶段开始 (1=基础驾驶, 2=静态避障, 3=慢速超车, 4=正常超车)')

    parser.add_argument('--num-npc', type=int, default=1, help='NPC障碍车数量 (0-3)')

    parser.add_argument('--random-start', action='store_true', default=True, help='启用随机起点')
    parser.add_argument('--no-random-start', action='store_false', dest='random_start')
    parser.add_argument('--map-switch-interval', type=int, default=50000, help='地图切换间隔步数')
    parser.add_argument('--max-throttle', type=float, default=0.3, help='最大油门')

    parser.add_argument('--delta-max', type=float, default=0.10, help='舵机slew-rate limit')
    parser.add_argument('--enable-lpf', action='store_true', default=True, help='启用低通滤波')
    parser.add_argument('--no-lpf', action='store_false', dest='enable_lpf')
    parser.add_argument('--beta', type=float, default=0.6, help='低通滤波系数')

    parser.add_argument('--w-d', type=float, default=0.25, help='|delta_steer| 权重')
    parser.add_argument('--w-dd', type=float, default=0.08, help='|jerk| 权重')
    parser.add_argument('--w-sat', type=float, default=0.15, help='saturation 权重')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
