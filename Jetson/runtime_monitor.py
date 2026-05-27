#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runtime_monitor.py - DonkeyCar noterowsnotedatanotemanage.py, note monkey-patch Vehicle.start notefirst
notedatanote Part, note manage.py note.

note:
  # note V8 notedata
  python runtime_monitor.py drive --model ~/mycar/models/v8_140000_steps_policy.pth --type v8

  # note(notemodel, note + note)
  python runtime_monitor.py drive

  # notefile
  python runtime_monitor.py drive --model xxx --type v8 --log-interval 0.2 --log-dir ~/mycar/monitor_logs

  # note
  python runtime_monitor.py drive --model xxx --type v8 --js

note manage.py drive note, note:
  --log-interval  note(note), default 0.5
  --log-dir       noteoutputdirectory, default ~/mycar/monitor_logs

output:
  monitor_logs/
    run_20260226_153000.csv      <- noterowsgeneratenote CSV
"""

import os
import sys
import time
import csv
import json
import math
import signal
import struct
import shlex
import threading
import subprocess
import numpy as np
import textwrap
from datetime import datetime

# ============================================================
# RP2040 notereadnote
# ============================================================
class RP2040SerialReader:
    """
    noteread RP2040 notedatanote.

    note (jetracer.cpp note):
      [0xAA, 0x55, frame_size, type, payload..., checksum]

    datanote payload (38 note):
      gyro_xyz(6) + accel_xyz(6) + euler_rpy(6) +
      odom_xy_yaw(6) + delta_xy_yaw(6) +
      motor_lvel(2) + motor_rvel(2) + motor_lset(2) + motor_rset(2)
    """

    HEAD1 = 0xAA
    HEAD2 = 0x55

    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._serial = None
        self._connected = False

        # notedata(noteread)
        self._data = {
            # IMU
            'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,       # rad/s
            'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,    # m/s^2
            'euler_roll': 0.0, 'euler_pitch': 0.0, 'euler_yaw': 0.0,  # degrees
            # note
            'odom_x': 0.0, 'odom_y': 0.0, 'odom_yaw': 0.0,     # meters / rad
            'delta_x': 0.0, 'delta_y': 0.0, 'delta_yaw': 0.0,  # meters / rad
            # note
            'motor_lvel': 0, 'motor_rvel': 0,                   # note
            'motor_lset': 0, 'motor_rset': 0,                   # note
            # notedata
            'frame_count': 0,
            'last_update': 0.0,
            'parse_errors': 0,
        }

    def start(self):
        """notereadnote"""
        try:
            import serial
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            self._connected = True
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"   RP2040 note: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"   RP2040 notefailed: {e}")
            self._connected = False
            return False

    def stop(self):
        """notereadnote"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._connected = False

    def get_data(self):
        """notedatanote(note)"""
        with self._lock:
            return dict(self._data)

    @property
    def is_connected(self):
        return self._connected

    def _read_loop(self):
        """note -- noteparsenote"""
        buf = bytearray(64)
        state = 'HEAD1'
        frame_size = 0

        while self._running:
            try:
                if state == 'HEAD1':
                    b = self._serial.read(1)
                    if len(b) == 0:
                        continue
                    if b[0] == self.HEAD1:
                        buf[0] = b[0]
                        state = 'HEAD2'

                elif state == 'HEAD2':
                    b = self._serial.read(1)
                    if len(b) == 0:
                        state = 'HEAD1'
                        continue
                    if b[0] == self.HEAD2:
                        buf[1] = b[0]
                        state = 'SIZE'
                    else:
                        state = 'HEAD1'

                elif state == 'SIZE':
                    b = self._serial.read(1)
                    if len(b) == 0:
                        state = 'HEAD1'
                        continue
                    frame_size = b[0]
                    buf[2] = b[0]
                    if frame_size < 5 or frame_size > 60:
                        state = 'HEAD1'
                        continue
                    state = 'DATA'

                elif state == 'DATA':
                    # read type + payload + checksum = frame_size - 3 note
                    remaining = frame_size - 3
                    chunk = self._serial.read(remaining)
                    if len(chunk)!= remaining:
                        state = 'HEAD1'
                        continue
                    buf[3:3 + remaining] = chunk

                    # note: sum(buf[0:frame_size-1]) == buf[frame_size-1]
                    calc_sum = sum(buf[:frame_size - 1]) & 0xFF
                    if calc_sum!= buf[frame_size - 1]:
                        with self._lock:
                            self._data['parse_errors'] += 1
                        state = 'HEAD1'
                        continue

                    # parsesucceeded
                    self._parse_frame(buf, frame_size)
                    state = 'HEAD1'

            except Exception as e:
                # note
                time.sleep(0.1)
                state = 'HEAD1'

    def _parse_frame(self, buf, size):
        """parse RP2040 datanote, note jetracer.cpp State_Handle note"""
        if size < 42:  # note motor data
            return

        d = buf

        # note (rad/s): raw/32768 * 2000 deg/s -> rad/s
        gyro_x = (struct.unpack('>h', bytes(d[4:6]))[0]) / 32768.0 * 2000.0 / 180.0 * 3.1415926
        gyro_y = (struct.unpack('>h', bytes(d[6:8]))[0]) / 32768.0 * 2000.0 / 180.0 * 3.1415926
        gyro_z = (struct.unpack('>h', bytes(d[8:10]))[0]) / 32768.0 * 2000.0 / 180.0 * 3.1415926

        # note (m/s^2): raw/32768 * 2g * 9.8
        accel_x = (struct.unpack('>h', bytes(d[10:12]))[0]) / 32768.0 * 2.0 * 9.8
        accel_y = (struct.unpack('>h', bytes(d[12:14]))[0]) / 32768.0 * 2.0 * 9.8
        accel_z = (struct.unpack('>h', bytes(d[14:16]))[0]) / 32768.0 * 2.0 * 9.8

        # note (note): raw / 10.0
        euler_roll  = (struct.unpack('>h', bytes(d[16:18]))[0]) / 10.0
        euler_pitch = (struct.unpack('>h', bytes(d[18:20]))[0]) / 10.0
        euler_yaw   = (struct.unpack('>h', bytes(d[20:22]))[0]) / 10.0

        # note (m): raw / 1000
        odom_x   = (struct.unpack('>h', bytes(d[22:24]))[0]) / 1000.0
        odom_y   = (struct.unpack('>h', bytes(d[24:26]))[0]) / 1000.0
        odom_yaw = (struct.unpack('>h', bytes(d[26:28]))[0]) / 1000.0

        # note (m): raw / 1000
        delta_x   = (struct.unpack('>h', bytes(d[28:30]))[0]) / 1000.0
        delta_y   = (struct.unpack('>h', bytes(d[30:32]))[0]) / 1000.0
        delta_yaw = (struct.unpack('>h', bytes(d[32:34]))[0]) / 1000.0

        # note
        motor_lvel = struct.unpack('>h', bytes(d[34:36]))[0]
        motor_rvel = struct.unpack('>h', bytes(d[36:38]))[0]
        motor_lset = struct.unpack('>h', bytes(d[38:40]))[0]
        motor_rset = struct.unpack('>h', bytes(d[40:42]))[0]

        with self._lock:
            self._data.update({
                'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
                'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
                'euler_roll': euler_roll, 'euler_pitch': euler_pitch, 'euler_yaw': euler_yaw,
                'odom_x': odom_x, 'odom_y': odom_y, 'odom_yaw': odom_yaw,
                'delta_x': delta_x, 'delta_y': delta_y, 'delta_yaw': delta_yaw,
                'motor_lvel': motor_lvel, 'motor_rvel': motor_rvel,
                'motor_lset': motor_lset, 'motor_rset': motor_rset,
                'last_update': time.time(),
            })
            self._data['frame_count'] += 1


# ============================================================
# ROS LiDAR readnote
# ============================================================
class RosLidarReader:
    """
    note ROS /scan note LaserScan, note scan note Python3 note.

    noterowsnote system python + rospy note, note runtime_monitor.py
    note Python3 note ROS note.
    """

    def __init__(self, topic='/scan',
                 ros_setup='/opt/ros/melodic/setup.bash',
                 workspace_setup='~/catkin_ws/devel/setup.bash',
                 python_cmd='/usr/bin/python'):
        self.topic = topic
        self.ros_setup = ros_setup
        self.workspace_setup = workspace_setup
        self.python_cmd = python_cmd
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._process = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._data = self._make_default_data()

    def _make_default_data(self):
        return {
            'frame_count': 0,
            'valid_points': 0,
            'points_total': 0,
            'last_update': 0.0,
            'scan_age_ms': -1.0,
            'nearest_min': -1.0,
            'parse_errors': 0,
            'angle_min': 0.0,
            'angle_max': 0.0,
            'angle_increment': 0.0,
            'range_min': 0.0,
            'range_max': 0.0,
            'ranges': [],
            'intensities': [],
        }

    def _build_helper_script(self):
        return textwrap.dedent(f"""
            import json
            import math
            import sys
            import time
            import rospy
            from sensor_msgs.msg import LaserScan
            from std_srvs.srv import Empty

            TOPIC = {self.topic!r}

            def encode_array(values):
                encoded = []
                for value in values:
                    if math.isnan(value) or math.isinf(value):
                        encoded.append(None)
                    else:
                        encoded.append(round(float(value), 4))
                return encoded

            def emit_message(msg, frame_count):
                valid_ranges = [
                    float(r) for r in msg.ranges
                    if (not math.isnan(r)) and (not math.isinf(r)) and r >= msg.range_min and r <= msg.range_max
                ]

                stamp = msg.header.stamp.to_sec()
                if stamp <= 0:
                    stamp = time.time()

                frame_count += 1
                payload = {{
                    'frame_count': frame_count,
                    'stamp': stamp,
                    'valid_points': len(valid_ranges),
                    'points_total': len(msg.ranges),
                    'nearest_min': -1.0 if not valid_ranges else round(min(valid_ranges), 4),
                    'angle_min': round(float(msg.angle_min), 6),
                    'angle_max': round(float(msg.angle_max), 6),
                    'angle_increment': round(float(msg.angle_increment), 6),
                    'range_min': round(float(msg.range_min), 4),
                    'range_max': round(float(msg.range_max), 4),
                    'ranges': encode_array(msg.ranges),
                    'intensities': encode_array(msg.intensities) if msg.intensities else [],
                }}

                sys.stdout.write(json.dumps(payload, separators=(',', ':')) + '\\n')
                sys.stdout.flush()
                return frame_count

            rospy.init_node('runtime_monitor_lidar_bridge', anonymous=True, disable_signals=True)
            try:
                rospy.wait_for_service('/start_motor', timeout=2.0)
                start_motor = rospy.ServiceProxy('/start_motor', Empty)
                start_motor()
            except Exception:
                pass
            frame_count = 0
            while not rospy.is_shutdown():
                try:
                    msg = rospy.wait_for_message(TOPIC, LaserScan, timeout=2.0)
                    frame_count = emit_message(msg, frame_count)
                except rospy.ROSException:
                    continue
        """).strip()

    def start(self):
        if self._running:
            return True

        helper_script = self._build_helper_script()
        command = (
            f"source {shlex.quote(self.ros_setup)} >/dev/null 2>&1 && "
            f"if [ -f {self.workspace_setup} ]; then source {self.workspace_setup} >/dev/null 2>&1; fi && "
            f"{self.python_cmd} -u -c {shlex.quote(helper_script)}"
        )

        try:
            self._process = subprocess.Popen(
                ['/bin/bash', '-lc', command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )
            self._running = True
            self._stdout_thread = threading.Thread(target=self._stdout_loop, daemon=True)
            self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
            self._stdout_thread.start()
            self._stderr_thread.start()
            self._connected = True
            print(f"   LiDAR ROS note: topic={self.topic} (raw /scan)")
            return True
        except Exception as e:
            print(f"   LiDAR ROS notefailed: {e}")
            self._connected = False
            self._running = False
            return False

    def stop(self):
        self._running = False
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except Exception:
                self._process.kill()
        if self._stdout_thread:
            self._stdout_thread.join(timeout=1.0)
        if self._stderr_thread:
            self._stderr_thread.join(timeout=1.0)
        self._connected = False

    def _stdout_loop(self):
        try:
            for line in self._process.stdout:
                if not self._running:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    with self._lock:
                        self._data['parse_errors'] += 1
                    continue

                with self._lock:
                    self._data['frame_count'] = payload.get('frame_count', self._data['frame_count'])
                    self._data['valid_points'] = payload.get('valid_points', 0)
                    self._data['points_total'] = payload.get('points_total', 0)
                    self._data['nearest_min'] = payload.get('nearest_min', -1.0)
                    self._data['last_update'] = payload.get('stamp', time.time())
                    self._data['scan_age_ms'] = max(0.0, (time.time() - self._data['last_update']) * 1000.0)
                    self._data['angle_min'] = payload.get('angle_min', 0.0)
                    self._data['angle_max'] = payload.get('angle_max', 0.0)
                    self._data['angle_increment'] = payload.get('angle_increment', 0.0)
                    self._data['range_min'] = payload.get('range_min', 0.0)
                    self._data['range_max'] = payload.get('range_max', 0.0)
                    self._data['ranges'] = payload.get('ranges', [])
                    self._data['intensities'] = payload.get('intensities', [])
                    self._connected = True
        finally:
            self._connected = False

    def _stderr_loop(self):
        try:
            for line in self._process.stderr:
                if not self._running:
                    break
                text = line.strip()
                if text:
                    print(f"   [LiDAR ROS] {text}")
        finally:
            return

    def get_data(self):
        with self._lock:
            data = dict(self._data)
            data['ranges'] = list(self._data['ranges'])
            data['intensities'] = list(self._data['intensities'])
            if data['last_update'] > 0:
                data['scan_age_ms'] = max(0.0, (time.time() - data['last_update']) * 1000.0)
            return data

    @property
    def is_connected(self):
        return self._connected and self._process is not None and self._process.poll() is None


# ============================================================
# datanote Part
# ============================================================
class DataCollector:
    """
    DonkeyCar Vehicle Part -- noterowsnotedatanoteVehicle note, noteread Memory notedata,
    note CSV file.
    """

    # note DonkeyCar note(note)
    # note: IMU/notedatanote RP2040 note, note DonkeyCar note
    INPUT_KEYS = [
        # notedata(note)
        'cam/image_array',      # note
        'user/angle',           # note
        'user/throttle',        # note
        'pilot/angle',          # AI note
        'pilot/throttle',       # AI note
        'angle',                # noterowsnote
        'throttle',             # noterowsnote
        'user/mode',            # note
        'recording',            # note
        'run_pilot',            # AI note
        'tub/num_records',      # note
    ]

    BASE_CSV_FIELDS = [
        # note
        'timestamp', 'sample_id', 'frame', 'elapsed_sec', 'loop_dt_ms', 'effective_fps',
        # note & note
        'mode', 'recording', 'run_pilot', 'tub_records',
        # noteinput
        'user_angle', 'user_throttle',
        # AI output
        'pilot_angle', 'pilot_throttle',
        # noterows
        'final_angle', 'final_throttle',
        # note(AI vs note)
        'angle_diff',
        # note
        'img_mean', 'img_std', 'img_brightness', 'img_contrast',
        # ── RP2040 note (note) ──
        # IMU note (rad/s)
        'gyro_x', 'gyro_y', 'gyro_z',
        # IMU note (m/s^2)
        'accel_x', 'accel_y', 'accel_z',
        # IMU note (note)
        'euler_roll', 'euler_pitch', 'euler_yaw',
        # note (m)
        'odom_x', 'odom_y', 'odom_yaw',
        # note
        'delta_x', 'delta_y', 'delta_yaw',
        # note (note/20ms)
        'motor_lvel', 'motor_rvel', 'motor_lset', 'motor_rset',
        # RP2040 note & parsenote
        'rp2040_frames', 'rp2040_errors',
        # ── Jetson note ──
        # note (7note)
        'cpu_temp', 'gpu_temp', 'ao_temp', 'pll_temp', 'fan_temp',
        'pmic_temp', 'wifi_temp',
        # note
        'gpu_load_pct', 'cpu_load_pct',
        # CPU/GPU note
        'cpu_freq_mhz', 'gpu_freq_mhz',
        # note & note
        'mem_used_mb', 'mem_total_mb', 'mem_used_pct',
        'swap_used_mb', 'swap_total_mb',
        # note
        'disk_used_pct',
        # note
        'fan_pwm',
        # WiFi note
        'wifi_rssi_dbm', 'wifi_link_quality',
        # ── note ──
        # INA219 note (I2C 0x41)
        'battery_voltage_v', 'battery_current_ma',
        # INA3221 Jetson note (mW)
        'power_in_mw', 'power_gpu_mw', 'power_cpu_mw',
    ]

    def __init__(self, log_path, log_interval=0.5, serial_reader=None, lidar_reader=None):
        self.log_path = log_path
        self.log_interval = log_interval
        self.last_log_time = 0
        self.frame_count = 0
        self.sample_count = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.serial_reader = serial_reader  # RP2040SerialReader note
        self.lidar_reader = lidar_reader
        self.lidar_fields = []
        self.lidar_raw_path = None
        self.lidar_raw_file = None
        if lidar_reader:
            self.lidar_fields = [
                'lidar_frames', 'lidar_valid_points', 'lidar_scan_age_ms',
                'lidar_points_total', 'lidar_nearest_min',
                'lidar_angle_min_rad', 'lidar_angle_max_rad',
                'lidar_angle_increment_rad',
                'lidar_range_min', 'lidar_range_max',
            ]
        self.csv_fields = list(self.BASE_CSV_FIELDS) + self.lidar_fields

        # note CSV
        self.csv_file = open(self.log_path, 'w', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fields)
        self.writer.writeheader()
        self.csv_file.flush()
        if lidar_reader:
            log_stem, _ = os.path.splitext(self.log_path)
            self.lidar_raw_path = f'{log_stem}_lidar_raw.jsonl'
            self.lidar_raw_file = open(self.lidar_raw_path, 'w')

        # note(note)
        self._cached_temps = {}
        self._last_temp_read = 0
        # notecompute CPU note /proc/stat note
        self._prev_cpu_stat = None
        # Vehicle note(note monkey-patch note)
        self.vehicle = None

        print(f"\n{'='*60}")
        print(f"DataCollector note")
        print(f"   notefile: {self.log_path}")
        print(f"   note: {self.log_interval}s")
        print(f"   DonkeyCar note: {len(self.INPUT_KEYS)} note")
        print(f"   RP2040 note: {'note' if serial_reader and serial_reader.is_connected else 'note'}")
        print(f"   LiDAR ROS: {'note' if lidar_reader and lidar_reader.is_connected else 'note'}")
        if self.lidar_raw_path:
            print(f"   LiDAR Raw: {self.lidar_raw_path}")
        print(f"   CSV note: {len(self.csv_fields)} note")
        print(f"{'='*60}\n")

    # ---- Jetson note ----

    def _read_cpu_load(self):
        """note /proc/stat notecompute CPU note(%)"""
        try:
            with open('/proc/stat') as f:
                line = f.readline()   # cpu  user nice system idle iowait...
            vals = list(map(int, line.split()[1:]))
            total = sum(vals)
            idle  = vals[3]
            if self._prev_cpu_stat is None:
                self._prev_cpu_stat = (total, idle)
                return -1.0
            prev_total, prev_idle = self._prev_cpu_stat
            self._prev_cpu_stat = (total, idle)
            d_total = total - prev_total
            d_idle  = idle  - prev_idle
            return round((1.0 - d_idle / d_total) * 100.0, 1) if d_total > 0 else -1.0
        except Exception:
            return -1.0

    def _read_mem(self):
        """read /proc/meminfo note (used_mb, total_mb, used_pct, swap_used_mb, swap_total_mb)"""
        try:
            info = {}
            with open('/proc/meminfo') as f:
                for line in f:
                    k, v = line.split(':', 1)
                    info[k.strip()] = int(v.split()[0])   # kB
            total = info.get('MemTotal', 0)
            avail = info.get('MemAvailable', info.get('MemFree', 0))
            used  = total - avail
            pct   = round(used / total * 100.0, 1) if total > 0 else -1.0
            swap_total = info.get('SwapTotal', 0)
            swap_free  = info.get('SwapFree', 0)
            swap_used  = swap_total - swap_free
            return (round(used / 1024, 1), round(total / 1024, 1), pct,
                    round(swap_used / 1024, 1), round(swap_total / 1024, 1))
        except Exception:
            return -1.0, -1.0, -1.0, -1.0, -1.0

    def _read_wifi(self):
        """read WiFi RSSI note, note (rssi_dbm, link_quality)"""
        try:
            with open('/proc/net/wireless') as f:
                lines = f.readlines()
            if len(lines) >= 3:
                parts = lines[2].split()
                link_quality = float(parts[2].rstrip('.'))
                rssi_dbm = float(parts[3].rstrip('.'))
                return rssi_dbm, link_quality
        except Exception:
            pass
        return -999.0, -1.0

    def _read_cpu_freq(self):
        """read CPU note (MHz)"""
        try:
            freqs = []
            for i in range(4):
                with open(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq') as f:
                    freqs.append(int(f.read().strip()) / 1000.0)  # kHz -> MHz
            return round(sum(freqs) / len(freqs), 0) if freqs else -1.0
        except Exception:
            return -1.0

    def _read_gpu_freq(self):
        """read GPU currentnote (MHz)"""
        try:
            with open('/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq') as f:
                return round(int(f.read().strip()) / 1e6, 0)  # Hz -> MHz
        except Exception:
            return -1.0

    def _read_fan_pwm(self):
        """readnote PWM note (0-255)"""
        try:
            with open('/sys/devices/pwm-fan/target_pwm') as f:
                return int(f.read().strip())
        except Exception:
            return -1

    def _read_disk_usage(self):
        """readnote (%)"""
        try:
            st = os.statvfs('/')
            total = st.f_blocks * st.f_frsize
            free  = st.f_bavail * st.f_frsize
            used_pct = round((1.0 - free / total) * 100.0, 1) if total > 0 else -1.0
            return used_pct
        except Exception:
            return -1.0

    def _read_ina219(self):
        """read INA219 note (I2C bus 1, addr 0x41)"""
        try:
            if not hasattr(self, '_smbus'):
                import smbus
                self._smbus = smbus.SMBus(1)
            bus = self._smbus
            # Bus voltage register (0x02) - big-endian
            raw = bus.read_word_data(0x41, 0x02)
            raw = ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)
            voltage = (raw >> 3) * 0.004  # LSB = 4mV
            # Shunt voltage register (0x01)
            raw = bus.read_word_data(0x41, 0x01)
            raw = ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)
            if raw & 0x8000:
                raw -= 65536
            shunt_mv = raw * 0.01  # LSB = 10uV
            # note 0.1 ohm note (noteconfiguration)
            current_ma = shunt_mv / 0.1
            return round(voltage, 3), round(current_ma, 1)
        except Exception:
            return -1.0, -1.0

    def _read_ina3221(self):
        """read INA3221 Jetson note (mW), note sudo note"""
        try:
            if not hasattr(self, '_ina3221_ok'):
                self._ina3221_ok = os.access(
                    '/sys/bus/iio/devices/iio:device0/in_power0_input', os.R_OK)
            if not self._ina3221_ok:
                return -1.0, -1.0, -1.0
            powers = []
            for i in range(3):
                with open(f'/sys/bus/iio/devices/iio:device0/in_power{i}_input') as f:
                    powers.append(int(f.read().strip()))
            return powers[0], powers[1], powers[2]  # IN, GPU, CPU
        except Exception:
            return -1.0, -1.0, -1.0

    def _read_thermal_zones(self):
        """noteread Jetson note, GPU/CPU note, note, note(note)"""
        now = time.time()
        if now - self._last_temp_read < 1.0 and self._cached_temps:
            return self._cached_temps

        zones = {
            'ao_temp':   0,   # AO-therm
            'cpu_temp':  1,   # CPU-therm
            'gpu_temp':  2,   # GPU-therm
            'pll_temp':  3,   # PLL-therm
            'pmic_temp': 4,   # PMIC-Die
            'fan_temp':  5,   # thermal-fan-est
            'wifi_temp': 6,   # iwlwifi
        }
        result = {}
        for name, zone_id in zones.items():
            try:
                with open(f'/sys/class/thermal/thermal_zone{zone_id}/temp') as f:
                    result[name] = int(f.read().strip()) / 1000.0
            except Exception:
                result[name] = -1.0

        # GPU load (%)
        try:
            with open('/sys/devices/gpu.0/load') as f:
                result['gpu_load_pct'] = int(f.read().strip()) / 10.0
        except Exception:
            result['gpu_load_pct'] = -1.0

        # CPU load (%)
        result['cpu_load_pct'] = self._read_cpu_load()

        # CPU/GPU note
        result['cpu_freq_mhz'] = self._read_cpu_freq()
        result['gpu_freq_mhz'] = self._read_gpu_freq()

        # note + note
        (result['mem_used_mb'], result['mem_total_mb'], result['mem_used_pct'],
         result['swap_used_mb'], result['swap_total_mb']) = self._read_mem()

        # note
        result['disk_used_pct'] = self._read_disk_usage()

        # note
        result['fan_pwm'] = self._read_fan_pwm()

        # WiFi
        result['wifi_rssi_dbm'], result['wifi_link_quality'] = self._read_wifi()

        # INA219 note
        result['battery_voltage_v'], result['battery_current_ma'] = self._read_ina219()

        # INA3221 note
        result['power_in_mw'], result['power_gpu_mw'], result['power_cpu_mw'] = self._read_ina3221()

        self._cached_temps = result
        self._last_temp_read = now
        return result

    # ---- DonkeyCar Part note ----

    def run(self, img_arr, user_angle, user_throttle,
            pilot_angle, pilot_throttle,
            final_angle, final_throttle,
            mode, recording, run_pilot, tub_records):
        """Vehicle noteINPUT_KEYS note(11notedata).
        IMU/note/notedatanote RP2040 note, note DonkeyCar note.
        """

        now = time.time()
        self.frame_count += 1
        loop_dt = now - self.prev_time
        self.prev_time = now
        elapsed = now - self.start_time

        # note
        if now - self.last_log_time < self.log_interval:
            return
        self.last_log_time = now
        self.sample_count += 1

        # -- note --
        img_mean = img_std = img_brightness = img_contrast = 0.0
        if img_arr is not None:
            flat = img_arr.astype(np.float32)
            img_mean = float(np.mean(flat))
            img_std = float(np.std(flat))
            # note = RGB note(note)
            img_brightness = float(np.mean(flat[:,:,:3]))
            # note = note
            gray = 0.299 * flat[:,:, 0] + 0.587 * flat[:,:, 1] + 0.114 * flat[:,:, 2]
            img_contrast = float(np.std(gray))

        # -- RP2040 notedata --
        if self.serial_reader and self.serial_reader.is_connected:
            sd = self.serial_reader.get_data()
        else:
            sd = {}  # note 0

        # -- LiDAR ROS data --
        if self.lidar_reader and self.lidar_reader.is_connected:
            ld = self.lidar_reader.get_data()
        else:
            ld = {}

        # -- notedata --
        temps = self._read_thermal_zones()

        # -- computenote --
        angle_diff = 0.0
        if pilot_angle is not None and user_angle is not None:
            angle_diff = (pilot_angle or 0) - (user_angle or 0)

        effective_fps = self.frame_count / elapsed if elapsed > 0 else 0

        # -- note CSV --
        row = {
            'timestamp':       f'{now:.3f}',
            'sample_id':       self.sample_count,
            'frame':           self.frame_count,
            'elapsed_sec':     f'{elapsed:.2f}',
            'loop_dt_ms':      f'{loop_dt * 1000:.1f}',
            'effective_fps':   f'{effective_fps:.1f}',
            'mode':            mode or 'unknown',
            'recording':       bool(recording),
            'run_pilot':       bool(run_pilot),
            'tub_records':     tub_records or 0,
            'user_angle':      f'{user_angle or 0:.4f}',
            'user_throttle':   f'{user_throttle or 0:.4f}',
            'pilot_angle':     f'{pilot_angle or 0:.4f}',
            'pilot_throttle':  f'{pilot_throttle or 0:.4f}',
            'final_angle':     f'{final_angle or 0:.4f}',
            'final_throttle':  f'{final_throttle or 0:.4f}',
            'angle_diff':      f'{angle_diff:.4f}',
            'img_mean':        f'{img_mean:.2f}',
            'img_std':         f'{img_std:.2f}',
            'img_brightness':  f'{img_brightness:.2f}',
            'img_contrast':    f'{img_contrast:.2f}',
            # RP2040 note
            'gyro_x':          f"{sd.get('gyro_x', 0):.4f}",
            'gyro_y':          f"{sd.get('gyro_y', 0):.4f}",
            'gyro_z':          f"{sd.get('gyro_z', 0):.4f}",
            'accel_x':         f"{sd.get('accel_x', 0):.4f}",
            'accel_y':         f"{sd.get('accel_y', 0):.4f}",
            'accel_z':         f"{sd.get('accel_z', 0):.4f}",
            'euler_roll':      f"{sd.get('euler_roll', 0):.2f}",
            'euler_pitch':     f"{sd.get('euler_pitch', 0):.2f}",
            'euler_yaw':       f"{sd.get('euler_yaw', 0):.2f}",
            'odom_x':          f"{sd.get('odom_x', 0):.4f}",
            'odom_y':          f"{sd.get('odom_y', 0):.4f}",
            'odom_yaw':        f"{sd.get('odom_yaw', 0):.4f}",
            'delta_x':         f"{sd.get('delta_x', 0):.4f}",
            'delta_y':         f"{sd.get('delta_y', 0):.4f}",
            'delta_yaw':       f"{sd.get('delta_yaw', 0):.4f}",
            'motor_lvel':      sd.get('motor_lvel', 0),
            'motor_rvel':      sd.get('motor_rvel', 0),
            'motor_lset':      sd.get('motor_lset', 0),
            'motor_rset':      sd.get('motor_rset', 0),
            'rp2040_frames':   sd.get('frame_count', 0),
            'rp2040_errors':   sd.get('parse_errors', 0),
            'cpu_temp':        f'{temps["cpu_temp"]:.1f}',
            'gpu_temp':        f'{temps["gpu_temp"]:.1f}',
            'ao_temp':         f'{temps["ao_temp"]:.1f}',
            'pll_temp':        f'{temps["pll_temp"]:.1f}',
            'fan_temp':        f'{temps["fan_temp"]:.1f}',
            'pmic_temp':       f'{temps["pmic_temp"]:.1f}',
            'wifi_temp':       f'{temps["wifi_temp"]:.1f}',
            'gpu_load_pct':    f'{temps["gpu_load_pct"]:.1f}',
            'cpu_load_pct':    f'{temps["cpu_load_pct"]:.1f}',
            'cpu_freq_mhz':    f'{temps["cpu_freq_mhz"]:.0f}',
            'gpu_freq_mhz':    f'{temps["gpu_freq_mhz"]:.0f}',
            'mem_used_mb':     f'{temps["mem_used_mb"]:.0f}',
            'mem_total_mb':    f'{temps["mem_total_mb"]:.0f}',
            'mem_used_pct':    f'{temps["mem_used_pct"]:.1f}',
            'swap_used_mb':    f'{temps["swap_used_mb"]:.0f}',
            'swap_total_mb':   f'{temps["swap_total_mb"]:.0f}',
            'disk_used_pct':   f'{temps["disk_used_pct"]:.1f}',
            'fan_pwm':         temps['fan_pwm'],
            'wifi_rssi_dbm':   f'{temps["wifi_rssi_dbm"]:.0f}',
            'wifi_link_quality': f'{temps["wifi_link_quality"]:.0f}',
            'battery_voltage_v':  f'{temps["battery_voltage_v"]:.3f}',
            'battery_current_ma': f'{temps["battery_current_ma"]:.1f}',
            'power_in_mw':     f'{temps["power_in_mw"]:.0f}',
            'power_gpu_mw':    f'{temps["power_gpu_mw"]:.0f}',
            'power_cpu_mw':    f'{temps["power_cpu_mw"]:.0f}',
        }
        if self.lidar_reader:
            row.update({
                'lidar_frames':        ld.get('frame_count', 0),
                'lidar_valid_points':  ld.get('valid_points', 0),
                'lidar_points_total':  ld.get('points_total', 0),
                'lidar_scan_age_ms':   f"{ld.get('scan_age_ms', -1.0):.1f}",
                'lidar_nearest_min':   f"{ld.get('nearest_min', -1.0):.4f}",
                'lidar_angle_min_rad': f"{ld.get('angle_min', 0.0):.6f}",
                'lidar_angle_max_rad': f"{ld.get('angle_max', 0.0):.6f}",
                'lidar_angle_increment_rad': f"{ld.get('angle_increment', 0.0):.6f}",
                'lidar_range_min':     f"{ld.get('range_min', 0.0):.4f}",
                'lidar_range_max':     f"{ld.get('range_max', 0.0):.4f}",
            })
        self.writer.writerow(row)
        self.csv_file.flush()
        if self.lidar_raw_file and ld.get('frame_count', 0) > 0:
            tub_record_index_est = -1
            if recording and tub_records:
                tub_record_index_est = max(int(tub_records) - 1, 0)
            raw_record = {
                'timestamp': round(now, 3),
                'sample_id': self.sample_count,
                'frame': self.frame_count,
                'elapsed_sec': round(elapsed, 3),
                'mode': mode or 'unknown',
                'recording': bool(recording),
                'run_pilot': bool(run_pilot),
                'tub_records': int(tub_records or 0),
                'tub_record_index_est': tub_record_index_est,
                'user_angle': float(user_angle or 0.0),
                'user_throttle': float(user_throttle or 0.0),
                'pilot_angle': float(pilot_angle or 0.0),
                'pilot_throttle': float(pilot_throttle or 0.0),
                'final_angle': float(final_angle or 0.0),
                'final_throttle': float(final_throttle or 0.0),
                'lidar': {
                    'frame_count': ld.get('frame_count', 0),
                    'stamp': ld.get('last_update', 0.0),
                    'scan_age_ms': round(ld.get('scan_age_ms', -1.0), 1),
                    'valid_points': ld.get('valid_points', 0),
                    'points_total': ld.get('points_total', 0),
                    'nearest_min': ld.get('nearest_min', -1.0),
                    'angle_min': ld.get('angle_min', 0.0),
                    'angle_max': ld.get('angle_max', 0.0),
                    'angle_increment': ld.get('angle_increment', 0.0),
                    'range_min': ld.get('range_min', 0.0),
                    'range_max': ld.get('range_max', 0.0),
                    'ranges': ld.get('ranges', []),
                    'intensities': ld.get('intensities', []),
                },
            }
            self.lidar_raw_file.write(json.dumps(raw_record, separators=(',', ':')) + '\n')
            self.lidar_raw_file.flush()

        # note(note60note)
        if self.frame_count % int(60 / max(self.log_interval, 0.01)) == 0:
            print(f"\n[{elapsed:.0f}s] frame={self.frame_count} fps={effective_fps:.1f} mode={mode}")
            print(f"   steer={final_angle or 0:+.3f}  thr={final_throttle or 0:.3f}  "
                  f"(pilot steer={pilot_angle or 0:+.3f} thr={pilot_throttle or 0:.3f})  "
                  f"diff={angle_diff:+.3f}")
            print(f"   CPU={temps['cpu_temp']:.0f}C  GPU={temps['gpu_temp']:.0f}C  "
                  f"AO={temps['ao_temp']:.0f}C  PLL={temps['pll_temp']:.0f}C  "
                  f"PMIC={temps['pmic_temp']:.0f}C  Fan={temps['fan_temp']:.0f}C")
            print(f"   CPUnote={temps['cpu_load_pct']:.0f}%@{temps['cpu_freq_mhz']:.0f}MHz  "
                  f"GPUnote={temps['gpu_load_pct']:.0f}%@{temps['gpu_freq_mhz']:.0f}MHz  "
                  f"FanPWM={temps['fan_pwm']}")
            print(f"   note={temps['mem_used_mb']:.0f}/{temps['mem_total_mb']:.0f}MB "
                  f"({temps['mem_used_pct']:.0f}%)  "
                  f"SWAP={temps['swap_used_mb']:.0f}/{temps['swap_total_mb']:.0f}MB  "
                  f"note={temps['disk_used_pct']:.0f}%")
            bv = temps['battery_voltage_v']
            bi = temps['battery_current_ma']
            pw = temps['power_in_mw']
            print(f"   note={bv:.2f}V/{bi:.0f}mA  "
                  f"note: IN={pw:.0f}mW GPU={temps['power_gpu_mw']:.0f}mW CPU={temps['power_cpu_mw']:.0f}mW  "
                  f"WiFi={temps['wifi_rssi_dbm']:.0f}dBm")
            print(f"   note={img_brightness:.0f}  note={img_contrast:.0f}  "
                  f"note={img_mean:.0f}  \u03c3={img_std:.0f}")
            if sd.get('frame_count', 0) > 0:
                print(f"   RP2040: gyro=({sd.get('gyro_x',0):.2f},{sd.get('gyro_y',0):.2f},{sd.get('gyro_z',0):.2f})  "
                      f"accel=({sd.get('accel_x',0):.1f},{sd.get('accel_y',0):.1f},{sd.get('accel_z',0):.1f})  "
                      f"yaw={sd.get('euler_yaw',0):.1f}deg")
                print(f"   odom=({sd.get('odom_x',0):.3f},{sd.get('odom_y',0):.3f})  "
                      f"motor L={sd.get('motor_lvel',0)}/R={sd.get('motor_rvel',0)}  "
                      f"frames={sd.get('frame_count',0)} errs={sd.get('parse_errors',0)}")
            if ld.get('frame_count', 0) > 0:
                print(f"   LiDAR: points={ld.get('valid_points', 0)}/{ld.get('points_total', 0)}  "
                      f"near={ld.get('nearest_min', -1.0):.2f}m  "
                      f"angle=[{ld.get('angle_min', 0.0):.2f},{ld.get('angle_max', 0.0):.2f}]  "
                      f"inc={ld.get('angle_increment', 0.0):.4f}rad  "
                      f"age={ld.get('scan_age_ms', -1.0):.0f}ms")

    def shutdown(self):
        self.csv_file.close()
        if self.lidar_raw_file:
            self.lidar_raw_file.close()
        if self.serial_reader:
            self.serial_reader.stop()
        if self.lidar_reader:
            self.lidar_reader.stop()
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"\n{'='*60}")
        print(f"DataCollector note")
        print(f"   note: {self.frame_count}")
        print(f"   noterowsnote: {elapsed:.1f}s")
        print(f"   note FPS: {avg_fps:.1f}")
        if self.serial_reader:
            sd = self.serial_reader.get_data()
            print(f"   RP2040 note: {sd.get('frame_count',0)}, parsenote: {sd.get('parse_errors',0)}")
        if self.lidar_reader:
            ld = self.lidar_reader.get_data()
            print(f"   LiDAR note: {ld.get('frame_count',0)}, noteobstacle: {ld.get('nearest_min',-1.0):.2f}m")
        print(f"   notesave: {self.log_path}")
        if self.lidar_raw_path:
            print(f"   LiDAR Raw: {self.lidar_raw_path}")
        print(f"{'='*60}\n")


# ============================================================
# Monkey-patch Vehicle.start note
# ============================================================
def install_monitor(log_dir, log_interval, serial_port='/dev/ttyACM0',
                    lidar_topic='/scan', enable_lidar=True):
    """
    Monkey-patch dk.vehicle.Vehicle.start, note Vehicle notefirst
    note DataCollector Part note RP2040 notereadnote.
    manage.py note.
    """
    import donkeycar as dk
    from donkeycar.vehicle import Vehicle

    # note RP2040 notereadnote
    serial_reader = RP2040SerialReader(port=serial_port)
    serial_ok = serial_reader.start()
    lidar_reader = None
    lidar_ok = False
    if enable_lidar:
        lidar_reader = RosLidarReader(topic=lidar_topic)
        lidar_ok = lidar_reader.start()

    _original_start = Vehicle.start

    def _patched_start(self, rate_hz=10, max_loop_count=None, verbose=False):
        # generatenotefilenote
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'run_{ts}.csv')

        # note(notereadnote)
        collector = DataCollector(log_path=log_path, log_interval=log_interval,
                                  serial_reader=serial_reader if serial_ok else None,
                                  lidar_reader=lidar_reader if lidar_ok else None)
        collector.vehicle = self
        self.add(collector,
                 inputs=DataCollector.INPUT_KEYS,
                 outputs=[],
                 threaded=False)

        print(f"DataCollector note Vehicle note (rate={rate_hz}Hz)")

        return _original_start(self, rate_hz=rate_hz,
                               max_loop_count=max_loop_count,
                               verbose=verbose)

    Vehicle.start = _patched_start


# ============================================================
# CLI entry point
# ============================================================
def main():
    """
    noterowsentry point -- note manage.py note, note.
    """
    # note stdout note(SSH note print noteoutput)
    import io
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), 'wb', 0), write_through=True)

    import argparse

    parser = argparse.ArgumentParser(
        description='DonkeyCar noterowsnotedatanote(note manage.py)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
note:
  python runtime_monitor.py drive --model ~/mycar/models/v8_140000_steps_policy.pth --type v8
  python runtime_monitor.py drive --model xxx --type v8 --js --log-interval 0.2
  python runtime_monitor.py drive  # note, note + notedata
        """)

    parser.add_argument('command', choices=['drive'], help='noterowsnote')
    parser.add_argument('--model', type=str, default=None, help='modelfilepath')
    parser.add_argument('--type', type=str, default=None,
                        choices=['linear', 'categorical', 'v8'],
                        help='modelclassnote')
    parser.add_argument('--js', action='store_true', help='note')
    parser.add_argument('--myconfig', type=str, default='myconfig.py',
                        help='noteconfigurationfile')
    parser.add_argument('--meta', type=str, nargs='*', default=[],
                        help='notedata key:value')

    # note
    parser.add_argument('--log-interval', type=float, default=0.5,
                        help='note(note), default 0.5')
    parser.add_argument('--log-dir', type=str,
                        default=os.path.expanduser('~/mycar/monitor_logs'),
                        help='noteoutputdirectory')
    parser.add_argument('--serial-port', type=str, default='/dev/ttyACM0',
                        help='RP2040 note (default /dev/ttyACM0)')
    parser.add_argument('--lidar-topic', type=str, default='/scan',
                        help='LiDAR LaserScan topic (default /scan)')
    parser.add_argument('--disable-lidar', action='store_true',
                        help='note LiDAR note')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("DonkeyCar Runtime Monitor + RP2040 Sensor Bridge")
    print("=" * 60)
    print(f"   model: {args.model or 'note(note)'}")
    print(f"   classnote: {args.type or 'notedetection'}")
    print(f"   note: {args.js}")
    print(f"   note: {args.log_interval}s")
    print(f"   notedirectory: {args.log_dir}")
    print(f"   RP2040 note: {args.serial_port}")
    print(f"   LiDAR: {'note' if args.disable_lidar else f'{args.lidar_topic} (raw /scan)'}")
    print("=" * 60 + "\n")

    import donkeycar as dk
    cfg = dk.load_config(myconfig=args.myconfig)

    # note DonkeyCar note IMU/note Part(notedatanote RP2040 note)
    cfg.HAVE_IMU = False
    cfg.HAVE_ODOM = False

    install_monitor(log_dir=args.log_dir, log_interval=args.log_interval,
                    serial_port=args.serial_port,
                    lidar_topic=args.lidar_topic,
                    enable_lidar=not args.disable_lidar)

    from manage import drive
    drive(cfg,
          model_path=args.model,
          use_joystick=args.js,
          model_type=args.type,
          camera_type='single',
          meta=args.meta or [])


if __name__ == '__main__':
    main()
