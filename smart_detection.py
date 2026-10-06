import cv2
import hmac
import os
import time
import json
import ssl
import numpy as np
import threading
import paho.mqtt.client as mqtt
import socketio
from gpiozero import LED
from ultralytics import YOLO
from datetime import datetime
from collections import deque
from dotenv import load_dotenv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

load_dotenv()

# ======================================================
# MQTT CONFIG
# ======================================================
MQTT_BROKER = "n1a44690.ala.asia-southeast1.emqxsl.com"
MQTT_PORT = 8883
MQTT_USERNAME = "testuser"
MQTT_PASSWORD = "testpass123"
DEVICE_IP_ADDRESS = "dmouv"

STATUS_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/status"
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"
SETTINGS_UPDATE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/settings/update"

# ======================================================
# SOCKET.IO CONFIG (Backend ML)
# ======================================================
SERVER_URL = os.getenv("SERVER_URL", "http://10.199.74.17:8001")
TOKEN = os.getenv("WS_AUTH_TOKEN", "")

# ======================================================
# GPIO CONFIG
# ======================================================
LAMP_PIN = int(os.getenv("LAMP_PIN", "26"))
FAN_PIN = int(os.getenv("FAN_PIN", "19"))
ACTIVE_LOW = os.getenv("RELAY_ACTIVE_LOW", "1") == "1"

# ======================================================
# STREAM CONFIG (MJPEG untuk aplikasi mobile)
# ======================================================
STREAM_ENABLED = os.getenv("STREAM_ENABLED", "1") == "1"
STREAM_PORT = int(os.getenv("STREAM_PORT", "8081"))
STREAM_TOKEN = os.getenv("STREAM_TOKEN", "")                    # kosong = tanpa token
STREAM_QUALITY = int(os.getenv("STREAM_QUALITY", "60"))         # kualitas JPEG 1-100
STREAM_FPS = float(os.getenv("STREAM_FPS", "10"))               # batas FPS per client
STREAM_ANNOTATED = os.getenv("STREAM_ANNOTATED", "1") == "1"    # 1 = dengan skeleton, 0 = video mentah
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "1") == "1"              # 0 jika Raspi tanpa monitor

# ======================================================
# MOTION CONFIG
# ======================================================
MOTION_CONFIG = {
    "enabled": False,
    "detection_duration": 0.5,
    "movement_threshold": 30.0,
    "position_buffer_size": 15,
    "confidence_threshold": 0.5,
    "stable_detection_frames": 8,
    "motion_cooldown": 1.0,
    "min_movement_points": 2,
    "relative_movement_threshold": 0.05,
    "keypoint_stability_threshold": 0.05,
    "min_stable_keypoints": 3
}

# ======================================================
# INISIALISASI KAMERA DAN PERANGKAT
# ======================================================
try:
    model_pose = YOLO("yolo11n-pose_ncnn_model", task="pose")
    resW, resH = 640, 480

    devices = {
        "lamp": {
            "instance": LED(LAMP_PIN),
            "state": 0,
            "mode": "auto",
            "schedule_on": None,
            "schedule_off": None,
            "is_person_reported": False
        },
        "fan": {
            "instance": LED(FAN_PIN),
            "state": 0,
            "mode": "auto",
            "schedule_on": None,
            "schedule_off": None,
            "is_person_reported": False
        }
    }

    motion_tracker = {
        "person_positions": deque(maxlen=MOTION_CONFIG["position_buffer_size"]),
        "position_timestamps": deque(maxlen=MOTION_CONFIG["position_buffer_size"]),
        "keypoint_history": deque(maxlen=MOTION_CONFIG["position_buffer_size"]),
        "is_motion_detected": False,
        "motion_start_time": None,
        "last_motion_time": None,
        "person_detected": False,
        "motion_triggered": False,
        "stable_pose_count": 0,
        "reference_keypoints": None
    }

    cam = cv2.VideoCapture(0)
    cam.set(3, resW)
    cam.set(4, resH)
    if not cam.isOpened():
        print("Gagal membuka kamera.")
        exit()

    print("Kamera siap.")

except Exception as e:
    print(f"Error inisialisasi: {e}")
    exit()

consecutive_detections = 0
fps_buffer = []
fps_avg_len = 50

# ======================================================
# FUNGSI KONTROL RELAY
# ======================================================
def apply_device(device_name, command):
    device = devices.get(device_name)
    if device is None:
        print(f"[Device] tidak dikenal: {device_name}")
        return False
    if str(command).upper() == "ON" and device["state"] == 0:
        device["instance"].on()
        device["state"] = 1
        print(f"[Device] {device_name} ON")
    elif str(command).upper() == "OFF" and device["state"] == 1:
        device["instance"].off()
        device["state"] = 0
        print(f"[Device] {device_name} OFF")
    return True

# ======================================================
# SOCKET.IO CLIENT
# ======================================================
sio = socketio.Client(
    reconnection=True,
    reconnection_delay=1,
    reconnection_delay_max=10,
)

@sio.event
def connect():
    print("[SocketIO] Terhubung ke backend ML")

@sio.event
def disconnect():
    print("[SocketIO] Terputus dari backend ML, reconnecting...")

@sio.on("device_command")
def on_device_command(data):
    device_name = data.get("device")
    command = data.get("command")
    print(f"[SocketIO] Perintah diterima: {device_name} -> {command}")
    if apply_device(device_name, command):
        sio.emit("device_ack", {
            "device": device_name,
            "state": str(command).upper(),
            "timestamp": time.time(),
        })

@sio.on("device_state")
def on_device_state(snapshot):
    for name, dev in (snapshot or {}).get("devices", {}).items():
        if dev.get("state") in ("ON", "OFF"):
            apply_device(name, dev["state"])

def connect_socketio():
    while True:
        try:
            sio.connect(
                SERVER_URL,
                auth={"token": TOKEN, "role": "raspi"},
                transports=["websocket"],
                wait_timeout=10,
            )
            sio.wait()
        except Exception as e:
            print(f"[SocketIO] Gagal konek: {e}, retry 5 detik...")
            time.sleep(5)

# Jalankan Socket.IO di thread terpisah
threading.Thread(target=connect_socketio, daemon=True).start()

# ======================================================
# MQTT CALLBACKS
# ======================================================
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        client.subscribe("dmouv/device/control")
        client.subscribe(ACTION_TOPIC)
        client.subscribe(SETTINGS_UPDATE_TOPIC)
        status_payload = json.dumps({"status": "online"})
        client.publish(STATUS_TOPIC, status_payload, retain=True)
        print("[MQTT] Terhubung ke EMQX Cloud")
    else:
        print(f"[MQTT] Gagal terhubung, kode: {rc}")

def on_message(client, userdata, msg):
    global devices

    if msg.topic == "dmouv/device/control":
        command = msg.payload.decode().strip()
        for name in devices:
            apply_device(name, command)
        return

    try:
        payload = json.loads(msg.payload.decode())

        if msg.topic == ACTION_TOPIC:
            device_name = payload.get("device")
            action = payload.get("action")
            if device_name in devices and action in ["turn_on", "turn_off"]:
                devices[device_name]["mode"] = "manual"
                apply_device(device_name, "ON" if action == "turn_on" else "OFF")

        elif msg.topic == SETTINGS_UPDATE_TOPIC:
            device_name = payload.get("device")
            if device_name in devices:
                if "mode" in payload:
                    new_mode = payload["mode"]
                    if new_mode in ["auto", "manual", "scheduled"]:
                        devices[device_name]["mode"] = new_mode
                        print(f"[MQTT] Mode {device_name} -> {new_mode}")
                if "schedule_on" in payload:
                    devices[device_name]["schedule_on"] = payload["schedule_on"]
                if "schedule_off" in payload:
                    devices[device_name]["schedule_off"] = payload["schedule_off"]

    except Exception as e:
        pass

# ======================================================
# SETUP MQTT
# ======================================================
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

last_will_payload = json.dumps({"status": "offline"})
mqtt_client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)

context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE
mqtt_client.tls_set_context(context)

mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# ======================================================
# FUNGSI MOTION DETECTION
# ======================================================
def get_stable_keypoints(keypoints):
    if keypoints is None or len(keypoints) == 0:
        return None
    kp = keypoints[0]
    stable_keypoints = []
    for i in range(len(kp)):
        if len(kp[i]) >= 3 and kp[i][2] > MOTION_CONFIG["confidence_threshold"]:
            stable_keypoints.append([kp[i][0], kp[i][1], kp[i][2]])
    return np.array(stable_keypoints) if len(stable_keypoints) >= MOTION_CONFIG["min_stable_keypoints"] else None

def calculate_pose_center(stable_keypoints):
    if stable_keypoints is None or len(stable_keypoints) == 0:
        return None
    return (np.mean(stable_keypoints[:, 0]), np.mean(stable_keypoints[:, 1]))

def calculate_relative_movement(current_keypoints, reference_keypoints):
    if current_keypoints is None or reference_keypoints is None:
        return 0.0
    if len(current_keypoints) != len(reference_keypoints):
        return 0.0
    total = 0.0
    valid = 0
    for i in range(len(current_keypoints)):
        curr = current_keypoints[i][:2]
        ref = reference_keypoints[i][:2]
        dist = np.sqrt(np.sum((curr - ref) ** 2))
        ref_dist = np.sqrt(ref[0]**2 + ref[1]**2)
        if ref_dist > 0:
            total += dist / ref_dist
            valid += 1
    return total / valid if valid > 0 else 0.0

def is_keypoints_stable(current_keypoints):
    if len(motion_tracker["keypoint_history"]) < 3:
        return False
    recent = list(motion_tracker["keypoint_history"])[-3:]
    for i in range(1, len(recent)):
        if recent[i] is None or recent[i-1] is None:
            return False
        if len(recent[i]) != len(recent[i-1]):
            return False
        if calculate_relative_movement(recent[i], recent[i-1]) > MOTION_CONFIG["keypoint_stability_threshold"]:
            return False
    return True

def detect_skeleton_motion():
    if not MOTION_CONFIG["enabled"] or len(motion_tracker["person_positions"]) < MOTION_CONFIG["min_movement_points"]:
        return False
    positions = list(motion_tracker["person_positions"])
    timestamps = list(motion_tracker["position_timestamps"])
    keypoint_history = list(motion_tracker["keypoint_history"])
    if len(keypoint_history) < 2:
        return False
    significant_movements = 0
    total_duration = 0
    for i in range(len(positions) - 1):
        if keypoint_history[i] is None or keypoint_history[i+1] is None:
            continue
        time_diff = timestamps[i+1] - timestamps[i]
        if time_diff <= 0 or time_diff > MOTION_CONFIG["detection_duration"]:
            continue
        rel_mov = calculate_relative_movement(keypoint_history[i+1], keypoint_history[i])
        pos_dist = np.sqrt((positions[i+1][0]-positions[i][0])**2 + (positions[i+1][1]-positions[i][1])**2)
        if rel_mov > MOTION_CONFIG["relative_movement_threshold"] and pos_dist > MOTION_CONFIG["movement_threshold"]:
            significant_movements += 1
            total_duration += time_diff
    return (significant_movements >= MOTION_CONFIG["min_movement_points"] and
            total_duration >= MOTION_CONFIG["detection_duration"])

def update_motion_detection(keypoints):
    global motion_tracker
    current_time = time.time()
    stable_keypoints = get_stable_keypoints(keypoints)
    motion_tracker["keypoint_history"].append(stable_keypoints)
    if stable_keypoints is not None:
        motion_tracker["person_detected"] = True
        if is_keypoints_stable(stable_keypoints):
            motion_tracker["stable_pose_count"] += 1
        else:
            motion_tracker["stable_pose_count"] = 0
        center_point = calculate_pose_center(stable_keypoints)
        if center_point is not None:
            motion_tracker["person_positions"].append(center_point)
            motion_tracker["position_timestamps"].append(current_time)
        if motion_tracker["stable_pose_count"] >= 5:
            if detect_skeleton_motion():
                if not motion_tracker["is_motion_detected"]:
                    motion_tracker["motion_start_time"] = current_time
                    motion_tracker["is_motion_detected"] = True
                motion_tracker["last_motion_time"] = current_time
                if (current_time - motion_tracker["motion_start_time"]) >= MOTION_CONFIG["detection_duration"]:
                    motion_tracker["motion_triggered"] = True
    else:
        motion_tracker["person_detected"] = False
        motion_tracker["stable_pose_count"] = 0
        if (motion_tracker["last_motion_time"] and
                current_time - motion_tracker["last_motion_time"] > MOTION_CONFIG["motion_cooldown"]):
            motion_tracker["is_motion_detected"] = False
            motion_tracker["motion_triggered"] = False
            motion_tracker["motion_start_time"] = None

# ======================================================
# MJPEG STREAM SERVER
# Link: http://<IP-RASPI>:STREAM_PORT/video  (tambah ?token=... jika STREAM_TOKEN diisi)
# ======================================================
stream_cond = threading.Condition()
stream_state = {"jpeg": None, "seq": 0, "clients": 0}

def publish_stream_frame(img):
    """Dipanggil dari main loop. Tidak meng-encode kalau tidak ada yang menonton."""
    if stream_state["clients"] <= 0:
        return
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
    if not ok:
        return
    with stream_cond:
        stream_state["jpeg"] = buf.tobytes()
        stream_state["seq"] += 1
        stream_cond.notify_all()

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # matikan log tiap request

    def _reply(self, code, text):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)

        if url.path == "/":
            return self._reply(200, "DMouv stream OK")

        if STREAM_TOKEN and not hmac.compare_digest(query.get("token", [""])[0], STREAM_TOKEN):
            return self._reply(401, "Unauthorized")

        if url.path == "/video":
            return self._stream()
        self._reply(404, "Not found")

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        with stream_cond:
            stream_state["clients"] += 1
        last_seq = -1
        min_gap = 1.0 / STREAM_FPS if STREAM_FPS > 0 else 0
        try:
            while True:
                with stream_cond:
                    stream_cond.wait_for(lambda: stream_state["seq"] != last_seq, timeout=5)
                    jpeg, last_seq = stream_state["jpeg"], stream_state["seq"]
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(min_gap)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # client (app) menutup koneksi
        finally:
            with stream_cond:
                stream_state["clients"] -= 1

stream_server = None
if STREAM_ENABLED:
    stream_server = ThreadingHTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler)
    stream_server.daemon_threads = True
    threading.Thread(target=stream_server.serve_forever, daemon=True).start()
    print(f"[Stream] Aktif: http://<IP-RASPI>:{STREAM_PORT}/video")

# ======================================================
# MAIN LOOP
# ======================================================
print("Sistem deteksi mulai berjalan. Tekan 'Q' untuk berhenti.")
try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            print("Gagal baca frame.")
            break

        results = model_pose.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        pose_found = len(results) > 0 and len(results[0].keypoints) > 0

        keypoints = results[0].keypoints.data.cpu().numpy() if pose_found else None
        update_motion_detection(keypoints)

        if pose_found:
            consecutive_detections = min(consecutive_detections + 1, 20)
        else:
            consecutive_detections = max(consecutive_detections - 2, 0)

        if MOTION_CONFIG["enabled"]:
            should_be_active = (consecutive_detections >= MOTION_CONFIG["stable_detection_frames"] and
                                motion_tracker["motion_triggered"] and
                                motion_tracker["stable_pose_count"] >= 5)
        else:
            should_be_active = consecutive_detections >= MOTION_CONFIG["stable_detection_frames"]

        should_be_inactive = consecutive_detections <= 0

        now = datetime.now().time()

        for name, device in devices.items():
            if device["mode"] == "auto":
                if should_be_active and device["state"] == 0:
                    device["instance"].on()
                    device["state"] = 1
                elif should_be_inactive and device["state"] == 1:
                    device["instance"].off()
                    device["state"] = 0

                if should_be_active and not device["is_person_reported"]:
                    device["is_person_reported"] = True
                    mqtt_client.publish(SENSOR_TOPIC, json.dumps({"device": name, "motion_detected": True}))
                elif should_be_inactive and device["is_person_reported"]:
                    device["is_person_reported"] = False
                    mqtt_client.publish(SENSOR_TOPIC, json.dumps({"device": name, "motion_cleared": True}))

            elif device["mode"] == "scheduled":
                try:
                    on_time = datetime.strptime(device["schedule_on"], "%H:%M").time()
                    off_time = datetime.strptime(device["schedule_off"], "%H:%M").time()
                    is_active_time = False
                    if on_time < off_time:
                        if on_time <= now < off_time:
                            is_active_time = True
                    else:
                        if now >= on_time or now < off_time:
                            is_active_time = True
                    if is_active_time and device["state"] == 0:
                        device["instance"].on()
                        device["state"] = 1
                    elif not is_active_time and device["state"] == 1:
                        device["instance"].off()
                        device["state"] = 0
                except (ValueError, TypeError):
                    if device["state"] == 1:
                        device["instance"].off()
                        device["state"] = 0

        y_pos = 30
        for name, device in devices.items():
            mode_text = f"{name.upper()} Mode: {device['mode'].upper()}"
            status_text = f"{name.upper()} Status: {'ON' if device['state'] == 1 else 'OFF'}"
            color_mode = (0, 255, 255)
            color_status = (0, 255, 0) if device['state'] == 1 else (0, 0, 255)
            cv2.putText(annotated_frame, mode_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, .6, color_mode, 2)
            y_pos += 30
            cv2.putText(annotated_frame, status_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, .6, color_status, 2)
            y_pos += 40

        t_stop = time.perf_counter()
        if (t_stop - t_start) > 0:
            frame_rate_calc = 1 / (t_stop - t_start)
            fps_buffer.append(frame_rate_calc)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_frame_rate = np.mean(fps_buffer)
            cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:.2f}', (resW - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)

        if STREAM_ENABLED:
            publish_stream_frame(annotated_frame if STREAM_ANNOTATED else frame)

        if SHOW_WINDOW:
            cv2.imshow("Smart Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    print("Membersihkan sumber daya...")
    mqtt_client.publish(STATUS_TOPIC, json.dumps({"status": "offline"}), retain=True)
    time.sleep(0.5)
    cam.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    if stream_server:
        stream_server.shutdown()
        stream_server.server_close()
    for device in devices.values():
        device["instance"].off()
        device["instance"].close()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    try:
        sio.disconnect()
    except:
        pass
    print("Selesai.")