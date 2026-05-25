import cv2
import os
import time
import json
import ssl
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import LED
from ultralytics import YOLO
from datetime import datetime

# EMQX Cloud Configuration
MQTT_BROKER = "n1a44690.ala.asia-southeast1.emqxsl.com"
MQTT_PORT = 8883  # TLS/SSL port
MQTT_USERNAME = "testuser"  # Your EMQX Cloud username
MQTT_PASSWORD = "testpass123"  # Your EMQX Cloud password
SYSTEM_NAME = "dmouv"

# MQTT Topics (same as before)
STATUS_TOPIC = f"iot/{SYSTEM_NAME}/status"
SENSOR_TOPIC = f"iot/{SYSTEM_NAME}/sensor"
ACTION_TOPIC = f"iot/{SYSTEM_NAME}/action"
SETTINGS_UPDATE_TOPIC = f"iot/{SYSTEM_NAME}/settings/update"
TEST_TOPIC = f"iot/{SYSTEM_NAME}/test"

# Initialize devices and model (same as before)
try:
    model_pose = YOLO("yolo11n-pose_ncnn_model", task="pose")
    cam_source = "usb0"
    resW, resH = 640, 480

    devices = {
        "lamp": {
            "instance": LED(26),
            "state": 0,  
            "mode": "auto", 
            "schedule_on": None, 
            "schedule_off": None,
            "is_person_reported": False
        },
        "fan": {
            "instance": LED(19),
            "state": 0,
            "mode": "auto",
            "schedule_on": None,
            "schedule_off": None,
            "is_person_reported": False
        }
    }

    # Camera initialization (same as before)
    if "usb" in cam_source:
        cam_idx = int(cam_source[3:])
        cam = cv2.VideoCapture(cam_idx)
        cam.set(3, resW)
        cam.set(4, resH)
        if not cam.isOpened():
            print("Gagal membuka kamera.")
            exit()
    else:
        print("Tidak ada kamera terdeteksi.")
        exit()
    print("Kamera siap.")

except Exception as e:
    print(f"Error saat inisialisasi: {e}")
    exit()

# Detection logic variables (same as before)
consecutive_detections = 0
fps_buffer = []
fps_avg_len = 50

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Terhubung ke EMQX Cloud di {MQTT_BROKER}!")
        client.subscribe(ACTION_TOPIC)
        print(f"SUBSCRIBE ke topik aksi: {ACTION_TOPIC}")
        client.subscribe(SETTINGS_UPDATE_TOPIC)
        print(f"SUBSCRIBE ke topik settings: {SETTINGS_UPDATE_TOPIC}")

        status_payload = json.dumps({"status": "online"})
        client.publish(STATUS_TOPIC, status_payload)
        print(f"PUBLISH: Mengirim status ONLINE ke {STATUS_TOPIC}")
    else:
        print(f"Gagal terhubung ke EMQX Cloud, kode error: {rc}")
        if rc == 1:
            print("Error: Incorrect protocol version")
        elif rc == 2:
            print("Error: Invalid client identifier")
        elif rc == 3:
            print("Error: Server unavailable")
        elif rc == 4:
            print("Error: Bad username or password")
        elif rc == 5:
            print("Error: Not authorized")

def on_message(client, userdata, msg):
    global devices
    print(f"PESAN DITERIMA di topik {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        
        # Manual mode
        if msg.topic == ACTION_TOPIC:
            device_name = payload.get("device")
            action = payload.get("action")
            
            if device_name in devices and action in ["turn_on", "turn_off"]:
                devices[device_name]["mode"] = "manual"
                if action == "turn_on":
                    devices[device_name]["instance"].on()
                    devices[device_name]["state"] = 1
                elif action == "turn_off":
                    devices[device_name]["instance"].off()
                    devices[device_name]["state"] = 0
                print(f"AKSI MANUAL: '{action}' pada '{device_name}'. Mode diubah ke MANUAL.")
        
        # Settings update
        elif msg.topic == SETTINGS_UPDATE_TOPIC:
            device_name = payload.get("device")
            if device_name in devices:
                if "mode" in payload:
                    new_mode = payload["mode"]
                    if new_mode in ["auto", "manual", "scheduled"]:
                        devices[device_name]["mode"] = new_mode
                        print(f"SETTINGS UPDATE: Mode '{device_name}' diubah menjadi {new_mode.upper()}")
                
                if "schedule_on" in payload:
                    devices[device_name]["schedule_on"] = payload["schedule_on"]
                    print(f"SETTINGS UPDATE: Jadwal ON '{device_name}' diatur ke {payload['schedule_on']}")
                if "schedule_off" in payload:
                    devices[device_name]["schedule_off"] = payload["schedule_off"]
                    print(f"SETTINGS UPDATE: Jadwal OFF '{device_name}' diatur ke {payload['schedule_off']}")

    except Exception as e:
        print(f"Error memproses pesan: {e}")

# Initialize MQTT Client with TLS/SSL
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

# Configure TLS/SSL
context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
# For production, you might want to verify certificates
# context.check_hostname = True
# context.verify_mode = ssl.CERT_REQUIRED

# For testing purposes, you can disable certificate verification
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

client.tls_set_context(context)

# Set last will and testament
last_will_payload = json.dumps({"status": "offline"})
client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)

# Connect to EMQX Cloud
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print("Connecting to EMQX Cloud...")
except Exception as e:
    print(f"Error connecting to EMQX Cloud: {e}")
    exit()

print("\nSistem deteksi mulai berjalan. Tekan 'Q' untuk berhenti.")
try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            print("Peringatan: Gagal mengambil frame.")
            break

        # Detection process (same as before)
        results = model_pose.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        pose_found = len(results) > 0 and len(results[0].keypoints) > 0

        if pose_found:
            consecutive_detections = min(consecutive_detections + 1, 10)
        else:
            consecutive_detections = max(consecutive_detections - 1, 0)
        
        should_be_active = consecutive_detections >= 8
        should_be_inactive = consecutive_detections <= 0
        
        now = datetime.now().time()

        # Device logic (same as before)
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
                    payload = json.dumps({"device": name, "motion_detected": True})
                    client.publish(SENSOR_TOPIC, payload)
                elif should_be_inactive and device["is_person_reported"]:
                    device["is_person_reported"] = False
                    payload = json.dumps({"device": name, "motion_cleared": True})
                    client.publish(SENSOR_TOPIC, payload)

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

        # Display information (same as before)
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

        # FPS calculation (same as before)
        t_stop = time.perf_counter()
        if (t_stop - t_start) > 0:
            frame_rate_calc = 1 / (t_stop - t_start)
            fps_buffer.append(frame_rate_calc)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_frame_rate = np.mean(fps_buffer)
            cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:.2f}', (resW - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)

        cv2.imshow("Smart Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nMembersihkan sumber daya...")
    status_payload = json.dumps({"status": "offline"})
    client.publish(STATUS_TOPIC, status_payload)
    print(f"PUBLISH: Mengirim status OFFLINE ke {STATUS_TOPIC}")
    time.sleep(0.5)

    cam.release()
    cv2.destroyAllWindows()
    for device in devices.values():
        device["instance"].close()
    client.loop_stop()
    client.disconnect()
    print("Selesai.")