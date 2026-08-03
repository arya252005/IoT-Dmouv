import cv2
import os
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import LED
from ultralytics import YOLO
from datetime import datetime
from collections import deque

MQTT_BROKER = "n1a44690.ala.asia-southeast1.emqxsl.com"
MQTT_PORT = 8883
MQTT_USERNAME = "testuser"
MQTT_PASSWORD = "testpass123"
DEVICE_IP_ADDRESS = "dmouv"

STATUS_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/status"
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"
SETTINGS_UPDATE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/settings/update"

MOTION_CONFIG = {
    "enabled": True,
    "detection_duration": 1.0,
    "movement_threshold": 85.0,
    "position_buffer_size": 15,
    "confidence_threshold": 0.5,
    "stable_detection_frames": 10,
    "motion_cooldown": 1.0,
    "min_movement_points": 3,
    "relative_movement_threshold": 0.15,
    "keypoint_stability_threshold": 0.05,
    "min_stable_keypoints": 5
}

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

    motion_tracker = {
        "person_positions": deque(maxlen=MOTION_CONFIG["position_buffer_size"]),
        "position_timestamps": deque(maxlen=MOTION_CONFIG["position_buffer_size"]),
        "keypoint_history": deque(maxlen=MOTION_CONFIG["position_buffer_size"]),
        "is_motion_detected": True,
        "motion_start_time": None,
        "last_motion_time": None,
        "person_detected": True,
        "motion_triggered": True,
        "stable_pose_count": 3,
        "reference_keypoints": None
    }

    if "usb" in cam_source:
        cam_idx = int(cam_source[3:])
        cam = cv2.VideoCapture(cam_idx)
        cam.set(3, resW)
        cam.set(4, resH)
        if not cam.isOpened():
            exit()
    else:
        exit()

except Exception as e:
    exit()

consecutive_detections = 0
fps_buffer = []
fps_avg_len = 50

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
        
    center_x = np.mean(stable_keypoints[:, 0])
    center_y = np.mean(stable_keypoints[:, 1])
    
    return (center_x, center_y)

def calculate_relative_movement(current_keypoints, reference_keypoints):
    if current_keypoints is None or reference_keypoints is None:
        return 0.0
    
    if len(current_keypoints) != len(reference_keypoints):
        return 0.0
    
    total_relative_movement = 0.0
    valid_comparisons = 0
    
    for i in range(len(current_keypoints)):
        curr_point = current_keypoints[i][:2]
        ref_point = reference_keypoints[i][:2]
        
        distance = np.sqrt(np.sum((curr_point - ref_point) ** 2))
        
        reference_distance = np.sqrt(ref_point[0]**2 + ref_point[1]**2)
        if reference_distance > 0:
            relative_distance = distance / reference_distance
            total_relative_movement += relative_distance
            valid_comparisons += 1
    
    return total_relative_movement / valid_comparisons if valid_comparisons > 0 else 0.0

def is_keypoints_stable(current_keypoints):
    if len(motion_tracker["keypoint_history"]) < 3:
        return False
    
    recent_keypoints = list(motion_tracker["keypoint_history"])[-3:]
    
    for i in range(1, len(recent_keypoints)):
        if recent_keypoints[i] is None or recent_keypoints[i-1] is None:
            return False
        
        if len(recent_keypoints[i]) != len(recent_keypoints[i-1]):
            return False
        
        movement = calculate_relative_movement(recent_keypoints[i], recent_keypoints[i-1])
        if movement > MOTION_CONFIG["keypoint_stability_threshold"]:
            return False
    
    return True

def detect_skeleton_motion():
    if not MOTION_CONFIG["enabled"] or len(motion_tracker["person_positions"]) < MOTION_CONFIG["min_movement_points"]:
        return False
    
    current_time = time.time()
    
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
        
        relative_movement = calculate_relative_movement(keypoint_history[i+1], keypoint_history[i])
        
        position_distance = np.sqrt(
            (positions[i+1][0] - positions[i][0])**2 + 
            (positions[i+1][1] - positions[i][1])**2
        )
        
        if (relative_movement > MOTION_CONFIG["relative_movement_threshold"] and 
            position_distance > MOTION_CONFIG["movement_threshold"]):
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

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        client.subscribe("dmouv/device/control")
        client.subscribe(ACTION_TOPIC)
        client.subscribe(SETTINGS_UPDATE_TOPIC)
        status_payload = json.dumps({"status": "online"})
        client.publish(STATUS_TOPIC, status_payload, retain=True)
        print(f"terhubung")

def on_message(client, userdata, msg):
    global devices

    # Handle ML command (string biasa, bukan JSON)
    if msg.topic == "dmouv/device/control":
        command = msg.payload.decode().strip()
        for name, device in devices.items():
            if command == "ON" and device["state"] == 0:
                device["instance"].on()
                device["state"] = 1
                print(f"[ML Command] {name} dinyalakan")
            elif command == "OFF" and device["state"] == 1:
                device["instance"].off()
                device["state"] = 0
                print(f"[ML Command] {name} dimatikan")
        return

    try:
        payload = json.loads(msg.payload.decode())
        
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
        
        elif msg.topic == SETTINGS_UPDATE_TOPIC:
            device_name = payload.get("device")
            if device_name in devices:
                if "mode" in payload:
                    new_mode = payload["mode"]
                    if new_mode in ["auto", "manual", "scheduled"]:
                        devices[device_name]["mode"] = new_mode
                
                if "schedule_on" in payload:
                    devices[device_name]["schedule_on"] = payload["schedule_on"]
                if "schedule_off" in payload:
                    devices[device_name]["schedule_off"] = payload["schedule_off"]

    except Exception as e:
        pass

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

last_will_payload = json.dumps({"status": "offline"})
client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)
import ssl
context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE
client.tls_set_context(context)

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            break

        results = model_pose.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        pose_found = len(results) > 0 and len(results[0].keypoints) > 0

        keypoints = results[0].keypoints.data.cpu().numpy() if pose_found else None
        update_motion_detection(keypoints)

        if pose_found:
            consecutive_detections = min(consecutive_detections + 1, 15)
        else:
            consecutive_detections = max(consecutive_detections - 1, 0)
        
        if MOTION_CONFIG["enabled"]:
            should_be_active = (consecutive_detections >= MOTION_CONFIG["stable_detection_frames"] and 
                              motion_tracker["motion_triggered"] and
                              motion_tracker["stable_pose_count"] >= 5)
        else:
            should_be_active = consecutive_detections >= MOTION_CONFIG["stable_detection_frames"]
        
        should_be_inactive = consecutive_detections <= 0 or not motion_tracker["person_detected"]
        
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
            cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:.2f}', (resW - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)

        cv2.imshow("Smart Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    status_payload = json.dumps({"status": "offline"})
    client.publish(STATUS_TOPIC, status_payload)
    time.sleep(0.5)

    cam.release()
    cv2.destroyAllWindows()
    for device in devices.values():
        device["instance"].close()
    client.loop_stop()
    client.disconnect()
