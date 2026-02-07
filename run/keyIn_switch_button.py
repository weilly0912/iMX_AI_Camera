import evdev
import os

# 設定檔案路徑
flag_file_path = "../data/tmp/flaginference.txt"
extra_loading_file_path = "../data/tmp/extra_loadingPicture.txt"

# 設定應用對應 (包含 `KP0` 到 `KP9`)
app_mapping = {
    "KEY_1": "objectdetect_mobinetssdv2",
    "KEY_2": "objectdetect_YOLOv5s",
    "KEY_3": "segmentation_YOLOv5s",
    "KEY_4": "posedetect_YOLOv8s",
    "KEY_5": "maskdetect_YOLOv5s",
    "KEY_6": "facemesh",
    "KEY_7": "ADAS_mobilnetssdv2",
    "KEY_8": "benchmark",
    "KEY_9": "contact",
    "KEY_0": "",
    "KEY_KP1": "objectdetect_mobinetssdv2",
    "KEY_KP2": "objectdetect_YOLOv5s",
    "KEY_KP3": "segmentation_YOLOv5s",
    "KEY_KP4": "posedetect_YOLOv8s",
    "KEY_KP5": "maskdetect_YOLOv5s",
    "KEY_KP6": "facemesh",
    "KEY_KP7": "ADAS_mobilnetssdv2",
    "KEY_KP8": "benchmark",
    "KEY_KP9": "contact",
    "KEY_KP0": ""
}

DEVICE_PATH = "/dev/input/event3"  # 你的鍵盤設備

def toggle_extra_loading():
    """切換 extra_loadingPicture.txt 內的值 (True / False)"""
    if os.path.exists(extra_loading_file_path):
        with open(extra_loading_file_path, "r") as f:
            current_value = f.read().strip()
        new_value = "False" if current_value == "True" else "True"
    else:
        new_value = "True"  # 如果檔案不存在，則預設為 True

    with open(extra_loading_file_path, "w") as f:
        f.write(new_value)
    
    print(f"🔄 切換 extra_loadingPicture.txt -> {new_value}\n")

def listen_keyboard():
    """監聽鍵盤輸入"""
    try:
        device = evdev.InputDevice(DEVICE_PATH)
        print(f"🎧 正在監聽 {device.name} ({DEVICE_PATH}) ... (按 Ctrl+C 停止)\n")

        for event in device.read_loop():
            if event.type == evdev.ecodes.EV_KEY and event.value == 1:  # 只處理按下事件
                key_code = event.code
                key_name = evdev.ecodes.KEY[key_code]

                if key_name in app_mapping:
                    app = app_mapping[key_name]
                    with open(flag_file_path, "w") as f:
                        f.write(app)  # 寫入對應的應用程式名稱
                    print(f"✅ 更新 flaginference.txt -> {app}\n")

                elif key_name == "KEY_KPASTERISK":  # 監聽數字鍵盤的 "*"
                    toggle_extra_loading()

    except FileNotFoundError:
        print(f"❌ 鍵盤設備 {DEVICE_PATH} 找不到，請確認設備路徑！\n")
    except PermissionError:
        print(f"❌ 權限不足，請用 `sudo python3 script.py` 運行！\n")

# 啟動鍵盤監聽
listen_keyboard()
