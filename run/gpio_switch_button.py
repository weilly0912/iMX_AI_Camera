import os
import time

# 設定 GPIO 參數
GPIO_PINS = {
    8: "objectdetect_YOLOv5s",       # GPIO 11 -> KEY_2
    9: "segmentation_YOLOv5s",       # GPIO 12 -> KEY_3
    10: "posedetect_YOLOv8s",         # GPIO 13 -> KEY_4
}

EXTRA_LOADING_PIN = 11  # 模擬 "*" 按鍵 (切換 `extra_loadingPicture.txt`)

# 設定檔案路徑
flag_file_path = "../data/tmp/flaginference.txt"
extra_loading_file_path = "../data/tmp/extra_loadingPicture.txt"

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
    
    print(f"🔄 切換 extra_loadingPicture.txt -> {new_value}")

def read_gpio_state(pin):
    """使用 `gpioget` 讀取 GPIO 狀態"""
    cmd = f"gpioget -c 0 {pin}"
    try:
        output = os.popen(cmd).read().strip()
        return output.endswith("=active")  # 只有當完整字串是 `=active` 才回傳 True
    except Exception as e:
        print(f"❌ 讀取 GPIO {pin} 失敗: {e}")
        return False

def monitor_gpio():
    """監聽 GPIO 按鍵事件"""
    print("🎧 正在監聽 GPIO (按 Ctrl+C 停止) ...")

    prev_state = {pin: False for pin in GPIO_PINS.keys()}  # 記錄前一次狀態
    prev_extra = False  # 記錄額外功能鍵狀態

    try:
        while True:
            for pin in GPIO_PINS.keys():
                state = read_gpio_state(pin)
                if state and not prev_state[pin]:  # 偵測按下事件
                    app = GPIO_PINS[pin]
                    with open(flag_file_path, "w") as f:
                        f.write(app)
                    print(f"✅ 更新 flaginference.txt -> {app}")

                prev_state[pin] = state  # 更新狀態
            
            # 處理 "*" (EXTRA_LOADING_PIN) 功能
            extra_state = read_gpio_state(EXTRA_LOADING_PIN)
            if extra_state and not prev_extra:
                toggle_extra_loading()
            prev_extra = extra_state

            time.sleep(0.1)  # 避免 CPU 過度負載
    except KeyboardInterrupt:
        print("\n🛑 停止監聽")

# 啟動 GPIO 監聽
monitor_gpio()
