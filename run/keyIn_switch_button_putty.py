import sys
import tty
import termios
import os
import time
import select  # 用來監聽輸入狀態

# 設定檔案路徑
flag_file_path = "../data/tmp/flaginference.txt"
extra_loading_file_path = "../data/tmp/extra_loadingPicture.txt"

# 設定應用對應
app_mapping = {
    "1": "objectdetect_mobinetssdv2",
    "2": "objectdetect_YOLOv5s",
    "3": "segmentation_YOLOv5s",
    "4": "posedetect_YOLOv8s",
    "5": "maskdetect_YOLOv5s",
    "6": "facemesh",
    "7": "ADAS_mobilnetssdv2",
    "8": "benchmark",
    "9": "contact",
    "0": ""
}

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

def listen_keyboard():
    """監聽鍵盤輸入（適用於 PuTTY / SSH，使用非阻塞模式）"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)  # 設定終端進入 raw 模式
        print("🎧 監聽鍵盤輸入中 ... (按 0~9 選擇應用，* 切換額外載入，ESC 結束程序，Ctrl+C 停止)")

        while True:
            # **使用 select.select() 來避免 read() 阻塞**
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)  # 最多等待 0.1 秒
            if rlist:  # 有輸入才執行
                key = sys.stdin.read(1)  # 讀取 1 個字元
                
                if key in app_mapping:
                    app = app_mapping[key]
                    with open(flag_file_path, "w") as f:
                        f.write(app)
                    print(f"✅ 更新 flaginference.txt -> {app}")

                elif key == "*":  # `*` 切換 extra_loading
                    toggle_extra_loading()

                elif key == "\x1b":  # `ESC` 的 ASCII 碼是 "\x1b"
                    print("\n🛑 按下 ESC，強制關閉所有 Python 進程！")
                    os.system("pkill -f python")
                    break  # 退出迴圈，確保程式結束

            time.sleep(0.01)  # 減少 CPU 佔用

    except KeyboardInterrupt:
        print("\n🛑 停止監聽")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # 恢復終端設定

# 啟動鍵盤監聽
listen_keyboard()
