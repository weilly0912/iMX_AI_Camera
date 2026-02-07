#!/bin/basha

export DISPLAY=:0
export GDK_BACKEND=wayland

nohup /home/root/ATU-Camera-GTK/run/demo.bin  -c 1 -cf "image/jpeg, width=1280,height=720, framerate=30/1 ! jpegdec ! videoconvert !" -wm 1 &
#python3 /root/ATU-Camera-GTK/run/demo.py -p 2 -c 1 -cf "image/jpeg, width=1280,height=720, framerate=30/1 ! jpegdec ! videoconvert !" -wm 1

sleep 2
python3 keyIn_switch_button.py

# -------------------------------------------------------------------------------------------------------------------
# Camera 
#   OP-Gyro(i.MX93) default is Video0 (MIPI or USB)
#   OP-Killer(i.MX8MP) default is Video2 (MIPI), VIdeo3 (USB)

# Camera format
#   video/x-raw,width=1920,heigt=1080,framerate=30/1 ! 
#   video/x-raw,width=1280,height=720,framerate=30/1 !  
#   image/jpeg, width=1280,height=720, framerate=30/1 ! jpegdec ! videoconvert !    

# windows setting 
#   [FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY, WINDOWS_MAIN_WIDTH, WINDOWS_MAIN_HEIGHT, IMAGE_LOGO_PATH", GTK_STYLE_CSS_PATH]
#   HDMI
#     1472, 828, 1920, 1000, ../data/logo/WPI_NXP_Logo_300x48.png, ../data/style.css
#   WXGA
#     960, 540, 1200, 720, ../data/logo/ATU_WPI_Logo_WXGA.png, ../data/style_WXGA.css   

# windows miximum
#   ws 1

# -------------------------------------------------------------------------------------------------------------------

# OP-Killer
# nohup /home/root/ATU-Camera-GTK/run/demo.bin -c 2

# OP-Gyro
# nohup /home/root/ATU-Camera-GTK/run/demo.bin -c 1 
# nohup /home/root/ATU-Camera-GTK/run/demo.bin -c 1 -cf "image/jpeg, width=1280,height=720, framerate=30/1 ! jpegdec ! videoconvert !" -wm 1
# nohup /home/root/ATU-Camera-GTK/run/demo.bin -c 1 -cf "image/jpeg, width=1280,height=720, framerate=30/1 ! jpegdec ! videoconvert !" -wm 1 -ws "960, 540, 1200, 720, ../data/logo/ATU_WPI_Logo_WXGA.png, ../data/style_WXGA.css"


#sudo login
trap '' SIGINT
while true; do
    read -p "Plase Input User : " username
    read -sp "Plase Input Password : " input
    echo ""
    if [ "$input" = "Wpi701702" ]; then
        echo "succeeded"
        break
    else
        echo "fail"
    fi
done
