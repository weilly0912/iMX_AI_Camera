2023/09/01 Ver 1.0 建立 GTK 框架
2023/09/15 Ver 2.0 加入 ISP 調整機制
2023/09/22 Ver 3.0 加入新模組(人臉識別、深度)
2023/10/20 Ver 4.0 加入雙鏡頭機制
   - 技術瓶頸 : (1) 使用 NNStreamer 方式會導致 buffer 畫面異常，但可以同時使用同一個模組
                        "v4l2src device=" + "/dev/video2 !" +
                        "imxvideoconvert_g2d  ! video/x-raw,format=RGB16,width=" + str(int(FRAME_WIDTH)) + ",height=" + str(int(FRAME_HEIGHT)) + ",framerate=40/1 ! " + 
                        "tee name=t t. ! queue max-size-buffers=10 leaky=2 ! appsink emit-signals=true name=sink1 " + 
                        " t. ! " +
                        "queue max-size-buffers=10 leaky=2 ! " + 
                        "videoconvert ! video/x-raw,format=RGB ! appsink emit-signals=true name=sink2" +
                        " " +
                        "v4l2src device=" + "/dev/video3 !" +
                        "imxvideoconvert_g2d  ! video/x-raw,format=RGB16,width=" + str(int(FRAME_WIDTH)) + ",height=" + str(int(FRAME_HEIGHT)) + ",framerate=40/1 ! " + 
                        "tee name=t1 t1. ! queue max-size-buffers=10 leaky=2 !" + 
                        " " +
                        "imxvideoconvert_g2d ! video/x-raw,width=300,height=300,format=RGBA !" +
                        "videoconvert ! video/x-raw,format=RGB !" +
                        "tensor_converter ! tensor_filter framework=tensorflow-lite model=/home/root/ATU-Camera-GTK/data/model/detect.tflite custom=Delegate:External,ExtDelegateLib:libvx_delegate.so !" +
                        "tensor_decoder mode=bounding_boxes option1=tf-ssd option2=/home/root/ATU-Camera-GTK/data/label/coco_labels.txt option3=0:1:2:3,50 option4=" + str(int(FRAME_WIDTH)) + ":" + str(int(FRAME_HEIGHT)) + " option5=300:300 ! " +
                        "mix. t1. ! queue max-size-buffers=10 leaky=2 ! " + 
                        "imxcompositor_g2d name=mix latency=30000000 min-upstream-latency=30000000 sink_0::zorder=2 sink_1::zorder=1 ! " + 
                        "imxvideoconvert_g2d  ! video/x-raw,format=RGB16,width=" + str(int(FRAME_WIDTH)) + ",height=" + str(int(FRAME_HEIGHT)) + ",framerate=40/1 ! " + 
                        "videoconvert ! queue max-size-buffers=10 leaky=2 ! appsink emit-signals=true name=sink3"
               (2) 使用 Python APP 會導致無法同時使用進行推理(不限定同一個模組)
                        "v4l2src device=" + "/dev/video2 !" +
                        "imxvideoconvert_g2d  ! video/x-raw,format=RGB16,width=" + str(int(FRAME_WIDTH)) + ",height=" + str(int(FRAME_HEIGHT)) + ",framerate=40/1 ! " + 
                        "tee name=t t. ! queue max-size-buffers=2 leaky=2 ! appsink emit-signals=true name=sink1 " + 
                        " t. ! " +
                        "queue max-size-buffers=2 leaky=2 ! " + 
                        "videoconvert ! video/x-raw,format=RGB ! appsink emit-signals=true name=sink2" +
                        " " +
                        "v4l2src device=" + "/dev/video3 !" +
                        "imxvideoconvert_g2d  ! video/x-raw,format=RGB16,width=" + str(int(FRAME_WIDTH)) + ",height=" + str(int(FRAME_HEIGHT)) + ",framerate=40/1 ! " + 
                        "tee name=t1 t1. ! queue max-size-buffers=2 leaky=2 ! appsink emit-signals=true name=sink3 " + 
                        " t1. ! " +
                        "identity silent=true !" +
                        "queue max-size-buffers=2 leaky=2 max-size-time=10000000! " + 
                        "videoconvert ! video/x-raw,format=RGB ! appsink emit-signals=true name=sink4" 
2024/12/03 Ver 6.0 修正當機問題, 重新彙整代碼, 找要當機點  self.IsLoadPicture 這段