# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 10.0
# * Code Date: 2025/02/06
# * Author   : Weilly Li
# * Modify fix break issue.
#--------------------------------------------------------------------------------------
# THIS SOFTWARE IS PROVIDED BY WPI-TW "AS IS" AND ANY EXPRESSED OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL WPI OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#--------------------------------------------------------------------------------------
# Ver 1.0 , Create for i.MX8MP (BSP 5.15.x) 
# Ver 2.0 , Add New platform : i.MX93 (BSP 5.15.x) 
# Ver 3.0 , Add New Function : ISP and AutoRun , 2023/12/30 by Weilly
# Ver 4.0 , Add New platform (OP-Gyro) and Screen (WXGA) , BSP (L6.1.55) , 2024/03/11 by Weilly
# Ver 5.0 , Add cairo surface of RGB888 output (segmentation_YOLOv5s still slow )
# Ver 6.0 , Re-Define detect algorithim output
# Ver 7.0 , Setting Restart Function
# Ver 8.0 , Update Restart Function
# Ver 9.0 , Fix can't display and crash issue , because x11
# ver 10.0 , Update Import Reference Parameters
# ver 11.0 , add "flaginference.txt" to change inference

import cairo
import gi
import gc
import re
import time
import numpy as np
import os
import tflite_runtime.interpreter as tflite
import math
import glob
import copy
import sys
import psutil
import utils
import logging
import argparse
import subprocess
from utils import *
from algorithm import *
from  settingISP import *

gi.require_version("Gtk", "3.0")
gi.require_version("Gst", "1.0")
from gi.repository import Gtk, Gst, Gio, GLib, Gdk

CAMERA_DUAL_MIPI          = False
REMOVE_SIDE_BUTTON        = True
CAMERA_ROTATION           = 0 #  AR0144 ( RPI-CAM-MIPI ) = 2
#os.environ['GDK_BACKEND'] = 'x11'

# ''' Resolution Setting (Defualt)''' 
FRAME_WIDTH = 1920 
FRAME_HEIGHT = 1080 
WINDOWS_START_WIDTH  = 500
WINDOWS_START_HEIGHT = 300
WINDOWS_TOUCH_FIX_X  = 420
WINDOWS_TOUCH_FIX_Y  = 160
BUTTON_WIDTH         = 140 # [WPI] add 2024/11/28
BUTTON_HEIGHT        = 70  # [WPI] add 2024/11/28

# ''' Camera Setting''' 
class MLVideoDemo(Gtk.Window):

    def __init__(self, CameraDevice):
        """Create the UI and start the video feed."""

        # Class variables
        super().__init__()

        # Window Set-up
        self.set_default_size(WINDOWS_MAIN_WIDTH, WINDOWS_MAIN_HEIGHT)  # [WPI] Screen Size
        self.set_resizable(True)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.show_logo = Gtk.Image.new_from_file(IMAGE_LOGO_PATH) # [WPI] self.fullscreen() #Error 71 (Protocol error) dispatching to Wayland display
        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(GTK_STYLE_CSS_PATH) # [WPI] load_from_data
        context = Gtk.StyleContext()
        screen  = Gdk.Screen.get_default()
        context.add_provider_for_screen(screen, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_USER)
        div = Gtk.Separator.new(Gtk.Orientation(0))

        # Window Maximum
        if WINDOWS_MAXIMUM == "1": self.maximize()

        # Initial
        self.title_app = ""
        self.LoadingImagePath = {}
        self.button_mapping = {}
        self.menu_button_mapping = {}
        self.FLAG_AI_Inference = '' #FLAG
        self.touch_signal       = 0
        self.touch_signal_count = 0
        self.touch_postionX     = 0
        self.touch_postionY     = 0
        self.Gstreamer_MaxSizebuffer="0"
        self.CairoContextTextSize_1 = 40
        self.cnt_crash = 0
        self.FLAG_AI_COUNTER   = 0
        self.BufferRGB565_1  = None
        self.BufferRGB565_2  = None
        self.Extra_loadingPictureSignal = False

        # -----------------------------------------------------------------------
        # GUI - AI Func
        # -----------------------------------------------------------------------
        # (0) Button Config
        self.Bclass = ["Restore", "Object", "Segment", "Features", "Others"]
        self.buttonsAPP_config = [
           # {"name": "restore", "Type":self.Bclass[0], "label": "Restore", "image_path": "../data/img/zidane.jpg", "image_loading":  False, "auto":False , "auto_speed_level" : 1},
            {"name": "objectdetect_mobinetssdv2", "Type":self.Bclass[1], "label": "Object Detect", "InputBuffer_Type" :"RGB565" , "image_path": "../data/img/Data_CC0/piexel_CC0_1.jpg", "image_loading":  False, "auto":True, "auto_speed_level" : 1},
            {"name": "objectdetect_YOLOv5s", "Type":self.Bclass[1], "label": "Object Detect(YOLOv5s)", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/Data_CC0/piexel_CC0_1.jpg", "image_loading":  False, "auto":True, "auto_speed_level" : 1},
            {"name": "objectdetect_YOLOv8n", "Type":self.Bclass[1], "label": "Object Detect(YOLOv8n)", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/Data_CC0/piexel_CC0_1.jpg", "image_loading":  False, "auto":True, "auto_speed_level" : 1},
            {"name": "segmentation_YOLOv5s", "Type":self.Bclass[2], "label": "Image Segmentation", "InputBuffer_Type" :"RGB888", "image_path": "../data/img/Data_CC0/piexel_CC0_1.jpg", "image_loading":  False, "auto":True, "auto_speed_level" : 3},
            {"name": "facedetect_mobilenetssdv2", "Type":self.Bclass[1], "label": "Face Detect", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/Data_Age/Didy.png", "image_loading":  False, "auto":False, "auto_speed_level" : 1},
            {"name": "facemesh", "Type":self.Bclass[3], "label": "Face Mesh", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/Data_CC0/piexel_CC0_2.jpg", "image_loading":  False, "auto":True, "auto_speed_level" : 1},
            {"name": "facerecongition", "Type":self.Bclass[4], "label": "Face Recognition", "InputBuffer_Type" :"RGB565", "image_path": "../data/Data_Age/img/Didy.png", "image_loading":  False, "auto":False, "auto_speed_level" : 1},
            {"name": "age_gender_recognition", "Type":self.Bclass[4], "label": "Age-Gender Detect", "InputBuffer_Type" :"RGB565", "image_path": "../data/Data_Age/black_woman.jpg", "image_loading":  False, "auto":False, "auto_speed_level" : 1},
            {"name": "maskdetect_YOLOv5s", "Type":self.Bclass[1], "label": "Mask Detect(YOLOv5s)", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/crowd.jpg", "image_loading":  False, "auto":True, "auto_speed_level" : 1},
            {"name": "hardhatdetect_YOLOv5s", "Type":self.Bclass[1], "label": "Hard-Hat Detect(YOLOv5s)", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/DataBase_HandHat/Test_1.jpg", "image_loading":  False, "auto":True, "auto_speed_level" : 1},
            {"name": "palmlandmark", "Type":self.Bclass[3], "label": "Palm Landmark", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/Data_Hand/hand-3.jpg", "image_loading":  False, "auto":False, "auto_speed_level" : 1},
            {"name": "posedetect_mobilenetssd", "Type":self.Bclass[3], "label": "Pose Detect(MobilentSSD)", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/Data_CC0/piexel_CC0_1.jpg", "image_loading":  True, "auto":False, "auto_speed_level" : 1},
            {"name": "posedetect_YOLOv8s", "Type":self.Bclass[3], "label": "Pose Detect(YOLOv8s)", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/Data_CC0/piexel_CC0_3.jpg", "image_loading":  False, "auto":True, "auto_speed_level" : 1},
            {"name": "PCBdetect_YOLOv8s", "Type":self.Bclass[1], "label": "PCB Detect(YOLOv8s)", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/ElectronicComponen.jpg", "image_loading":  True, "auto":False, "auto_speed_level" : 1},
            {"name": "ADAS_mobilnetssdv2", "Type":self.Bclass[1], "label": "ADAS (LDW+FCW)", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/zidane.jpg", "image_loading":  True, "auto":True, "auto_speed_level" : 3},
            {"name": "fruitdetect_YOLOv5s", "Type":self.Bclass[1], "label": "Fruit Detect(YOLOv5s)", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/DataBase_Fruit/fruits.jpg", "image_loading":  True, "auto":False, "auto_speed_level" : 1},
            {"name": "depthestimation_MegaDepth", "Type":self.Bclass[4], "label": "Depth Estimation(MGNet)", "InputBuffer_Type" :"RGB565", "image_path": "../data/img/dog.bmp", "image_loading":  True, "auto":False, "auto_speed_level" : 1},
            {"name": "dual_objectdetect_mobinetssdv2", "Type":self.Bclass[1], "label": "Object Detect(Dual Camera)", "InputBuffer_Type" :"RGB565x2", "image_path": "../data/img/zidane.jpg", "image_loading":  False, "auto":False, "auto_speed_level" : 1},
            {"name": "benchmark", "Type":self.Bclass[4], "label": "Benchmark", "InputBuffer_Type" :"RGB565", "image_path": "../../Benchmark_IMX93.png", "image_loading":  True, "auto":True, "auto_speed_level" : 4},
            {"name": "contact", "Type":self.Bclass[4], "label": "Contact Us", "InputBuffer_Type" :"RGB565", "image_path": "../../ContactUs_IMX93.png", "image_loading":  True, "auto":False, "auto_speed_level" : 4},]

        # (1) Generate Dynamic setup_methods
        self.generate_dynamic_setup_methods() #[WPI] modfiy fun => def setup_objectdetect_mobinetssdv2(self, unused)

        # (2) Button and Grid Setting by APP
        for config in self.buttonsAPP_config:
            # Button
            button = Gtk.Button.new_with_label(config["label"])
            button.set_size_request(BUTTON_WIDTH, BUTTON_HEIGHT) 
            setattr(self, config["name"] + "_btn", button) #[WPI] modfiy fun => self.objectdetect_mobinetssdv2_btn = Gtk.Button.new_with_label("objectdetect")
            button.connect("clicked", config["callback"])
            self.LoadingImagePath[config["name"]] = config["image_path"]
            self.button_mapping[config["name"]] = button

            # Grid
            grid = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
            grid.pack_start(button, True, True, 0)
            setattr(self, config["name"] + "_grid", grid) #[WPI] modfiy fun => self.objectdetect_mobinetssdv2_grid = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)

        # (3) Switch Platform
        if 1 :
            if (PLATFORM=="IMX8MPLUSEVK") :
                self.depthestimation_MegaDepth_btn.set_sensitive(False) ; 
                self.depthestimation_MegaDepth_btn.get_style_context().add_class("Button-action-stop")
                self.title_app = "NXP i.MX93 Demos"
                for config in self.buttonsAPP_config:
                    if config["name"] == "age_gender_recognition": config["auto"] = True

            if (PLATFORM=="IMX93EVK" ) :
                self.age_gender_recognition_btn.set_sensitive(False)
                self.age_gender_recognition_btn.get_style_context().add_class("Button-action-stop")
                self.title_app = "NXP i.MX8M Plus Demos"
                #self.PCBdetect_YOLOv8s_btn.set_sensitive(False) # [WPI] It will fail before BSP 6.1.55 
                #self.PCBdetect_YOLOv8s_btn.get_style_context().add_class("Button-action-stop")

            if (PLATFORM=="IMX95EVK" ) :  
                self.segmentation_YOLOv5s_btn.set_sensitive(False)
                self.facemesh_btn.set_sensitive(False)
                self.palmlandmark_btn.set_sensitive(False)
                self.posedetect_mobilenetssd_btn.set_sensitive(False)
                self.posedetect_YOLOv8s_btn.set_sensitive(False)
                self.depthestimation_MegaDepth_btn.set_sensitive(False)
                self.segmentation_YOLOv5s_btn.get_style_context().add_class("Button-action-stop")
                self.facemesh_btn.get_style_context().add_class("Button-action-stop")    
                self.palmlandmark_btn.get_style_context().add_class("Button-action-stop")
                self.posedetect_mobilenetssd_btn.get_style_context().add_class("Button-action-stop")
                self.posedetect_YOLOv8s_btn.get_style_context().add_class("Button-action-stop")
                self.depthestimation_MegaDepth_btn.get_style_context().add_class("Button-action-stop")
                self.title_app = "NXP i.MX95 Demos"

            # Switch Dual Camera   
            if (CAMERA_DUAL_MIPI == 0 ) :
                self.dual_objectdetect_mobinetssdv2_btn.set_sensitive(False)
                self.dual_objectdetect_mobinetssdv2_btn.get_style_context().add_class("Button-action-stop")

        # (4) Button and Grid Setting by FUNC
        if 1 :
            # Create a custom header/title
            header = Gtk.HeaderBar()
            header.set_title("AI : Shaping a Smarter Tomorrow")
            header.get_style_context().add_class("header_AI")
            header.set_subtitle(self.title_app)
            self.set_titlebar(header)

            # Button to quit
            quit_button           = Gtk.Button()
            quit_icon             = Gio.ThemedIcon(name="save-alt")
            quit_image            = Gtk.Image.new_from_gicon(quit_icon, Gtk.IconSize.BUTTON)
            quit_button.add(quit_image)
            header.pack_end(quit_button)
            quit_button.connect("clicked", Gtk.main_quit)
            quit_button.get_style_context().add_class("Button-quit")

            # Button of SettingISP
            if PLATFORM == "IMX8MPLUSEVK":  # [WPI] ISP Function
                self.settingISP = VeriSilicon_ISP_SettingsWindow(CameraDevice)
                self.settingISP_WINDOWS = False
                settingISP_button = create_button_with_icon(icon_name="applications-system-symbolic", style_class="Button-settingISP", callback=self.open_settingVeriSiliconISP, header=header )

            # Button of Message
            self.MsgResult = ""
            self.debugMsg = MsgWindow()
            self.debugMsg_WINDOWS = False
            debugMsg_button = create_button_with_icon(icon_name="debug-breakpoint-unsupported", style_class="Button-debugMsg", callback=self.open_debugMsg,  header=header)

            # Button of SaveImage
            self.generalFunc = SaveLoadImageWindow(self.LoadingImagePath, self.FLAG_AI_Inference)
            self.SaveImageSignal = False
            generalFunc_button = create_button_with_icon(icon_name="pan-up-symbolic",  style_class="Button-saveImg",   callback=self.open_saveImg,  header=header)

            # Button of Special Function 
            specialFunc_button = create_button_with_icon(icon_name="emblem-important-symbolic", style_class="Button-specialFunc", callback=self.specialFunc_Change,  header=header)

            # Button of Auto-Run Function
            AutoRunFunc_button = create_button_with_icon(icon_name="emblem-favorite-symbolic", style_class="Button-specialFunc",  callback=self.autoRunFunc_Change,  header=header)
            
        # (5) Draw AREA
        # Area to display video
        self.draw_area = Gtk.DrawingArea.new()
        self.draw_area.set_hexpand(True)
        self.draw_area.set_size_request( FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY )
        self.draw_area.connect("draw", self.draw_cb)# Tell GTK what function to use to draw

        # -----------------------------------------------------------------------
        # GUI - Grid
        # -----------------------------------------------------------------------
        # Add video and label to window
        # Define the main menu and category-specific grids
        self.host_grid         = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.menu_grid         = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.menu_top_grid     = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.restore_grid      = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.object_grid       = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.segmentation_grid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.feature_grid      = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.other_grid        = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.category_grid     = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # add function of button
        self.back_button = Gtk.Button(label="Back")
        self.back_button.connect("clicked", lambda _: self.update_bacK_menu())

        # button layout
        self.menu_buttons_configs = [
            {"name": self.Bclass[0], "label": " Restore", "icon": "../data/icon/recycle_24.png", "category_grid": self.restore_grid},
            {"name": self.Bclass[1], "label": " Object Detect", "icon": "../data/icon/detector_24.png", "category_grid": self.object_grid},
            {"name": self.Bclass[2], "label": "Segmentation", "icon": "../data/icon/segment_24.png", "category_grid": self.segmentation_grid},
            {"name": self.Bclass[3], "label": "Features", "icon": "../data/icon/feature_24.png", "category_grid": self.feature_grid},
            {"name": self.Bclass[4], "label": " Others", "icon": "../data/icon/other_24.png", "category_grid": self.other_grid}, ]
        for menu_buttons_config in self.menu_buttons_configs:
            menu_button = Gtk.Button(label=menu_buttons_config["label"])
            menu_button.set_image(Gtk.Image.new_from_file(menu_buttons_config["icon"]))
            menu_button.set_always_show_image(True)
            menu_button.set_size_request(BUTTON_WIDTH, BUTTON_HEIGHT)
            # connect
            if menu_buttons_config["name"] == "Restore" :
                menu_button.connect("clicked",self.setup_restore)
            else :
                menu_button.connect("clicked", lambda _, grid=menu_buttons_config["category_grid"]: self.update_category(grid))
            # menu pack
            self.menu_grid.pack_start(menu_button, False, True, 0) 
            self.menu_button_mapping[menu_buttons_config["name"]] = menu_button
            
        # main layout
        self.menu_top_grid.set_margin_top(20)
        self.menu_top_grid.pack_start(self.show_logo, False, True, 15)
        self.menu_top_grid.pack_start(self.restore_grid, False, True, 0)
        self.host_grid.set_margin_left(10)
        self.host_grid.pack_start(self.menu_top_grid, False, True, 0)
        self.host_grid.pack_start(self.menu_grid, False, True, 0)

        # Populate each category with specific grids
        def populate_category(category_grid, grids):
            """
            Populates a category with its grids and dynamically adds a unique Back button.
            """
            for grid in grids:
                category_grid.pack_start(grid, False, True, 3) # 是否固定小, 間距
   
            # Create a unique Back button for each category
            back_button = Gtk.Button(label="Back")
            back_button.connect("clicked", lambda _: self.update_bacK_menu())
            back_button.set_size_request(BUTTON_WIDTH, BUTTON_HEIGHT) 
            category_grid.pack_start(back_button, False, True, 0) 
            
        # Assign grids to categories
        populate_category(self.object_grid, [
            self.objectdetect_mobinetssdv2_grid,
            self.objectdetect_YOLOv5s_grid,
            #self.objectdetect_YOLOv8n_grid,
            self.maskdetect_YOLOv5s_grid,
            self.hardhatdetect_YOLOv5s_grid,
            self.fruitdetect_YOLOv5s_grid,
            #self.PCBdetect_YOLOv8s_grid, # Fial by opgyro 6.1.55 
            self.ADAS_mobilnetssdv2_grid,
        ])

        populate_category(self.segmentation_grid, [
            self.segmentation_YOLOv5s_grid,
        ])

        populate_category(self.feature_grid, [
            self.posedetect_mobilenetssd_grid,
            self.posedetect_YOLOv8s_grid,
            self.facemesh_grid,
            #self.palmlandmark_grid,
        ])

        populate_category(self.other_grid, [
            self.depthestimation_MegaDepth_grid,
            self.benchmark_grid,
            self.contact_grid,
        ])

        # Add the main menu grid to the scrollable area
        self.scroll_grid = Gtk.ScrolledWindow()
        self.scroll_grid.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.ALWAYS)
        self.scroll_grid.add(self.host_grid)

        # Create the main layout
        self.main_grid = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.main_grid.pack_start(self.scroll_grid, True, True, 0)

        # Camera view or video area
        self.draw_area_alignment = Gtk.Alignment() 
        self.draw_area_alignment.set(0.5, 0.5, 0, 0)
        self.draw_area_alignment.add(self.draw_area)
        self.draw_area_alignment.set_margin_start(50)
        self.draw_area_alignment.set_margin_end(50) 
        self.draw_area_alignment.set_valign(Gtk.Align.CENTER)
        self.main_grid.pack_start(self.draw_area_alignment, True, True, 0)  # Camera/Video area
        self.add(self.main_grid)
        self.connect("motion-notify-event", self.on_touch_event)

        # -----------------------------------------------------------------------
        # Loading AI mode
        # -----------------------------------------------------------------------
        if (PLATFORM=="IMX8MPLUSEVK") : 
            self.FLAG_AI_Inference         = ''
            self.drawout_details           = ""
            self.drawout_frame             = ''
            self.ImageNumber               = 0
            self.ImageNumberTemp           = 0
            self.SpecialSignal             = True
            self.IsLoadPicture             = False
            self.AutoRunFunc               = False
            self.IPS_list                  = []
            self.CPU_percentage_list       = []
            self.AI_model_obj              = GetInferenceMode("../data/model/ssd_mobilenet_quant_224x224.tflite","vx")
            self.AI_model_obj_yolov5s      = GetInferenceMode("../data/model/yolov5s-objectdetect_integer_quant_256x256.tflite","vx")
            self.AI_model_seg_yolov5s      = GetInferenceMode("../data/model/yolov5s-segmentation_integer_quant_256x256.tflite","vx")
            self.AI_model_face             = GetInferenceMode("../data/model/ssd_mobilenet_facedetect_quant_224x224.tflite","vx")
            self.AI_model_facemesh         = GetInferenceMode("../data/model/mediapipe_facemesh_quant_192x192.tflite","vx")
            self.AI_model_facialEmotion    = GetInferenceMode("../data/model/facial_expression_detection.tflite","vx")
            self.AI_model_facialAgeGender  = GetInferenceMode("../data/model/facial_age_gender_detection.tflite","vx")
            self.AI_model_faceRecognition  = GetInferenceMode("../data/model/sface.tflite","vx")
            self.AI_model_facemask_yolov5  = GetInferenceMode("../data/model/yolov5s-maskdetect_integer_quant_256x256.tflite","vx")
            self.AI_model_hardhat_yolov5   = GetInferenceMode("../data/model/yolov5s-hardhatdetect_integer_quant_256x256.tflite","vx")
            self.AI_model_palm             = GetInferenceMode("../data/model/palm_detection_builtin_256_integer_quant.tflite","vx")
            self.AI_model_landmark         = GetInferenceMode("../data/model/hand_landmark_3d_256_integer_quant.tflite","vx")
            self.AI_model_lane             = GetInferenceMode("../data/model/LDW_tusimple_integer_quant.tflite","vx")
            self.AI_model_personSeg        = GetInferenceMode("../data/model/bodypix_concrete.tflite","vx")
            self.AI_model_PCBdetect_yolov8s= GetInferenceMode("../data/model/yolov8s-PCBdetect_integer_quant_256x256.tflite","vx") 
            self.AI_model_posedetect_yolov8s= GetInferenceMode("../data/model/yolov8s-pose_integer_quant_v2.tflite","vx")
            self.AI_model_posedetect_mobilenetssd= GetInferenceMode("../data/model/posenet_mobilenet_v1_075_481_641_quant.tflite","vx")  # [WPI] It is operator fail before BSP L5.15.72
            self.AI_model_fruitdetect_yolov5s= GetInferenceMode("../data/model/yolov5s-fruitdetect_integer_quant_256x256.tflite","vx")    # [WPI] It is operator fail before BSP L5.15.72
            self.AI_model_depthestimation_megadepth = GetInferenceMode("../data/model/megadepth_integer_quant_192x256.tflite","vx")   
            self.Benchmark_img = "../data/Benchmark_IMX8MP.png"
            self.contact_img = "../data/ContactUs_IMX8MP.png"
            self.FLAG_AI_SPEED         = 40
            self.debugMsg.change_msg(os.popen('dmesg | tail -n20').read()) 
        if (PLATFORM=="IMX93EVK") : 
            self.FLAG_AI_Inference         = 'objectdetect_mobinetssdv2'
            self.drawout_details           = ""
            self.drawout_frame             = ''
            self.ImageNumber               = 0
            self.ImageNumberTemp           = 0 
            self.SpecialSignal             = True
            self.IsLoadPicture             = False
            self.AutoRunFunc               = False
            self.IPS_list                  = []
            self.CPU_percentage_list       = []
            self.AI_model_obj              = GetInferenceMode("../data/model_vela/ssd_mobilenet_quant_224x224_vela.tflite","ethosu")
            self.AI_model_obj_yolov5s      = GetInferenceMode("../data/model_vela/yolov5s-objectdetect_integer_quant_256x256_vela.tflite","ethosu")
            self.AI_model_seg_yolov5s      = GetInferenceMode("../data/model_vela/yolov5s-segmentation_integer_quant_256x256_vela.tflite","ethosu")
            self.AI_model_face             = GetInferenceMode("../data/model_vela/ssd_mobilenet_facedetect_quant_224x224_vela.tflite","ethosu")
            self.AI_model_facemesh         = GetInferenceMode("../data/model_vela/mediapipe_facemesh_quant_192x192_vela.tflite","ethosu")
            #self.AI_model_facialEmotion    = GetInferenceMode("../data/model_vela/facial_expression_detection_vela.tflite","ethosu")
            #self.AI_model_facialAgeGender  = GetInferenceMode("../data/model_vela/facial_age_gender_detection_vela.tflite","ethosu") # [WPI] It is operator fail before BSP L6.1.55
            self.AI_model_faceRecognition  = GetInferenceMode("../data/model_vela/facenet_qunat_vela.tflite","ethosu")
            self.AI_model_facemask_yolov5  = GetInferenceMode("../data/model_vela/yolov5s-maskdetect_integer_quant_256x256_vela.tflite","ethosu")
            self.AI_model_hardhat_yolov5   = GetInferenceMode("../data/model_vela/yolov5s-hardhatdetect_integer_quant_256x256_vela.tflite","ethosu")
            self.AI_model_palm             = GetInferenceMode("../data/model_vela/palm_detection_builtin_256_integer_quant_vela.tflite","ethosu")
            self.AI_model_landmark         = GetInferenceMode("../data/model_vela/hand_landmark_3d_256_integer_quant_vela.tflite","ethosu")
            self.AI_model_lane             = GetInferenceMode("../data/model_vela/LDW_tusimple_integer_quant_vela.tflite","ethosu")
            self.AI_model_PCBdetect_yolov8s= GetInferenceMode("../data/model_vela/yolov8s-PCBdetect_integer_quant_256x256_vela.tflite","ethosu") 
            self.AI_model_posedetect_yolov8s= GetInferenceMode("../data/model_vela/yolov8s-pose_integer_quant_vela.tflite","ethosu") # [WPI] It is operator fail before BSP L6.1.36
            self.AI_model_posedetect_mobilenetssd= GetInferenceMode("../data/model_vela/posenet_mobilenet_v1_075_481_641_quant_vela.tflite","ethosu") 
            self.AI_model_fruitdetect_yolov5s= GetInferenceMode("../data/model_vela/yolov5s-fruitdetect_integer_quant_256x256_vela.tflite","ethosu") 
            self.AI_model_depthestimation_megadepth = GetInferenceMode("../data/model_vela/megadepth_integer_quant_192x256_vela.tflite","ethosu") # [WPI] It is operator fail before BSP L6.1.36
            self.Benchmark_img = "../data/Benchmark_IMX93.png"
            self.contact_img = "../data/ContactUs_IMX93.png"
            self.FLAG_AI_SPEED         = 40
            self.debugMsg.change_msg(os.popen('dmesg | tail -n20').read()) 
        if (PLATFORM=="IMX95EVK") : 
            self.FLAG_AI_Inference         = 'objectdetect_YOLOv8n'
            self.drawout_details           = ""
            self.drawout_frame             = ''
            self.ImageNumber               = 0
            self.ImageNumberTemp           = 0 
            self.SpecialSignal             = True
            self.IsLoadPicture             = False
            self.AutoRunFunc               = False
            self.IPS_list                  = []
            self.CPU_percentage_list       = []
            #self.AI_model_obj              = GetInferenceMode("../data/model_neutron/ssd_mobilenet_quant_224x224_vela.tflite","neutrons")
            #self.AI_model_obj_yolov5s      = GetInferenceMode("../data/model_neutron/yolov5s-objectdetect_integer_quant_256x256_vela.tflite","neutrons")
            #self.AI_model_seg_yolov5s      = GetInferenceMode("../data/model_neutron/yolov5s-segmentation_integer_quant_256x256_vela.tflite","neutrons")
            self.AI_model_obj_yolov8n       = GetInferenceMode("../data/model_neutron/yolov8n-objectdetect_full_integer_quant-320-neutron.tflite","neutrons")
            #self.AI_model_face             = GetInferenceMode("../data/model_neutron/ssd_mobilenet_facedetect_quant_224x224_vela.tflite","neutrons")
            #self.AI_model_facemesh         = GetInferenceMode("../data/model_neutron/mediapipe_facemesh_quant_192x192_vela.tflite","neutrons")
            #self.AI_model_facialEmotion    = GetInferenceMode("../data/model_neutron/facial_expression_detection_vela.tflite","neutrons")
            #self.AI_model_facialAgeGender  = GetInferenceMode("../data/model_neutron/facial_age_gender_detection_vela.tflite","neutrons") # [WPI] It is operator fail before BSP L6.1.55
            #self.AI_model_faceRecognition  = GetInferenceMode("../data/model_neutron/facenet_qunat_vela.tflite","neutrons")
            #self.AI_model_facemask_yolov5  = GetInferenceMode("../data/model_neutron/yolov5s-maskdetect_integer_quant_256x256_vela.tflite","neutrons")
            self.AI_model_hardhat_yolov5   = GetInferenceMode("../data/model_neutron/yolov5s-hardhatdetect_integer_quant_256x256-neutron.tflite","neutrons")
            #self.AI_model_palm             = GetInferenceMode("../data/model_neutron/palm_detection_builtin_256_integer_quant_vela.tflite","neutrons")
            #self.AI_model_landmark         = GetInferenceMode("../data/model_neutron/hand_landmark_3d_256_integer_quant_vela.tflite","neutrons")
            #self.AI_model_lane             = GetInferenceMode("../data/model_neutron/LDW_tusimple_integer_quant_vela.tflite","neutrons")
            #self.AI_model_PCBdetect_yolov8s= GetInferenceMode("../data/model_neutron/yolov8s-PCBdetect_integer_quant_256x256_vela.tflite","neutrons") 
            #self.AI_model_posedetect_yolov8s= GetInferenceMode("../data/model_neutron/yolov8s-pose_integer_quant_vela.tflite","neutrons") # [WPI] It is operator fail before BSP L6.1.36
            #self.AI_model_posedetect_mobilenetssd= GetInferenceMode("../data/model_neutron/posenet_mobilenet_v1_075_481_641_quant_vela.tflite","neutrons") 
            self.AI_model_fruitdetect_yolov5s= GetInferenceMode("../data/model_neutron/yolov5s-fruitdetect_integer_quant_256x256_neutron.tflite","neutrons") 
            #self.AI_model_depthestimation_megadepth = GetInferenceMode("../data/model_neutron/megadepth_integer_quant_192x256_vela.tflite","neutrons") # [WPI] It is operator fail before BSP L6.1.36
            self.Benchmark_img = "../data/Benchmark_IMX95.png"
            self.contact_img = "../data/ContactUs_IMX95.png"
            self.FLAG_AI_SPEED         = 40
            self.debugMsg.change_msg(os.popen('dmesg | tail -n20').read()) 

        # -----------------------------------------------------------------------
        # Pre-Algorithm
        # -----------------------------------------------------------------------
        # 提取人臉資訊
        if(0):
            self.FaceRecognitionFeature = []
            self.FaceRecognitionImage   = []
            self.database_path      = load_faceRecognitionDataBase(self.AI_model_faceRecognition, "../data/img/DataBase_Face_Recognition/", self.FaceRecognitionFeature, self.FaceRecognitionImage)
            self.database_path_keep          = copy.copy(self.database_path)
            self.FaceRecognitionFeature_keep = copy.copy(self.FaceRecognitionFeature)
            self.FaceRecognitionImage_keep   = copy.copy(self.FaceRecognitionImage)

        # -----------------------------------------------------------------------
        # GStreamer
        # -----------------------------------------------------------------------        
        # Note that the format is in RGB16. I'm not sure if this is the only format that can be used, but it seems the most straight forward
        if (PLATFORM=="IMX8MPLUSEVK") : 
            # Select Dual Camera
            if (CAMERA_DUAL_MIPI) :
                cam_pipeline = ( #(dual-camera) # https://github.com/nnstreamer/nnstreamer
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
                        "tensor_converter ! tensor_filter framework=tensorflow-lite model=/home/root/ATU-Camera-GTK/data/model/ssd_mobilenet_quant_224x224.tflite custom=Delegate:External,ExtDelegateLib:libvx_delegate.so !" +
                        "tensor_decoder mode=bounding_boxes option1=tf-ssd option2=/home/root/ATU-Camera-GTK/data/label/coco_labels.txt option3=0:1:2:3,50 option4=" + str(int(FRAME_WIDTH)) + ":" + str(int(FRAME_HEIGHT)) + " option5=300:300 ! " +
                        "mix. t1. ! queue max-size-buffers=10 leaky=2 ! " + 
                        "imxcompositor_g2d name=mix latency=30000000 min-upstream-latency=30000000 sink_0::zorder=2 sink_1::zorder=1 ! " + 
                        "imxvideoconvert_g2d  ! video/x-raw,format=RGB16,width=" + str(int(FRAME_WIDTH)) + ",height=" + str(int(FRAME_HEIGHT)) + ",framerate=40/1 ! " + 
                        "videoconvert ! queue max-size-buffers=10 leaky=2 ! appsink emit-signals=true name=sink3"
                        )
            else :
                cam_pipeline = (
                        "v4l2src device=" + VIDEO + " ! " + CAMERA_FORMAT + "imxvideoconvert_g2d " +
                        "! video/x-raw,format=RGB16,width=" + str(int(FRAME_WIDTH_DISPLAY)) + ",height=" + str(int(FRAME_HEIGHT_DISPLAY)) + " ! " + 
                        "tee name=t t. ! queue max-size-buffers=0 leaky=2 ! appsink emit-signals=true name=sink1 " +
                        "t. ! queue max-size-buffers=0 leaky=2 ! " + 
                        "videoconvert ! video/x-raw,format=RGB ! appsink emit-signals=true name=sink2 "
                        )
        if (PLATFORM=="IMX93EVK") : 
            cam_pipeline = (
               "v4l2src device=" + VIDEO + " ! " + CAMERA_FORMAT + " imxvideoconvert_pxp " +               
               "! video/x-raw,format=RGB16 ,width=" + str(int(FRAME_WIDTH_DISPLAY)) + ",height=" + str(int(FRAME_HEIGHT_DISPLAY)) + "! " +
               "imxvideoconvert_pxp rotation=" + str(int(CAMERA_ROTATION)) + " !" + 
               "tee name=t t. ! "+
               "queue max-size-buffers=0 leaky=2 ! " +
               "appsink emit-signals=true name=sink1 sync=true " +
               " t. ! queue max-size-buffers=0 leaky=2 !  "+
               "videoconvert ! video/x-raw,format=RGB ! "+ 
               "appsink emit-signals=true name=sink2 sync=true "
                )
        if (PLATFORM=="IMX95EVK") : 
            cam_pipeline = (
               "v4l2src device=" + VIDEO + " ! " + CAMERA_FORMAT + " imxvideoconvert_g2d " +               
               "! video/x-raw,format=RGB16 ,width=" + str(int(FRAME_WIDTH_DISPLAY)) + ",height=" + str(int(FRAME_HEIGHT_DISPLAY)) + "! " +
               "imxvideoconvert_g2d rotation=" + str(int(CAMERA_ROTATION)) + " !" + 
               "tee name=t t. ! "+
               "queue max-size-buffers=0 leaky=2 ! " +
               "appsink emit-signals=true name=sink1 sync=true " +
               " t. ! queue max-size-buffers=0 leaky=2 !  "+
               "videoconvert ! video/x-raw,format=RGB ! "+ 
               "appsink emit-signals=true name=sink2 sync=true "
                )

        # update flaginference (initial)
        writeFlagInference(self.FLAG_AI_Inference)

        # update extra loading picture
        writeExtraLoadingPictue(str(self.Extra_loadingPictureSignal))

        # update clock
        self.refresh_clock = time.perf_counter()

        # Parse the above pipeline
        self.pipeline = Gst.parse_launch(cam_pipeline)

        # Set a callback function to get the frame
        tensor_sink1  = self.pipeline.get_by_name('sink1')
        tensor_sink2  = self.pipeline.get_by_name('sink2')
        tensor_sink1.connect('new-sample', self.on_data_original)  # [WPI] RAW Image
        tensor_sink2.connect('new-sample', self.on_data_inference) # [WPI] Detect  Camera 0 ( inference )

        # Set a callback function to get the frame (dual-camera)
        if (CAMERA_DUAL_MIPI) :
            tensor_sink3  = self.pipeline.get_by_name('sink3')
            tensor_sink3.connect('new-sample', self.on_data_original) # [WPI] RAW Image
        
        # Run the pipeline
        self.frame_count = 0
        self.timer = time.perf_counter()
        state_change_return = self.pipeline.set_state(Gst.State.PLAYING) # checnk states when switch
        if state_change_return == Gst.StateChangeReturn.FAILURE:
            logging.error("Failed to set pipeline to PLAYING")

        # Anomaly detect  , by 2024/12/20 Weilly Li
        self.last_draw_cb_time = time.monotonic()
        GLib.timeout_add_seconds(30, self.check_draw_cb_activity)

        # Print Show
        print("----------------------------------------------------------")
        print("The Code of WPI offer by version 11")
        print("Please follow licence")
        print("----------------------------------------------------------")
        print("Platform [ IMX8MPLUSEVK, IMX93EVK, IMX95EVK] : ", PLATFORM)
        print("----------------------------------------------------------")
        print("Camera [/dev/Video*] :", VIDEO)
        print("Camera Format :", CAMERA_FORMAT)
        print("Camera Dual [ Disable , Enable] :", CAMERA_DUAL_MIPI)
        print("----------------------------------------------------------")
        print("Logo Image:" , IMAGE_LOGO_PATH  )
        print("CSS Style:" , GTK_STYLE_CSS_PATH     )
        print("Frame Size : (",  FRAME_WIDTH_DISPLAY  , " , ",  FRAME_HEIGHT_DISPLAY, ")" )
        print("Windows Size : (",  WINDOWS_MAIN_WIDTH     , " , ",  WINDOWS_MAIN_HEIGHT  , ")" )
        print("Windows Maximum [ Disable , Enable] :", WINDOWS_MAXIMUM)
        print("----------------------------------------------------------")

    # --------------------------------------------
    # Original 
    # --------------------------------------------    
    def on_data_original(self, element): 
        """
        Get the new frame from the GStreamer pipeline and signal the redraw.

        This function retrieves a sample (frame) from the specified GStreamer appsink 
        and assigns it to the appropriate buffer (BufferRGB565_1 or BufferRGB565_2) 
        depending on the element's name. After processing, it clears the buffer to free 
        memory and signals the drawing area to redraw the updated frame.

        Args:
            element (Gst.Element): The GStreamer element emitting the 'new-sample' signal.

        Returns:
            int: Always returns 0 to indicate successful processing.
        """
        try:
            # Determine which buffer to update based on the element name
            if element.name == "sink1":
                self.BufferRGB565_1 = element.emit('pull-sample')  # Retrieve sample from sink1
            else:
                self.BufferRGB565_2 = element.emit('pull-sample')  # Retrieve sample from sink2
            
            # Retrieve the GStreamer buffer associated with the sample
            buffer = self.BufferRGB565_1.get_buffer() if element.name == "sink1" else self.BufferRGB565_2.get_buffer()

            # Free memory associated with the buffer
            del buffer

            # Signal the drawing area to update the display with the new frame
            self.draw_area.queue_draw()

        except Exception as e:
            # Log any exceptions encountered during frame processing
            logging.error(f"Error in on_data_original: {e}", exc_info=True)
            print("Error in on_data_original")
        return 0
    # --------------------------------------------
    # AI Inference
    # --------------------------------------------
    def handle_auto_run(self):
        """Handle AutoRun functionality."""
        auto_configs = [config for config in self.buttonsAPP_config if config.get("auto", False)]

        # Calculate the current index for AutoRun
        n = self.FLAG_AI_COUNTER // self.FLAG_AI_SPEED
        if self.FLAG_AI_COUNTER >= self.FLAG_AI_SPEED * len(auto_configs):
            self.FLAG_AI_COUNTER = 1  # Reset counter
        else:
            if n < len(auto_configs):
                selected_button = auto_configs[n]
                self.FLAG_AI_Inference = selected_button["name"]
                self.IsLoadPicture = selected_button["image_loading"]
                self.FLAG_AI_COUNTER += selected_button["auto_speed_level"]
            else:
                self.FLAG_AI_COUNTER = 1  # Reset counter if out of range
    def run_selected_algorithm(self, frame):
        """Run the selected algorithm based on FLAG_AI_Inference."""
        try:
            if (self.FLAG_AI_Inference == "objectdetect_mobinetssdv2") :
                frame_rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                self.PicturePath = self.LoadingImagePath["objectdetect_mobinetssdv2"]
                self.drawout_details = detect_objectdetect_mobilnetv2_ssd(self.AI_model_obj, frame_rgb, self.PicturePath , self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "objectdetect_YOLOv5s") :
                self.PicturePath = self.LoadingImagePath["objectdetect_YOLOv5s"]
                self.drawout_details = detect_objectdetect_YOLOv5s(self.AI_model_obj_yolov5s, frame, self.PicturePath , self.IsLoadPicture) 
            if (self.FLAG_AI_Inference == "objectdetect_YOLOv8n") :
                self.PicturePath = self.LoadingImagePath["objectdetect_YOLOv8n"]
                self.drawout_details = detect_objectdetect_YOLOv8n_NXP(self.AI_model_obj_yolov8n, frame, self.PicturePath , self.IsLoadPicture) 
            if (self.FLAG_AI_Inference == "segmentation_YOLOv5s") :
                self.PicturePath = self.LoadingImagePath["segmentation_YOLOv5s"]
                self.drawout_details = detect_segmentation_YOLOv5s(self.AI_model_seg_yolov5s, frame, self.PicturePath , self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "facedetect_mobilenetssdv2") :
                frame_rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.PicturePath     = self.LoadingImagePath["facedetect_mobilenetssdv2"]
                self.drawout_details = detect_facedetect_mobilnetv2_ssd(self.AI_model_face, frame_rgb, self.PicturePath , self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "facemesh") :
                frame_rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.PicturePath     = self.LoadingImagePath["facemesh"]
                self.drawout_details = detect_facemesh_mediapipe(self.AI_model_face, self.AI_model_facemesh, frame_rgb, self.PicturePath, self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "facerecongition") :
                self.SpecialSignal = self.IsLoadPicture
                if(self.SpecialSignal) : 
                    self.database_path          = copy.copy(self.database_path_keep)
                    self.FaceRecognitionFeature = copy.copy(self.FaceRecognitionFeature_keep)
                    self.FaceRecognitionImage   = copy.copy(self.FaceRecognitionImage_keep)
                    self.Special_Signal = False # [WPI] Close Status
                self.IsLoadPicture = self.SpecialSignal 
                touch_signal         = [self.touch_signal, self.touch_postionX, self.touch_postionY]
                frame_rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.drawout_details = detect_faceRecognition( self.AI_model_face, self.AI_model_faceRecognition, self.database_path,  frame_rgb, self.FaceRecognitionFeature, self.FaceRecognitionImage, touch_signal)
            if (self.FLAG_AI_Inference == "age_gender_recognition") :
                self.PicturePath     = self.LoadingImagePath["age_gender_recognition"]
                self.drawout_details = detect_age_gender_recognition(self.AI_model_face, self.AI_model_facialAgeGender, self.AI_model_facialEmotion, frame, self.PicturePath, self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "maskdetect_YOLOv5s") :
                self.PicturePath     = self.LoadingImagePath["maskdetect_YOLOv5s"]
                self.drawout_details = detect_facemask_YOLOv5s(self.AI_model_facemask_yolov5, frame, self.PicturePath, self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "hardhatdetect_YOLOv5s") :
                self.PicturePath     = self.LoadingImagePath["hardhatdetect_YOLOv5s"]
                #self.ImageNumberTemp = 1 if self.ImageNumberTemp > 445 else self.ImageNumberTemp + 5 ; self.PicturePath = "../data/img/DataBase_HardHat/worker_"+ str(self.ImageNumberTemp) +".jpg" 
                self.drawout_details = detect_hardhat_YOLOv5s(self.AI_model_hardhat_yolov5, frame, self.PicturePath, self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "palmlandmark") :
                self.PicturePath     = self.LoadingImagePath["palmlandmark"]
                self.drawout_details = detect_palm_and_landmark(self.AI_model_palm, self.AI_model_landmark, frame, self.PicturePath, self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "posedetect_mobilenetssd") :
                self.ImageNumberTemp = 1 if self.ImageNumberTemp > 319 else self.ImageNumberTemp + 10
                self.PicturePath = "../data/img/DataBase_Pose_Dencer/dancer_"+ str(self.ImageNumberTemp) +".jpg"
                self.drawout_details = detect_posedetect_mobilenetssd(self.AI_model_posedetect_mobilenetssd, frame, self.PicturePath, self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "posedetect_YOLOv8s") :
                self.PicturePath = self.LoadingImagePath["posedetect_YOLOv8s"]
                self.drawout_details = detect_posedetect_YOLOv8s(self.AI_model_posedetect_yolov8s, frame, self.PicturePath , self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "PCBdetect_YOLOv8s") :
                self.PicturePath = self.LoadingImagePath["PCBdetect_YOLOv8s"]
                self.drawout_details = detect_PCBdetect_YOLOv8s(self.AI_model_PCBdetect_yolov8s, frame, self.PicturePath , self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "ADAS_mobilnetssdv2") :
                self.ImageNumberTemp = 1 if self.ImageNumberTemp > 2000 else self.ImageNumberTemp + 10
                self.PicturePath = "../data/img/DataBase_ADAS_HighWay/adas_"+ str(self.ImageNumberTemp) +".jpg"
                self.drawout_details = detect_ADAS( self.AI_model_lane ,self.AI_model_obj, frame, self.PicturePath , self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "fruitdetect_YOLOv5s") :
                self.PicturePath = self.LoadingImagePath["fruitdetect_YOLOv5s"]
                #self.ImageNumberTemp = 1 if self.ImageNumberTemp > 350 else self.ImageNumberTemp + 15
                #self.PicturePath = "../data/img/DataBase_FreshFruit/fruit_"+ str(self.ImageNumberTemp) +".jpg"
                self.drawout_details = detect_fruitdetect_YOLOv5s(self.AI_model_fruitdetect_yolov5s, frame,  self.PicturePath , self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "depthestimation_MegaDepth") :
                self.PicturePath = self.LoadingImagePath["depthestimation_MegaDepth"]
                self.drawout_details = detect_depthestimation_MegaDepth(self.AI_model_depthestimation_megadepth, frame, self.PicturePath,  self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "dual-objectdetect_mobinetssdv2") :
                frame_rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                self.PicturePath = self.LoadingImagePath["dual-objectdetect_mobinetssdv2"]
                self.drawout_details = detect_objectdetect_mobilnetv2_ssd(self.AI_model_obj, frame_rgb, self.PicturePath , self.IsLoadPicture)
            if (self.FLAG_AI_Inference == "benchmark") :
                self.drawout_details = LoadingImage_OutputRGB565( self.Benchmark_img , int(FRAME_WIDTH_DISPLAY) , int(FRAME_HEIGHT_DISPLAY) )
            if (self.FLAG_AI_Inference == "contact") :
                self.drawout_details = LoadingImage_OutputRGB565( self.contact_img , int(FRAME_WIDTH_DISPLAY) , int(FRAME_HEIGHT_DISPLAY) )
        except Exception as e:
            logging.error(f"Error running algorithm {self.FLAG_AI_Inference}: {e}", exc_info=True)
            print("Error running algorithm")
    def on_data_inference(self, element):
        """Get the new frame and run inference."""
        vision_time = time.monotonic()

        try:
            # Retrieve buffer and caps based on element name
            if element.name == "sink2":
                self.BufferRGB888_1 = element.emit('pull-sample')
                buffer = self.BufferRGB888_1.get_buffer()
                caps = self.BufferRGB888_1.get_caps()
            else:
                self.BufferRGB888_2 = element.emit('pull-sample')
                buffer = self.BufferRGB888_2.get_buffer()
                caps = self.BufferRGB888_2.get_caps()

            # Map buffer and extract frame details
            ret, mem_buf = buffer.map(Gst.MapFlags.READ)
            if not ret:
                logging.error("Failed to map buffer")
                return 0

            height = caps.get_structure(0).get_value("height")
            width = caps.get_structure(0).get_value("width")
            frame_org = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=mem_buf.data)
            frame = frame_org[..., ::-1].copy()  # Convert to BGR format

            # Update application button styles
            self.update_styles_app_button()

            # Handle AutoRun functionality
            if self.AutoRunFunc:
                self.handle_auto_run()

            # Select algorithm based on FLAG_AI_Inference
            self.run_selected_algorithm(frame)

            # Calculate and update FPS
            self.FPS_Inference = str(round(1 / (time.monotonic() - vision_time), 2))[:2]

            # Update image info
            self.update_image_info(frame)

            # Update flag state
            self.update_flag_inference_state()

        except Exception as e:
            logging.error(f"Error during inference: {e}", exc_info=True)
            print("Error on_data_inference")

        finally:
            # Ensure buffer is unmapped
            if 'mem_buf' in locals():
                buffer.unmap(mem_buf)

            # Explicitly delete large objects and run garbage collection
            del frame_org, frame

        return 0
    # --------------------------------------------
    # AI Result 
    # --------------------------------------------
    def draw_cb(self, widget, context):    
        self.last_draw_cb_time = time.monotonic()
        try : 
            """Draw the frame in the GUI."""
            # 確保 Buffer 可用
            if self.BufferRGB565_1 is None:
                context.set_source_rgb(0, 0, 0)
                context.paint()
                return

            # 提取 Buffer 與幀的尺寸
            input_buffer_type = next((config["InputBuffer_Type"] for config in self.buttonsAPP_config if config["name"] == self.FLAG_AI_Inference),  "RGB565" )
            if input_buffer_type == "RGB888":
                self.buffer_1 = self.BufferRGB888_1.get_buffer()
                caps = self.BufferRGB888_1.get_caps()
            elif input_buffer_type == "RGB565x2":
                self.buffer_1 = self.BufferRGB565_1.get_buffer()
                self.buffer_2 = self.BufferRGB565_2.get_buffer()
                caps = self.BufferRGB565_1.get_caps()
            else:
                self.buffer_1 = self.BufferRGB565_1.get_buffer()
                caps = self.BufferRGB565_1.get_caps()

            height = caps.get_structure(0).get_value("height")
            width = caps.get_structure(0).get_value("width")

            # load special cases
            self.load_special_cases()

            # load picture_mode
            self.load_picture_mode()

            # 映射 GStreamer Buffer
            ret, mem_buf_1 = self.buffer_1.map(Gst.MapFlags.READ)
            if not ret:
                logging.error("Failed to map self.buffer_1")
                return
            if input_buffer_type == "RGB565x2":
                ret, mem_buf_2 = self.buffer_2.map(Gst.MapFlags.READ)
                if not ret:
                    logging.error("Failed to map self.buffer_2")
                    self.buffer_1.unmap(mem_buf_1)
                    return

            # Create context using Buffer
            if input_buffer_type == "RGB565x2":
                # 處理雙鏡頭輸入
                frame1 = np.ndarray(shape=(height, width), dtype=np.uint16, buffer=mem_buf_1.data).copy()
                frame2 = np.ndarray(shape=(height, width), dtype=np.uint16, buffer=mem_buf_2.data).copy()
                frame_combined = np.zeros((height, width), dtype=np.uint16)
                frame_combined[:, :width // 2] = frame1[:, ::2]
                frame_combined[:, width // 2:] = frame2[:, ::2]
                surface = cairo.ImageSurface.create_for_data(frame_combined, cairo.Format.RGB16_565, width, height)
                context.set_source_surface(surface, 0, 0)
                context.paint()
            elif input_buffer_type == "RGB888":
                # 處理 RGB888 幀數據
                frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=mem_buf_1.data).copy()
                frame_rgba = np.zeros((height, width, 4), dtype=np.uint8)
                frame_rgba[:, :, :3] = frame
                surface = cairo.ImageSurface.create_for_data(frame_rgba, cairo.FORMAT_ARGB32, width, height, width * 4)
                context.set_source_surface(surface, 0, 0)
                context.paint()
            else:
                # 處理單一鏡頭輸入
                frame = np.ndarray(shape=(height, width), dtype=np.uint16, buffer=mem_buf_1.data).copy()
                surface = cairo.ImageSurface.create_for_data(frame, cairo.Format.RGB16_565, width, height)
                context.set_source_surface(surface, 0, 0)
                context.paint()

            # AI DrawOut
            if (self.drawout_details != ""):
                try:
                    # Check if the list has enough elements to access index 1
                    if len(self.drawout_details) > 1:
                        if (self.FLAG_AI_Inference == "objectdetect_mobinetssdv2") :
                            draw_objectdetect_mobilnetv2_ssd(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "objectdetect_YOLOv5s"):
                            draw_objectdetect_YOLOv5s(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "objectdetect_YOLOv8n"):
                            draw_objectdetect_YOLOv8n_NXP(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "segmentation_YOLOv5s"):
                            self.drawout_frame = draw_segmentation_YOLOv5s(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "facedetect_mobilenetssdv2") :
                            draw_facedetect_mobilnetv2_ssd(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "facemesh") :
                            draw_facemesh_mediapipe(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "facerecongition") :
                            draw_faceRecognition(context, frame, self.drawout_details[1], "../data/img/DataBase_Face_Recognition/BillGates_1.jpg")
                        if (self.FLAG_AI_Inference == "age_gender_recognition") :
                            draw_age_gender_recognition(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "maskdetect_YOLOv5s") :
                            draw_facemask_YOLOv5s(context, frame, self.drawout_details[1]) 
                        if (self.FLAG_AI_Inference == "hardhatdetect_YOLOv5s") :
                            draw_hardhat_YOLOv5s(context, frame, self.drawout_details[1]) 
                        if (self.FLAG_AI_Inference == "palmlandmark") :
                            draw_palm_and_landmark(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "PCBdetect_YOLOv8s") :
                            draw_PCBdetect_YOLOv8s(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "fruitdetect_YOLOv5s") :
                            draw_fruitdetect_YOLOv5s(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "posedetect_YOLOv8s") :
                            draw_posedetect_YOLOv8s(context, frame, self.drawout_details[1])
                        if (self.FLAG_AI_Inference == "posedetect_mobilenetssd") :
                            draw_posedetect_mobilenetssd(context, frame, self.drawout_details[1]) 
                        if (self.FLAG_AI_Inference == "ADAS_mobilnetssdv2") :
                            draw_ADAS(context, frame, self.drawout_details[1]) 
                        if (self.FLAG_AI_Inference == "depthestimation_MegaDepth") :
                            draw_depthestimation_MegaDepth(context, frame, self.drawout_details[1]) 
                        if (self.FLAG_AI_Inference == "dual-objectdetect_mobinetssdv2") :
                            draw_objectdetect_mobilnetv2_ssd_ByDaulCamera(context, frame, self.drawout_details[1])
                    else:
                        # If the list does not have enough elements, log a warning or handle the case
                        print(f"Warning: drawout_details has insufficient elements for {self.FLAG_AI_Inference}")
                except Exception as e:
                    # Handle unexpected errors gracefully
                    print(f"Error processing inference for {self.FLAG_AI_Inference}: {e}")

            # Draw Infomation
            self.update_info_ips(context, (round(width*0.85,0), 70))
            self.draw_text_camera_dual(context) # Title of Dual Camera

            # Clean up
            if (self.FLAG_AI_Inference == "dual-objectdetect_mobinetssdv2") : 
                self.buffer_2.unmap(mem_buf_2) #CAMERA_DUAL_MIPI
                del mem_buf_2
                del self.buffer_2
            else :
                self.buffer_1.unmap(mem_buf_1)
                del mem_buf_1
                del self.buffer_1

            # Update Info
            self.generalFunc.update_Result_info(surface, self.ImageNumber, self.FLAG_AI_Inference)

            # Touch Siganl
            self.touch_signal_count = self.touch_signal_count + 1
            if (self.touch_signal_count>2):
                self.touch_signal = False
                self.touch_signal_count = 0

            # Msgbox
            if self.MsgResult != os.popen('dmesg | tail -n1').read() :
                self.MsgResult = os.popen('dmesg | tail -n1').read()
                self.debugMsg.change_msg(self.MsgResult)
        except Exception as e:
            logging.error(f"Error in on_data_inference: {e}", exc_info=True)
            print("Error in on_data_inference")
        return 0
    # --------------------------------------
    # Update & Other
    # --------------------------------------
    def generate_dynamic_setup_methods(self):
        """
        Dynamically generates setup_* methods and updates the buttonsAPP_config
        with the corresponding callback functions.
        """
        # Dynamically generate setup_* methods
        for config in self.buttonsAPP_config:
            method_name = f"setup_{config['name']}"  # Create method name

            # Function factory to create individual setup_* methods
            def create_setup_method(flag_name):
                def setup_method(self, unused):
                    self.FLAG_AI_Inference = flag_name
                    self.IPS_list = []
                    self.IsLoadPicture = [config["image_loading"] for config in self.buttonsAPP_config if config["name"] == flag_name][0]
                    writeFlagInference(self.FLAG_AI_Inference)
                    print(f"{method_name} called")
                return setup_method

            # Dynamically create and attach the method to the class
            setattr(MLVideoDemo, method_name, create_setup_method(config["name"]))

        # Update buttonsAPP_config with the generated callbacks
        for config in self.buttonsAPP_config:
            method_name = f"setup_{config['name']}"  # Find the corresponding method name
            # Add the method reference as the callback in the configuration
            config["callback"] = getattr(self, method_name)
    def setup_restore(self, unused):
        self.FLAG_AI_Inference = '' ; writeFlagInference(self.FLAG_AI_Inference)
        self.IPS_list = []
        self.update_styles_menu_button()
        self.menu_button_mapping["Restore"].get_style_context().add_class("Button-blue")
        print("setup_restore")
    def update_styles_app_button(self):
        for name, button in self.button_mapping.items():
            if self.FLAG_AI_Inference == name:
                button.get_style_context().add_class("Button-action")
            else:
                button.get_style_context().remove_class("Button-action")
    def update_styles_menu_button(self):
        for name, button in self.menu_button_mapping.items():
            type_value = next((config["Type"] for config in self.buttonsAPP_config if config["name"] == self.FLAG_AI_Inference), None)    
            if type_value == name or ( self.FLAG_AI_Inference=='' and name==self.Bclass[0]) :
                button.get_style_context().add_class("Button-action")
            else:
                button.get_style_context().remove_class("Button-action")
    def update_bacK_menu(self):
        """
        Display the main menu.
        """
        self.host_grid.remove(self.category_grid)
        self.host_grid.pack_start(self.menu_grid, False, True, 0)
        self.scroll_grid.get_child().destroy()  # Clear current content
        self.scroll_grid.add(self.host_grid)
        self.scroll_grid.show_all()
        self.update_styles_menu_button()
    def update_category(self, category_grid):
        """
        Display a specific category.
        """
        print("update_category")
        self.host_grid.remove(self.menu_grid)
        self.category_grid = category_grid
        self.host_grid.pack_start(category_grid, False, True, 0)
        self.scroll_grid.get_child().destroy()  # Clear current content
        self.scroll_grid.add(self.host_grid)
        self.scroll_grid.show_all()
        self.update_styles_menu_button()
    def update_image_info(self, frame):
        """Update image-related information."""
        self.ImageNumber += 1

        # update loagin image path
        self.generalFunc.update_Image_info(frame, self.ImageNumber, self.LoadingImagePath, self.FLAG_AI_Inference)
        self.LoadingImagePath = self.generalFunc.LoadingImagePath

        # update loading image state
        if self.generalFunc.IsLoadPicture and not self.IsLoadPicture:
            self.IsLoadPicture = self.generalFunc.IsLoadPicture
            self.generalFunc.IsLoadPicture = False
    def update_flag_inference_state(self):
        # Update from TXT
        tmp_FLAG_AI_Inference = getFlagInference()
        if tmp_FLAG_AI_Inference !=  self.FLAG_AI_Inference :
            self.FLAG_AI_Inference = tmp_FLAG_AI_Inference
            writeFlagInference(tmp_FLAG_AI_Inference)

        # update
        if self.AutoRunFunc == False :
            self.Extra_loadingPictureSignal = getExtraLoadingPictue()
            if self.Extra_loadingPictureSignal == "True":
                self.IsLoadPicture = True
            else:
                # update image loading stare
                for config in self.buttonsAPP_config:
                    if config["name"] == self.FLAG_AI_Inference:
                        self.IsLoadPicture = config.get("image_loading", False)
                        break

        # update button styles
        self.update_styles_app_button()
        self.update_styles_menu_button()
    def update_info_cpu(self, context):
        if (len(self.CPU_percentage_list)==100):
            self.CPU_percentage_list.pop(0)
            self.CPU_percentage_list.append(psutil.cpu_percent(interval=None))
            #DrawObjLabelText_cairo(context, width - 280 , 70, (0,40,255), ("cpu(%):" + str((sum(self.CPU_percentage_list) / len(self.CPU_percentage_list))))[:9], self.CairoContextTextSize_1)
        else :
            #DrawObjLabelText_cairo(context, width - 280 , 70, (0,40,255), ("cpu(%):" + str(psutil.cpu_percent(interval=None)))[:9], self.CairoContextTextSize_1)
            self.CPU_percentage_list.append(psutil.cpu_percent(interval=None))
    def update_info_ips(self, context, position):
        if (self.drawout_details != ""):
            IPS = 1 / (self.drawout_details[2])
            IPS = int(IPS)
            # calcuate IPS
            if (len(self.IPS_list) ==30):
                self.IPS_list.pop(0)
                self.IPS_list.append(IPS)
                IPS_average_30  = str(round(sum(self.IPS_list) / len(self.IPS_list),0))
                DrawObjLabelText_cairo(context, position[0]  , position[1], (0,40,255), "IPS:" + IPS_average_30[:5], self.CairoContextTextSize_1)
            else : 
                DrawObjLabelText_cairo(context, position[0]  , position[1], (0,40,255), "IPS:" + "N/A", self.CairoContextTextSize_1)
                self.IPS_list.append(IPS)
    def open_settingVeriSiliconISP(self, unused):
        if self.settingISP_WINDOWS == True :
            self.settingISP_WINDOWS = False
            self.settingISP.hide()# [WPI] self.settingISP.quit_button.emit("clicked")
        else:
            self.settingISP.set_keep_above(True)
            self.settingISP.set_position(Gtk.WindowPosition.CENTER_ALWAYS)
            self.settingISP.set_gravity(Gdk.Gravity.NORTH_WEST)
            self.settingISP_WINDOWS = True
            GLib.idle_add(self.settingISP.show_all)
    def open_debugMsg(self, unused):
        if self.debugMsg_WINDOWS == True :
            self.debugMsg_WINDOWS = False
            self.debugMsg.hide()
        else:
            self.debugMsg.set_keep_above(True)
            self.debugMsg.set_position(Gtk.WindowPosition.CENTER_ALWAYS)
            self.debugMsg.set_gravity(Gdk.Gravity.NORTH_WEST)
            self.debugMsg_WINDOWS = True
            GLib.idle_add(self.debugMsg.show_all)
    def open_saveImg(self, unused):
        if self.SaveImageSignal == True :
            self.SaveImageSignal = False
            self.generalFunc.hide()
        else:
            self.generalFunc.set_keep_above(True)
            self.generalFunc.set_position(Gtk.WindowPosition.CENTER_ALWAYS)
            self.generalFunc.set_gravity(Gdk.Gravity.NORTH_WEST)
            self.SaveImageSignal = True
            GLib.idle_add(self.generalFunc.show_all)
    def specialFunc_Change(self, unused):
        if getExtraLoadingPictue() == "True" :
            self.Extra_loadingPictureSignal  = False
        else :
            self.Extra_loadingPictureSignal = True
        writeExtraLoadingPictue(str(self.Extra_loadingPictureSignal))
    def autoRunFunc_Change(self, unused):
        if self.AutoRunFunc == True :
            self.AutoRunFunc = False
        else :
            self.AutoRunFunc = True
            self.IsLoadPicture = False
            self.FLAG_AI_COUNTER = 0
        print("autoRunFunc_Change")
    def on_touch_event(self,widget, event): # [WPI] for touch's postion
        if (self.touch_postionX != event.x or self.touch_postionY != event.y  ): 
            self.touch_signal = True
            self.touch_postionX = event.x - WINDOWS_TOUCH_FIX_X
            self.touch_postionY = event.y - WINDOWS_TOUCH_FIX_Y
    def draw_text_camera_dual(self, context):
        if (self.FLAG_AI_Inference == "dual-objectdetect_mobinetssdv2"):
            DrawObjRectFull_cairo(context, 10, 10, 190, 70, (0,0,0), 2)
            DrawObjRectFull_cairo(context, int(width/2) + 10, 10, 280, 70, (0,0,0), 2)
            DrawObjLabelText_cairo(context, 15 , 70, (255,255,255), "Python", self.CairoContextTextSize_1)
            DrawObjLabelText_cairo(context, int(width/2) + 20, 70, (255,255,255), "Gstreamer", self.CairoContextTextSize_1)
    def load_picture_mode(self):
        """
        Safely loads picture mode data and handles exceptions to prevent crashes.
        """
        # Define the list of modes that support picture loading
        flags_picture_mode = [
            "ADAS_mobilnetssdv2", "age_gender_recognition", "objectdetect_YOLOv5s","objectdetect_YOLOv8n", "facemesh",
            "maskdetect_YOLOv5s", "hardhatdetect_YOLOv5s", "palmlandmark", "posedetect_mobilenetssd",
            "posedetect_YOLOv8s", "PCBdetect_YOLOv8s", "fruitdetect_YOLOv5s", "objectdetect_mobinetssdv2",
            "facedetect_mobilenetssdv2"
        ]
        # Check if the current mode is picture mode
        if self.IsLoadPicture and self.FLAG_AI_Inference in flags_picture_mode:
            try:
                # Ensure drawout_details is initialized and has data
                if not self.drawout_details or len(self.drawout_details) == 0:
                    raise ValueError("drawout_details is empty or not initialized.")

                # Extract data from drawout_details
                data = self.drawout_details[0]
                bytebuffer = data.tobytes()
                
                # Check if bytebuffer is valid
                if not bytebuffer:
                    raise ValueError("Bytebuffer is empty or invalid.")

                # Create Gst.Buffer
                self.buffer_1 = Gst.Buffer.new_wrapped_full(
                    Gst.MemoryFlags.READONLY, bytebuffer, len(bytebuffer), 0, None, None
                )

                # [Optional: Add additional logic for buffer handling here]
                #print("Picture mode loaded successfully.")

            except ValueError as e:
                print(f"ValueError encountered: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    def load_special_cases(self):
        special_cases = {
             "segmentation_YOLOv5s": "RGB888",
            "depthestimation_MegaDepth": "RGB565",
            "benchmark": "RGB565",
            "contact": "RGB565"
        }

        if self.FLAG_AI_Inference in special_cases:
            try:
                data_format = special_cases[self.FLAG_AI_Inference]
                data = self.drawout_details[0]
                bytebuffer = data.tobytes()
                self.buffer_1 = Gst.Buffer.new_wrapped_full(Gst.MemoryFlags.READONLY, bytebuffer, len(bytebuffer), 0, None, None)
            except ValueError as e:
                print(f"ValueError encountered: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    
    # --------------------------------------
    # Gstreamer API by carsh fix  
    # --------------------------------------
    def check_draw_cb_activity(self):
        """
        Re-Gstreamer Pipline if check_draw_cb_activity isnt running in 5 second
        """
        current_time = time.monotonic()
        if current_time - self.last_draw_cb_time > 5: 
            logging.warning("draw_cb has not been called for more than 5 seconds. Restarting pipeline.")
            # Stop NPU
            subprocess.run(["echo", "stop", ">", "/sys/class/remoteproc/remoteproc0/state"],shell=True, check=True,);time.sleep(1)
            
            # Stop the pipeline
            if self.pipeline: 
                self.pipeline.set_state(Gst.State.NULL)

                # Ensure buffers and elements are cleared
                for element_name in ["sink1", "sink2", "sink3"]:
                    element = self.pipeline.get_by_name(element_name)
                    if element:
                        element.set_property("emit-signals", False)

                # Clear buffer references
                self.BufferRGB565_1 = None
                self.BufferRGB565_2 = None
            
            # Restart USB Camera
            if 0 :
                subprocess.run(["sudo", "modprobe", "-r", "uvcvideo"], check=True);time.sleep(1)
                subprocess.run(["sudo", "modprobe", "uvcvideo"], check=True);time.sleep(1)

            # Restart the script
            gc.collect()
            Gtk.main_quit() 
            time.sleep(5)
            print("check_draw_cb_activity is timeout")
            logging.warning("draw_cb has not been called for more than 5 seconds. Restarting pipeline.")

            # restart
            script_name = os.path.basename(sys.argv[0]) 
            if script_name.endswith(".py"):
                os.execv(sys.executable, [sys.executable, script_name] + sys.argv[1:])   
            else:
                os.execv(script_name, [script_name] + sys.argv[1:])
        return True  

# --------------------------------------
# GUI Windows   
# --------------------------------------
class SaveLoadImageWindow(Gtk.Window):
    def __init__(self, LoadingImagePath, Inference_Model ):
        """Create the UI to selct camera."""
        super().__init__()

        # Main layout
        side_grid    = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5) 
        side_grid.set_margin_start(5)
        side_grid.set_margin_end(5)
        side_grid.set_margin_top(5)
        side_grid.set_margin_bottom(5)

        LoadImage_grid= Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        LoadImage_btn  = Gtk.Button.new_with_label(" Load Image")
        LoadImage_btn.connect("clicked", self.image_load_signal)

        SaveOgnal_grid = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        SaveOgnal_btn  = Gtk.Button.new_with_label(" Save Image (Orignal)")
        SaveOgnal_btn.connect("clicked", self.image_save_signal)
        SaveResult_grid= Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        SaveResult_btn = Gtk.Button.new_with_label(" Save Image (Result) ")
        SaveResult_btn.connect("clicked", self.result_save_signal)

        side_grid.pack_start(LoadImage_grid , True, True, 0)
        side_grid.pack_start(SaveOgnal_grid , True, True, 0)
        side_grid.pack_start(SaveResult_grid, True, True, 0)
        LoadImage_grid.pack_start(LoadImage_btn , True, True, 0)
        SaveOgnal_grid.pack_start(SaveOgnal_btn , True, True, 0)
        SaveResult_grid.pack_start(SaveResult_btn, True, True, 0)

        # Setting GUI
        self.add(side_grid)
        self.set_title("Save Image")
        self.frame = ""
        self.result= ""
        self.Num   = 0
        self.selectFilePath = ""
        self.set_position(Gtk.WindowPosition.CENTER)  # Center the window on the screen
        self.set_default_size(400, 300)              # Optional: Set a default window size

        # output param
        self.IsLoadPicture = False
        self.LoadingImagePath = ""
        self.FLAG_AI_INFER_MODEL = Inference_Model

    def on_header_button_press(self, widget, event):
        """Handle button press event to start dragging."""
        self.begin_move_drag(event.button, event.x_root, event.y_root, event.time)

    def on_header_motion_notify(self, widget, event):
        """Handle motion event while dragging."""
        pass  # Optional: Add custom behavior during dragging
        
    def update_Image_info(self,frame, Num, Loading_ImagePath, Inference_Model):
        self.frame  = frame
        self.Num    = Num
        self.LoadingImagePath    = Loading_ImagePath
        self.FLAG_AI_INFER_MODEL = Inference_Model
    def update_Result_info(self,surface, Num, Inference_Model):
        self.result = surface
        self.Num    = Num
        self.FLAG_AI_INFER_MODEL = Inference_Model
    def image_save_signal(self, unused):
        cv2.imwrite( "../data/save/img_" + str(self.Num) +".jpg",self.frame)
        print("It's saved orignal image No." + str(self.Num))
    def image_load_signal(self, button):
        dialog = Gtk.FileChooserDialog("Please choose a file", None,
            Gtk.FileChooserAction.OPEN,(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        response = dialog.run()
        self.IsLoadPicture = True
        if response == Gtk.ResponseType.OK:
            self.selectFilePath = dialog.get_filename()
            self.LoadingImagePath[self.FLAG_AI_INFER_MODEL] = self.selectFilePath
            print("Selected file: ", self.selectFilePath)
            print("Inference Model" , self.FLAG_AI_INFER_MODEL)
            print("Inference Model" , self.LoadingImagePath[self.FLAG_AI_INFER_MODEL])
        else:
            print("File selection cancelled.")
            
        dialog.destroy()
    def result_save_signal(self, unused):
        self.result.write_to_png( "../data/save/result_" + str(self.Num) +".png")
        print("It's saved result image No." + str(self.Num))

class MsgWindow(Gtk.Window):
    def __init__(self):
        """Create the UI to selct camera."""
        super().__init__()
        Msg_main_grid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_title("Msgbox")
        self.add(Msg_main_grid)
        self.set_default_size(800, 200)
        self.debug_buffer = Gtk.TextBuffer.new(None)
        self.debug_buffer.insert_at_cursor("Loading")
        self.debug_output = Gtk.TextView.new_with_buffer(self.debug_buffer)
        self.debug_output.set_wrap_mode(1)
        self.debug_output.set_editable(False)
        self.debug_output.set_cursor_visible(False)
        self.debug_window = Gtk.ScrolledWindow.new()
        self.debug_window.add(self.debug_output)
        self.debug_window.set_size_request(100, 100)
        self.debug_window.set_visible(True)
        self.set_keep_above(True)
        self.set_position(Gtk.WindowPosition.CENTER_ALWAYS)
        self.set_gravity(Gdk.Gravity.NORTH_WEST)
        Msg_main_grid.pack_start(self.debug_window, True, True, 0)
    def change_msg(self, command):
        """Send a command to the ISP."""
        self.debug_buffer.insert_at_cursor("\n" + command)
        self.debug_output.scroll_to_mark(self.debug_buffer.get_insert(), 0.0, True, 0.5, 0.5)

class StartWindow(Gtk.Window):
    """A window that lets a user select the camera."""
    def __init__(self):
        """Create the UI to selct camera."""

        # Parser
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = argparse.ArgumentParser()
        parser.add_argument( '-p' ,"--platform", default="1", help=" 0 : IMX8MPLUSEVK , 1 : IMX93EVK,  2 : IMX95EVK")
        parser.add_argument( '-c' ,"--camera", default="0", help=" /dev/video* ")
        parser.add_argument( '-cf' ,"--camera_format", default=" video/x-raw,width=1920,heigt=1080,framerate=30/1 !  ", help=" Input Camera foramt [YUV/RGB/MJPEG] ")
        parser.add_argument( '-cd' ,"--camera_dual", default="0", help=" Enable Dual-Camera Display ")
        parser.add_argument( '-ws' ,"--windows_set", default="960, 540, 1200, 720, ../data/logo/ATU_WPI_Logo_WXGA.png , ../data/style_WXGA.css", help=" Setting Windows Size ")
        parser.add_argument( '-wm' ,"--windows_maximum", default="1", help=" Setting Windows Maximum ")
        args = parser.parse_args()

        """
        /dev/video*
           OP-Gyro(i.MX93) default is Video0 (MIPI or USB)
           OP-Killer(i.MX8MP) default is Video2 (MIPI), VIdeo3 (USB)
        format
          video/x-raw,width=1920,heigt=1080,framerate=30/1 ! 
          video/x-raw,width=1280,height=720,framerate=30/1 !  
          image/jpeg, width=1280,height=720, framerate=30/1 ! jpegdec ! videoconvert !    

        windows setting [FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY, WINDOWS_MAIN_WIDTH, WINDOWS_MAIN_HEIGHT, IMAGE_LOGO_PATH", GTK_STYLE_CSS_PATH]
            * HDMI
                1472, 828, 1920, 1000, ../data/logo/WPI_NXP_Logo_300x48.png, ../data/style.css
            * WXGA
                960, 540, 1200, 720, ../data/logo/ATU_WPI_Logo_WXGA.png, ../data/style_WXGA.css   
        """

        # Init
        super().__init__()
        self.set_default_size(WINDOWS_START_WIDTH, WINDOWS_START_HEIGHT)
        self.set_resizable(False)

        # Header
        header = Gtk.HeaderBar()
        header.set_title("AI Camera Demo")
        if (int(args.platform)==0) : header.set_subtitle("i.MX8M Plus Demos")
        if (int(args.platform)==1) : header.set_subtitle("i.MX93 Demos")
        self.set_titlebar(header)

        quit_button = Gtk.Button()
        quit_icon   = Gio.ThemedIcon(name="application-exit-symbolic")
        quit_image  = Gtk.Image.new_from_gicon(quit_icon, Gtk.IconSize.BUTTON)
        quit_button.add(quit_image)
        header.pack_end(quit_button)
        quit_button.connect("clicked", Gtk.main_quit)

        # Label
        platform_label = Gtk.Label.new("Platform: ")
        vid_label      = Gtk.Label.new("Video device: ")
        height_label   = Gtk.Label.new("Height: ")
        width_label    = Gtk.Label.new("Width: ")

        #platform
        platform = [ "IMX8MPLUSEVK", "IMX93EVK" , "IMX95EVK"  ]
        self.platform_select = Gtk.ComboBoxText()
        self.platform_select.set_entry_text_column(0)
        self.platform_select.set_hexpand(True)
        for option in platform:
            self.platform_select.append_text(option)
        self.platform_select.set_active(int(args.platform))

        # Camera
        devices = []
        devices_all = ['/dev/video0', '/dev/video1', '/dev/video2', '/dev/video3', '/dev/video4']
        devices_all.reverse()
        
        for device in devices_all :
            devices.append(device)
        self.source_select = Gtk.ComboBoxText()
        self.source_select.set_entry_text_column(0)
        self.source_select.set_hexpand(True)
        for option in devices:
            self.source_select.append_text(option)

        if not args.camera.isdigit():  
            self.selected_camera = args.camera  
        else:
            device_name = f"/dev/video{args.camera}"
            if device_name in devices:
                index = devices.index(device_name)
                self.source_select.set_active(index)
                self.selected_camera = self.source_select.get_active_text()
            else:
                self.selected_camera = device_name

        # Dual-CAM
        self.camera_dual_enable = int(args.camera_dual)

        # Camera format
        self.camera_format = args.camera_format

        # Camera[HxW]
        self.height_spin = Gtk.SpinButton.new_with_range(0, 1080, 10)
        self.height_spin.set_value(FRAME_HEIGHT)
        self.width_spin = Gtk.SpinButton.new_with_range(0, 1920, 10)
        self.width_spin.set_value(FRAME_WIDTH)

        # Windows 
        self.windows_maximum = args.windows_maximum
        self.windows_params = [param.strip() for param in args.windows_set.split(",")]

        # Button
        self.button = Gtk.Button.new_with_label("Start")
        self.button.connect("clicked", self.start)
        self.button.emit("clicked") #[WPI] Auto Trigger
        self.width_spin.set_sensitive(False)
        self.height_spin.set_sensitive(False)

        # Grid
        grid = Gtk.Grid.new()
        grid.attach(platform_label, 0, 0, 1, 1) 
        grid.attach(self.platform_select, 1, 0, 1, 1)
        grid.attach(vid_label, 0, 1, 1, 1)
        grid.attach(self.source_select, 1, 1, 1, 1)
        grid.attach(height_label, 0, 2, 1, 1)
        grid.attach(self.height_spin, 1, 2, 1, 1)
        grid.attach(width_label, 0, 3, 1, 1)
        grid.attach(self.width_spin, 1, 3, 1, 1)
        grid.attach(self.button, 0, 4, 2, 1)
        grid.props.margin = 30
        grid.set_column_spacing(30)
        grid.set_row_spacing(30)

        self.add(grid)

    def start(self, unused):
        """Start the video feed"""
        global VIDEO
        global PLATFORM
        global FRAME_WIDTH
        global FRAME_HEIGHT
        global CAMERA_DUAL_MIPI
        global CAMERA_FORMAT
        global WINDOWS_MAXIMUM
        global FRAME_WIDTH_DISPLAY
        global FRAME_HEIGHT_DISPLAY
        global WINDOWS_MAIN_WIDTH
        global WINDOWS_MAIN_HEIGHT
        global IMAGE_LOGO_PATH
        global GTK_STYLE_CSS_PATH

        VIDEO            = self.selected_camera
        PLATFORM         = self.platform_select.get_active_text()
        CAMERA_DUAL_MIPI = self.camera_dual_enable
        CAMERA_FORMAT    = self.camera_format
        WINDOWS_MAXIMUM  = self.windows_maximum
        
        FRAME_WIDTH_DISPLAY  = int(self.windows_params[0])
        FRAME_HEIGHT_DISPLAY = int(self.windows_params[1])
        WINDOWS_MAIN_WIDTH   = int(self.windows_params[2]) 
        WINDOWS_MAIN_HEIGHT  = int(self.windows_params[3])
        IMAGE_LOGO_PATH      = self.windows_params[4]
        GTK_STYLE_CSS_PATH   = self.windows_params[5]
        
        self.button.set_sensitive(False)
        self.width_spin.set_sensitive(False)
        self.height_spin.set_sensitive(False)
        GLib.idle_add(self.launch)

    def launch(self):
        """Launch demo"""
        window = MLVideoDemo(self.source_select.get_active_text())
        window.show_all()
        self.close()

if __name__ == "__main__":
    # 設定日誌檔案
    logging.basicConfig(
        filename='app_debug.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # 初始化 GStreamer
        logging.info("Initializing GStreamer...")
        Gst.init(None)

        # 顯示主視窗
        logging.info("Launching StartWindow...")
        window = StartWindow()
        window.show_all()

        # 啟動 GTK 主迴圈
        logging.info("Running GTK main loop...")
        Gtk.main()

    except Exception as e:
        # 捕捉未預期錯誤並記錄詳細訊息
        logging.error("An unexpected error occurred", exc_info=True)

    finally:
        # 程式執行完成
        logging.info("Application has terminated.")
