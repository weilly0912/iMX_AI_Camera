
import gi
import os
gi.require_version("Gtk", "3.0")
gi.require_version("Gst", "1.0")
from gi.repository import Gtk, Gst, Gio, GLib, Gdk

COLOR_DEFAULE_VALUE = 168
GUI_LABEL_SPACE = 10

class VeriSilicon_ISP_SettingsWindow(Gtk.Window):
    """A class that contains the UI and camera elements."""

    def __init__(self, CAM):
        """Create the UI for settings."""
        super().__init__()
        self.CamearDevice = CAM
        self.set_default_size(400, 150)
        self.move(100,100)
        self.set_position(Gtk.WindowPosition.CENTER_ON_PARENT)
        self.set_gravity(Gdk.Gravity.NORTH_WEST)#
        self.set_resizable(True)

        header = Gtk.HeaderBar()
        header.set_title("Settings")
        header.set_subtitle("AI Camera Demo")
        self.set_titlebar(header)

        quit_button = Gtk.Button()
        quit_icon = Gio.ThemedIcon(name="application-exit-symbolic")
        quit_image = Gtk.Image.new_from_gicon(quit_icon, Gtk.IconSize.BUTTON)
        quit_button.add(quit_image)
        header.pack_end(quit_button)
        quit_button.connect("clicked", self.close_window)

        # -----------------------------------------------------------------------
        # ISP
        # -----------------------------------------------------------------------
        ISP_main_grid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add(ISP_main_grid)
        self.content_stack = Gtk.Stack()

        options = [
            "Black Level Subtraction",
            "Dewarp and FPS",
            "Auto White Balance",
            "Color Processing",
            "Demosaicing and Gamma Control",
            "Filtering"
        ]

        options_combo = Gtk.ComboBoxText()
        options_combo.set_entry_text_column(0)
        for option in options:
            options_combo.append_text(option)
        options_combo.set_active(0)
        options_combo.set_size_request(150,40)
        options_combo.get_style_context().add_class("OptionsBox-ISP")
        options_combo.connect('changed', self.on_options_change)

        self.debug_buffer = Gtk.TextBuffer.new(None)
        self.debug_output = Gtk.TextView.new_with_buffer(self.debug_buffer)
        self.debug_output.set_wrap_mode(1)
        self.debug_output.set_editable(False)
        self.debug_output.set_cursor_visible(False)
        self.debug_window = Gtk.ScrolledWindow.new()
        self.debug_window.add(self.debug_output)
        self.debug_window.set_size_request(100, 100)
        self.debug_window.set_visible(False)

        ISP_main_grid.pack_start(options_combo, True, True, 0)
        ISP_main_grid.pack_start(self.content_stack, True, True, 0)
        ISP_main_grid.set_vexpand(True)
        
        #-----------------------------------------------------------
        # BLS
        #-----------------------------------------------------------
        r_label = Gtk.Label.new("Red")
        r_label.set_halign(1)
        r_label.set_margin_start(GUI_LABEL_SPACE)
        gr_label = Gtk.Label.new("Green.R")
        gr_label.set_halign(1)
        gr_label.set_margin_start(GUI_LABEL_SPACE)
        gb_label = Gtk.Label.new("Green.B")
        gb_label.set_halign(1)
        gb_label.set_margin_start(GUI_LABEL_SPACE)
        b_label = Gtk.Label.new("Blue")
        b_label.set_halign(1)
        b_label.set_margin_start(GUI_LABEL_SPACE)
        self.r_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 255, 1)
        self.r_scale.set_value(COLOR_DEFAULE_VALUE)
        self.r_scale.connect('value-changed', self.on_change_bls)
        self.r_scale.set_hexpand(True)
        self.gr_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 255, 1)
        self.gr_scale.set_value(COLOR_DEFAULE_VALUE)
        self.gr_scale.connect('value-changed', self.on_change_bls)
        self.gr_scale.set_hexpand(True)
        self.gb_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 255, 1)
        self.gb_scale.set_value(COLOR_DEFAULE_VALUE)
        self.gb_scale.connect('value-changed', self.on_change_bls)
        self.gb_scale.set_hexpand(True)
        self.b_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 255, 1)
        self.b_scale.set_value(COLOR_DEFAULE_VALUE)
        self.b_scale.connect('value-changed', self.on_change_bls)
        self.b_scale.set_hexpand(True)

        # BLS Layout
        bls_grid = Gtk.Grid()
        bls_grid.attach(r_label, 0, 1, 1, 1)
        bls_grid.attach(gr_label, 0, 2, 1, 1)
        bls_grid.attach(gb_label, 0, 3, 1, 1)
        bls_grid.attach(b_label, 0, 4, 1, 1)
        bls_grid.attach(self.r_scale, 1, 1, 1, 1)
        bls_grid.attach(self.gr_scale, 1, 2, 1, 1)
        bls_grid.attach(self.gb_scale, 1, 3, 1, 1)
        bls_grid.attach(self.b_scale, 1, 4, 1, 1)
        self.content_stack.add_named(bls_grid, "Black Level Subtraction")

        #-----------------------------------------------------------
        # Dewarp
        #-----------------------------------------------------------
        dewarp_grid = Gtk.Grid(row_homogeneous=True, column_spacing=15, row_spacing=15)
        dewarp_grid.set_margin_end(GUI_LABEL_SPACE)
        dewarp_grid.set_margin_start(GUI_LABEL_SPACE)
        dwe_label = Gtk.Label.new("Enable Dewarp")
        dwe_label.set_halign(1)
        self.dwe_switch = Gtk.Switch()
        self.dwe_dropdown = Gtk.ComboBoxText()
        self.dwe_switch.connect("notify::active", self.on_change_dwe)
        self.dwe_switch.set_active(True)
        self.dwe_switch.set_halign(1)
        self.dwe_switch.set_valign(3)
        dwe_mode_label = Gtk.Label.new("Dewarp Mode")
        dwe_mode_label.set_halign(1)
        self.dwe_options = [{"name": "Lens Distortion","id": 1},
                            {"name": "Fisheye Expand", "id": 2},
                            {"name": "Split Screen","id": 4},
                            {"name": "Fisheye Dewarp","id": 8}]
        self.dwe_dropdown.set_entry_text_column(0)
        for option in self.dwe_options:
            self.dwe_dropdown.append_text(option["name"])
        self.dwe_dropdown.set_active(0)
        self.dwe_dropdown.connect('changed', self.on_change_dwe_mode)
        self.dwe_dropdown.set_hexpand(True)
        vflip_label = Gtk.Label.new("Vertical Flip")
        vflip_label.set_halign(1)
        self.vflip_switch = Gtk.Switch()
        self.vflip_switch.set_active(False)
        self.vflip_switch.connect("notify::active", self.on_change_vflip)
        self.vflip_switch.set_halign(1)
        self.vflip_switch.set_valign(3)
        hflip_label = Gtk.Label.new("Horizontal Flip")
        hflip_label.set_halign(1)
        self.hflip_switch = Gtk.Switch()
        self.hflip_switch.connect("notify::active", self.on_change_hflip)
        self.hflip_switch.set_active(False)
        self.hflip_switch.set_halign(1)
        self.hflip_switch.set_valign(3)
        fps_label = Gtk.Label.new("FPS Max")
        fps_label.set_halign(1)
        self.fps_scale = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 1, 60, 1)
        self.fps_scale.set_value(COLOR_DEFAULE_VALUE)
        self.fps_scale.connect('value-changed', self.on_change_fps)
        self.fps_scale.set_hexpand(True)

        # Dewarp Layout
        dewarp_grid.attach(dwe_label, 0, 1, 1, 1)
        dewarp_grid.attach(dwe_mode_label, 0, 2, 1, 1)
        dewarp_grid.attach(vflip_label, 0, 3, 1, 1)
        dewarp_grid.attach(hflip_label, 0, 4, 1, 1)
        dewarp_grid.attach(fps_label, 0, 5, 1, 1)
        dewarp_grid.attach(self.dwe_switch, 1, 1, 1, 1)
        dewarp_grid.attach(self.dwe_dropdown, 1, 2, 1, 1)
        dewarp_grid.attach(self.vflip_switch, 1, 3, 1, 1)
        dewarp_grid.attach(self.hflip_switch, 1, 4, 1, 1)
        dewarp_grid.attach(self.fps_scale, 1, 5, 1, 1)
        self.content_stack.add_named(dewarp_grid, "Dewarp and FPS")

        #-----------------------------------------------------------
        # AWB
        #-----------------------------------------------------------
        awb_grid = Gtk.Grid(row_homogeneous=True,column_spacing=15,row_spacing=15)
        awb_grid.set_margin_end(10)
        awb_grid.set_margin_start(10)
        awb_enable = Gtk.Label.new("Enable AWB")
        awb_enable.set_halign(1)
        awb_r_label = Gtk.Label.new("Red")
        awb_r_label.set_halign(1)
        awb_gr_label = Gtk.Label.new("Green.R")
        awb_gr_label.set_halign(1)
        awb_gb_label = Gtk.Label.new("Green.B")
        awb_gb_label.set_halign(1)
        awb_b_label = Gtk.Label.new("Blue")
        awb_b_label.set_halign(1)
        self.awb_switch = Gtk.Switch()
        self.awb_switch.set_active(True)
        self.awb_switch.connect("notify::active", self.on_change_awb)
        self.awb_switch.set_halign(1)
        self.awb_switch.set_valign(3)
        self.awb_r_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.2, 3.9, 0.1)
        self.awb_r_scale.set_value(1.0)
        self.awb_r_scale.connect('value-changed', self.on_change_awb_set)
        self.awb_r_scale.set_hexpand(True)
        self.awb_r_scale.set_sensitive(False)
        self.awb_gr_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.2, 3.9, 0.1)
        self.awb_gr_scale.set_value(1.0)
        self.awb_gr_scale.connect('value-changed', self.on_change_awb_set)
        self.awb_gr_scale.set_hexpand(True)
        self.awb_gr_scale.set_sensitive(False)
        self.awb_gb_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.2, 3.9, 0.1)
        self.awb_gb_scale.set_value(1.0)
        self.awb_gb_scale.connect('value-changed', self.on_change_awb_set)
        self.awb_gb_scale.set_hexpand(True)
        self.awb_gb_scale.set_sensitive(False)
        self.awb_b_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.2, 3.9, 0.1)
        self.awb_b_scale.set_value(1.0)
        self.awb_b_scale.connect('value-changed', self.on_change_awb_set)
        self.awb_b_scale.set_hexpand(True)
        self.awb_b_scale.set_sensitive(False)

        # AWB Layout
        awb_grid.attach(awb_enable, 0, 1, 1, 1)
        awb_grid.attach(awb_r_label, 0, 2, 1, 1)
        awb_grid.attach(awb_gr_label, 0, 3, 1, 1)
        awb_grid.attach(awb_gb_label, 0, 4, 1, 1)
        awb_grid.attach(awb_b_label, 0, 5, 1, 1)
        awb_grid.attach(self.awb_switch, 1, 1, 1, 1)
        awb_grid.attach(self.awb_r_scale, 1, 2, 1, 1)
        awb_grid.attach(self.awb_gr_scale, 1, 3, 1, 1)
        awb_grid.attach(self.awb_gb_scale, 1, 4, 1, 1)
        awb_grid.attach(self.awb_b_scale, 1, 5, 1, 1)
        self.content_stack.add_named(awb_grid, "Auto White Balance")

        #-----------------------------------------------------------
        # CPROC
        #-----------------------------------------------------------
        cproc_grid = Gtk.Grid(row_homogeneous=True,column_spacing=15,row_spacing=15)
        cproc_grid.set_margin_end(10)
        cproc_grid.set_margin_start(10)
        cproc_enable = Gtk.Label.new("Enable CPROC")
        cproc_enable.set_halign(1)
        brightness_label = Gtk.Label.new("Brightness")
        brightness_label.set_halign(1)
        contrast_label = Gtk.Label.new("Contrast")
        contrast_label.set_halign(1)
        saturation_label = Gtk.Label.new("Saturation")
        saturation_label.set_halign(1)
        hue_label = Gtk.Label.new("Hue")
        hue_label.set_halign(1)
        self.cproc_switch = Gtk.Switch()
        self.cproc_switch.set_active(True)
        self.cproc_switch.connect("notify::active", self.on_change_cproc)
        self.cproc_switch.set_halign(1)
        self.cproc_switch.set_valign(3)
        self.brightness_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -127, 127, 1)
        self.brightness_scale.set_value(0)
        self.brightness_scale.connect('value-changed', self.on_change_cproc_set)
        self.brightness_scale.set_hexpand(True)
        self.contrast_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.00, 1.99, 0.01)
        self.contrast_scale.set_value(1.00)
        self.contrast_scale.connect('value-changed', self.on_change_cproc_set)
        self.contrast_scale.set_hexpand(True)
        self.saturation_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 1.99, 0.01)
        self.saturation_scale.set_value(1.00)
        self.saturation_scale.connect('value-changed', self.on_change_cproc_set)
        self.saturation_scale.set_hexpand(True)
        self.hue_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -127, 127, 1)
        self.hue_scale.set_value(0)
        self.hue_scale.connect('value-changed', self.on_change_cproc_set)
        self.hue_scale.set_hexpand(True)

        # CPROC Layout
        cproc_grid.attach(cproc_enable, 0, 1, 1, 1)
        cproc_grid.attach(brightness_label, 0, 2, 1, 1)
        cproc_grid.attach(contrast_label, 0, 3, 1, 1)
        cproc_grid.attach(saturation_label, 0, 4, 1, 1)
        cproc_grid.attach(hue_label, 0, 5, 1, 1)
        cproc_grid.attach(self.cproc_switch, 1, 1, 1, 1)
        cproc_grid.attach(self.brightness_scale, 1, 2, 1, 1)
        cproc_grid.attach(self.contrast_scale, 1, 3, 1, 1)
        cproc_grid.attach(self.saturation_scale, 1, 4, 1, 1)
        cproc_grid.attach(self.hue_scale, 1, 5, 1, 1)
        self.content_stack.add_named(cproc_grid, "Color Processing")

        #-----------------------------------------------------------
        # Demosaicing 
        #-----------------------------------------------------------
        demo_grid = Gtk.Grid(row_homogeneous=True,column_spacing=15,row_spacing=15)
        demo_grid.set_margin_end(10)
        demo_grid.set_margin_start(10)
        demo_enable = Gtk.Label.new("Enable Demosaicing")
        demo_enable.set_halign(1)
        demo_threshold = Gtk.Label.new("Threshold")
        demo_threshold.set_halign(1)
        gamma_enable = Gtk.Label.new("Enable Gamma")
        gamma_enable.set_halign(1)
        gamma_mode = Gtk.Label.new("Gamma Mode")
        gamma_mode.set_halign(1)
        self.demo_switch = Gtk.Switch()
        self.demo_switch.set_active(True)
        self.demo_switch.connect("notify::active", self.on_change_demo)
        self.demo_switch.set_halign(1)
        self.demo_switch.set_valign(3)
        self.demo_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 255, 1)
        self.demo_scale.set_value(64)
        self.demo_scale.connect('value-changed', self.on_change_demo)
        self.demo_scale.set_hexpand(True)
        self.gamma_switch = Gtk.Switch()
        self.gamma_switch.set_active(True)
        self.gamma_switch.connect("notify::active", self.on_change_gamma)
        self.gamma_switch.set_halign(1)
        self.gamma_switch.set_valign(3)
        gamma_options = ["Logarithmic", "Equidistant"]
        self.gamma_dropdown = Gtk.ComboBoxText()
        self.gamma_dropdown.set_entry_text_column(0)
        for option in gamma_options:
            self.gamma_dropdown.append_text(option)
        self.gamma_dropdown.set_active(0)
        self.gamma_dropdown.connect('changed', self.on_change_gamma_set)
        self.gamma_dropdown.set_hexpand(True)

        # Demosaicing Layout
        demo_grid.attach(demo_enable, 0, 1, 1, 1)
        demo_grid.attach(demo_threshold, 0, 2, 1, 1)
        demo_grid.attach(gamma_enable, 0, 3, 1, 1)
        demo_grid.attach(gamma_mode, 0, 4, 1, 1)
        demo_grid.attach(self.demo_switch, 1, 1, 1, 1)
        demo_grid.attach(self.demo_scale, 1, 2, 1, 1)
        demo_grid.attach(self.gamma_switch, 1, 3, 1, 1)
        demo_grid.attach(self.gamma_dropdown, 1, 4, 1, 1)

        self.content_stack.add_named(
            demo_grid, "Demosaicing and Gamma Control")

        #-----------------------------------------------------------
        # Filter 
        #-----------------------------------------------------------
        filter_grid = Gtk.Grid(row_homogeneous=True,column_spacing=15,row_spacing=15)
        filter_grid.set_margin_end(10)
        filter_grid.set_margin_start(10)
        filter_enable = Gtk.Label.new("Enable Filter")
        filter_enable.set_halign(1)
        denoise_label = Gtk.Label.new("Denoise")
        denoise_label.set_halign(1)
        sharpen_label = Gtk.Label.new("Sharpeness")
        sharpen_label.set_halign(1)
        self.filter_switch = Gtk.Switch()
        self.filter_switch.set_active(True)
        self.filter_switch.connect("notify::active", self.on_change_filter)
        self.filter_switch.set_halign(1)
        self.filter_switch.set_valign(3)
        self.denoise_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1, 10, 1)
        self.denoise_scale.set_value(1)
        self.denoise_scale.connect('value-changed', self.on_change_filter_set)
        self.denoise_scale.set_hexpand(True)
        self.sharpen_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1, 10, 1)
        self.sharpen_scale.set_value(3)
        self.sharpen_scale.connect('value-changed', self.on_change_filter_set)
        self.sharpen_scale.set_hexpand(True)

        # Filter Layout
        filter_grid.attach(filter_enable, 0, 1, 1, 1)
        filter_grid.attach(denoise_label, 0, 2, 1, 1)
        filter_grid.attach(sharpen_label, 0, 3, 1, 1)
        filter_grid.attach(self.filter_switch, 1, 1, 1, 1)
        filter_grid.attach(self.denoise_scale, 1, 2, 1, 1)
        filter_grid.attach(self.sharpen_scale, 1, 3, 1, 1)
        self.content_stack.add_named(filter_grid, "Filtering")

        # WINDOWS Setting
        self.set_size_request(400,200)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_keep_above(True)

    # -----------------------------------------------------------------------
    # ISP Event Handlers
    # -----------------------------------------------------------------------
    def save(self, unused):
        """Save selections"""
        self.close_window(None)
    
    def close_window(self, unused):
        self.hide()

    def on_change_bls(self, unused):
        """Set the BLS values (Red, Green.R/B, Blue)."""
        red = self.r_scale.get_value()
        green_r = self.gr_scale.get_value()
        green_b = self.gb_scale.get_value()
        blue = self.b_scale.get_value()
        self.change_isp(
            "<id>:<bls.s.cfg>;<blue>:" + str(int(blue)) + ";<green.b>:" +
            str(int(green_b)) + ";<green.r>:" + str(int(green_r)) +
            ";<red>:" + str(int(red)))

    def on_change_dwe(self, widget, unused):
        """Flip the camera output vertically."""
        if widget.get_active():
            self.dwe_dropdown.set_sensitive(True)
            self.change_isp("<id>:<dwe.s.bypass>; <dwe>:{<bypass>:false}")
        else:
            self.dwe_dropdown.set_sensitive(False)
            self.change_isp("<id>:<dwe.s.bypass>; <dwe>:{<bypass>:true}")
    
    def on_change_dwe_mode(self, widget):
        """Set the mode for dwe control."""
        mode = self.dwe_dropdown.get_active()
        print(mode)
        mode_num = self.dwe_options[mode]["id"]
        self.change_isp("<id>:<dwe.s.type>; <dwe>:{<type>:" + str(mode_num) + "}")

    def on_change_vflip(self, widget, unused):
        """Flip the camera output vertically."""
        if widget.get_active():
            self.change_isp("<id>:<dwe.s.vflip>; <dwe>:{<vflip>:true}")
        else:
            self.change_isp("<id>:<dwe.s.vflip>; <dwe>:{<vflip>:false}")

    def on_change_hflip(self, widget, unused):
        """Flip the camera output horizontally."""
        if widget.get_active():
            self.change_isp("<id>:<dwe.s.hflip>; <dwe>:{<hflip>:true}")
        else:
            self.change_isp("<id>:<dwe.s.hflip>; <dwe>:{<hflip>:false}")

    def on_change_fps(self, widget):
        """Change the output FPS."""
        self.change_isp("<id>:<s.fps>; <fps>:" + str(int(widget.get_value())))

    def on_change_awb(self, widget, unused):
        print(widget)
        """Enable or disable AWB."""
        if widget.get_active():
            self.change_isp("<id>:<awb.s.en>; <enable>:true")
            if self.awb_r_scale is not None:
                self.awb_r_scale.set_sensitive(False)
                self.awb_gr_scale.set_sensitive(False)
                self.awb_gb_scale.set_sensitive(False)
                self.awb_b_scale.set_sensitive(False)
        else:
            self.change_isp("<id>:<awb.s.en>; <enable>:false")
            self.awb_r_scale.set_sensitive(True)
            self.awb_gr_scale.set_sensitive(True)
            self.awb_gb_scale.set_sensitive(True)
            self.awb_b_scale.set_sensitive(True)
            self.on_change_awb_set(None)

    def on_change_cproc(self, widget, unused):
        """Enable or disable CPROC."""
        if widget.get_active():
            self.change_isp("<id>:<cproc.s.en>; <enable>:true")
            if self.brightness_scale is not None:
                self.brightness_scale.set_sensitive(True)
                self.contrast_scale.set_sensitive(True)
                self.saturation_scale.set_sensitive(True)
                self.hue_scale.set_sensitive(True)
        else:
            self.change_isp("<id>:<cproc.s.en>; <enable>:false")
            self.brightness_scale.set_sensitive(False)
            self.contrast_scale.set_sensitive(False)
            self.saturation_scale.set_sensitive(False)
            self.hue_scale.set_sensitive(False)
            self.on_change_cproc_set(None)

    def on_change_awb_set(self, unused):
        """Set the values for AWB."""
        red = self.awb_r_scale.get_value()
        green_r = self.awb_gr_scale.get_value()
        green_b = self.awb_gb_scale.get_value()
        blue = self.awb_b_scale.get_value()
        self.change_isp(
            "<id>:<awb.s.gain>;<red>:" + str(red) + ";<green.r>:" +
            str(green_r) + ";<green.b>:" + str(green_b) + ";<blue>:" +
            str(blue))

    def on_change_cproc_set(self, unused):
        """Set the values for CPROC."""
        bright = self.brightness_scale.get_value()
        cont = self.contrast_scale.get_value()
        sat = self.saturation_scale.get_value()
        hue = self.hue_scale.get_value()
        self.change_isp(
            "<id>:<cproc.s.cfg>;<brightness>:" + str(int(bright)) +
            ";<contrast>:" + str(cont) + ";<saturation>:" + str(sat) +
            ";<hue>:" + str(int(hue)) +
            ";<luma.in>:1;<luma.out>:1;<chroma.out>:1")

    def on_change_demo(self, unused, unused_2=None):
        """Enable or disable and set threshold for demosaicing."""
        if self.demo_switch.get_active():
            demo_en = "1"
        else:
            demo_en = "0"
        self.change_isp(
            "<id>:<dmsc.s.cfg>;<mode>:" + demo_en + ";<threshold>:" +
            str(int(self.demo_scale.get_value())))

    def on_change_gamma(self, widget, unused):
        """Enable or disable gamma control."""
        if widget.get_active():
            self.change_isp("<id>:<gc.s.en>; <enable>:true")
            if self.brightness_scale is not None:
                self.gamma_dropdown.set_sensitive(True)
        else:
            self.change_isp("<id>:<gc.s.en>; <enable>:false")
            self.gamma_dropdown.set_sensitive(False)

    def on_change_gamma_set(self, widght):
        """Set the mode for gamma control."""
        mode = widght.get_active_text()
        if mode == "Logarithmic":
            mode_num = "1"
        else:
            mode_num = "0"
        self.change_isp("<id>:<gc.s.cfg>; <gc.mode>:" + mode_num)

    def on_change_filter(self, widget, unused):
        """Enable or disable the filter."""
        if widget.get_active():
            self.change_isp("<id>:<filter.s.en>; <enable>:true")
            if self.brightness_scale is not None:
                self.denoise_scale.set_sensitive(True)
                self.sharpen_scale.set_sensitive(True)
        else:
            self.change_isp("<id>:<filter.s.en>; <enable>:false")
            self.denoise_scale.set_sensitive(False)
            self.sharpen_scale.set_sensitive(False)

    def on_change_filter_set(self, unused):
        """Set the sharpness and denoise values."""
        den = self.denoise_scale.get_value()
        sharp = self.sharpen_scale.get_value()
        self.change_isp(
            "<id>:<filter.s.cfg>;<auto>:false;<denoise>:" + str(int(den)) +
            ";<sharpen>:" + str(int(sharp)))

    def change_isp(self, command):
        """Send a command to the ISP."""
        os.system(
            "v4l2-ctl -d " + self.CamearDevice + " -c viv_ext_ctrl='{" + command + "}'")
        self.debug_buffer.insert_at_cursor("\n" + command)
        self.debug_output.scroll_to_mark(
            self.debug_buffer.get_insert(), 0.0, True, 0.5, 0.5)

    def on_options_change(self, widght):
        """Change the currently visable options."""
        self.content_stack.set_visible_child_name(widght.get_active_text())

    def on_debug(self, widget):
        """Shows and hides the debug output."""
        if self.debug_window.get_visible():
            self.debug_window.set_visible(False)
            window.set_size_request(440, 280)
        else:
            self.debug_window.set_visible(True)
