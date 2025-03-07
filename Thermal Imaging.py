from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys
import cv2
import numpy as np
import pyqtgraph as pg
import qdarktheme
import time
import os
import io
from datetime import datetime
# import win32com.client
from threading import Condition
import json
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import xlsxwriter

from seekcamera import (
    SeekCameraIOType,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekCameraPipelineMode,
    SeekCamera,
    SeekFrame,
)

try:
    from mcculw import ul
    from mcculw.enums import InfoType, BoardInfo, AiChanType, TcType, TempScale, TInOptions
    from mcculw.ul import ULError
    THERMOCOUPLE_AVAILABLE = True
except:
    print("Could not import mcculw")
    THERMOCOUPLE_AVAILABLE = False

# TODO: Try to make the SV1 pipeline local to the class
USE_SV1_PIPELINE = False


# ===================================================================================================
# ThermographyData
# ===================================================================================================

class ThermographyData():
    """Contains raw and processed thermography data, and methods to process and export the data."""

    def __init__(self, slope=1.0, intercept=0.0):
        self.temperature_range = (-40, 330) # (-40, 330) for the Seek Compact Pro

        self.sample_name = ''
        self.data_type = '' # 'image' or 'video'
        self.created = datetime.now().strftime("%Y%m%d-%H%M%S%f")

        self.time = []
        self.raw_frames = []
        self.average_frame = None

        self.thermocouple = []

        self.calibration_slope = slope
        self.calibration_intercept = intercept

    def update_average_frame(self):
        """Calculates the average frame from the raw frames."""
        for i, frame in enumerate(self.raw_frames):
            if i == 0:
                average_frame = frame
            else:
                alpha = 1.0/(i + 1)
                beta = 1.0 - alpha
                average_frame = cv2.addWeighted(frame, alpha, average_frame, beta, 0.0)
        self.average_frame = average_frame

    def calibrate_frame(self, frame):
        """Calibrates a frame using the calibration slope and intercept."""

        # For backwards compatibility, add slope, and intercept if not present
        if not hasattr(self, 'calibration_slope'):
            self.calibration_slope = 1.0
        if not hasattr(self, 'calibration_intercept'):
            self.calibration_intercept = 0.0

        return frame * self.calibration_slope + self.calibration_intercept

    def to_pkl(self, filename):
        """Serialises the data to pkl."""

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def to_png(self, filename=None, frame=None, include_color_bar=True, color_map='inferno', fixed_range=None, return_image=False):
        """Exports the data to png."""

        if filename is None:
            filename = f"{self.created} - image - {self.sample_name}.png"

        if frame is None:
            frame = self.average_frame

        if fixed_range is not None:
            vmin, vmax = fixed_range
        else:
            vmin, vmax = None, None

        frame = self.calibrate_frame(frame)

        frame_fig = plt.figure()
        frame_ax = frame_fig.add_subplot(111)
        frame_ax.pcolormesh(frame, cmap=color_map, vmin=vmin, vmax=vmax)
        frame_ax.set_aspect('equal')
        frame_ax.invert_yaxis()
        frame_ax.axis('off')
        if include_color_bar:
            frame_fig.colorbar(frame_ax.pcolormesh(frame, cmap=color_map, vmin=vmin, vmax=vmax), ax=frame_ax)
        if return_image:
            frame_fig.canvas.draw()
            frame_arr = np.array(frame_fig.canvas.renderer._renderer)
            plt.close(frame_fig)
            return frame_arr
        else:
            frame_fig.savefig(filename, dpi=600)
        
        plt.close(frame_fig)

    def to_avi(self, filename=None, include_color_bar=True, color_map='inferno', fixed_range=None):
        """Exports the data to avi."""

        if filename is None:
            filename = f"{self.created} - video - {self.sample_name}.avi"

        try:
            fps = self.fps
        except:
            fps = np.mean([1/(self.time[i+1] - self.time[i]) for i in range(len(self.time)-1)])

        w, h = self.raw_frames[0].shape[1], self.raw_frames[0].shape[0]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(filename, fourcc, fps, (2*w, 2*h))

        for frame in self.raw_frames:
            f = self.to_png(frame=frame, include_color_bar=include_color_bar, color_map=color_map, fixed_range=fixed_range, return_image=True)
            f = cv2.resize(f, (2*w, 2*h))
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            video.write(f)
        video.release()

    def export_data(self):
        """Exports the data."""

        filename_raw_frames = f"exported data/{self.created} - {self.data_type} - {self.sample_name} - raw frames.csv"
        filename_average_frame = f"exported data/{self.created} - {self.data_type} - {self.sample_name} - average frame.csv"
        filename_roi_data = f"exported data/{self.created} - {self.data_type} - {self.sample_name} - roi data.csv"

        with open(filename_raw_frames, 'w') as f:
            for frame in self.raw_frames:
                np.savetxt(f, frame, delimiter=",")
                f.write("\n")

        with open(filename_average_frame, 'w') as f:
            np.savetxt(f, self.average_frame, delimiter=",")
            f.write("\n")

        with open(filename_roi_data, 'w') as f:
            f.write("x, y, w, h\n")
            f.write(f"{self.roi_data['pos'][0], self.roi_data['pos'][1], self.roi_data['size'][0], self.roi_data['size'][1]}\n")
            f.write("##########")
            f.write("Time (s), Min (°C), Max (°C), Mean (°C), Center (°C)\n")
            for i, time in enumerate(self.time):
                f.write(f"{time}, {self.roi_data['min'][i]}, {self.roi_data['max'][i]}, {self.roi_data['mean'][i]}, {self.roi_data['center'][i]}\n")
        
class Renderer:
    """Contains camera and image data required to render images to the screen."""

    def __init__(self):
        self.busy = False
        self.frame = SeekFrame()
        self.camera = SeekCamera()
        self.frame_condition = Condition()
        self.first_frame = True

def seek_on_frame(camera, camera_frame, renderer):
    """Async callback fired whenever a new frame is available"""

    # Acquire the condition variable and notify the main thread
    # that a new frame is ready to render. This is required since
    # all rendering done by OpenCV needs to happen on the main thread.
    with renderer.frame_condition:
        renderer.frame = camera_frame.thermography_float
        renderer.frame_condition.notify()

    if USE_SV1_PIPELINE:
        camera.pipeline_mode = SeekCameraPipelineMode.SEEKVISION
    else:
        camera.pipeline_mode = SeekCameraPipelineMode.LITE

def seek_on_event(camera, event_type, event_status, renderer):
    """Async callback fired whenever a camera event occurs."""

    print("{}: {}".format(str(event_type), camera.chipid))

    if event_type == SeekCameraManagerEvent.CONNECT:
        if renderer.busy:
            return
        # Claim the renderer. This is required in case of multiple cameras.
        renderer.busy = True
        renderer.camera = camera
        # Indicate the first frame has not come in yet. This is required to properly resize the rendering window.
        renderer.first_frame = True
        # Start imaging and provide a custom callback to be called every time a new frame is received.
        camera.register_frame_available_callback(seek_on_frame, renderer)
        camera.capture_session_start(SeekCameraFrameFormat.THERMOGRAPHY_FLOAT)        

    elif event_type == SeekCameraManagerEvent.DISCONNECT:
        # Check that the camera disconnecting is one actually associated with the renderer. This is required in case of multiple cameras.
        if renderer.camera == camera:
            # Stop imaging and reset all the renderer state.
            camera.capture_session_stop()
            renderer.camera = None
            renderer.frame = None
            renderer.busy = False

    elif event_type == SeekCameraManagerEvent.ERROR:
        print("{}: {}".format(str(event_status), camera.chipid))

    elif event_type == SeekCameraManagerEvent.READY_TO_PAIR:
        return
    
# ===================================================================================================
# ThermalImagingApp
# ===================================================================================================
    
class ThermalImagingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thermal Imaging")

        app_icon = QIcon()
        app_icon.addFile('assets\\Logo.png', QSize(256, 256))
        self.setWindowIcon(app_icon)

        self.setGeometry(100, 100, 1200, 800)
        self.showMaximized()
        self.EnableMouseTracking = True

        self.init_parameters()
        self.init_ui()
        self.init_seek_camera()
        
        # Horrible hack to give the camera time to connect
        time.sleep(1)

        # Start the timer for refreshing the app
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(self.time_step)

    def init_parameters(self):

        # Initial parameters for display
        self.frame_index = 0
        
        # Initial parameters for ROI
        self.roi_time_0 = time.time()
        self.roi_data = {
            "pos": [160, 110],
            "size": [20, 20],
            "time": [],
            "max": [],
            "min": [],
            "mean": [],
            "center": []
        }

        self.current_loaded_data = ThermographyData()
        self.files_in_raw_data = []

        # Initial parameters for setup
        self.directory = os.path.dirname(os.path.realpath(__file__))
        # self.sh = win32com.client.gencache.EnsureDispatch('Shell.Application', 0)
        # self.ns = self.sh.NameSpace(self.directory)

        required_directories = ['thumbnails', 'raw data', 'exported data']

        for directory in required_directories:
            if not os.path.exists(f"{self.directory}/{directory}"):
                os.makedirs(f"{self.directory}/{directory}")

        self.recording = False
        
        self.calibration_mode = 'None'
        self.use_thermocouple = False
        self.thermocouple_temperature = 0.0
        self.calibration_temperature_1 = 30.0
        self.calibration_temperature_2 = 50.0
        self.measured_temperature_1 = 30.0
        self.measured_temperature_2 = 50.0
        self.time_step = 200
        self.temperature_unit = '°C'
        self.emmisivity = 0.95
        self.app_mode = 'Capture'

        self.temperature_range_mode = 'Auto'
        self.temperature_range = (15, 35)
        self.frame_mode = 'Individual Frames'
        self.color_maps = ['viridis', 'plasma', 'inferno', 'magma']
        self.color_map = 'inferno'
        self.mpl_cmap = mpl.colors.Colormap(self.color_map)
        self.pg_cmap = pg.colormap.getFromMatplotlib(self.color_map)

        self.sample_name = 'Sample 1'
        
        initial_setup_params = [
            {'name': 'Directory Selection', 'type': 'file', 'value': self.directory, 'fileMode': 'Directory'},
            # {'name': 'Time Step (ms)', 'type': 'int', 'value': self.time_step, 'limits':[150, 1000]},
            # {'name': 'Temperature Unit', 'type': 'list', 'values': ['°C', '°F', '°K'], 'value': self.temperature_unit},
            # {'name': 'Emmisivity', 'type': 'float', 'value': self.emmisivity, 'step': 0.01, 'limits': [0.0, 1.0]},
            {'name': 'Use SV1 Pipeline', 'type': 'bool', 'value': USE_SV1_PIPELINE},
            {'name': 'Use Thermocouple', 'type': 'bool', 'value': self.use_thermocouple, 'visible': THERMOCOUPLE_AVAILABLE},
            {'name': 'Thermocouple Temperature (°C)', 'type': 'float', 'value': self.thermocouple_temperature, 'visible': self.use_thermocouple, 'readonly': True},
            {'name': 'Calibration', 'type': 'group', 'children': [
                {'name': 'Calibration Mode', 'type': 'list', 'values': ['None', 'One Point', 'Two Point'], 'value': self.calibration_mode},
                {'name': 'Calibration Temperature 1 (°C)', 'type': 'float', 'value': self.calibration_temperature_1, 'limits': [0, 100], 'visible': False},
                {'name': 'Set Point 1', 'type': 'action', 'visible': False },
                {'name': 'Calibration Temperature 2 (°C)', 'type': 'float', 'value': self.calibration_temperature_2, 'limits': [0, 100], 'visible': False},
                {'name': 'Set Point 2', 'type': 'action', 'visible': False},
                {'name': 'Calibration Slope', 'type': 'float', 'value': self.current_loaded_data.calibration_slope, 'readonly': True},
                {'name': 'Calibration Intercept', 'type': 'float', 'value': self.current_loaded_data.calibration_intercept, 'readonly': True},
            ]
            },
            {'name': 'App Mode', 'type': 'str', 'value': self.app_mode, 'readonly': True},
            {'name': 'Reload Camera', 'type': 'action', },
            {'name': 'Return to Capture Mode', 'type': 'action', 'visible': False},
        ]

        initial_process_params = [
            {'name': 'Sample Details', 'type': 'group', 'children': [
                {'name': 'Time Stamp', 'type': 'str', 'value': self.current_loaded_data.created, 'readonly': True},
                {'name': 'File Type', 'type': 'str', 'value': self.current_loaded_data.data_type, 'readonly': True, 'visible': False},
                {'name': 'Sample Name', 'type': 'str', 'value': self.sample_name}
            ]
            },
            {'name': 'Capture Image', 'type': 'group', 'children': [
                {'name': 'Number of Repeats', 'type': 'int', 'value': 3},
                {'name': 'Images per Repeat', 'type': 'int', 'value': 3},
                {'name': 'Capture Image', 'type': 'action'}
            ]
            },
            {'name': 'Capture Video', 'type': 'group', 'children': [
                {'name': 'Duration (seconds)', 'type': 'int', 'value': 5},
                {'name': 'Frames per second', 'type': 'int', 'value': 5, 'limits': [1, 5]},
                {'name': 'Capture Video', 'type': 'action'},
                {'name': 'Stop Capture', 'type': 'action', 'visible': False}
            ]
            },
            {'name': 'Region of Interest', 'type': 'group', 'children': [
                {'name': 'Position', 'type': 'group', 'children': [
                    {'name': 'X', 'type': 'int', 'value': self.roi_data['pos'][0]},
                    {'name': 'Y', 'type': 'int', 'value': self.roi_data['pos'][1]}
                ]
                },
                {'name': 'Size', 'type': 'group', 'children': [
                    {'name': 'X', 'type': 'int', 'value': self.roi_data['size'][0]},
                    {'name': 'Y', 'type': 'int', 'value': self.roi_data['size'][1]}
                ]
                }
            ]
            },
            {'name': 'Display', 'type': 'group', 'children': [
                {'name': 'Frame Mode', 'type': 'list', 'values': ['Individual Frames', 'Average Frame'], 'value':self.frame_mode},
                {'name': 'Color Map', 'type': 'list',
                    'values': self.color_maps, 'value': self.color_map},
                {'name': 'Display Color Bar', 'type': 'bool', 'value': True},
                {'name': 'Temperature Range', 'type': 'group', 'children': [
                    {'name': 'Mode', 'type': 'list', 'values': ['Auto', 'Fixed'], 'value': self.temperature_range_mode},
                    {'name': 'Min', 'type': 'int', 'value': self.temperature_range[0]},
                    {'name': 'Max', 'type': 'int', 'value': self.temperature_range[1]},
                ]
                },
            ]
            },
        ]

        initial_export_params = [
            {'name': 'Export Data', 'type': 'group', 'children': [
                {'name': 'Export Data', 'type': 'action', }
            ]
            },
            {'name': 'Export Image', 'type': 'group', 'children': [
                # {'name': 'Frame selection', 'type': 'list', 'values': ['Average', 'Current'], 'value': 'Average'},
                {'name': 'Export Image', 'type': 'action', }
            ]
            },
            {'name': 'Export Video', 'type': 'group', 'children': [
                {'name': 'Export Video', 'type': 'action', }
            ]
            },
        ]

        # Initial parameters for setup
        self.setup_params = pg.parametertree.Parameter.create(name='setup_params', type='group', children=initial_setup_params)
        self.setup_tree = pg.parametertree.ParameterTree()
        self.setup_tree.setParameters(self.setup_params, showTop=False)

        # Initial parameters for processing
        self.process_params = pg.parametertree.Parameter.create(name='process_params', type='group', children=initial_process_params)
        self.process_tree = pg.parametertree.ParameterTree()
        self.process_tree.setParameters(self.process_params, showTop=False)

        # Initial parameters for export
        self.export_params = pg.parametertree.Parameter.create(name='export_params', type='group', children=initial_export_params)
        self.export_tree = pg.parametertree.ParameterTree()
        self.export_tree.setParameters(self.export_params, showTop=False)

        # Connect actions for setup
        self.setup_params.param('Directory Selection').sigValueChanged.connect(self.update_directory)
        # self.setup_params.param('Time Step (ms)').sigValueChanged.connect(self.update_time_step)
        # self.setup_params.param('Temperature Unit').sigValueChanged.connect(self.update_temperature_unit)
        # self.setup_params.param('Emmisivity').sigValueChanged.connect(self.update_emmisivity)
        # self.setup_params.param('App Mode').sigValueChanged.connect(self.update_app_mode)
        self.setup_params.param('Use SV1 Pipeline').sigValueChanged.connect(self.update_use_sv1_pipeline)
        self.setup_params.param('Use Thermocouple').sigValueChanged.connect(self.update_use_thermocouple)
        self.setup_params.param('Calibration', 'Calibration Mode').sigValueChanged.connect(self.update_calibration_mode)
        self.setup_params.param('Calibration', 'Set Point 1').sigActivated.connect(self.update_calibration_temeraure_1)
        self.setup_params.param('Calibration', 'Set Point 2').sigActivated.connect(self.update_calibration_temeraure_2)
        self.setup_params.param('Reload Camera').sigActivated.connect(self.init_seek_camera)
        self.setup_params.param('Return to Capture Mode').sigActivated.connect(self.return_to_capture)

        # Connect actions for processing
        self.process_params.param('Sample Details', 'Sample Name').sigValueChanged.connect(self.update_sample_name)
        self.process_params.param('Capture Image', 'Capture Image').sigActivated.connect(self.capture_image)
        self.process_params.param('Capture Video', 'Capture Video').sigActivated.connect(self.capture_video)
        self.process_params.param('Capture Video', 'Stop Capture').sigActivated.connect(self.stop_capture)
        self.process_params.param('Region of Interest', 'Position', 'X').sigValueChanged.connect(self.update_roi_via_text)
        self.process_params.param('Region of Interest', 'Position', 'Y').sigValueChanged.connect(self.update_roi_via_text)
        self.process_params.param('Region of Interest', 'Size', 'X').sigValueChanged.connect(self.update_roi_via_text)
        self.process_params.param('Region of Interest', 'Size', 'Y').sigValueChanged.connect(self.update_roi_via_text)
        self.process_params.param('Display', 'Frame Mode').sigValueChanged.connect(self.update_frame_mode)
        self.process_params.param('Display', 'Color Map').sigValueChanged.connect(self.update_color_map)
        self.process_params.param('Display', 'Display Color Bar').sigValueChanged.connect(self.update_display_color_bar)
        self.process_params.param('Display', 'Temperature Range', 'Mode').sigValueChanged.connect(self.update_temperature_range_mode)
        self.process_params.param('Display', 'Temperature Range', 'Min').sigValueChanged.connect(self.update_temperature_range)
        self.process_params.param('Display', 'Temperature Range', 'Max').sigValueChanged.connect(self.update_temperature_range)

        # Connect actions for export
        self.export_params.param('Export Data', 'Export Data').sigActivated.connect(self.current_loaded_data_export_data)
        self.export_params.param('Export Image', 'Export Image').sigActivated.connect(self.current_loaded_data_to_png)
        self.export_params.param('Export Video', 'Export Video').sigActivated.connect(self.current_loaded_data_to_avi)
        
    def init_ui(self):
        central_splitter = QSplitter()

        # Create Widgets for Splitter
        left_column_splitter = QSplitter(Qt.Vertical)
        middle_column_splitter = QSplitter(Qt.Vertical)
        right_column_splitter = QSplitter(Qt.Vertical)

        # ===================================================================================================
        # ---------- Left column ----------
        # ===================================================================================================

        # Setup parameters
        left_column_splitter.addWidget(self.setup_tree)

        # Show a table of all directory images and videos if in analyze mode
        directory_group = QGroupBox("Directory")
        self.directory_layout = QVBoxLayout()
        directory_group.setLayout(self.directory_layout)

        directory_scroll = QScrollArea()
        directory_scroll.setWidgetResizable(True)
        directory_scroll.setWidget(directory_group)

        refesh_button = QPushButton("Refresh")
        refesh_button.clicked.connect(self.refresh_directory)
        self.directory_layout.addWidget(refesh_button)

        left_column_splitter.addWidget(directory_scroll)
        self.refresh_directory()

        # # Show information about the current loaded file
        # current_loaded_data_info_group = QGroupBox("Loaded File")
        # self.current_loaded_data_info_layout = QVBoxLayout()
        # current_loaded_data_info_group.setLayout(self.current_loaded_data_info_layout)
        # layout_left.addWidget(current_loaded_data_info_group)

        # layout_left.addStretch(1)

        # ===================================================================================================
        # ---------- Middle column ----------
        # ===================================================================================================

        # Container for current image
        current_image_group = QGroupBox("Current Image")
        current_image_layout = QVBoxLayout()
        current_image_group.setLayout(current_image_layout)

        # Information about the current image, such as timestamp and recording status
        self.current_image_info = QLabel()
        current_image_layout.addWidget(self.current_image_info)

        # Display area for current image. Either live feed or loaded image/video
        current_image_window = pg.GraphicsLayoutWidget()
        current_image_window.setBackground(None)
        self.current_image_image = pg.ImageItem(edgecolors=None, antialiasing=False, colorMap=self.pg_cmap)
        self.current_image_image.setOpts(axisOrder='row-major')
        current_image_view = pg.ViewBox(lockAspect=True, invertY=True)
        current_image_view.addItem(self.current_image_image)
        self.current_image_plot = pg.PlotItem(viewBox=current_image_view)
        current_image_window.addItem(self.current_image_plot)


        # Add histogram to current image
        self.current_image_histogram = pg.HistogramLUTItem()
        self.current_image_histogram.gradient.loadPreset(self.color_map)
        self.current_image_histogram.setImageItem(self.current_image_image)
        current_image_window.addItem(self.current_image_histogram, 0,1,1,1)

        # Add Region of Interest to current image
        x = self.process_params.param('Region of Interest', 'Position', 'X').value()
        y = self.process_params.param('Region of Interest', 'Position', 'Y').value()
        w = self.process_params.param('Region of Interest', 'Size', 'X').value()
        h = self.process_params.param('Region of Interest', 'Size', 'Y').value()
        self.roi = pg.RectROI([x, y], [w, h], pen='w')
        self.roi.addScaleHandle([0.5, 0.5], [0.5, 0.5])
        self.roi.sigRegionChanged.connect(self.update_roi_via_plot)

        current_image_view.addItem(self.roi)
        current_image_layout.addWidget(current_image_window)

        # Add crosshair to current image and display temperature
        self.current_image_vline = pg.InfiniteLine(angle=90, movable=False, pen='w')
        self.current_image_hline = pg.InfiniteLine(angle=0, movable=False, pen='w')
        self.current_image_pos_label = pg.LabelItem()
        self.current_image_pos_label.setAttr('size', '12pt')
        self.current_image_pos_label.setAttr('color', 'white')
        self.current_image_plot.addItem(self.current_image_vline, ignoreBounds=True)
        self.current_image_plot.addItem(self.current_image_hline, ignoreBounds=True)
        self.current_image_plot.addItem(self.current_image_pos_label)
        self.current_image_vline.hide()
        self.current_image_hline.hide()
        self.current_image_pos_label.hide()
        self.current_image_plot.setCursor(Qt.CrossCursor)
        self.current_image_plot.scene().sigMouseMoved.connect(self.current_image_mouse_moved)

        # Add a frame selection slider and hide it unless in analyze video mode
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.valueChanged.connect(self.update_frame_index)
        current_image_layout.addWidget(self.frame_slider)

        self.frame_slider.hide()

        self.analyze_video_roi_button = QPushButton("Analyze Video ROI")
        self.analyze_video_roi_button.clicked.connect(self.analyze_video_roi)
        current_image_layout.addWidget(self.analyze_video_roi_button)

        self.analyze_video_roi_button.hide()

        # Region of Interest
        roi_group = QGroupBox("Region of Interest")
        roi_layout = QHBoxLayout()
        roi_group.setLayout(roi_layout)

        self.roi_data_window = pg.GraphicsLayoutWidget()
        self.roi_data_window.setBackground(None)
        
        # Display the ROI image
        self.roi_image = pg.ImageItem(edgecolors=None, antialiasing=False, colorMap=self.pg_cmap)
        self.roi_image.setOpts(axisOrder='row-major')
        self.roi_image_view = pg.ViewBox(lockAspect=True, invertY=True)
        self.roi_image_view.addItem(self.roi_image)
        self.roi_image_plot = pg.PlotItem(viewBox=self.roi_image_view)
        self.roi_image_plot.hideAxis('left')
        self.roi_image_plot.hideAxis('bottom')
        self.roi_image_plot.setFixedWidth(200)

        # Plot ROI data
        self.roi_data_plot = pg.PlotItem()
        self.roi_data_plot._exportOpts = {'f'}
        self.roi_data_plot.setLabel('left', 'Temperature', units='°C')
        self.roi_data_plot.setLabel('bottom', 'Time', units='s')
        self.roi_data_plot.showGrid(x=True, y=True)

        # Add histogram to ROI image
        self.roi_histogram = pg.HistogramLUTItem()
        self.roi_histogram.gradient.loadPreset(self.color_map)
        self.roi_histogram.setImageItem(self.roi_image)
        self.roi_data_window.addItem(self.roi_histogram)

        self.roi_data_window.addItem(self.roi_image_plot)
        self.roi_data_window.addItem(self.roi_data_plot)
        self.roi_data_window.addItem(self.roi_histogram)

        roi_layout.addWidget(self.roi_data_window)

        middle_column_splitter.addWidget(current_image_group)
        middle_column_splitter.addWidget(roi_group)

        # ===================================================================================================
        # ---------- Right column ----------
        # ===================================================================================================

        # Processing parameters
        right_column_splitter.addWidget(self.process_tree)
        right_column_splitter.addWidget(self.export_tree)


        # Setup splitter sizes
        left_column_splitter.setSizes([400,800])
        middle_column_splitter.setSizes([600,600])
        right_column_splitter.setSizes([800,400])

        central_splitter.addWidget(left_column_splitter)
        central_splitter.addWidget(middle_column_splitter)
        central_splitter.addWidget(right_column_splitter)

        central_splitter.setSizes([200, 800, 200])

        # Set Splitter as Central Widget
        self.setCentralWidget(central_splitter)

        
        # Status bar
        self.statusBar().showMessage('Ready', 2000)

    # ===================================================================================================
    # ---------- Functions ----------
    # ===================================================================================================


    # ===================================================================================================
    # Functions to handle changed to setup and processing parameters
    # ===================================================================================================

    def update_directory(self):
        self.directory = self.setup_params.param('Directory Selection').value()
        # self.ns = self.sh.NameSpace(self.directory)

        required_directories = ['thumbnails', 'raw data', 'exported data']

        for directory in required_directories:
            if not os.path.exists(f"{self.directory}/{directory}"):
                os.makedirs(f"{self.directory}/{directory}")
            
        self.refresh_directory()

    def update_use_sv1_pipeline(self):
        global USE_SV1_PIPELINE
        USE_SV1_PIPELINE = self.setup_params.param('Use SV1 Pipeline').value()
        self.init_seek_camera()

    def update_use_thermocouple(self):
        self.use_thermocouple = self.setup_params.param('Use Thermocouple').value()
        if self.use_thermocouple:
            self.setup_params.param('Thermocouple Temperature (°C)').show()
        else:
            self.setup_params.param('Thermocouple Temperature (°C)').hide()

    def update_calibration_mode(self):
        self.calibration_mode = self.setup_params.param('Calibration', 'Calibration Mode').value()
        if self.calibration_mode == 'None':
            self.setup_params.param('Calibration', 'Calibration Temperature 1 (°C)').hide()
            self.setup_params.param('Calibration', 'Set Point 1').hide()
            self.setup_params.param('Calibration', 'Calibration Temperature 2 (°C)').hide()
            self.setup_params.param('Calibration', 'Set Point 2').hide()
            self.current_loaded_data.calibration_slope = 1.0
            self.current_loaded_data.calibration_intercept = 0.0
        elif self.calibration_mode == 'One Point':
            self.setup_params.param('Calibration', 'Calibration Temperature 1 (°C)').show()
            self.setup_params.param('Calibration', 'Set Point 1').show()
            self.setup_params.param('Calibration', 'Calibration Temperature 2 (°C)').hide()
            self.setup_params.param('Calibration', 'Set Point 2').hide()
            self.current_loaded_data.calibration_slope = 1.0
        elif self.calibration_mode == 'Two Point':
            self.setup_params.param('Calibration', 'Calibration Temperature 1 (°C)').show()
            self.setup_params.param('Calibration', 'Set Point 1').show()
            self.setup_params.param('Calibration', 'Calibration Temperature 2 (°C)').show()
            self.setup_params.param('Calibration', 'Set Point 2').show()

        self.setup_params.param('Calibration', 'Calibration Slope').setValue(self.current_loaded_data.calibration_slope)
        self.setup_params.param('Calibration', 'Calibration Intercept').setValue(self.current_loaded_data.calibration_intercept)

    def update_calibration_temeraure_1(self):
        if self.use_thermocouple:
            self.calibration_temperature_1 = self.setup_params.param('Thermocouple Temperature (°C)').value()
            self.setup_params.param('Calibration', 'Calibration Temperature 1 (°C)').setValue(self.calibration_temperature_1)
        else:
            self.calibration_temperature_1 = self.setup_params.param('Calibration', 'Calibration Temperature 1 (°C)').value()

        self.measured_temperature_1 = self.mean_raw
        self.calibrate()

    def update_calibration_temeraure_2(self):
        if self.use_thermocouple:
            self.calibration_temperature_2 = self.setup_params.param('Thermocouple Temperature (°C)').value()
            self.setup_params.param('Calibration', 'Calibration Temperature 2 (°C)').setValue(self.calibration_temperature_2)
        else:
            self.calibration_temperature_2 = self.setup_params.param('Calibration', 'Calibration Temperature 2 (°C)').value()
    
        self.measured_temperature_2 = self.mean_raw
        self.calibrate()

    def calibrate(self):
        self.calibration_mode = self.setup_params.param('Calibration', 'Calibration Mode').value()
        if self.calibration_mode == 'None':
            return
        elif self.calibration_mode == 'One Point':
            self.current_loaded_data.calibration_intercept = self.calibration_temperature_1 - self.measured_temperature_1
        elif self.calibration_mode == 'Two Point':
            self.current_loaded_data.calibration_slope = (self.calibration_temperature_2 - self.calibration_temperature_1) / (self.measured_temperature_2 - self.measured_temperature_1)
            self.current_loaded_data.calibration_intercept = self.calibration_temperature_1 - self.measured_temperature_1 * self.current_loaded_data.calibration_slope
        self.setup_params.param('Calibration', 'Calibration Intercept').setValue(self.current_loaded_data.calibration_intercept)
        self.setup_params.param('Calibration', 'Calibration Slope').setValue(self.current_loaded_data.calibration_slope)

    # def update_time_step(self):
    #     self.time_step = int(self.setup_params.param('Time Step (ms)').value())
    #     self.timer.setInterval(self.time_step)

    # def update_temperature_unit(self):
    #     self.temperature_unit = self.setup_params.param('Temperature Unit').value()
    #     self.roi_data_plot.setLabel('left', 'Temperature', units=self.temperature_unit)
        # self.current_image_color_bar.setLabel(f"Temperature ({self.temperature_unit})")
        # self.roi_color_bar.setLabel(f"Temperature ({self.temperature_unit})")

    # def update_emmisivity(self):
    #     self.emmisivity = self.setup_params.param('Emmisivity').value()

    def update_temperature_range_mode(self):
        self.temperature_range_mode = self.process_params.param('Display', 'Temperature Range', 'Mode').value()
        self.update_current_image()
        self.update_roi_image()

    def update_temperature_range(self):
        self.temperature_range = (
            self.process_params.param('Display', 'Temperature Range', 'Min').value(),
            self.process_params.param('Display', 'Temperature Range', 'Max').value()
        )
        self.update_current_image()
        self.update_roi_image()

        if self.temperature_range_mode == 'Fixed':
            self.current_image_histogram.setLevels(min=self.temperature_range[0], max=self.temperature_range[1])
            self.roi_histogram.setLevels(min=self.temperature_range[0], max=self.temperature_range[1])

    def update_app_mode(self):
        if self.app_mode == 'Capture':
            # Restart the timer
            self.timer.start(self.time_step)

            # Make adjustments to the UI
            self.frame_slider.hide()
            self.analyze_video_roi_button.hide()
            if THERMOCOUPLE_AVAILABLE:
                self.setup_params.param('Use Thermocouple').show()
            self.setup_params.param('Return to Capture Mode').hide()
            self.setup_params.param('Reload Camera').show()
            self.setup_params.param('Calibration', 'Calibration Mode').show()
            self.process_params.param('Sample Details', 'Sample Name').setOpts(readonly=False)
            self.process_params.param('Sample Details', 'File Type').hide()
            self.update_calibration_mode()

        elif self.app_mode == 'Analysis':
            # Stop the timer
            self.timer.stop()

            # Make adjustments to the UI
            self.frame_slider.show()
            self.analyze_video_roi_button.show()
            self.setup_params.param('Use Thermocouple').hide()
            self.setup_params.param('Return to Capture Mode').show()
            self.setup_params.param('Reload Camera').hide()
            self.setup_params.param('Calibration', 'Calibration Mode').hide()
            self.setup_params.param('Calibration', 'Calibration Temperature 1 (°C)').hide()
            self.setup_params.param('Calibration', 'Set Point 1').hide()
            self.setup_params.param('Calibration', 'Calibration Temperature 2 (°C)').hide()
            self.setup_params.param('Calibration', 'Set Point 2').hide()
            self.process_params.param('Sample Details', 'Sample Name').setOpts(readonly=True)
            self.process_params.param('Sample Details', 'File Type').show()
            
        self.setup_params.param('App Mode').setValue(self.app_mode)

    def update_sample_name(self):
        self.sample_name = self.process_params.param('Sample Details', 'Sample Name').value()

    def update_roi_via_text(self):
        x = self.process_params.param('Region of Interest', 'Position', 'X').value()
        y = self.process_params.param('Region of Interest', 'Position', 'Y').value()
        w = self.process_params.param('Region of Interest', 'Size', 'X').value()
        h = self.process_params.param('Region of Interest', 'Size', 'Y').value()
        self.roi.setPos([x, y])
        self.roi.setSize([w, h])
        self.roi_data['pos'] = [x, y]
        self.roi_data['size'] = [w, h]
        if not self.recording:
            self.clear_roi_data()
            self.update_roi_data()

    def update_roi_via_plot(self):
        x = int(self.roi.pos()[0])
        y = int(self.roi.pos()[1])
        w = int(self.roi.size()[0])
        h = int(self.roi.size()[1])
        self.roi_data['pos'] = [x, y]
        self.roi_data['size'] = [w, h]
        self.process_params.param('Region of Interest', 'Position', 'X').setValue(x)
        self.process_params.param('Region of Interest', 'Position', 'Y').setValue(y)
        self.process_params.param('Region of Interest', 'Size', 'X').setValue(w)
        self.process_params.param('Region of Interest', 'Size', 'Y').setValue(h)
        if not self.recording:
            self.clear_roi_data()
            self.update_roi_data()

    def update_color_map(self):
        self.color_map = self.process_params.param('Display', 'Color Map').value()
        self.mpl_cmap = mpl.colors.Colormap(self.color_map)
        self.pg_cmap = pg.colormap.getFromMatplotlib(self.color_map)

        self.current_image_image.setColorMap(self.color_map)
        self.current_image_histogram.gradient.loadPreset(self.color_map)
        self.roi_image.setColorMap(self.color_map)
        self.roi_histogram.gradient.loadPreset(self.color_map)

    def update_display_color_bar(self):
        self.display_color_bar = self.process_params.param('Display', 'Display Color Bar').value()
        if self.display_color_bar:
            self.current_image_histogram.show()
            self.roi_histogram.show()
        else:
            self.current_image_histogram.hide()
            self.roi_histogram.hide()

    def update_frame_index(self):
        self.frame_index = int(self.frame_slider.value())
        try:
            self.roi_data_plot.removeItem(self.roi_data_plot_time_marker)
        except:
            pass
        try:
            x = self.roi_data["time"][self.frame_index]
        except:
            self.analyze_video_roi()
            x = self.roi_data["time"][self.frame_index]
        self.roi_data_plot_time_marker = pg.InfiniteLine(pos=x, angle=90, movable=False)
        self.roi_data_plot.addItem(self.roi_data_plot_time_marker)

        self.update_current_image()
        self.update_roi_image()

    def return_to_capture(self):
        self.app_mode = 'Capture'
        self.update_app_mode()
        self.clear_roi_data()

    def current_loaded_data_to_png(self):
        if self.frame_mode == 'Average Frame':
            frame = self.current_loaded_data.average_frame
            suffix = 'average frame'
        elif self.frame_mode == 'Individual Frames':
            frame = self.current_loaded_data.raw_frames[self.frame_index]
            suffix = f'frame {self.frame_index}'

        filename = f"exported data/{self.current_loaded_data.created} - image - {self.current_loaded_data.sample_name} - {suffix}.png"

        include_color_bar = self.process_params.param('Display', 'Display Color Bar').value()
        color_map = self.color_map
        if self.temperature_range_mode == 'Auto':
            fixed_range = None
        elif self.temperature_range_mode == 'Fixed':
            fixed_range = (
                self.process_params.param('Display', 'Temperature Range', 'Min').value(),
                self.process_params.param('Display', 'Temperature Range', 'Max').value()
            )

        self.current_loaded_data.to_png(filename=filename, frame=frame, include_color_bar=include_color_bar, color_map=color_map, fixed_range=fixed_range)
    
    def current_loaded_data_to_avi(self):
        filename = f"exported data/{self.current_loaded_data.created} - video - {self.current_loaded_data.sample_name}.avi"
        include_color_bar = self.process_params.param('Display', 'Display Color Bar').value()
        color_map = self.color_map
        if self.temperature_range_mode == 'Auto':
            fixed_range = None
        elif self.temperature_range_mode == 'Fixed':
            fixed_range = (
                self.process_params.param('Display', 'Temperature Range', 'Min').value(),
                self.process_params.param('Display', 'Temperature Range', 'Max').value()
            )

        self.current_loaded_data.to_avi(filename=filename, include_color_bar=include_color_bar, color_map=color_map, fixed_range=fixed_range)
    
    def current_loaded_data_export_data(self):
        self.current_loaded_data.export_data()

    def update_frame_mode(self):
        self.frame_mode = self.process_params.param('Display', 'Frame Mode').value()
        if self.app_mode == 'Capture':
            return
        if self.frame_mode == 'Individual Frames':
            self.frame_slider.show()
            self.analyze_video_roi_button.show()
        elif self.frame_mode == 'Average Frame':
            self.frame_slider.hide()
            self.analyze_video_roi_button.hide()
        self.update_current_image()

    def current_image_mouse_moved(self, evt):
        pos = evt
        mouse_point = self.current_image_plot.vb.mapSceneToView(pos)
        x = int(mouse_point.x())
        y = int(mouse_point.y())
        if x >= 0 and x < self.current_image_image.image.shape[1] and y >= 0 and y < self.current_image_image.image.shape[0]:
            self.current_image_vline.show()
            self.current_image_hline.show()
            self.current_image_pos_label.show()
            t = self.current_image_image.image[y, x]
            t = self.current_loaded_data.calibration_slope * t + self.current_loaded_data.calibration_intercept
            self.current_image_pos_label.setText(f"{t:.2f}°C")
            self.current_image_pos_label.setPos(mouse_point)
            self.current_image_vline.setPos(mouse_point.x())
            self.current_image_hline.setPos(mouse_point.y())
        else:
            self.current_image_vline.hide()
            self.current_image_hline.hide()
            self.current_image_pos_label.hide()
    

    # ===================================================================================================
    # Functions to handle application setup and refresh
    # ===================================================================================================

    def refresh(self):
        if self.app_mode == 'Capture':
            self.update_thermocouple()
            self.update_current_image()
            self.refresh_crosshair()
            self.update_roi_image()
            self.analyze_roi()
            self.plot_roi_data()

            self.process_params.param('Sample Details', 'Time Stamp').setValue(datetime.now().strftime("%Y%m%d-%H%M%S%f"))

    def init_seek_camera(self):
        try: 
            self.seek_manager.destroy()
        except:
            pass
        self.clear_roi_data()
        self.seek_renderer = Renderer()
        self.seek_manager = SeekCameraManager(SeekCameraIOType.USB)
        self.seek_manager.register_event_callback(seek_on_event, self.seek_renderer)            
        
    def get_live_frame(self):
        retry_count = 0
        max_retries = 10
        while retry_count < max_retries:
            try:
                with self.seek_renderer.frame_condition:
                    # if self.seek_renderer.frame_condition.wait(self.time_step):
                    frame = self.seek_renderer.frame.data
                return np.copy(frame)
            except:
                retry_count += 1
                frame = np.random.normal(0.5, 0.1, (480, 640))
                cv2.putText(frame, "CAMERA NOT DETECTED", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
                frame = np.interp(frame, (0, 1), self.temperature_range)
                return frame
    
    def update_current_image(self):
        if self.app_mode == 'Capture':
            frame = self.get_live_frame()
        elif self.app_mode == 'Analysis' and self.frame_mode == 'Individual Frames':
            frame = self.current_loaded_data.raw_frames[self.frame_index]
        elif self.app_mode == 'Analysis' and self.frame_mode == 'Average Frame':
            frame = self.current_loaded_data.average_frame
        self.current_image_image.setImage(frame)

        if self.temperature_range_mode == 'Fixed':
            self.current_image_histogram.setLevels(min=self.temperature_range[0], max=self.temperature_range[1])

    def refresh_directory(self):
        try :
            self.directory_layout.removeWidget(self.directory_content)
            self.directory_content.deleteLater()
        except:
            pass

        self.directory_content = QWidget()
        directory_content_layout = QVBoxLayout()
        self.directory_content.setLayout(directory_content_layout)

        directory_content_layout.addWidget(QLabel("<b>Images</b>"))
        image_list = QFrame()
        image_list_layout = QVBoxLayout()
        image_list.setLayout(image_list_layout)
        directory_content_layout.addWidget(image_list)
        directory_content_layout.addWidget(QLabel("<b>Videos</b>"))
        video_list = QFrame()
        video_list_layout = QVBoxLayout()
        video_list.setLayout(video_list_layout)
        directory_content_layout.addWidget(video_list)
        directory_content_layout.addStretch(1)

        for file in os.listdir(f"{self.directory}/raw data"):
            if not file.endswith(".pkl"):
                continue

            with open(f"{self.directory}/raw data/{file}", 'rb') as f:
                self.files_in_raw_data.append(file)
                # data = pickle.load(f)
                # self.all_loaded_data[file] = data

                if not os.path.exists(f"{self.directory}/thumbnails/{file}.png"):
                    data = pickle.load(f)
                    frame = data.raw_frames[0]
                    data.to_png(f"{self.directory}/thumbnails/{file}.png", frame)

                # if data.data_type == 'image':
                if 'image' in file:
                    image_thumbnail = f"<img src='{self.directory}/thumbnails/{file}' width='400'/>"
                    image_label = QLabel(file)
                    image_label.setToolTip(image_thumbnail)
                    image_label.mousePressEvent = lambda event, arg=file: self.handel_directory_click(event, arg)
                    image_list_layout.addWidget(image_label)

                # elif data.data_type == 'video':
                elif 'video' in file:
                    video_thumbnail = f"<img src='{self.directory}/thumbnails/{file}.png' width='400'/>"
                    video_label = QLabel(file)
                    video_label.setToolTip(video_thumbnail)
                    video_label.mousePressEvent = lambda event, arg=file: self.handel_directory_click(event, arg)
                    video_list_layout.addWidget(video_label)

        self.directory_layout.addWidget(self.directory_content)
    
    def handel_directory_click(self, event, file):
        if event.button() == Qt.LeftButton:
            self.load_data_for_analysis(file)
        if event.button() == Qt.RightButton:
            # Copy file name to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(file)
            self.statusBar().showMessage(f"File name copied to clipboard: {file}", 3000)

    def refresh_crosshair(self):
        x = int(self.current_image_vline.value())
        y = int(self.current_image_hline.value())
        t = self.current_image_image.image[y, x]
        t = self.current_loaded_data.calibration_slope * t + self.current_loaded_data.calibration_intercept
        self.current_image_pos_label.setText(f"{t:.2f}°C")


    # ===================================================================================================
    # Functions to handle data collection
    # ===================================================================================================

    def update_thermocouple(self):
        if self.app_mode == 'Analysis':
            return

        if not self.use_thermocouple:
            self.current_loaded_data.thermocouple.append(np.nan)
            return
        
        board_num = 0
        channel = 0

        try:
            # Get the devices name...
            board_name = ul.get_board_name(board_num)
        except Exception as e:
            if ul.ErrorCode(1):
                # No board at that number throws error
                print("\nNo board found at Board 0.")
                print(e)
                return
        
        try:
            ul.set_config(
                InfoType.BOARDINFO, board_num, channel, BoardInfo.CHANTCTYPE, TcType.K)
            ul.set_config(
                InfoType.BOARDINFO, board_num, channel, BoardInfo.TEMPSCALE, TempScale.CELSIUS)
            ul.set_config(
                InfoType.BOARDINFO, board_num, channel, BoardInfo.ADDATARATE, 10)
            options = TInOptions.NOFILTER
            value_temperature = ul.t_in(board_num, channel, TempScale.CELSIUS, options)

            self.thermocouple_temperature = value_temperature
            self.setup_params.param('Thermocouple Temperature (°C)').setValue(self.thermocouple_temperature)
            # print("Channel{:d}:  {:.3f}°C.".format(channel, value_temperature))

            self.current_loaded_data.thermocouple.append(self.thermocouple_temperature)

        except Exception as e:
            self.current_loaded_data.thermocouple.append(np.nan)
            print('\n', e)

    def capture_image(self):
        repeat = self.process_params.param(
            'Capture Image', 'Number of Repeats').value()
        number = self.process_params.param(
            'Capture Image', 'Images per Repeat').value()
        
        for i in range(int(repeat)):
            data = ThermographyData(slope=self.current_loaded_data.calibration_slope, intercept=self.current_loaded_data.calibration_intercept)
            data.sample_name = self.sample_name
            data.data_type = 'image'

            for j in range(int(number)):
                frame = self.get_live_frame()
                data.raw_frames.append(frame)
                data.time.append(time.time())
                time.sleep(self.time_step/1000)
            data.update_average_frame()
            data.to_pkl(f"{self.directory}/raw data/{data.created} - image - {data.sample_name}.pkl")

        self.refresh_directory()
    
    def stop_capture(self):
        self.recording = False
        self.process_params.param('Capture Video', 'Capture Video').setOpts(visible=True)
        self.process_params.param('Capture Video', 'Stop Capture').setOpts(visible=False)

    def capture_video(self):
        """Capture a video for a specified duration and framerate.
        This funtion is responsible for starting the timer that will capture the video frames."""

        self.timer.stop()
        self.capture_duration = int(self.process_params.param(
            'Capture Video', 'Duration (seconds)').value())
        fps = int(self.process_params.param(
            'Capture Video', 'Frames per second').value())
        
        self.process_params.param('Capture Video', 'Capture Video').setOpts(visible=False)
        self.process_params.param('Capture Video', 'Stop Capture').setOpts(visible=True)
        
        self.recording = True
        
        self.current_video_capture = ThermographyData(slope=self.current_loaded_data.calibration_slope, intercept=self.current_loaded_data.calibration_intercept)
        self.current_video_capture.sample_name = self.sample_name
        self.current_video_capture.data_type = 'video'
        self.current_video_capture.start_time = time.time()

        self.captue_elapsed = QElapsedTimer()
        self.captue_elapsed.start()
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_video_execute)
        self.capture_timer.start(int(1000/fps))

    def capture_video_execute(self):
        """Capture a video for a specified duration and framerate.
        This funtion is executed by 'capture_video' and is responsible for capturing the video frames and ending the capture"""

        if self.capture_duration != 0 \
            and self.captue_elapsed.elapsed() > self.capture_duration * 1000:
            self.stop_capture()

        if self.recording:
            frame = self.get_live_frame()
            self.update_current_image()
            self.refresh_crosshair()
            self.update_roi_data()
            self.current_video_capture.thermocouple.append(self.thermocouple_temperature)
            self.current_video_capture.raw_frames.append(frame)
            time_stamp = time.time() - self.current_video_capture.start_time
            self.current_video_capture.time.append(time_stamp)
            status_text = f"Recording {datetime.fromtimestamp(time_stamp).strftime('%H:%M:%S.%f')[:-3]}"
            self.current_image_info.setText(status_text)
            self.current_image_info.setStyleSheet("color: red")

        else:
            self.capture_timer.stop()
            self.current_video_capture.update_average_frame()
            self.current_video_capture.to_pkl(f"{self.directory}/raw data/{self.current_video_capture.created} - video - {self.current_video_capture.sample_name}.pkl")
            self.refresh_directory()
            self.timer.start(self.time_step)
            self.current_image_info.setText("")

    # ===================================================================================================
    # Functions to handle loading images and videos for analysis
    # ===================================================================================================

    def load_data_for_analysis(self, file):

        self.app_mode = 'Analysis'
        self.update_app_mode()

        with open(f"{self.directory}/raw data/{file}", 'rb') as f:
            self.current_loaded_data = pickle.load(f)

        time_stamp = self.current_loaded_data.created
        file_type = self.current_loaded_data.data_type
        sample_name = self.current_loaded_data.sample_name

        # Sample name may contain "-"
        file_parse = file.split(' - ')
        time_stamp = file_parse[0]
        file_type = file_parse[1]
        sample_name = ' - '.join(file_parse[2:])
        # time_stamp, file_type, sample_name = file.split(' - ')
        sample_name = sample_name.split('.')[0]
        self.sample_name = sample_name
        self.process_params.param('Sample Details', 'Time Stamp').setValue(time_stamp)
        self.process_params.param('Sample Details', 'File Type').setValue(file_type)
        self.process_params.param('Sample Details', 'Sample Name').setValue(self.sample_name)

        self.total_frames = len(self.current_loaded_data.raw_frames)
        self.frame_index = 0
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.setValue(0)

        # For backwards compatibility, add thermocouple data, slope, and intercept if not present
        if not hasattr(self.current_loaded_data, 'thermocouple'):
            self.current_loaded_data.thermocouple = []
        if not hasattr(self.current_loaded_data, 'calibration_slope'):
            self.current_loaded_data.calibration_slope = 1.0
        if not hasattr(self.current_loaded_data, 'calibration_intercept'):
            self.current_loaded_data.calibration_intercept = 0.0

        self.setup_params.param('Calibration', 'Calibration Slope').setValue(self.current_loaded_data.calibration_slope)
        self.setup_params.param('Calibration', 'Calibration Intercept').setValue(self.current_loaded_data.calibration_intercept)

        # self.get_current_loaded_data_metadata()
        # self.current_loaded_data_metadata['Total Frames'] = str(self.total_frames)
        # self.current_loaded_data_metadata['Frame Rate'] = str(self.current_loaded_data.time[-1] / self.total_frames)
        # self.update_current_loaded_data_metadata()
        self.analyze_video_roi()

    # def get_current_loaded_data_metadata(self):
    #     self.current_loaded_data_metadata = {}

    #     item = self.ns.ParseName(self.current_loaded_data_name)
    #     col_num = 0
    #     cols = []
    #     while True:
    #         col_name = self.ns.GetDetailsOf(None, col_num)
    #         if not col_name:
    #             break
    #         cols.append(col_name)
    #         col_num += 1
    #     for col_num in range(len(cols)):
    #         col_val = self.ns.GetDetailsOf(item, col_num)
    #         if col_val:
    #             self.current_loaded_data_metadata[cols[col_num]] = col_val
            
    # def update_current_loaded_data_metadata(self):
    #     try:
    #         self.current_loaded_data_info_layout.removeWidget(self.current_loaded_data_info)
    #         self.current_loaded_data_info.deleteLater()
    #     except:
    #         pass
        
    #     self.current_loaded_data_info = QTableWidget()
    #     self.current_loaded_data_info.setColumnCount(2)
    #     self.current_loaded_data_info.setHorizontalHeaderLabels(['Property', 'Value'])
    #     self.current_loaded_data_info.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    #     for i, key in enumerate(self.current_loaded_data_metadata.keys()):
    #         self.current_loaded_data_info.insertRow(i)
    #         self.current_loaded_data_info.setItem(i, 0, QTableWidgetItem(key))
    #         self.current_loaded_data_info.setItem(i, 1, QTableWidgetItem(self.current_loaded_data_metadata[key]))

    #     self.current_loaded_data_info_layout.addWidget(self.current_loaded_data_info)
        

    # ===================================================================================================
    # Functions to handle roi analysis
    # ===================================================================================================

    def update_roi_data(self):
        self.update_roi_image()
        self.analyze_roi()
        self.update_thermocouple()
        self.plot_roi_data()

    def clear_roi_data(self):
        self.roi_time_0 = time.time()
        self.roi_data = {
            "pos": [self.roi.pos()[0], self.roi.pos()[1]],
            "size": [self.roi.size()[0], self.roi.size()[1]],
            "time": [],
            "max": [],
            "min": [],
            "mean": [],
            "center": [],
        }

        if self.app_mode == 'Capture':
            self.current_loaded_data.thermocouple = []

    def update_roi_image(self):
        self.roi_image.setImage(self.roi.getArrayRegion(
            self.current_image_image.image, self.current_image_image))
        
        if self.temperature_range_mode == 'Fixed':
            self.roi_histogram.setLevels(min=self.temperature_range[0], max=self.temperature_range[1])

        w, h = self.roi.size()
        self.roi_image_plot.clear()
        self.roi_image_plot.plot([int(w/2)], [int(h/2)], pen='w', symbol='o', symbolSize=10, symbolBrush=None)

    def analyze_roi(self):
        roi_image = self.roi_image.image
        w, h, = roi_image.shape

        self.mean_raw = np.mean(roi_image)

        # Calibrate
        roi_image = roi_image * self.current_loaded_data.calibration_slope + self.current_loaded_data.calibration_intercept

        self.roi_max = np.max(roi_image)
        self.roi_min = np.min(roi_image)
        self.roi_mean = np.mean(roi_image)
        self.roi_center = roi_image[int(w / 2), int(h / 2)]

        t = time.time() - self.roi_time_0
        self.roi_data["time"].append(t)
        self.roi_data["max"].append(self.roi_max)
        self.roi_data["min"].append(self.roi_min)
        self.roi_data["mean"].append(self.roi_mean)
        self.roi_data["center"].append(self.roi_center)

    def plot_roi_data(self):
        self.roi_data_plot.clear()

        min_label = pg.TextItem(f"Min: {self.roi_min:.2f} {self.temperature_unit}", color='cornflowerblue')
        mean_label = pg.TextItem(f"Mean: {self.roi_mean:.2f} {self.temperature_unit}", color='silver')
        center_label = pg.TextItem(f"Center: {self.roi_center:.2f} {self.temperature_unit}", color='gold')
        max_label = pg.TextItem(f"Max: {self.roi_max:.2f} {self.temperature_unit}", color='crimson')

        x = np.array(self.roi_data["time"])
        if x.size > 1:
            self.roi_data_plot.plot(x, self.roi_data["max"], pen='crimson', name="Max")
            self.roi_data_plot.plot(x, self.roi_data["min"], pen='cornflowerblue', name="Min")
            self.roi_data_plot.plot(x, self.roi_data["mean"], pen='silver', name="Mean")
            self.roi_data_plot.plot(x, self.roi_data["center"], pen='gold', name="Center")
            
            min_label.setPos(x[-1], self.roi_min)
            mean_label.setPos(x[-1], self.roi_mean)
            center_label.setPos(x[-1], self.roi_center)
            max_label.setPos(x[-1], self.roi_max)

        else:
            max_bar = pg.BarGraphItem(x=3, height=self.roi_data['max'], width=0.5, brush='crimson', pen=None)
            min_bar = pg.BarGraphItem(x=0, height=self.roi_data['min'], width=0.5, brush='cornflowerblue', pen=None)
            mean_bar = pg.BarGraphItem(x=1, height=self.roi_data['mean'], width=0.5, brush='silver', pen=None)
            center_bar = pg.BarGraphItem(x=2, height=self.roi_data['center'], width=0.5, brush='gold', pen=None)

            self.roi_data_plot.addItem(max_bar)
            self.roi_data_plot.addItem(min_bar)
            self.roi_data_plot.addItem(mean_bar)
            self.roi_data_plot.addItem(center_bar)

            offset = self.roi_max * 0.1

            min_label.setPos(-0.25, self.roi_min+offset)
            mean_label.setPos(0.75, self.roi_mean+offset)
            center_label.setPos(1.75, self.roi_center+offset)
            max_label.setPos(2.75, self.roi_max+offset)

        self.roi_data_plot.addItem(min_label)
        self.roi_data_plot.addItem(mean_label)
        self.roi_data_plot.addItem(center_label)
        self.roi_data_plot.addItem(max_label)

        try:
            thermocouple_label = pg.TextItem(f"Thermocouple: {self.current_loaded_data.thermocouple[-1]:.2f} {self.temperature_unit}", color='forestgreen')
            if x.size > 1:
                self.roi_data_plot.plot(x, self.current_loaded_data.thermocouple, pen='forestgreen', name="Thermocouple")
                thermocouple_label.setPos(x[-1], self.current_loaded_data.thermocouple[-1])

            else:
                thermocouple_bar = pg.BarGraphItem(x=4, height=self.current_loaded_data.thermocouple, width=0.5, brush='forestgreen', pen=None)
                self.roi_data_plot.addItem(thermocouple_bar)
                thermocouple_label.setPos(3.25, self.current_loaded_data.thermocouple[-1]+offset)
                
            self.roi_data_plot.addItem(thermocouple_label)
        except:
            pass

    def analyze_video_roi(self):
        self.clear_roi_data()
        data = self.current_loaded_data
        for frame in data.raw_frames:
            self.current_image_image.setImage(frame)
            self.update_roi_image()
            self.analyze_roi()
        self.roi_data["time"] = data.time
        
        self.update_current_image()
        self.update_roi_image()
        self.plot_roi_data()

        data.roi_data = self.roi_data


# ===================================================================================================
# Main Application
# ===================================================================================================

def main():
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    window = ThermalImagingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

