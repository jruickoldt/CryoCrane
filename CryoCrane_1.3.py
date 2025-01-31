import sys
import matplotlib
matplotlib.use('Qt5Agg')

import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import mrcfile
import tifffile

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import re



def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def get_xy_rotated(xml_file, offsetx, offsety, angle = 170):
    angle = angle/360*2*np.pi
    file = open(xml_file, "r")
    meta= file.read()
    file.close()
    xml_string = xml_file.as_posix()
    if str(xml_string[-3:])=="xml":
        String_list_xml= ['<BeamShift xmlns:a="http://schemas.datacontract.org/2004/07/Fei.Types">',
                          "</BeamShift>",
                          "</B><X>",
                          "</X>",
                          "<a:_x>",
                          "</a:_x><a:_y>",
                          "<Y>",
                          "</Y>",
                          "<a:_y>",
                          "</a:_y>",
                          '<a:Key>AppliedDefocus</a:Key><a:Value i:type="b:double" xmlns:b="http://www.w3.org/2001/XMLSchema">',
                          '</a:Value></a:KeyValueOfstringanyType></CustomData>']

        Start = String_list_xml[0]
        End = String_list_xml[1]
        Beamshift=meta[meta.index(Start)+len(Start):meta.index(End)]
        #Beam shift in µm?
        
        #Read the x and y coordinate from the xml file
        Start = String_list_xml[2]
        End = String_list_xml[3]
        
        x=float(meta[meta.index(Start)+len(Start):meta.index(End)])*1000*1000
        Start = String_list_xml[4]
        End = String_list_xml[5]
        x_shift=float(Beamshift[Beamshift.index(Start)+len(Start):Beamshift.index(End)])*20 #Empirically determined, could be wrong

        Start = String_list_xml[6]
        End = String_list_xml[7]
        y=float(meta[meta.index(Start)+len(Start):meta.index(End)])*1000*1000
        
        Start = String_list_xml[8]
        End = String_list_xml[9]
        y_shift=float(Beamshift[Beamshift.index(Start)+len(Start):Beamshift.index(End)])*20 #Empirically determined, could be wrong

        #get the applied defocus: 
        Start = String_list_xml[10]
        End = String_list_xml[11]
        df=float(meta[meta.index(Start)+len(Start):meta.index(End)])*1000*1000

    
    if str(xml_string[-4:])=="mdoc":
        
        String_list_xml= ["StagePosition = ",
                          "StageZ = ",
                          'Defocus = ',
                          'ImageShift = ',
                          "RotationAngle = "
                         ]

        #Read the x and y coordinate from the mdoc file
        Start = String_list_xml[0]
        End = String_list_xml[1]

        Stage = meta[meta.index(Start)+len(Start):meta.index(End)]
        x = float(Stage[0:Stage.index(" ")])
        y = float(Stage[Stage.index(" "):])

        #get the applied defocus: 
        Start = String_list_xml[2]
        End = String_list_xml[3]
        df = float(meta[meta.index(Start)+len(Start):meta.index(End)])

        Start = String_list_xml[3]
        End = String_list_xml[4]
        Shift = meta[meta.index(Start)+len(Start):meta.index(End)]
        x_shift = float(Shift[0:Shift.index(" ")])
        y_shift = float(Shift[Shift.index(" "):])
 
                      
    
    x += x_shift
    x += offsetx

    y += y_shift
    y += offsety
    
    #Calculate the rotated coordinates
    x_rot = np.cos(angle) * x - np.sin(angle) * y
    y_rot = np.sin(angle) * x + np.cos(angle) * y



    
    
    values =[x_rot,y_rot,df]
    return values

def read_atlas_mrc(filename):
    """Reads an MRC file and returns the image data."""
    with mrcfile.open(filename, permissive=True) as mrc:
        return mrc.data

def read_atlas_meta(meta_filename):
    """
    Reads metadata from an XML file and returns coordinates and pixel size.

    The meta file is assumed to have strings like:
        <PixelSize>value</PixelSize>
        <Coordinates>x,y</Coordinates>

    Parameters:
        meta_filename (str): Path to the metadata file.

    Returns:
        tuple: (pixel_size (float), coordinates (tuple of float))
    """
    with open(meta_filename, 'r') as f:
        content = f.read()

    # Search for pixel size and coordinates in the XML content
    pixel_size_match = re.search(r'<pixelSize><x><numericValue>(.*?)</numericValue>', content)
    x_match = re.search(r'</A><B>0</B><X>(.*?)</X><Y>', content)
    y_match = re.search(r'</X><Y>(.*?)</Y><Z>', content)

    if not pixel_size_match or not x_match or not y_match:
        raise ValueError(f"Metadata file {meta_filename} is missing required fields.")

    pixel_size = float(pixel_size_match.group(1))
    coordinates = float(x_match.group(1)),float(y_match.group(1))

    return pixel_size, coordinates

def stitch_tiles(tiles, tile_coords, pixel_size):
    """
    Stitch tiles together based on coordinates and pixel size.

    Parameters:
        tiles (list of ndarray): List of tile image data arrays.
        tile_coords (list of tuples): Coordinates of the top-left corner of each tile (x, y).
        pixel_size (float): The pixel size in the same unit as coordinates.

    Returns:
        ndarray: Stitched image.
    """
    # Convert coordinates to pixel units
    pixel_coords = [(int(x / pixel_size), int(y / pixel_size)) for x, y in tile_coords]
    print(pixel_coords)

    
    # Calculate bounds of the coordinate space
    min_x = min(px - int(tile.shape[1] / 2) for (px, py), tile in zip(pixel_coords, tiles))
    max_x = max(px + int(tile.shape[1] / 2) for (px, py), tile in zip(pixel_coords, tiles))
    min_y = min(py - int(tile.shape[0] / 2) for (px, py), tile in zip(pixel_coords, tiles))
    max_y = max(py + int(tile.shape[0] / 2) for (px, py), tile in zip(pixel_coords, tiles))

    # Compute the offset to center the image around (0, 0)
    offset_x = -min_x
    offset_y = -min_y

    # Compute the final stitched image size
    stitched_width = max_x - min_x
    stitched_height = max_y - min_y

    # Initialize the stitched image
    stitched_image = np.zeros((stitched_height, stitched_width), dtype=float)
    
    # Initialize a weight matrix to handle overlapping regions
    weight_matrix = np.zeros_like(stitched_image, dtype=float)
   
    # Place each tile in the stitched image
    for tile, (px, py) in zip(tiles, pixel_coords):
        # Adjust coordinates to center around (0, 0)
        px += offset_x
        py += offset_y

        # Calculate region in the stitched image
        start_y = max(0, py - int(tile.shape[0] / 2))
        end_y = min(stitched_image.shape[0], py + int(tile.shape[0] / 2))
        start_x = max(0, px - int(tile.shape[1] / 2))
        end_x = min(stitched_image.shape[1], px + int(tile.shape[1] / 2))

        # Calculate corresponding region in the tile
        tile_start_y = max(0, - (py - int(tile.shape[0] / 2)))
        tile_end_y = tile_start_y + (end_y - start_y)
        tile_start_x = max(0, - (px - int(tile.shape[1] / 2)))
        tile_end_x = tile_start_x + (end_x - start_x)

        # Place the tile in the stitched image
        stitched_image[start_y:end_y, start_x:end_x] += tile[tile_start_y:tile_end_y, tile_start_x:tile_end_x]
        weight_matrix[start_y:end_y, start_x:end_x] += 1

    # Normalize overlapping regions
    non_zero_mask = weight_matrix > 0
    stitched_image[non_zero_mask] /= weight_matrix[non_zero_mask]

    return stitched_image

def stitch_atlas(tile_folder):
    # Path to the folder containing the tiles
    tile_files = sorted([os.path.join(tile_folder, f) for f in os.listdir(tile_folder) if f.endswith('.mrc') and f.startswith("Tile")])

    # Read tiles and their metadata
    tiles = []
    tile_coords = []
    pixel_size = None
    
    #print(tile_files)
    

    for tile_file in tile_files:
        tile_file = str(tile_file)
        meta_file = tile_file.replace('.mrc', '.xml')
        
        tile_pixel_size, coordinates = read_atlas_meta(meta_file)
        if pixel_size is None:
            pixel_size = tile_pixel_size
        elif pixel_size != tile_pixel_size:
            raise ValueError("Inconsistent pixel sizes in metadata files.")
        tiles.append(read_atlas_mrc(tile_file))
        tile_coords.append(coordinates)
        print(tile_coords)
    
    
    # Stitch tiles together
    stitched_image = stitch_tiles(tiles, tile_coords, pixel_size)
    
    return stitched_image
    
def load_mic_data(angle, offsetx, offsety, Micpath,Atlaspath, TIFF = False, MDOC = False):
    #Create an empty dictionary
    Data_rot = {}
    p = Path(Micpath)
    #Load the exposures should be located in a Data folder
    if MDOC == False:
        No_foilhole = [path for path in list(p.glob('**/*.xml')) if ("Data" in path.parts or "Images" in path.parts)]
    else:
        No_foilhole = [path for path in list(p.glob('**/*.mdoc'))]
        
    for i in No_foilhole:
        Data_rot[str(i)]=get_xy_rotated(i, offsetx, offsety, angle)
    
    pd.set_option("display.precision", 4)
    #Rearange the data base file
    Locations_rot=pd.DataFrame(Data_rot).T
    Locations_rot.round(4)
    #Extract list of x and y for plotting purposed in the plot function
    x, y, df = list(zip(*Data_rot.values()))
    
    #Reorder the data base new column contains the path of the exposures
    Locations_rot = Locations_rot.reset_index().rename(columns={"index":"xml"})
    if MDOC == False:
        if TIFF == False:
            Locations_rot = Locations_rot.assign(JPG = [w.replace(".xml",".mrc") for w in list(Locations_rot.iloc[:,0])])
    
        else:
            Locations_rot = Locations_rot.assign(JPG = [w.replace(".xml",".tiff") for w in list(Locations_rot.iloc[:,0])])
    else:
        Locations_rot = Locations_rot.assign(JPG = [w.replace(".mdoc","") for w in list(Locations_rot.iloc[:,0])])

    
    #open the Atlas, which should be located somewhere in the Micpath
    if Atlaspath[-3:] == "mrc":
        if "/" in Atlaspath:
            Atlas_path = Atlaspath
        else:
            Atlas_list = [str(path) for path in list(p.glob('**/*.mrc')) if Atlaspath in path.parts]
            Atlas_path = str(Atlas_list[0])

        with mrcfile.open(Atlas_path) as mrc:
            Atlas=mrc.data[:]
    elif Atlaspath[-4:] == "tiff":
        if "/" in Atlaspath:
            Atlas_path = Atlaspath
        else:
            Atlas_list = [str(path) for path in list(p.glob('**/*.tiff')) if Atlaspath in path.parts]
            Atlas_path = str(Atlas_list[0])
        Atlas=tifffile.imread(Atlas_path)
    else:
        assert "/" in Atlaspath
        print(Atlaspath)
        Atlas = stitch_atlas(Atlaspath)
        print(Atlas)
        
        
    print(Atlas.shape)

    return x, y, df, Locations_rot, Atlas

def calc_distance(x1,y1, x2,y2):
    distance = ((x2-x1)**2+(y2-y1)**2)*0.5
    return distance

def contrast_normalization(arr_bin, tile_size = 128):
    '''
    Computes the minimum and maximum contrast values to use
    by calculating the median of the 2nd/98th percentiles
    of the mic split up into tile_size * tile_size patches.
    :param arr_bin: the micrograph represented as a numpy array
    :type arr_bin: list
    :param tile_size: the size of the patch to split the mic by 
        (larger is faster)
    :type tile_size: int
    '''
    ny,nx = arr_bin.shape
    # set up start and end indexes to make looping code readable
    tile_start_x = np.arange(0, nx, tile_size)
    tile_end_x = tile_start_x + tile_size
    tile_start_y = np.arange(0, ny, tile_size)
    tile_end_y = tile_start_y + tile_size
    num_tile_x = len(tile_start_x)
    num_tile_y = len(tile_start_y)
    
    # initialize array that will hold percentiles of all patches
    tile_all_data = np.empty((num_tile_y*num_tile_x, 2), dtype=np.float32)

    index = 0
    for y in range(num_tile_y):
        for x in range(num_tile_x):
            # cut out a patch of the mic
            arr_tile = arr_bin[tile_start_y[y]:tile_end_y[y], tile_start_x[x]:tile_end_x[x]]
            # store 2nd and 98th percentile values
            tile_all_data[index:,0] = np.percentile(arr_tile, 98)
            tile_all_data[index:,1] = np.percentile(arr_tile, 2)
            index += 1

    # calc median of non-NaN percentile values
    all_tiles_98_median = np.nanmedian(tile_all_data[:,0])
    all_tiles_2_median = np.nanmedian(tile_all_data[:,1])
    vmid = 0.5*(all_tiles_2_median+all_tiles_98_median)
    vrange = abs(all_tiles_2_median-all_tiles_98_median)
    extend = 1.5
    # extend vmin and vmax enough to not include outliers
    vmin = vmid - extend*0.5*vrange
    vmax = vmid + extend*0.5*vrange

    return vmin, vmax




# Handle high resolution displays:
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class NavigationToolbar(NavigationToolbar):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save')]


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = fig.add_subplot(111)
        self.canvas = fig.canvas
        fig.tight_layout(pad=0)
        fig.patch.set_facecolor('#FFFFFF00')
        #Adding a colorbar to the figure...struggled to redraw the colorbar 
        #self.ax1.colorbar = fig.colorbar

        


        super(MplCanvas, self).__init__(fig)
        
class SubplotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = fig.add_subplot(111)
        self.canvas = fig.canvas
        fig.tight_layout(pad=0)
        fig.patch.set_facecolor('#FFFFFF00')
        self.ax2 = inset_axes(self.ax1,"25%","25%", loc= "upper right", borderpad=0)
        self.ax2.set_aspect('equal')



        super(SubplotCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        weite = 4
        self.setStyleSheet('background-color: white; color: black; font-family: Arial')

        sshFile="style_sheet.qss"
        with open(sshFile,"r") as fh:
            self.setStyleSheet(fh.read())
        #Create the plot areas
        
        self.sc = MplCanvas(self, width=weite, height=weite, dpi=200)

        self.Mic = SubplotCanvas(self, width=weite, height=weite, dpi=200)
        self.flagged = pd.DataFrame(columns=['x', 'y'])
        self.counter = 0


        # Create toolbars and input lines

        self.toolbar_Atlas = NavigationToolbar(self.sc, self)
        toolbar_Mic = NavigationToolbar(self.Mic, self)
        self.input_xml = QtWidgets.QLineEdit(self, width=3)
        self.label_offset = QtWidgets.QLabel(self, text="Offset x,y (µm):")
        
        
        
        
        
        self.input_offsetx = QtWidgets.QLineEdit(self, width=1)
        self.label_Atlas = QtWidgets.QLabel(self, text="Name of the Atlas image (.tiff or .mrc):")
        self.label_Atlas.setToolTip("The atlas image has to be in the directory of the micrographs. It is sufficient to copy the image file.")
        self.input_Atlas = QtWidgets.QLineEdit(self, width=1)
        self.input_Atlas.setToolTip("The atlas image has to be in the directory of the micrographs. It is sufficient to copy the image file.")
        self.input_offsety = QtWidgets.QLineEdit(self, width=1)
        self.label_angle = QtWidgets.QLabel(self, text="Angle of atlas rotation (°):")
        self.input_angle = QtWidgets.QLineEdit(self, width=1)
        self.label_mm = QtWidgets.QLabel(self, text="Extent of atlas (µm):")
        self.input_mm= QtWidgets.QLineEdit(self, width=1)
        self.label_xml = QtWidgets.QLabel(self, text='Path to the data set:')
        self.label_xml.setToolTip("""This directory should contain the atlas image and the micrographs and meta data files.
        If you use .xml as meta file, only those micrographs and meta files in a sub directory called Data will be considered.""")
        self.input_xml.setToolTip("""This directory should contain the atlas image and the micrographs and meta data files.
        If you use .xml as meta file, only those micrographs and meta files in a sub directory called Data will be considered.""")
        self.label_xy = QtWidgets.QLabel(self, text="        ")
        self.label_error = QtWidgets.QLabel(self, text="")
        self.FFT_box = QtWidgets.QCheckBox(text="Calculate FFT")
        self.label_bin = QtWidgets.QLabel(self, text="Binning factor:")
        self.input_bin= QtWidgets.QLineEdit(self, width=2)
        self.Scale_box = QtWidgets.QCheckBox(text="Show scale")
        self.input_pix = QtWidgets.QLineEdit(self, width=2)
        self.label_pix = QtWidgets.QLabel(self, text="Pixel size (Å)")
        self.input_length = QtWidgets.QLineEdit(self, width=2)
        self.label_scale = QtWidgets.QLabel(self, text="Scale length (Å)")
        self.input_res = QtWidgets.QLineEdit(self, width=2)
        self.label_res = QtWidgets.QLabel(self, text="FFT resolution ring (Å)")
        self.dummy = QtWidgets.QLabel(self, text="")
        flag_label = QtWidgets.QLabel(self, text="Mark locations on the atlas: ")
        
        self.label_Dataset = QtWidgets.QLabel(self, text="Data set options")
        self.label_Atlas_alignment = QtWidgets.QLabel(self, text="Atlas alignment")
        self.label_FS_options = QtWidgets.QLabel(self, text="FFT, scalebar and binning options")
        
        self.label_Dataset.setStyleSheet("font-weight: bold;")
        self.label_Atlas_alignment.setStyleSheet("font-weight: bold;")
        self.label_FS_options.setStyleSheet("font-weight: bold;")
        
        self.mrc_button = QtWidgets.QRadioButton(self, text=".mrc")
        self.tif_button = QtWidgets.QRadioButton(self, text=".tiff")
        self.label_tif_mrc = QtWidgets.QLabel(self, text="Micrograph file type")
        tif_mrc = QtWidgets.QButtonGroup(self)
        tif_mrc.addButton(self.mrc_button)
        tif_mrc.addButton(self.tif_button)
        tif_mrc.setExclusive(True)

        self.xml_button = QtWidgets.QRadioButton(self, text=".xml")
        self.mdoc_button = QtWidgets.QRadioButton(self, text=".mdoc")
        self.label_xml_mdoc = QtWidgets.QLabel(self, text="Meta data file type")
        xml_mdoc = QtWidgets.QButtonGroup(self)
        xml_mdoc.addButton(self.xml_button)
        xml_mdoc.addButton(self.mdoc_button)
        xml_mdoc.setExclusive(True)
        
        # Create a Angle_slider for displaying the slider's value
        self.angle_slider_label = QtWidgets.QLabel("Angle (°):")  # Initial label text

        # Create a QSlider instance
        self.angle_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.angle_slider.setRange(0, 360)  # Set the range from 0 to 360
        self.angle_slider.setValue(84)       # Initial value
        
        
        self.angle_spinbox = QtWidgets.QDoubleSpinBox()
        self.angle_spinbox.setRange(0, 360)
        self.angle_spinbox.setSingleStep(1)  # Set the step size to 1
        self.angle_spinbox.setValue(84)  # Initial value
        
        self.angle_spinbox.valueChanged.connect(lambda value: self.angle_slider.setValue(int(value)))
        self.angle_slider.valueChanged.connect(self.angle_spinbox.setValue)
        
        # Create a Extend_slider for displaying the slider's value
        self.extend_slider_label = QtWidgets.QLabel("Atlas extension (µm): ")  # Initial label text
        


        # Create a QSlider instance
        self.extend_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.extend_slider.setRange(800, 1200)  # Set the range from 0 to 360
        self.extend_slider.setValue(910)       # Initial value
        
        self.extend_spinbox = QtWidgets.QDoubleSpinBox()
        self.extend_spinbox.setRange(800, 1200)
        self.extend_spinbox.setSingleStep(1)  # Set the step size to 1
        self.extend_spinbox.setValue(910)  # Initial value
        
        self.extend_spinbox.valueChanged.connect(lambda value: self.extend_slider.setValue(int(value)))
        self.extend_slider.valueChanged.connect(self.extend_spinbox.setValue)
        
        # Create a Extend_slider for displaying the slider's value
        self.offset_x_slider_label = QtWidgets.QLabel("Offset in x (µm): ")  # Initial label text

        # Create a QSlider instance
        self.offset_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.offset_x_slider.setRange(-100, 100)  # Set the range
        self.offset_x_slider.setValue(0)       # Initial value
        
        self.offset_x_spinbox = QtWidgets.QDoubleSpinBox()
        self.offset_x_spinbox.setRange(-100, 100)
        self.offset_x_spinbox.setSingleStep(1)  # Set the step size to 1
        self.offset_x_spinbox.setValue(0)  # Initial value
        
        self.offset_x_spinbox.valueChanged.connect(lambda value: self.offset_x_slider.setValue(int(value)))
        self.offset_x_slider.valueChanged.connect(self.offset_x_spinbox.setValue)
        # Create a Extend_slider for displaying the slider's value
        self.offset_y_slider_label = QtWidgets.QLabel("Offset in y (µm):")  # Initial label text

        # Create a QSlider instance
        self.offset_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.offset_y_slider.setRange(-100, 100)  # Set the range
        self.offset_y_slider.setValue(0)       # Initial value
        
        self.offset_y_spinbox = QtWidgets.QDoubleSpinBox()
        self.offset_y_spinbox.setRange(-100, 100)
        self.offset_y_spinbox.setSingleStep(1)  # Set the step size to 1
        self.offset_y_spinbox.setValue(0)  # Initial value
        
        self.offset_y_spinbox.valueChanged.connect(lambda value: self.offset_y_slider.setValue(int(value)))
        self.offset_y_slider.valueChanged.connect(self.offset_y_spinbox.setValue)
        
        plot_button = QtWidgets.QPushButton(parent=self, text="")
        plot_button.setIcon(QtGui.QIcon('CryoCrane_logo.png'))
        
        plot_button.setToolTip("Plot the Atlas and foilhole locations")
        plot_button.setIconSize(QtCore.QSize(128, 128))
        
        align_button = QtWidgets.QPushButton(parent=self, text="align atlas")
        flag_good = QtWidgets.QPushButton(parent=self, text="Good micrograph")
        flag_bad = QtWidgets.QPushButton(parent=self, text="Bad micrograph")
        
        #Default values
        self.input_mm.setText("910")
        self.input_angle.setText("84.5")
        self.input_offsetx.setText("-14")
        self.input_offsety.setText("2")
        self.input_bin.setText("2")
        self.input_Atlas.setText("Atlas_1.mrc")
        #Default values
        

        #Set the Layout
        

        self.setWindowTitle("CryoCrane 1.3 - Correlate atlas and exposures")
        self.setWindowIcon(QtGui.QIcon('CryoCrane_logo.png'))
        
        #Toolbars and canvas
        layout1 = QtWidgets.QVBoxLayout()
        layout0 = QtWidgets.QGridLayout()
        layout0.addWidget(self.toolbar_Atlas,0,0)
        layout0.addWidget(toolbar_Mic,0,1)
        layout1.addLayout( layout0 )
        
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.sc)
        layout2.addWidget(self.Mic)



        layout1.addLayout( layout2 )
        
        layout_X = QtWidgets.QGridLayout()

            
        layout_X.addWidget(flag_label,0,0, QtCore.Qt.AlignRight)
        layout_X.addWidget(flag_good,0,3, QtCore.Qt.AlignRight)
        layout_X.addWidget(flag_bad,0,4, QtCore.Qt.AlignRight)
        for i in range(5,9):        
            layout_X.addWidget(self.dummy,0,i, QtCore.Qt.AlignRight)
        layout_X.addWidget(self.label_xy,0,9, QtCore.Qt.AlignRight)
        
        layout1.addLayout( layout_X )
        
        #Buttons below the plot area
        
        layout3 = QtWidgets.QGridLayout()
        
        # Headings
        layout3.addWidget(plot_button,2,0,4,1)
        layout3.addWidget(self.label_Dataset,1,1,1,3, QtCore.Qt.AlignCenter)
        layout3.addWidget(self.label_Atlas_alignment,1,4,1,2, QtCore.Qt.AlignCenter)
        layout3.addWidget(self.label_FS_options,1,7,1,3, QtCore.Qt.AlignCenter)
        
        
        #Data set options
        layout3.addWidget(self.label_xml,2,1)
        layout3.addWidget(self.input_xml,3,1,1,3)
        
        layout3.addWidget(self.label_tif_mrc,4,1)
        layout3.addWidget(self.mrc_button,4,2)
        layout3.addWidget(self.tif_button,4,3)
        layout3.addWidget(self.label_xml_mdoc,5,1)
        layout3.addWidget(self.xml_button,5,2)
        layout3.addWidget(self.mdoc_button,5,3)
        layout3.addWidget(self.label_Atlas,6,1)
        layout3.addWidget(self.input_Atlas,6,2)
        
        #Atlas alignment options

        layout3.addWidget(self.angle_slider_label,2,4)
        layout3.addWidget(self.angle_slider,2,6)
        layout3.addWidget(self.angle_spinbox,2,5)
        layout3.addWidget(self.extend_slider_label,3,4)
        layout3.addWidget(self.extend_slider,3,6)
        layout3.addWidget(self.extend_spinbox,3,5)
        
        layout3.addWidget(self.offset_x_slider_label,4,4)
        layout3.addWidget(self.offset_x_slider,4,6)
        layout3.addWidget(self.offset_x_spinbox,4,5)                                         

        layout3.addWidget(self.offset_y_slider_label,5,4)
        layout3.addWidget(self.offset_y_spinbox,5,5)                                          
        layout3.addWidget(self.offset_y_slider,5,6)
        layout3.addWidget(align_button,6,4)
        layout3.addWidget(self.label_error,7,8,QtCore.Qt.AlignRight)

        #FFT and scalebar options
        layout3.addWidget(self.FFT_box,2,7,QtCore.Qt.AlignLeft)
        layout3.addWidget(self.label_bin,2,8)
        layout3.addWidget(self.input_bin,2,9,1,1)
        layout3.addWidget(self.Scale_box,3,7,QtCore.Qt.AlignLeft)
        layout3.addWidget(self.label_pix,3,8,QtCore.Qt.AlignLeft)
        layout3.addWidget(self.input_pix,3,9,QtCore.Qt.AlignLeft)
        layout3.addWidget(self.label_scale,4,7,QtCore.Qt.AlignLeft)
        layout3.addWidget(self.input_length,4,8,QtCore.Qt.AlignLeft)
        layout3.addWidget(self.label_res,5,7,QtCore.Qt.AlignLeft)
        layout3.addWidget(self.input_res,5,8,QtCore.Qt.AlignLeft)


        layout1.addLayout( layout3 )
        
        # Reset the appearance of the GUI
        
        self.input_pix.setDisabled(True)
        self.input_res.setDisabled(True)
        self.input_length.setDisabled(True)

        
        
        self.sc.ax1.set_axis_off()
        self.Mic.ax1.set_axis_off()
        self.Mic.ax2.set_axis_off()
        self.Mic.draw()
        self.sc.draw()

        #Actions

        plot_button.clicked.connect(self.plot_Data)
        align_button.clicked.connect(self.realign)
        flag_good.clicked.connect(self.flag_good_hole)
        flag_bad.clicked.connect(self.flag_bad_hole)
        
        self.angle_slider.valueChanged.connect(self.realign)
        self.angle_spinbox.valueChanged.connect(self.realign)
        self.extend_slider.valueChanged.connect(self.realign)
        self.offset_x_slider.valueChanged.connect(self.realign)
        self.offset_y_slider.valueChanged.connect(self.realign)
        self.Scale_box.clicked.connect(self.turn_on_pixel_input)

        self.sc.canvas.mpl_connect('button_press_event', self.onclick)
        
        #Shortcuts
        
        plot_button.setShortcut("Ctrl+P")
        flag_good.setShortcut("Ctrl+G")
        flag_bad.setShortcut("Ctrl+B")
        
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout1)
        self.setCentralWidget(widget)
        self.showMaximized() 

        self.show()
        

    def plot_Data(self):
        #Reads the data set and plots it. Is triggered by pressing the plot button
        try:
            load_mic_data(offsetx = float(self.input_offsetx.text()),
                          offsety = float(self.input_offsety.text()),
                          angle = float(self.input_angle.text()),
                          Micpath = self.input_xml.text(),
                          Atlaspath = self.input_Atlas.text(),
                          TIFF = self.tif_button.isChecked(),
                          MDOC = self.mdoc_button.isChecked()
                         )
            float(self.input_mm.text())
        except:
            self.label_error.setText("Error: Invalid path or atlas parameters")
            self.label_error.setStyleSheet("color: red;")
            self.Locations_rot = []


        else:
            x, y, df, self.Locations_rot, Atlas = load_mic_data(offsetx = float(self.input_offsetx.text()),
                                                                offsety = float(self.input_offsety.text()),
                                                                angle = float(self.input_angle.text()),
                                                                Micpath = self.input_xml.text(),
                                                                Atlaspath = self.input_Atlas.text(),
                                                                TIFF = self.tif_button.isChecked(),
                                                                MDOC = self.mdoc_button.isChecked()
                                                               )

            self.label_error.setText("")

            self.sc.ax1.cla()
            
            scale = float(self.input_mm.text())
            
            self.sc.ax1.imshow(Atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale])
            exposures = self.sc.ax1.scatter(x, y, c = df, s = 0.5, cmap = "GnBu")
            #self.cbar = self.sc.ax1.colorbar(exposures, orientation='vertical',location = "left",extend='both', shrink = 0.66, pad = 0.01, label = "defocus (µm)", cax = self.sc.ax2)
            #self.cbar.ax.tick_params(labelsize="xx-small")
            #self.cbar.set_label(label='defocus (µm)', size='xx-small', weight='bold')
            self.current_hole = self.sc.ax1.scatter(scale,scale, c = "red", s = 0.8, alpha = 0.0)
            self.sc.ax1.set_axis_off()
            self.sc.draw()
            
            self.angle =  float(self.input_angle.text())/360*2*np.pi
            self.offset_x = float(self.input_offsetx.text())
            self.offset_y = float(self.input_offsety.text())
            self.df = df
            self.Atlas = Atlas
   
        return self.Locations_rot, self.angle, self.offset_x, self.offset_y, self.df, self.Atlas

    
    def realign(self):
        try:
            test = self.Atlas
            float(self.angle)
        except:
            self.label_error.setText("You should plot an atlas before aligning it")
            self.label_error.setStyleSheet("color: red;")
        else: 
            self.label_error.setText("")
            print(self.angle)

            #Calculate the unrotated coordinates
            # x_rot = np.cos(angle) * x - np.sin(angle) * y
            # y_rot = np.sin(angle) * x + np.cos(angle) * y
            # x = (x_rot + np.sin(angle) * y)/np.cos(angle)
            # y = (y_rot - np.sin(angle) * x)/ np.cos(angle)
            x_rot = self.Locations_rot[0]
            y_rot = self.Locations_rot[1]

            self.Locations_rot[0] = np.cos(-1*self.angle) * x_rot - np.sin(-1*self.angle) * y_rot
            self.Locations_rot[1] = np.sin(-1*self.angle) * x_rot + np.cos(-1*self.angle) * y_rot

            self.Locations_rot[0] -= self.offset_x
            self.Locations_rot[1] -= self.offset_y

            # Read the new shifts and angles

            self.offset_x = float(self.offset_x_slider.value())
            self.offset_y = float(self.offset_y_slider.value())
            #self.angle = float(self.angle_slider.value())/360*2*np.pi

            self.angle = float(self.angle_spinbox.value())/360*2*np.pi

            # Add the new shifts

            self.Locations_rot[0] += self.offset_x 
            self.Locations_rot[1] += self.offset_y         

            #Calculate the rotated coordinates
            # x_rot = np.cos(angle) * x - np.sin(angle) * y
            # y_rot = np.sin(angle) * x + np.cos(angle) * y
            x_unrot = self.Locations_rot[0]
            y_unrot = self.Locations_rot[1]
            self.Locations_rot[0] = np.cos(self.angle) * x_unrot - np.sin(self.angle) * y_unrot
            self.Locations_rot[1] = np.sin(self.angle) * x_unrot + np.cos(self.angle) * y_unrot
            print(self.Locations_rot[0])
            x,y = self.Locations_rot[0], self.Locations_rot[1]
            self.xlim = self.sc.ax1.get_xlim()
            self.ylim = self.sc.ax1.get_ylim()
            self.sc.ax1.cla()

            self.scale = float(self.extend_slider.value())
            scale = self.scale
            self.sc.ax1.imshow(self.Atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale])
            exposures = self.sc.ax1.scatter(x, y, c = self.df, s = 0.5, cmap = "GnBu")
            #self.cbar = self.sc.ax1.colorbar(exposures, orientation='vertical',location = "left",extend='both', shrink = 0.66, pad = 0.01, label = "defocus (µm)", cax = self.sc.ax2)
            #self.cbar.ax.tick_params(labelsize="xx-small")
            #self.cbar.set_label(label='defocus (µm)', size='xx-small', weight='bold')
            self.current_hole = self.sc.ax1.scatter(scale,scale, c = "red", s = 0.8, alpha = 0.0)
            self.sc.ax1.set_axis_off()
            self.sc.ax1.set_xlim(self.xlim)
            self.sc.ax1.set_ylim(self.ylim)
            self.sc.draw()


            #self.angle_slider_label.setText(f"Angle: {self.angle*360/2/np.pi} °")
            #self.extend_slider_label.setText(f"Alas extension: {self.scale} µm")
            #self.offset_y_slider_label.setText(f"Offset in y: {self.offset_y} µm")
            #self.offset_x_slider_label.setText(f"Offset in x: {self.offset_x} µm")


            return self.Locations_rot, self.offset_x, self.offset_y, self.angle, self.scale
        
    def turn_on_pixel_input(self):
        self.input_pix.setEnabled(self.Scale_box.isChecked())
        self.input_res.setEnabled(self.Scale_box.isChecked())
        self.input_length.setEnabled(self.Scale_box.isChecked())
        
    def turn_on_fft_input(self):
        self.input_bin.setEnabled(self.FFT_box.isChecked())
    
    def onclick(self, event):
        print([event.xdata, event.ydata])
        self.x = event.xdata
        self.y = event.ydata
        x1,y1 = float(self.x), float(self.y)
        dataframe = self.Locations_rot

        dataframe = dataframe.assign(distance = [calc_distance(x1,y1,x2,y2) for x2,y2 in zip(list(dataframe.iloc[:,1]),list(dataframe.iloc[:,2]))])

        minimum = min(dataframe.iloc[:,-1])
        print(minimum)
        hits = dataframe[dataframe.iloc[:,-1] == minimum]
        print(hits.iloc[0,0])
        if len(hits) == 1:
            self.current_hole.remove()
            self.x_hole, self.y_hole = hits.iloc[0,1],hits.iloc[0,2]
            self.current_hole = self.sc.ax1.scatter(hits.iloc[0,1],hits.iloc[0,2], c = "red", s =2, alpha = 0.7)

            self.Mic.ax1.cla()
            self.Mic.ax2.cla()
            if self.tif_button.isChecked() == False:
                with mrcfile.open(hits.iloc[0,4]) as mrc:
                    Micrograph=mrc.data[:]
            else:
                Micrograph = tifffile.imread(hits.iloc[0,4])
            print(Micrograph.shape)
            if len(np.shape(Micrograph)) > 2: #summing movies
                #chunks = np.array_split(Micrograph, 16, axis=0)
                #result_chunks = movie_pool.map(sum_movie, chunks)
                #Micrograph = np.sum(result_chunks, axis=0)
                Micrograph = np.sum(Micrograph, axis=np.argmin(Micrograph.shape)) #sums over the smallest axis, should be time
                print(Micrograph.shape)

            try: 
                bin_factor = float(self.input_bin.text())
                Bin = rebin(Micrograph, (int(Micrograph.shape[0]/bin_factor), int(Micrograph.shape[1]/bin_factor)))
            except:
                self.label_error.setText("Error: Invalid binning factor")
                self.label_error.setStyleSheet("color: red;")
            else:
                self.label_error.setText(" ")
                bin_factor = float(self.input_bin.text())
                Pix_x= Micrograph.shape[1]/bin_factor
                Pix_y= Micrograph.shape[0]/bin_factor
                Bin = rebin(Micrograph, (int(Micrograph.shape[0]/bin_factor), int(Micrograph.shape[1]/bin_factor)))
                
                if self.FFT_box.isChecked()==True:


                        ft = np.fft.ifftshift(Bin)

                        ft = np.fft.fft2(ft)

                        #Thon = np.log(np.abs(np.fft.fftshift(ft)))
                        Thon = np.log(np.abs(np.fft.fftshift(ft)))
                        #Thon = rebin(Thon, (int(Thon.shape[0]/4), int(Thon.shape[1]/4)))
                        vmin,vmax = contrast_normalization(Thon)
                        print(vmin)
                        print(vmax)
                        self.Mic.ax2.imshow(Thon, cmap ="gray", extent=[-1,1,-1,1], filternorm= True, vmin=vmin, vmax=vmax)
                        
                self.Mic.ax1.imshow(Bin, cmap ="gray")
                   
                self.Mic.ax1.text(0.05*Pix_x, 0.05*Pix_y ,"appl. defocus {:.1f} µm".format(hits.iloc[0,3]), c ="white",size = "xx-small", horizontalalignment='left', verticalalignment='bottom')
                print(hits.iloc[0,3])

                if self.Scale_box.isChecked()==True:
                    try:
                        test = float(self.input_pix.text())
                        test = float(self.input_res.text())
                        test = float(self.input_length.text())

                    except:
                        self.label_error.setText("Invalid input for scale options")
                        self.label_error.setStyleSheet("color: red;")

                    else:
                        self.label_error.setText(" ")
                        offset = Pix_x*0.05
                        offsety = Pix_y*0.05

                        size = float(self.input_length.text())
                        pixelsize = float(self.input_pix.text())*bin_factor

                        x_marker = [Pix_x - size/pixelsize - offset,Pix_x - offset]
                        y_marker = [Pix_y-offsety,Pix_y-offsety]
                        print(x_marker)
                        print(y_marker)
                        self.Mic.ax1.plot(x_marker, y_marker, c = "white", linewidth = 1)
                        self.Mic.ax1.text(np.mean(x_marker), np.mean(y_marker)*0.99,"{:.0f} Å".format(round(size,0)), horizontalalignment='center', verticalalignment='bottom', c = "white", size = "xx-small")
                        
                        if self.FFT_box.isChecked()==True:
                            radius = (2*pixelsize)/float(self.input_res.text())
                            Res_ring = plt.Circle((0, 0), radius, color='w', fill=False, lw = 0.33)
                            self.Mic.ax2.add_patch(Res_ring)
                            self.Mic.ax2.text(0, radius*-1.03 ,"{:.1f} Å".format(float(self.input_res.text())), c ="white", fontsize=4, horizontalalignment='center', verticalalignment='top')
            self.Mic.ax1.set_axis_off()
            self.Mic.ax2.set_axis_off()
            self.Mic.draw()
     
            #self.Mic.ax1.set_title(hits.iloc[0,0].rsplit('/', 1)[-1])
            #self.label_xy.setText("X:{:.1f} µm, Y: {:.1f} µm".format(hits.iloc[0,1],hits.iloc[0,2]))
            self.label_xy.setText("{}".format(hits.iloc[0,4].rsplit('/', 1)[-1]))
        else: 
            print("Something is wrong with the coordinates")

        return self.x, self.y, self.x_hole, self.y_hole
    

    
    def flag_good_hole(self):
        try:
            self.x_hole, self.y_hole
        except:
            print("Click on the micrograph first.")
            
        else:
            
            self.flagged.loc[self.counter] = [self.x_hole, self.y_hole]
            print(self.flagged)
            Mark_ring = plt.Circle((self.flagged.iloc[self.counter,0],self.flagged.iloc[self.counter,1]), 1, color='gold', fill=False, lw = 1)
            self.sc.ax1.add_patch(Mark_ring)
            self.sc.draw()
            #self.sc.ax1.scatter(self.flagged.iloc[:,0],self.flagged.iloc[:,1], c = "green", s =4, alpha = 0.)
            self.counter += 1
        return self.counter, self.flagged
    
    def flag_bad_hole(self):
        try:
            self.x_hole, self.y_hole
        except:
            print("Click on the micrograph first.")
            
        else:
            self.flagged.loc[self.counter] = [self.x_hole, self.y_hole]
            print(self.flagged)
            Mark_ring = plt.Circle((self.flagged.iloc[self.counter,0],self.flagged.iloc[self.counter,1]), 1, color='saddlebrown', fill=False, lw = 1)

            self.sc.ax1.add_patch(Mark_ring)
            self.sc.draw()
            #self.sc.ax1.scatter(self.flagged.iloc[:,0],self.flagged.iloc[:,1], c = "green", s =4, alpha = 0.)
            self.counter += 1
        return self.counter, self.flagged
        
 


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
