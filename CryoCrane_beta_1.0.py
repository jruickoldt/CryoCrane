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



def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def get_xy_rotated(xml_file, offsetx, offsety, angle = 170):
    angle = angle/360*2*np.pi
    file = open(xml_file, "r")
    meta= file.read()
    file.close()
    
    Start = '<BeamShift xmlns:a="http://schemas.datacontract.org/2004/07/Fei.Types">'
    End = "</BeamShift>"
    Beamshift=meta[meta.index(Start)+len(Start):meta.index(End)]
    #Beam shift in µm?
    
    #Read the x and y coordinate from the xml file
    Start = "</B><X>"
    End = "</X>"
    
    x=float(meta[meta.index(Start)+len(Start):meta.index(End)])*1000*1000
    Start = "<a:_x>"
    End = "</a:_x><a:_y>"
    x_shift=float(Beamshift[Beamshift.index(Start)+len(Start):Beamshift.index(End)])
    x += x_shift
    x += offsetx
    Start = "<Y>"
    End = "</Y>"
    y=float(meta[meta.index(Start)+len(Start):meta.index(End)])*1000*1000
    
    Start = "<a:_y>"
    End = "</a:_y>"
    y_shift=float(Beamshift[Beamshift.index(Start)+len(Start):Beamshift.index(End)])
    y += y_shift
    y += offsety
    
    #Calculate the rotated coordinates
    x_rot = np.cos(angle) * x - np.sin(angle) * y
    y_rot = np.sin(angle) * x + np.cos(angle) * y

    #get the applied defocus: 
    Start = '<a:Key>AppliedDefocus</a:Key><a:Value i:type="b:double" xmlns:b="http://www.w3.org/2001/XMLSchema">'
    End = '</a:Value></a:KeyValueOfstringanyType></CustomData>'

    df=float(meta[meta.index(Start)+len(Start):meta.index(End)])*1000*1000
    
    
    values =[x_rot,y_rot,df]
    return values
    
def load_mic_data(angle, offsetx, offsety, Micpath,Atlaspath, TIFF = False):
    #Create an empty dictionary
    Data_rot = {}
    p = Path(Micpath)
    #Load the exposures should be located in a Data folder
    No_foilhole = [path for path in list(p.glob('**/*.xml')) if ("Data" in path.parts or "Images" in path.parts)]
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
    if TIFF == False:
        Locations_rot = Locations_rot.assign(JPG = [w.replace(".xml",".mrc") for w in list(Locations_rot.iloc[:,0])])

    else:
        Locations_rot = Locations_rot.assign(JPG = [w.replace(".xml",".tiff") for w in list(Locations_rot.iloc[:,0])])
    
    #open the Atlas, which should be located somewhere in the Micpath
    if Atlaspath[-3:] == "mrc":
        Atlas_list = [str(path) for path in list(p.glob('**/*.mrc')) if Atlaspath in path.parts]
        Atlas_path = str(Atlas_list[0])
        with mrcfile.open(Atlas_path) as mrc:
            Atlas=mrc.data[:]
    if Atlaspath[-4:] == "tiff":
        Atlas_list = [str(path) for path in list(p.glob('**/*.tiff')) if Atlaspath in path.parts]
        Atlas_path = str(Atlas_list[0])
        Atlas=tifffile.imread(Atlas_path)
        
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
        fig.patch.set_facecolor('white')
        #Adding a colorbar to the figure...struggled to redraw the colorbar 
        #self.ax1.colorbar = fig.colorbar

        


        super(MplCanvas, self).__init__(fig)
        
class SubplotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = fig.add_subplot(111)
        self.canvas = fig.canvas
        fig.tight_layout(pad=0)
        fig.patch.set_facecolor('white')
        self.ax2 = inset_axes(self.ax1,"25%","25%", loc= "upper right", borderpad=0)
        self.ax2.set_aspect('equal')



        super(SubplotCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        weite = 4
        self.setStyleSheet('background-color: white; color: black; font-family: Arial')
        
        #Create the plot areas
        
        self.sc = MplCanvas(self, width=weite, height=weite, dpi=200)

        self.Mic = SubplotCanvas(self, width=weite, height=weite, dpi=200)


        # Create toolbars and input lines
        toolbar_Atlas = NavigationToolbar(self.sc, self)
        toolbar_Mic = NavigationToolbar(self.Mic, self)
        self.input_xml = QtWidgets.QLineEdit(self, width=3)
        self.label_offset = QtWidgets.QLabel(self, text="Offset x,y (µm):")
        
        
        
        
        
        self.input_offsetx = QtWidgets.QLineEdit(self, width=1)
        self.label_Atlas = QtWidgets.QLabel(self, text="Name of the Atlas image (.tiff or .mrc):")
        self.input_Atlas = QtWidgets.QLineEdit(self, width=1)
        self.input_offsety = QtWidgets.QLineEdit(self, width=1)
        self.label_angle = QtWidgets.QLabel(self, text="Angle of atlas rotation (°):")
        self.input_angle = QtWidgets.QLineEdit(self, width=1)
        self.label_mm = QtWidgets.QLabel(self, text="Extent of atlas (µm):")
        self.input_mm= QtWidgets.QLineEdit(self, width=1)
        self.label_xml = QtWidgets.QLabel(self, text='Path to the data set:')
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
        
        self.label_Dataset = QtWidgets.QLabel(self, text="Data set options")
        self.label_Atlas_alignment = QtWidgets.QLabel(self, text="Atlas alignment")
        self.label_FS_options = QtWidgets.QLabel(self, text="FFT, scalebar and binning options")
        
        self.label_Dataset.setStyleSheet("font-weight: bold;")
        self.label_Atlas_alignment.setStyleSheet("font-weight: bold;")
        self.label_FS_options.setStyleSheet("font-weight: bold;")
        
        self.mrc_button = QtWidgets.QRadioButton(self, text=".mrc")
        self.tif_button = QtWidgets.QRadioButton(self, text=".tiff")
        tif_mrc = QtWidgets.QButtonGroup(self)
        tif_mrc.addButton(self.mrc_button)
        tif_mrc.addButton(self.tif_button)
        tif_mrc.setExclusive(True)
        
        
        
        plot_button = QtWidgets.QPushButton(parent=self, text="")
        plot_button.setIcon(QtGui.QIcon('CryoCrane_logo.png'))
        
        plot_button.setToolTip("Plot the Atlas and foilhole locations")
        plot_button.setIconSize(QtCore.QSize(128, 128))
        
        #Default values
        self.input_mm.setText("910")
        self.input_angle.setText("84.5")
        self.input_offsetx.setText("-14")
        self.input_offsety.setText("2")
        self.input_bin.setText("2")
        self.input_Atlas.setText("Atlas_1.mrc")
        #Default values
        

        #Set the Layout
        

        self.setWindowTitle("CryoCrane 1.1 - Correlate atlas and exposures")
        self.setWindowIcon(QtGui.QIcon('CryoCrane_logo.png'))
        
        #Toolbars and canvas
        layout1 = QtWidgets.QVBoxLayout()
        layout0 = QtWidgets.QGridLayout()
        layout0.addWidget(toolbar_Atlas,0,0)
        layout0.addWidget(toolbar_Mic,0,1)
        layout1.addLayout( layout0 )
        
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.sc)
        layout2.addWidget(self.Mic)


        layout1.addLayout( layout2 )
        
        #Buttons below the plot area
        
        layout3 = QtWidgets.QGridLayout()
        
        # Headings
        layout3.addWidget(plot_button,2,0,4,1)
        layout3.addWidget(self.label_Dataset,1,1,1,3, QtCore.Qt.AlignCenter)
        layout3.addWidget(self.label_Atlas_alignment,1,4,1,2, QtCore.Qt.AlignCenter)
        layout3.addWidget(self.label_FS_options,1,6,1,3, QtCore.Qt.AlignCenter)
        layout3.addWidget(self.label_xy,1,8, QtCore.Qt.AlignRight)
        
        #Data set options
        layout3.addWidget(self.label_xml,2,1)
        layout3.addWidget(self.input_xml,3,1,1,2)
        layout3.addWidget(self.mrc_button,4,1)
        layout3.addWidget(self.tif_button,4,2)
        layout3.addWidget(self.label_Atlas,5,1)
        layout3.addWidget(self.input_Atlas,5,2)
        
        #Atlas alignment options

        layout3.addWidget(self.label_angle,2,4)
        layout3.addWidget(self.input_angle,2,5)
        layout3.addWidget(self.label_mm,3,4)
        layout3.addWidget(self.input_mm,3,5)
        
        layout3.addWidget(self.label_offset,4,4)
        layout3.addWidget(self.input_offsetx,5,4)

        layout3.addWidget(self.input_offsety,5,5)
        layout3.addWidget(self.label_error,7,8,QtCore.Qt.AlignRight)

        #FFT and scalebar options
        layout3.addWidget(self.FFT_box,2,6,QtCore.Qt.AlignLeft)
        layout3.addWidget(self.label_bin,2,7)
        layout3.addWidget(self.input_bin,2,8,1,1)
        layout3.addWidget(self.Scale_box,3,6,QtCore.Qt.AlignLeft)
        layout3.addWidget(self.label_pix,3,7,QtCore.Qt.AlignLeft)
        layout3.addWidget(self.input_pix,3,8,QtCore.Qt.AlignLeft)
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
        self.Scale_box.clicked.connect(self.turn_on_pixel_input)

        self.sc.canvas.mpl_connect('button_press_event', self.onclick)
        
        #Shortcuts
        
        plot_button.setShortcut("Ctrl+P")



        
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
                          TIFF = self.tif_button.isChecked())
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
                                                                TIFF = self.tif_button.isChecked())

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
   
        return self.Locations_rot
        
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
        print(hits)
        if len(hits) == 1:
            self.current_hole.remove()
            self.current_hole = self.sc.ax1.scatter(hits.iloc[0,1],hits.iloc[0,2], c = "red", s =2, alpha = 0.7)

            self.Mic.ax1.cla()
            self.Mic.ax2.cla()
            if self.tif_button.isChecked() == False:
                with mrcfile.open(hits.iloc[0,4]) as mrc:
                    Micrograph=mrc.data[:]
            else:
                Micrograph = tifffile.imread(hits.iloc[0,4])
            if len(np.shape(Micrograph)) > 2: #summing movies
                Micrograph = np.sum(Micrograph, axis=np.argmin(Micrograph.shape)) #sums over the smallest axis, should be time

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
     
            
            self.label_xy.setText("X:{:.1f} µm, Y: {:.1f} µm".format(hits.iloc[0,1],hits.iloc[0,2]))
        else: 
            print("Something is wrong with the coordinates")

        return self.x, self.y
        
 


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
