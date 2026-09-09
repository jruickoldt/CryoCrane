# CryoCrane Tutorial - Basics

This tutorial will teach you how to use CryoCrane to supervise, evaluate and improve your data collection on-the-fly. The best time to start CryoCrane is after you have set up a data collection on a few squares. You can use CryoCrane to evaluate the first micrographs and let it guide you to the most promising areas of your grid. 

As a first step, we will install CryoCrane2

## Installation

Create a clean python environment (tested with Python 3.10) and install the following packages manually: 

```
 conda create -n CryoCrane python=3.10
 conda activate CryoCrane
```

The program can be fetched from github and started by the following commands:
```
cd /path/to/your/desired/directory
git clone https://github.com/jruickoldt/CryoCrane/
cd CryoCrane

conda activate CryoCrane
pip install .
python3 src/CryoCrane2.py
```

To start the program activate the environment and run the main program by:

```
conda activate CryoCrane
python3 src/CryoCrane2.py
```

## Starting a session

 After the installation you should see something like this:

<img src="https://github.com/jruickoldt/CryoCrane/blob/main/CryoCrane2_overview.png?raw=true"
     alt="CryoCrane GUI"
     width="800"
     align="center" />


To load data into CryoCrane you need to specify the path to the folder containing the exposures (preferably the summed images) and the path to the atlas (.mrc or .tiff). See the [documentation](./documentation.md#Data-organisation) for more details about the expected data structure.
<img src="https://github.com/jruickoldt/CryoCrane/blob/main/CryoCrane_logo.png?raw=true"
     alt="CryoCrane logo"
     width="120"
     align="right" />
     
The paths can be either pasted into the field or you can use your file browser via the "browse" button. After specifying the information, click on the CryoCrane: 



All existing exposures will be loaded into CryoCrane. Furthermore, CryoCrane determines the position of the grid squares on the atlas (takes around 40 s). 

Afterwards, the left image will show the atlas. The position of the exposures will be shown as dots colored by the applied defocus (color code is shown at the bottom of the window). Upon changing the dropdown menu at "Colour by:" to grid squares. The atlas will be shown with the overlayed grid squares in red. 

## Inspecting micrographs

Upon clicking on the atlas the corresponding micrograph closest to the click will be shown on the left. The micrograph image can be easily modified by clicking on "Micrograph options" below the micrograph. You can specify the binning factor, pixel size and whether the power spectrum should be shown. Micrographs can be saved by clicking on the floppy disk symbol on the top. You can also perform measurements on the micrographs (more details [here](./documentation.md#Micrograph-options)). 

## Aligning exposures and atlas

### Global alignment
The alignment process has two steps. First globally for all exposures, you have to specify a rotation angle (usually around 0° for Titan Krios and 90° for Talos microscopes) and a x and y offset. Furthermore, you have to specify the scale of the atlas. This is very easy by comparing the spacing of the exposures and the holes on the atlas. Here, usually a value around 900 µm works well. 

### Grid square alignment
After the global alignment, you will notice that exposures from different grid squares have different offsets. To account for this you can click "cluster and auto align grid squares". This function will cluster the exposures into grid squares and apply an offset to same. If you are not satisfied with the automated procedure, you can also modify the results manually. More details for this [here.](./documentation.md#.Cluster-exposures-in-grid-squares). After clustering of the exposures a new option "cluster" will appear in the "Colour by" menu. By selecting this the exposures will be colored according to the grid square cluster that they were assigned to.







