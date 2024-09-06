# CryoCrane
Correlate Atlas and Exposures - a GUI for the analysis of cryo-EM screening data

## Description
Screening of cryo EM samples is essential for the generation of high-resolution cryo-EM data. Often, it is cumbersome to correlate the appearance of specific grid squares and micrograph quality. Here, we present a visualization tool for cryo-EM screening data: CryoCrane. It is aimed to provide an intuitive way of visualization of micrographs and to speed up data analysis. 

## Installation

Create a clean python environment (tested with Python 3.10) and install the following packages: 

```
conda create -n CryoCrane
conda activate CryoCrane

conda install matplotlib
conda install tifffile
conda install pandas
pip install mrcfile
pip install PyQt5

```

After downloading CryoCrane.py and the logo file, the program can be started by navigating to the directory containing the source code, activating the python environment and typing: 
```
cd /path/to/CryoCrane.py
conda activate CryoCrane
python3 CryoCrane.py
```

## Data organization
The provided path should contain the Atlas_1.mrc and the directories containing the summed images (not the movies!) outputted by EPU. GOLM searches the provided path for an "Atlas_1.mrc" file and for directories named "Data". The first found Atlas_1.mrc file will be used to show the atlas. All micrographs and their .xml meta data files in any "Data" directory will be displayed as well.   


## Usage

The rotation angle, extent of the atlas image and x and y shifts have to be determined empirically and are different for every microscope. If you have found a set of parameters that suit your needs, you can set them as defaults by modifying the values after   "#Default values"-line. These will then be shown after starting the GUI. 

## Citation

If you found CryoCrane useful please cite: 
