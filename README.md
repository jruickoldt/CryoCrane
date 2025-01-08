# CryoCrane
Correlate Atlas and Exposures - a GUI for the analysis of cryo-EM screening data

![alt text](https://github.com/jruickoldt/CryoCrane/blob/main/CryoCrane_example_1.png?raw=true)


<video src="CryoCrane.mp4"></video>

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

The progam can be fetched from github and started by the following commands:
```
cd /path/to/your/desired/directory
git clone https://github.com/jruickoldt/CryoCrane/
cd CryoCrane
conda activate CryoCrane
python3 CryoCrane_1.3.py
```

## Usage
### Data organisation


For data sets recored with EPU CryoCrane searches the provided path for the provided Atlas filename (or the absolute path) and for directories named "Data". The first found Atlas file will be used to show the atlas. All micrographs and their .xml meta data files in any "Data" directory will be displayed as well.
For data sets recorded with SerialEM the given directory and its subdirectories will be searched for .mdoc files and the respective image files. You can either specify the absolute path to the atlas image or simply its file name, if it is contained in the same directory as the data set.  
The GUI works both with summed images and movies. However, plotting of movies takes a while.


### Aligning stage and atlas coordinates

After loading your dataset you should align the atlas and stage coordinates. The default files were determined for a Talos F200C microscope and might be different for your set up. It is convenient to first determine the rotation angle, then the extent of the atlas (correlate the hole spacings) and to finally set the x and y offsets. If you have found a set of parameters that suit your needs, you can set them as defaults by modifying the values after "#Default values"-line. These will then be shown after starting the GUI. 

### Analysing the grid

You can zoom, pan and move around on the atlas image. Upon clicking on the atlas on the left side the micrograph at the nearest position will be shown. You can add a scalebar and an 2D-FFT to that micrograph in the lower panel. The micrograph can be zoomed and saved in various formats with the navigation toolbar. 


## Citation

If you found CryoCrane useful please cite the following article:

J. Ruickoldt and P. Wendler (2025). Acta Cryst. F81, https://doi.org/10.1107/S2053230X25000081

 
