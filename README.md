# CryoCrane
Correlate Atlas and Exposures - a GUI for the analysis of cryo-EM screening data


<video src="CryoCrane.mp4"></video>

## Descriptionz
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

## Data organization

The GUI works both with summed images and movies. However, plotting of movies takes a while.
For data sets recored with EPU CryoCrane searches the provided path for the provided Atlas filename and for directories named "Data". The first found Atlas file will be used to show the atlas. All micrographs and their .xml meta data files in any "Data" directory will be displayed as well.
For data recorded with SerialEM the given directory will be searched for .mdoc files and the provided atlas file.  


## Usage

After loading your dataset you should align the atlas and stage coordinates. The default files were determined for a Talos F200C microscope and might be different for your set up. It is convenient to first determine the rotation angle, then the extent of the atlas (correlate the hole spacings) and to finally set the x and y offsets. If you have found a set of parameters that suit your needs, you can set them as defaults by modifying the values after   "#Default values"-line. These will then be shown after starting the GUI. 

## Citation

If you found CryoCrane useful please cite: 
