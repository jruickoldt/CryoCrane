<img src="https://github.com/jruickoldt/CryoCrane/blob/main/CryoCrane_logo.png?raw=true"
     alt="CryoCrane logo"
     width="120"
     align="right" />

# CryoCrane
Correlate Atlas and Exposures – a GUI for the analysis of cryo-EM screening data


## Description
Screening of cryo EM samples is essential for the generation of high-resolution cryo-EM data. Often, it is cumbersome to correlate the appearance of specific grid squares and micrograph quality. Here, we present a visualization tool for cryo-EM screening data: CryoCrane. It is aimed to provide an intuitive way of visualization of micrographs and to speed up data analysis. 
 
CryoCrane 2.0 now incorporates the CryoPike suite as well. CryoPike is a set of programs for the automated scoring of cryo-EM exposures. The CryoPike networks were trained on a diverse data set of cryo-EM micrographs rated by experts based on the presence of contaminations, aggregation or crystalline ice, on the particle distribution and image contrast. You can also train your own neural-network within CryoCrane allowing the prediction of suitable grid areas for data acquisition.   

## Disclaimer

The authors takes no liability and grants no warranty for the usage of this program. It is not advised to run CryoCrane on the computer controlling the microscope. Although the program was developed with greatest care, out-of-memory issue might occur upon unintended usage. 

## Installation

Create a clean python environment (tested with Python 3.10) and install the following packages: 

```
 conda env create -f CryoCrane.yml 

```

If that does not work, create a clean python environment (tested with Python 3.10) and install the following packages manually: 

```
 conda create -n CryoCrane python=3.10
 conda activate CryoCrane

 pip install matplotlib
 pip install torch torchvision
 pip install tqdm
 pip install mrcfile
 pip install PyQt5
 pip install pandas
 pip install scikit-learn
 pip install tifffile
 pip install "numpy<2"
```

The progam can be fetched from github and started by the following commands:
```
cd /path/to/your/desired/directory
git clone https://github.com/jruickoldt/CryoCrane/
cd CryoCrane
conda activate CryoCrane
python3 CryoCrane2.py
```

This installation has been succesfully tested on Windows 11 and MacOS Sonoma. 

## Usage
### Data organisation
#### EPU data collections

The following data structure works best for EPU data sets for the summed images. 
```bash
───your_data_set
   ├───your_data_set
   │   ├───Images-Disc1
   │   │   ├───GridSquare_23261350
   │   │   │   ├───Data
   │   │   │   │ └─── *.mrc/.tiff (not the _Fractions.mrc/.tiff)
   │   │   │   └───FoilHoles
   │   │   └───GridSquare_23261372
   │   │       ├───Data
   │   │       │ └─── *.mrc/.tiff (not the _Fractions.mrc/.tiff)
   │   │       └───FoilHoles
   │   └───Metadata
   └───Atlas
```
CryoCrane searches the provided path for the provided atlas image filename and for directories named "Data". The first found Atlas file will be used to show the atlas. All micrographs and their .xml meta data files in any "Data" directory will be displayed as well. 
You can also store the atlas image in any other directory. In this case specify the absolute path to the atlas image. Unfortunately, the meta files of the *_Fractions.mrc images do not contain the stage coordinates. 


#### SerialEM data collections

For data sets recorded with SerialEM the given directory and its subdirectories will be searched for .mdoc files and the respective image files. You can either specify the absolute path to the atlas image or simply its file name, if it is contained in the same directory as the data set.  
The GUI works both with summed images and movies. However, plotting of movies takes a while.

### Analysing the grid

You can zoom, pan and move around on the atlas image. Upon clicking on the atlas on the left side the micrograph at the nearest position will be shown. The red dot marks the location of the clicked foil hole. You can add a scalebar and an 2D-FFT to that micrograph in the panel below the micrograph. The micrograph can be zoomed and saved in various formats with the navigation toolbar. You can furthermore mark locations as good (golden) or bad (brown) using the buttons below the atlas image. 

![alt text](https://github.com/jruickoldt/CryoCrane/blob/main/CryoCrane_example_1.png?raw=true)

#### Aligning stage and atlas coordinates

After loading your dataset you should align the atlas and stage coordinates. The default values were determined for a Talos F200C microscope and might be different for your setup.

#### Clustering and aligning grid squares

After entering an integer in the field "Number of grid squares", the coordinates will be clustered and you can align the x- and y-offset for each cluster seperately on the atlas. To do so select a cluster from the dropdown menu. This cluster will then be highlighted by a red dot and you can align it using the sliders for the x- and y-offset.

#### Rating the micrographs

Select a model from the drowdown list and click on "Start prediction". This will start a prediction thread running in the background. After the prediction has finished, you can start a training on the dataset for the atlas prediction.
The dropdown list shows all models in the "./weights" folder. Please ensure that the weights-file has the format {Model_type}_{image_size}_{dropout}_{any_text}.pth

CryoCrane comes currently shipped with the following models trained on a data set of 2800 images from 37 data sets of various proteins and various microscopes:

|weights | validation mean average error | purpose |
| ------- | ------- | ------- |
| ResNet8_256_0.2_full.pth | 0.091 | fast analysis |
| ResNet10_512_0.2_full.pth  | 0.089 | more precise analysis |
| ResNet12_1024_0.2_full.pth  | 0.090 | more precise analysis |




### Saving and loading a session

You can save your session at any point. This is especially advised after you have aligned and rated the grid. The session information will be stored in a .csv file. You can load and restore your session then from that .csv file. 

### Training a model on the data set for atlas prediction

If the micrographs have been rated, you can start a training on the data set. After supplying all information and click the "Start Training" button, the training will start. During training the weights with the lowest validation loss will be saved to "./atlas_weights" as {Model_type}_{image_size}_{dropout}_{any_text}.pth and will then be found in the dropdown list for the atlas prediction model.  

CryoCrane comes currently shipped with the following model. However, you are better off training a model yourself:

|weights | grid type | validation MAE* |
 ------- | ------- | ------- |
| CoordNet8_32_0.2_NcNR_C.pth | R1.2/1.3| - |

*mean absolute error

### Predicting the score of an atlas

You can either use a pre-trained model or an model trained on the specific data set. If you use a pretrained model, this should be at least be trained on a similar grid type and imaging mode (e.g. non-filtered vs. plasmon imaging). Ideally, it is trained on the same sample as well. 

### Supervising a session on-the-fly

This function is intended for the use during a live data collection session. Initially, load the data set, select a model for the score prediction and run the prediction. You can then update the session with the "update" button. CryoCrane will then search for any new micrographs and plot their locations using the currently applied angles and shifts. Furthermore, the score of the new micrographs will be predicted.   


## Citation

If you found CryoCrane useful please cite the following article:

J. Ruickoldt and P. Wendler (2025). Acta Cryst. F81, https://doi.org/10.1107/S2053230X25000081

 
