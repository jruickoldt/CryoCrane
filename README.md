<img src="https://github.com/jruickoldt/CryoCrane/blob/main/CryoCrane_logo.png?raw=true"
     alt="CryoCrane logo"
     width="120"
     align="right" />

# CryoCrane
Correlate Atlas and Exposures – a GUI for the analysis of cryo-EM screening data

## Description

Screening of cryo EM samples is essential for the generation of high-resolution cryo-EM data. Often, it is cumbersome to correlate the appearance of specific grid squares and micrograph quality. Here, we present a visualization tool for cryo-EM screening data: CryoCrane. It is aimed to provide an intuitive way of visualization of micrographs and to speed up data analysis. 
 
CryoCrane 2.0.5 now incorporates the CryoPike suite as well. CryoPike is a set of programs for the automated scoring of cryo-EM exposures. The CryoPike networks were trained on a diverse data set of cryo-EM micrographs rated by experts based on the presence of contaminations, aggregation or crystalline ice, on the particle distribution and image contrast. You can also train your own neural-network within CryoCrane allowing the prediction of suitable grid areas for data acquisition.

If you want to learn, how you can use CryoCrane to supervise, evaluate and improve your data collection on-the-fly follow this [tutorial.](tutorial.md) 

More advanced tutorials can be found here:
- [Automated micrograph evaluation](./tutorial_micrgraph_eval.md)
- [Improving data collections with AtlasPike](./tutorial_atlas_pike.md)
- [Reporting and data cleaning](./tutorial_after.md)

## Citation

If you found CryoCrane useful please cite the following article:

J. Ruickoldt and P. Wendler (2025). Acta Cryst. F81, https://doi.org/10.1107/S2053230X25000081

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

This installation has been successfully tested on Windows 11 and MacOS Tahoe. 

---
**Disclaimer**

The authors take no liability and grants no warranty for the usage of this program. It is not advised to run CryoCrane on the computer controlling the microscope. Although the program was developed with greatest care, out-of-memory issue might occur upon unintended usage. 

---



 
