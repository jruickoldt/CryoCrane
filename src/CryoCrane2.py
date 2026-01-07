#!/usr/bin/env python3
VERSION = "2.0.2"

import sys
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'  # keeps text as text in SVG
import time
from tqdm import tqdm
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True' #should prevent any crashes
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
import mrcfile
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from Training import CoordNet8, ResNet4, ResNet6, ResNet8, ResNet10, ResNet12, ResNet34, ResNet50, ResNet152  # Import your ResNet model (adjust as needed)
from utilities import *
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch.utils.data import Dataset, random_split, DataLoader
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import re
from sklearn.cluster import KMeans
import queue
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QCheckBox, 
    QDialog, QLineEdit, QLabel, QFormLayout, QProgressBar, QComboBox, QFileDialog,
    QPlainTextEdit
)
from scipy.ndimage import label, mean, center_of_mass
from scipy.ndimage import sum as ndi_sum, maximum as ndi_max

torch.set_num_threads(1) #might prevent crashes
print("All packages loaded")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Detected {device} for pytorch calculations.")




# Handle high resolution displays:
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)






# ---------------- Atlas prediction thread ----------------
class Atlas_predictionThread(QThread):
    """Runs Atlas prediction in a separate thread to keep PyQt5 responsive."""
    progress_signal = pyqtSignal(int)  # Signal for UI updates
    finished_signal = pyqtSignal() # Signal for completion
    data_signal = pyqtSignal(object)  # Signal to send the DataFrame Does it work for an array?
    
    def __init__(self, Atlas, model_weights):
        super().__init__() 
        self.Atlas = Atlas
        self.model_weights = model_weights
        self._is_running = True  # Flag to control thread execution
    
    def stop(self):
        """Stops the thread execution."""
        self._is_running = False

        
    def run(self):
        success = True
        try:
            image = self.Atlas
            

        except:
            print("Atlas prediction failed due to faulty atlas parameters.")
            success = False
        try:  
            clean_weights = self.model_weights
            directory = "./atlas_weights/"
            self.model_weights = directory + self.model_weights
            print(f"Loaded a model from {self.model_weights}")
            split_list = clean_weights.split("_")
            device = "cpu"
            model_Atlas = load_model(self.model_weights, device, split_list[0], dropout=float(split_list[2]))
            model_Atlas.eval()
            size = int(split_list[1])
            patch_size = size
            step = patch_size // 4
            print(f"Loaded a model from {self.model_weights} for a patch size of {patch_size} and step of {step}.")

        except Exception as e:
            print("Atlas prediction failed due to faulty models.")
            print(f"Error 6: {e}")
            success = False

        if success:
            pred_scores = []
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
            i = 1
            image_height, image_width = image.shape[0],image.shape[1]
            #for patch in tqdm(prediction_patches):
            patches_y = (image_height - patch_size) // step + 1
            patches_x = (image_width - patch_size) // step + 1
            total_patches = patches_y * patches_x
            print("Start Atlas prediction")
            for patch, center in extract_overlapping_patches_generator(image, patch_size=patch_size, step=step):
                if self._is_running == True:
                    self.progress_signal.emit(int((i / total_patches) * 100))
                    i += 1
                    flattened = [item for row in patch for item in row]
                    average = sum(flattened) / len(flattened)
                    # calculate the average of each patch. If it is less than the threshold (e.g. dark), it will be ignored for the inference step 
                    if average > 0.1:
                        image_tensor = preprocess_patch(patch)
                        pred_score = predict(model_Atlas, image_tensor, device = "cpu")

                        pred_scores.append(pred_score)
                    else:
                        pred_scores.append(0)
                    pass
                else:
                    return

            # Create a heatmap from predictions
            heatmap = np.zeros(image.shape, dtype=np.float32)
            count = np.zeros(image.shape, dtype=np.int32)

            # Fill heatmap by averaging overlapping patches
            idx = 0
            print(f"Shape of the atlas: {image.shape[0]}, step size: {step}, patch_size: {patch_size}")
            for y in range(0, image.shape[0] - patch_size + 1, step):
                for x in range(0, image.shape[1] - patch_size + 1, step):
                    heatmap[y:y+patch_size, x:x+patch_size] += pred_scores[idx]
                    count[y:y+patch_size, x:x+patch_size] += 1
                    idx += 1


            # Normalize by the number of overlaps
            heatmap /= (count + 1e-8)  # Avoid division by zero
            self.heatmap = heatmap


            
            self.finished_signal.emit() 
            self.data_signal.emit(self.heatmap)
 


class BatchGenerationThread(QThread):
    """Generates batches of images for prediction."""
    finished_signal = pyqtSignal()  # Signal for completion

    def __init__(self, images_to_predict, size, batch_size, batch_queue, Fourier = False):
        super().__init__()
        self.images_to_predict = images_to_predict
        self.size = size
        self.batch_size = batch_size
        self.batch_queue = batch_queue
        #print(f"[DEBUG] Queue type: {type(self.batch_queue)}")
        self._is_running = True
        self.Fourier = Fourier
        
    
    def put_batch(self, batch_images, debug=True):
        while self._is_running:
            try:
                self.batch_queue.put(batch_images, timeout=2)
                if debug:
                    print(f"Batch of size {len(batch_images)} put into queue.")
                    print(f"Queue size: {self.batch_queue.qsize()}")
                break  # success
            except queue.Full:
                if debug:
                    print("Queue full. Waiting to retry...")
                time.sleep(1)

    def stop(self):
        """Stops the thread execution."""
        self._is_running = False

    def run(self):
        batch_images = []
        for image_path in tqdm(self.images_to_predict):
            if not self._is_running:
                print("Batch generation thread exiting early.")
                return
            
            image_tensor = preprocess_mrc(image_path, self.size, Fourier=self.Fourier)  # Preprocess the image
            batch_images.append(image_tensor)

            # If the batch is full, put it in the queue
            if len(batch_images) == self.batch_size:
                #self.batch_queue.put(batch_images)
                self.put_batch(batch_images, debug=False)
                batch_images = []  # Reset the batch
                

        # Put any remaining images in the last batch
        if batch_images:
            #self.batch_queue.put(batch_images)
            self.put_batch(batch_images, debug=False)
        print("Finished batching of the images")
        self.finished_signal.emit()
        self.stop()

class NyquistPredictionThread(QThread):
    """Runs hole prediction in a separate thread to keep PyQt5 responsive."""
    progress_signal = pyqtSignal(int)  # Signal for UI updates
    finished_signal = pyqtSignal()  # Signal for completion
    data_signal = pyqtSignal(object)  # Signal to send the DataFrame
    
    def __init__(self, Locations_rot, batch_queue):
        super().__init__() 
        self.Locations_rot = Locations_rot
        self.batch_queue = batch_queue
        self._is_running = True  # Flag to control thread execution
        self.model_weights = "./nyquist_weights/RCoordNet8_1024_0.2_full_fourier.pth"
    
    def stop(self):
        """Stops the thread execution."""
        self._is_running = False
        print("Prediction thread stopping early.")

    def run(self):
        clean_weights = self.model_weights.replace("./nyquist_weights/", "")
        #print(clean_weights)
        split_list = clean_weights.split("_")
        device = "cpu"
        model = load_model(self.model_weights, device, split_list[0], dropout=float(split_list[2]))
        model.eval()
        size = int(split_list[1])
        
        pred_scores = []
        i = 1

        while self._is_running and i < len(self.Locations_rot)+1:
            try:
                print("Waiting for a new batch")
                batch_images = self.batch_queue.get(timeout=60)
                print("....received new batch")            # Wait for a batch
 
            except queue.Empty:
                continue  # Continue if no batch is available
            if batch_images is None:
                break  # Exit if no more batches
                


            batch_tensor = torch.stack(batch_images).to(device)
            batch_scores = [] #create empty batch_score file
            with torch.no_grad():
                batch_scores = predict(model, batch_tensor, device=device, batch=True)
            pred_scores.extend(batch_scores)
            i += len(batch_scores)
            self.progress_signal.emit(int((i / len(self.Locations_rot)) * 100))
                
        print("Either forced to stop or predicted all images. Will exit the prediction thread")
        print(f"Current index is {i} out of {len(self.Locations_rot)+1}")
        if i == len(self.Locations_rot)+1:
            print("finished prediction")
            pred_scores = [score.item() for score in pred_scores]
            self.Locations_rot["ctf_estimate"] = pred_scores
            self.data_signal.emit(pred_scores)
            self.finished_signal.emit()
            self.stop()
        
class PredictionThread(QThread):
    """Runs hole prediction in a separate thread to keep PyQt5 responsive."""
    progress_signal = pyqtSignal(int)  # Signal for UI updates
    finished_signal = pyqtSignal()  # Signal for completion
    data_signal = pyqtSignal(object)  # Signal to send the DataFrame
    
    def __init__(self, Locations_rot, model_weights, batch_queue):
        super().__init__() 
        self.Locations_rot = Locations_rot
        self.model_weights = model_weights
        self.batch_queue = batch_queue
        self._is_running = True  # Flag to control thread execution
    
    def stop(self):
        """Stops the thread execution."""
        self._is_running = False
        print("Prediction thread stopping early.")

    def run(self):
        clean_weights = self.model_weights.replace("./weights/", "")
        #print(clean_weights)
        split_list = clean_weights.split("_")
        device = "cpu"
        model = load_model(self.model_weights, device, split_list[0], dropout=float(split_list[2]))
        model.eval()
        size = int(split_list[1])
        
        pred_scores = []
        i = 1

        while self._is_running and i < len(self.Locations_rot)+1:
            try:
                print("Waiting for a new batch")
                batch_images = self.batch_queue.get(timeout=60)
                print("....received new batch")            # Wait for a batch
 
            except queue.Empty:
                continue  # Continue if no batch is available
            if batch_images is None:
                break  # Exit if no more batches
                


            batch_tensor = torch.stack(batch_images).to(device)
            batch_scores = [] #create empty batch_score file
            with torch.no_grad():
                batch_scores = predict(model, batch_tensor, device=device, batch=True)
            pred_scores.extend(batch_scores)
            i += len(batch_scores)
            self.progress_signal.emit(int((i / len(self.Locations_rot)) * 100))
                
        print("Either forced to stop or predicted all images. Will exit the prediction thread")
        print(i)
        if i == len(self.Locations_rot)+1:
            print("finished prediction")
            pred_scores = [score.item() for score in pred_scores]
            self.Locations_rot["score"] = pred_scores
            self.data_signal.emit(pred_scores)
            self.finished_signal.emit()
            self.stop()

 
class CTFEstimationThread(QThread):
    """Runs powerspectrum signal estimation in a separate thread to keep PyQt5 responsive."""
    progress_signal = pyqtSignal(int)  # Signal for UI updates
    finished_signal = pyqtSignal()  # Signal for completion
    data_signal = pyqtSignal(object)  # Signal to send the DataFrame

    def __init__(self, Locations_rot):
        super().__init__() 
        self.Locations_rot = Locations_rot
        self._is_running = True  # Flag to control thread execution

    def stop(self):
        """Stops the thread execution."""
        self._is_running = False
        print("Prediction thread stopping early.")

    def run(self):
        ctf_estimates = []
        i = 1
        start = time.time()
        while self._is_running and i < len(self.Locations_rot)+1:
            micrograph_path = self.Locations_rot.iloc[i-1]["JPG"]
            ctf_estimates.append(estimate_ctf_extent(micrograph_path))
            i += 1
            self.progress_signal.emit(int((i / len(self.Locations_rot)) * 100))
        end = time.time()
        print(f"CTF estimation of {i} micrographs took {end - start:.2f} seconds.")
                
        print("Either forced to stop or predicted all images. Will exit the prediction thread")
        print(f"Number of processed images: {i} out of {len(self.Locations_rot)}")

        if i == len(self.Locations_rot)+1:
            print("finished prediction")
            self.Locations_rot["ctf_estimate"] = ctf_estimates
            self.data_signal.emit(ctf_estimates)
            self.finished_signal.emit()
            self.stop()


        
        
# ---------------- PyTorch Training Thread ----------------
class TrainingThread(QThread):
    """Runs training in a separate thread to keep PyQt5 responsive."""
    update_signal = pyqtSignal(int, float, float)  # Signal for UI updates
    finished_signal = pyqtSignal()  # Signal for completion
    
    def __init__(self, Locations_rot, atlas, input_mm, dropout, patch_size, learning_rate, epochs, model_type, name, dark, light, training_data):
        super().__init__()
        self.Locations_rot = Locations_rot
        self.atlas = atlas
        self.input_mm = input_mm
        self.dropout = dropout
        self.patch_size = patch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.atlas_model = model_type
        self.name = name
        self.dark = dark
        self.light = light
        self.training_data = training_data
        self._is_running = True  # Flag to control thread execution
    
    def stop(self):
        """Stops the thread execution."""
        self._is_running = False

        
    def run(self):
        success = True
        try:
            image = self.atlas
        except:
            success = False
            print("Load the atlas first!")
        try:
            scale_extent = self.input_mm
        except:
            success = False
            print("Provide a reasonable atlas extent!")
        try:
            dropout_rate =self.dropout
            patch_size= self.patch_size
            lr = self.learning_rate
            epochs = self.epochs 
            name = self.name

        except:
            success = False
            print("Invalid training parameters")
        try:
            if self.training_data == "powerspectrum signal":
                coord_list=list(zip(self.Locations_rot["x"], self.Locations_rot["y"], self.Locations_rot["ctf_estimate"]))
                suffix = "ps"
            elif self.training_data == "predicted CryoPike score":
                coord_list=list(zip(self.Locations_rot["x"], self.Locations_rot["y"], self.Locations_rot["score"]))
                suffix = "score"
            elif self.training_data == "both scores":
                coord_list=list(zip(self.Locations_rot["x"], self.Locations_rot["y"], (self.Locations_rot["score"]+self.Locations_rot["ctf_estimate"])/2))
                suffix = "combined"
            print(coord_list[:5])
        except Exception as e:
            success = False
            print(f"Failed to extract coordinates and scores from the provided dataframe. Error 7: {e}")
        if success:
            #print(image.shape[0])
            print(f"Minimum value of the atlas: {np.min(image)}, maximum {np.max(image)}")
            recalculated_coord_list = recalculate_coordinates(coord_list, image.shape[0], image.shape[0], scale_extent)
            if (
                (self.atlas_model == "CoordNet8" and patch_size >= 16)
                or (self.atlas_model == "ResNet8" and patch_size >= 16) 
                or (self.atlas_model == "ResNet10" and patch_size >= 32) 
                or (self.atlas_model == "ResNet12" and patch_size >= 64)
            ):
                patches, scores = extract_patches(image, recalculated_coord_list, patch_size=patch_size, dark=self.dark, light=self.light)

                # Load dataset

                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

                dataset = AtlasScoreDataset(patches, scores)

                if len(dataset) > 0: 
                    # Split dataset (80% train, 20% test)
                    train_size = int(0.8 * len(dataset))
                    test_size = len(dataset) - train_size

                    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

                    # Create DataLoaders
                    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
                    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(test_dataset)}")

                    model_type = self.atlas_model
                    cls = {
                        "ResNet8": ResNet8, "ResNet10": ResNet10, "ResNet12": ResNet12, "CoordNet8": CoordNet8
                    }.get(model_type)
                    if cls is None:
                        raise ValueError(f"Unknown model: {model_type}")

                    model_Atlas = cls(dropout_rate)
                        
                    # Define loss and optimizer

                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model_Atlas.parameters(), lr=lr)


                    def train_atlas_model(model, train_loader, criterion, optimizer, criterion_val = nn.L1Loss(), epochs=30):
                        best_val_loss = 1000000  # Initialize best loss as infinite
                        model.train()
                        val_loss_history = []
                        train_loss_history = []
                        for epoch in range(epochs):
                            if not self._is_running:  # Check if stop was requested
                                print("Training stopped.")
                                return  # Exit run() safely
                            total_loss = 0.0
                            for images, scores in train_loader:
                                optimizer.zero_grad()
                                outputs = model(images).squeeze(1) #to get rid of the warning
                                loss = criterion_val(outputs, scores)
                                loss.backward()
                                optimizer.step()
                                total_loss += loss.item()
                            train_loss = total_loss/len(train_loader)
                            val_loss = 0.0
                            for images, scores in test_loader:
                                outputs = model(images).squeeze(1) #to get rid of the warning
                                loss = criterion_val(outputs, scores)
                                val_loss += loss.item()
                            validation_loss = val_loss/len(test_loader)
                            val_loss_history.append(validation_loss)
                            train_loss_history.append(train_loss)

                            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Validation MAE: {val_loss/len(test_loader):.4f}")
                            # Send update to UI
                            self.update_signal.emit(epoch + 1, train_loss, validation_loss)
                            if validation_loss < best_val_loss:
                                best_val_loss = validation_loss
                                model_path = f"./atlas_weights/{model_type}_{patch_size}_{dropout_rate}_{name}_{suffix}.pth"
                                torch.save(model.state_dict(), model_path)
                                print(f"saved model to {model_path} with a validation loss of {best_val_loss:.4f}")
                        return 

                    train_atlas_model(model_Atlas, train_loader, criterion, optimizer, epochs=epochs)
                else:
                    print("Please provide a larger patch size. Currently, the patch size is smaller than the receptive field of the model.")

        self.finished_signal.emit()  # Notify UI when training is done
        

class mic_options_dialog(QDialog):
    def __init__(self, mic_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Micrograph plotting parameters")
        self.setGeometry(100, 100, 300, 200)
        self.mic_params = mic_params

        sshFile="style_sheet.qss"
        with open(sshFile,"r") as fh:
            self.setStyleSheet(fh.read())
        # Create layout
        layout = QVBoxLayout()
        # Input fields for parameters
        form_layout = QFormLayout()
        self.binning_input = QLineEdit(self)
        self.pixelsize_input = QLineEdit(self)
        self.FFT_box = QtWidgets.QCheckBox(text="Calculate FFT")
        self.Scale_box = QtWidgets.QCheckBox(text="Show scale")
        self.scale_input = QLineEdit(self)
        self.FFT_scale_input = QLineEdit(self)
        
        self.submit_button = QPushButton("Save parameters", self)
        self.submit_button.clicked.connect(self.on_submit)

        form_layout.addRow(QLabel("Binning factor:"), self.binning_input)
        form_layout.addRow(self.FFT_box)
        form_layout.addRow(QLabel("Pixel size (Å):"), self.pixelsize_input)
        form_layout.addRow(self.Scale_box)
        form_layout.addRow(QLabel("Scale bar length (Å):"), self.scale_input)
        form_layout.addRow(QLabel("FFT scale (Å):"), self.FFT_scale_input)
        layout.addLayout(form_layout)
        layout.addWidget(self.submit_button)
        self.setLayout(layout)
        #Set the current values from the dictionary
        self.binning_input.setText(str(self.mic_params["binning_factor"]))
        self.pixelsize_input.setText(str(self.mic_params["pixel_size"]))
        self.scale_input.setText(str(self.mic_params["scale_length"]))
        self.FFT_scale_input.setText(str(self.mic_params["FFT_scale"]))
        if self.mic_params["FFT"] == True:
            self.FFT_box.setChecked(True)
        else:
            self.FFT_box.setChecked(False)
            
        if self.mic_params["draw_scale"] == True:
            self.Scale_box.setChecked(True)
        else:
            self.Scale_box.setChecked(False)
            
    def on_submit(self):
        """Handle input values."""
        try:
            self.binning_factor = int(self.binning_input.text())
        except:
            print("invalid binning factor")
        try:
            self.FFT = self.FFT_box.isChecked()
            self.pixelsize = float(self.pixelsize_input.text())
        except:
            print("invalid pixel size")
        try:
            self.draw_scale = self.Scale_box.isChecked()
            self.scale = float(self.scale_input.text())
        except:
            print("invalid scalebar parameter")
        try:
            self.FFT_scale = float(self.FFT_scale_input.text())
        except:
            print("invalid fourier-space scale parameter")

        else:
            self.accept()
            
    def get_parameters(self):
        """Retrieve the values as a dictionary"""
        mic_params = {
            "binning_factor":  int(self.binning_input.text()),
            "FFT" : self.FFT_box.isChecked(),
            "pixel_size" : float(self.pixelsize_input.text()),
            "draw_scale" : self.Scale_box.isChecked(),
            "scale_length" : float(self.scale_input.text()),
            "FFT_scale" : float(self.FFT_scale_input.text())
        }
        print(mic_params)
        return mic_params

        
        

class Atlas_training_Dialog(QDialog):
    def __init__(self, Atlas, scale, Locations_rot, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training parameters")
        self.setGeometry(100, 100, 500, 400)
        self.Atlas = Atlas
        self.scale = scale
        self.Locations_rot = Locations_rot
        sshFile="style_sheet.qss"
        with open(sshFile,"r") as fh:
            self.setStyleSheet(fh.read())
        # Create layout
        layout = QVBoxLayout()


        # Input fields for parameters
        form_layout = QFormLayout()
        self.patch_size_input = QLineEdit(self)
        self.dropout_input = QLineEdit(self)
        self.epochs_input = QLineEdit(self)
        self.learning_rate_input = QLineEdit(self)
        self.model_input = QComboBox(self)
        self.score_input = QComboBox(self)
        self.name_input = QLineEdit(self)
        self.light_check = QCheckBox(text="Augment light patches")
        self.dark_check = QCheckBox(text="Augment dark patches")
        self.status = QLabel("")
        
        for i in ["CoordNet8","ResNet8","ResNet10","ResNet12"]:
            self.model_input.addItem(i)

        if "ctf_estimate" in self.Locations_rot.columns:
            self.score_input.addItem("powerspectrum signal")
            ctf = True
        if "score" in self.Locations_rot.columns:
            self.score_input.addItem("predicted CryoPike score")
            score = True
        if all(["ctf_estimate" in self.Locations_rot.columns, "score" in self.Locations_rot.columns]):
            self.score_input.addItem("both scores")

        form_layout.addRow(QLabel("Model:"), self.model_input)
        form_layout.addRow(QLabel("Train on:"), self.score_input)
        form_layout.addRow(QLabel("Patch Size:"), self.patch_size_input)
        form_layout.addRow(QLabel("Dropout rate:"), self.dropout_input)
        form_layout.addRow(QLabel("Number of Epochs:"), self.epochs_input)
        form_layout.addRow(QLabel("Learning Rate:"), self.learning_rate_input)
        form_layout.addRow(QLabel("Name:"), self.name_input)
        form_layout.addRow(QLabel(""), self.dark_check)
        form_layout.addRow(QLabel(""), self.light_check)
        form_layout.addRow(QLabel(""), self.status)
        
        
        self.patch_size_input.setText("32")
        self.epochs_input.setText("100")
        self.learning_rate_input.setText("1e-5")
        self.name_input.setText("default")
        self.dropout_input.setText("0.2")
        self.dark_check.setChecked(True)
        self.light_check.setChecked(True)
        
        # Start Button
        self.submit_button = QPushButton("Start training", self)
        self.submit_button.clicked.connect(self.on_submit)
        
        # Stop Button
        self.stop_button = QPushButton("Stop training", self)
        self.stop_button.clicked.connect(self.on_stop)
        
        # Close Button
        self.close_button = QPushButton("Stop training and close dialog", self)
        self.close_button.clicked.connect(self.stop_and_accept)



        
        # Create a Matplotlib figure for loss tracking
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        
                # Add widgets to layout
        layout.addWidget(self.canvas)
        layout.addLayout(form_layout)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.close_button)

        self.setLayout(layout)

        self.training_loss = []
        self.validation_loss = []
        self.thread = None  # Thread will be created later
        if hasattr(self, 'train_thread'):
            self.show_loading_animation()

    def get_parameters(self):
        """Retrieve the values as a dictionary"""
        return {
            "patch_size": int(self.patch_size_input.text()) or 6,
            "dropout": 0,
            "model_name": self.model_input.currentText(),
            "name": self.name_input.text(),
        }

 
    def update_plot(self, epoch, train_loss, val_loss):
        """Update the plot dynamically with new training and validation loss."""
        self.training_loss.append(train_loss)
        self.validation_loss.append(val_loss)
        

        self.ax.clear()
        self.ax.plot(range(1, len(self.training_loss) + 1), self.training_loss, label="Training Loss", color="blue")
        self.ax.plot(range(1, len(self.validation_loss) + 1), self.validation_loss, label="Validation Loss", color="red")
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("mean absolute error")
        self.ax.legend()
        #self.ax.set_title("Training and Validation Loss")

        self.canvas.draw()
        
    def show_loading_animation(self):
        self.loading_label = QLabel(self)
        self.loading_label.setFixedSize(64, 64)  # Adjust size as needed
        self.loading_label.setAlignment(Qt.AlignBottom | Qt.AlignRight)

        # Load and start animation
        self.movie = QMovie("pike.gif")  # Use a valid path to your GIF
        self.loading_label.setMovie(self.movie)
        self.movie.start()

        # Add to layout (can be more sophisticated with layout structure)
        self.layout.addWidget(self.loading_label, alignment=Qt.AlignBottom | Qt.AlignRight)
        
    def on_stop(self):
        """Stops the training thread."""
        print("trying to stop the thread")
        if hasattr(self, 'train_thread') and self.train_thread.isRunning():
            self.train_thread.stop()
            self.status.setText("Training stopped.")
            
    def stop_and_accept(self):
        """Stops the training thread and closes the dialog"""
        #self.train_thread.stop()
        print("trying to stop the thread")
        if hasattr(self, 'train_thread') and self.train_thread.isRunning():
            self.train_thread.stop()
        self.accept()


    def on_submit(self):
        """Handle input values."""
        success = True  # Track whether all inputs are valid

        try:
            MainWindow.patch_size = int(self.patch_size_input.text())
        except Exception as e:
            print("Invalid patch size")
            success = False

        try:
            MainWindow.epochs = int(self.epochs_input.text())
        except Exception as e:
            print("Invalid parameter provided for epochs")
            success = False
        try:
            MainWindow.dropout =  float(self.dropout_input.text())
            if MainWindow.dropout > 1 or MainWindow.dropout < 0:
                raise ValueError("Dropout has to be in the intervall 0,1.")
        except Exception as e:
            print("Invalid dropout provided")
            print(f"Error 8: {e}")
            success = False

        try:
            MainWindow.learning_rate = float(self.learning_rate_input.text())
            if MainWindow.learning_rate > 0.1:
                print("Learning rate is too high. It has to be lower than 0.1.")
                success = False
        except Exception as e:
            print("Invalid learning rate")
            success = False

        try:
            MainWindow.model_name = self.model_input.currentText()
            MainWindow.name = self.name_input.text()
        except Exception as e:
            print("Invalid model name")
            print(f"Error 9: {e}")
            success = False
            
        try: 
            light =  self.light_check.isChecked() 
            dark = self.dark_check.isChecked()
        except Exception as e:
            print("Invalid augementation options")
            print(f"Error 10: {e}")
            success = False
        try:
            training_data = self.score_input.currentText()
        except Exception as e:
            print("Invalid training data option")
            print(f"Error 11: {e}")
            success = False

        if success:
            self.status.setText("Training started. Checking for other threads.")

            print(f"Patch Size: {MainWindow.patch_size}, Epochs: {MainWindow.epochs}, Learning Rate: {MainWindow.learning_rate}, Dropout: {MainWindow.dropout}")
            #self.accept()
            if hasattr(self, 'train_thread'):
                print("Trying to stop current training thread")
                self.train_thread.stop()
                self.train_thread.wait()
            patch_size, epochs, learning_rate, dropout, model_name, name = MainWindow.patch_size, MainWindow.epochs, MainWindow.learning_rate, MainWindow.dropout, MainWindow.model_name, MainWindow.name
            atlas = self.Atlas
            input_mm = self.scale
            Locations_rot = self.Locations_rot

            #print(Locations_rot)
            self.train_thread = TrainingThread(Locations_rot, atlas, input_mm, dropout, patch_size, learning_rate, epochs, model_name, name, dark, light, training_data)
            self.training_loss = []
            self.validation_loss = []
            self.train_thread.update_signal.connect(self.update_plot)
            #self.thread.finished_signal.connect(self.training_done)
            self.status.setText("Training running. Extracting patches...")
            self.train_thread.start()
            return  MainWindow.patch_size, MainWindow.epochs, MainWindow.learning_rate, MainWindow.dropout, MainWindow.model_name

        
class NavigationToolbar(NavigationToolbar):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save')]
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self._home_xlim = self.canvas.ax1.get_xlim()
        self._home_ylim = self.canvas.ax1.get_ylim()
        print(f"Limits: {self._home_ylim}, {self._home_xlim}")
    
    def save_figure(self, *args):
        self.canvas.draw()

        ax = self.canvas.ax1
        fig = ax.figure

        bbox = ax.get_position()
        fig_width, fig_height = fig.get_size_inches()
        left = bbox.x0 * fig_width
        bottom = bbox.y0 * fig_height
        width = bbox.width * fig_width
        height = bbox.height * fig_height

        filename, _ = QFileDialog.getSaveFileName(
            self.parentWidget(),
            "Save figure", "",
            "SVG files (*.svg);;TIFF files (*.tif);;PNG files (*.png);;PDF files (*.pdf);;All Files (*)"
        )
        if filename:
            # Save original size
            orig_size = fig.get_size_inches()

            # Resize figure to match axes size
            fig.set_size_inches(width, height)

            # Save figure
            fig.savefig(filename, bbox_inches='tight')

            # Restore original size
            fig.set_size_inches(orig_size)
            self.canvas.draw()  # redraw canvas with original size

class ZoomNavigationToolbar(NavigationToolbar):
    toolitems = [t for t in NavigationToolbar.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save')]
        
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self._home_xlim = self.canvas.ax1.get_xlim()
        self._home_ylim = self.canvas.ax1.get_ylim()
        print(f"Limits: {self._home_ylim}, {self._home_xlim}")

    def home(self, *args):
        ax = self.canvas.ax1
        ax.set_xlim(self._home_xlim)
        ax.set_ylim(self._home_ylim)
        self.canvas.draw()

    def save_figure(self, *args):
        self.canvas.draw()

        ax = self.canvas.ax1
        fig = ax.figure

        bbox = ax.get_position()
        fig_width, fig_height = fig.get_size_inches()
        left = bbox.x0 * fig_width
        bottom = bbox.y0 * fig_height
        width = bbox.width * fig_width
        height = bbox.height * fig_height

        filename, _ = QFileDialog.getSaveFileName(
            self.parentWidget(),
            "Save figure", "",
            "SVG files (*.svg);;TIFF files (*.tif);;PNG files (*.png);;PDF files (*.pdf);;All Files (*)"
        )
        if filename:
            # Save original size
            orig_size = fig.get_size_inches()

            # Resize figure to match axes size
            fig.set_size_inches(width, height)

            # Save figure
            fig.savefig(filename, bbox_inches='tight')

            # Restore original size
            fig.set_size_inches(orig_size)
            self.canvas.draw()  # redraw canvas with original size

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_xlim(-910, 910)
        self.ax1.set_ylim(-910, 910)
        self.fig.tight_layout(pad=0)
        self.fig.patch.set_facecolor('#FFFFFF00')

        super(MplCanvas, self).__init__(self.fig)
        #self.setParent(parent)
        self.canvas = self.fig.canvas 

    def wheelEvent(self, event):
        # Determine zoom direction
        zoom_in = event.angleDelta().y() > 0

        # Get current mouse position in pixels
        mouse_x = event.pos().x()
        mouse_y = event.pos().y()

        # Convert to axes coordinates
        display_coord = (mouse_x, self.height() - mouse_y)
        xdata, ydata = self.ax1.transData.inverted().transform(display_coord)
        #print(f"Current center: {xdata}, {ydata}")
        # Current limits
        xlim = self.ax1.get_xlim()
        ylim = self.ax1.get_ylim()

        # Zoom factor
        scale_factor = 0.9 if zoom_in else 1.1

        # Calculate new limits
        new_xlim = [
            xdata - (xdata - xlim[0]) * scale_factor,
            xdata + (xlim[1] - xdata) * scale_factor,
        ]
        new_ylim = [
            ydata - (ydata - ylim[0]) * scale_factor,
            ydata + (ylim[1] - ydata) * scale_factor,
        ]

        # Set new limits
        self.ax1.set_xlim(new_xlim)
        self.ax1.set_ylim(new_ylim)

        self.draw()
        
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
        weite = 8
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

        self.toolbar_Atlas = ZoomNavigationToolbar(self.sc, self)
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
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)  # optional: limit log size
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

        
        self.label_Dataset = QtWidgets.QLabel(self, text="Data set options")
        self.label_Atlas_alignment = QtWidgets.QLabel(self, text="Atlas alignment")
        self.label_FS_options = QtWidgets.QLabel(self, text="CryoPike options")
        
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
        self.label_Atlas_resolution = QtWidgets.QLabel(self, text="Display stitched image in:")
        self.Atlas_resolution = QComboBox(self)
        self.Atlas_resolution.addItem("low resolution")

        self.xml_button = QtWidgets.QRadioButton(self, text=".xml")
        self.mdoc_button = QtWidgets.QRadioButton(self, text=".mdoc")
        self.label_xml_mdoc = QtWidgets.QLabel(self, text="Meta data file type")
        xml_mdoc = QtWidgets.QButtonGroup(self)
        xml_mdoc.addButton(self.xml_button)
        xml_mdoc.addButton(self.mdoc_button)
        xml_mdoc.setExclusive(True)
        label_colormap = QtWidgets.QLabel(self, text="Colour by:")
        self.colormap = QtWidgets.QComboBox()

        self.ctf_button = QPushButton("Start powerspectrum signal\n estimation")
        
        self.save_button = QPushButton("Save session")
        self.load_button = QPushButton("Load session")
        self.browse_micrographs_button = QPushButton("browse")
        self.browse_atlas_button = QPushButton("browse")
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
        
        self.plot_button = QtWidgets.QPushButton(parent=self, text="")
        self.plot_button.setIcon(QtGui.QIcon('CryoCrane_logo.png'))
        
        self.plot_button.setToolTip("Plot the Atlas and foilhole locations")
        self.plot_button.setIconSize(QtCore.QSize(128, 128))
        
        update_button = QtWidgets.QPushButton(parent=self, text="update")
        
        self.align_button = QtWidgets.QPushButton(parent=self, text="align atlas")

        
        #Default values
        self.input_mm.setText("910")
        self.input_angle.setText("84.5")
        self.input_offsetx.setText("0")
        self.input_offsety.setText("0")
        self.input_bin.setText("2")
        self.input_Atlas.setText("Atlas_1.mrc")
        self.grid_offset_x=0
        self.grid_offset_y=0
        self.Cluster = 0
        #Default values
        
        #Buttons for gridsquare alignment
        self.cluster_button = QtWidgets.QPushButton(parent=self, text="Cluster in grids quares")
        self.input_squares = QtWidgets.QLineEdit(self, width=0.5)
        self.label_squares = QtWidgets.QLabel(self, text="Number of grid squares")
        self.squares_box = QtWidgets.QComboBox()
        
        # Create a QSlider instance
        self.grid_y_slider_label = QtWidgets.QLabel("Offset in y (µm): ")
        self.grid_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.grid_y_slider.setRange(-100, 100)  # Set the range
        self.grid_y_slider.setValue(0)       # Initial value
        
        self.grid_y_spinbox = QtWidgets.QDoubleSpinBox()
        self.grid_y_spinbox.setRange(-100, 100)
        self.grid_y_spinbox.setSingleStep(1)  # Set the step size to 1
        self.grid_y_spinbox.setValue(0)  # Initial value
        
        self.grid_y_spinbox.valueChanged.connect(lambda value: self.grid_y_slider.setValue(int(value)))
        self.grid_y_slider.valueChanged.connect(self.grid_y_spinbox.setValue)
        
        # Create a QSlider instance
        self.grid_x_slider_label = QtWidgets.QLabel("Offset in x (µm): ")
        self.grid_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.grid_x_slider.setRange(-100, 100)  # Set the range
        self.grid_x_slider.setValue(0)       # Initial value
        
        self.grid_x_spinbox = QtWidgets.QDoubleSpinBox()
        self.grid_x_spinbox.setRange(-100, 100)
        self.grid_x_spinbox.setSingleStep(1)  # Set the step size to 1
        self.grid_x_spinbox.setValue(0)  # Initial value
        
        self.grid_x_spinbox.valueChanged.connect(lambda value: self.grid_x_slider.setValue(int(value)))
        self.grid_x_slider.valueChanged.connect(self.grid_x_spinbox.setValue)

        #Buttons and settings for prediction#
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)  # Set initial progress
        self.weights_label = QtWidgets.QLabel("Choose model weights for the score prediction: ")
        self.weights_combobox = QtWidgets.QComboBox()
        self.atlas_weights_label = QtWidgets.QLabel("Choose model weights for the atlas prediction: ")
        self.atlas_weights_combobox = QtWidgets.QComboBox()
        self.populate_weights_combobox() #Adds an item for every .pth file found in the weights folder
        self.predict_button = QtWidgets.QPushButton(parent=self, text="Predict scores")
        
        self.train_atlas_parameters = QtWidgets.QPushButton(parent=self, text="Train model for atlas prediction")
        self.Epochs_label = QtWidgets.QLabel("Epochs: ")
        self.Epochs = QtWidgets.QLineEdit(self, width=0.5)
        
        self.num_squares_spinbox = QtWidgets.QDoubleSpinBox()
        self.num_squares_spinbox.setRange(1, 400)  # Set the range
        self.num_squares_spinbox.setSingleStep(1)  # Set the step size to 1
        self.num_squares_spinbox.setValue(10)  # Initial value

        self.num_squares = 10

        self.num_squares_label = QtWidgets.QLabel("Number of squares to label: ")

        self.predict_atlas_button = QtWidgets.QPushButton(parent=self, text="Predict atlas")
        mic_parameters = QtWidgets.QPushButton(parent=self, text="Micrograph options")
        #Set the Layout
        

        self.setWindowTitle(f"CryoCrane {VERSION} - Correlate atlas and exposures")
        self.setWindowIcon(QtGui.QIcon('CryoCrane_logo.png'))
        
        #Toolbars and canvas
        layout1 = QtWidgets.QVBoxLayout()
        layout0 = QtWidgets.QGridLayout()
        layout0.addWidget(self.toolbar_Atlas,0,0)
        layout0.addWidget(toolbar_Mic,0,1)
        layout1.addLayout( layout0, stretch=1 )
        
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.sc)
        layout2.addWidget(self.Mic)



        layout1.addLayout( layout2, stretch=8 )
        
        layout_X = QtWidgets.QGridLayout()

            

        for i in range(5,9):        
            layout_X.addWidget(self.dummy,0,i, QtCore.Qt.AlignRight)
        layout_X.addWidget(self.label_xy,0,9, QtCore.Qt.AlignRight)
        
        layout1.addLayout( layout_X, stretch=1 )
        
        #Buttons below the plot area
        
        layout3 = QtWidgets.QGridLayout()
        
        # Headings
        layout3.addWidget(self.plot_button,2,0,4,1)
        layout3.addWidget(update_button,6,0)
        layout3.addWidget(self.ctf_button,7,0)
        layout3.addWidget(self.label_Dataset,1,1,1,3, QtCore.Qt.AlignCenter)
        layout3.addWidget(self.label_Atlas_alignment,1,4,1,2, QtCore.Qt.AlignCenter)
        layout3.addWidget(self.label_FS_options,1,7,1,3, QtCore.Qt.AlignCenter)

        
        
        #Data set options
        layout3.addWidget(self.label_xml,2,1)
        layout3.addWidget(self.input_xml,3,1,1,2)
        layout3.addWidget(self.browse_micrographs_button,3,3)
        layout3.addWidget(self.browse_atlas_button,6,3)
        
        layout3.addWidget(self.label_tif_mrc,4,1)
        layout3.addWidget(self.mrc_button,4,2)
        layout3.addWidget(self.tif_button,4,3)
        layout3.addWidget(self.label_xml_mdoc,5,1)
        layout3.addWidget(self.xml_button,5,2)
        layout3.addWidget(self.mdoc_button,5,3)
        layout3.addWidget(self.label_Atlas,6,1)
        layout3.addWidget(self.input_Atlas,6,2)

        layout3.addWidget(label_colormap, 7, 1)
        layout3.addWidget(self.colormap, 7, 2)
        layout3.addWidget(self.label_Atlas_resolution, 8, 1)
        
        layout3.addWidget(self.Atlas_resolution,8,2)
        layout3.addWidget(self.save_button,9,1)
        layout3.addWidget(self.load_button,9,2)

        # colormap limits display (three colored boxes + values)
        self.colormap_limits = QtWidgets.QLabel(self)
        self.colormap_limits_label = QtWidgets.QLabel("Color code:")
        #self.colormap_limits.setFixedHeight(40)
        self.colormap_limits.setText("") 
        self.colormap_limits.setAlignment(QtCore.Qt.AlignCenter)
        layout3.addWidget(self.colormap_limits_label, 10, 1)      
        layout3.addWidget(self.colormap_limits, 10, 2)
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
        layout3.addWidget(self.align_button,6,4)


        
        layout3.addWidget(self.label_squares,7,4)
        layout3.addWidget(self.input_squares,7,5)
        layout3.addWidget(self.squares_box ,7,6)
        
        layout3.addWidget(self.grid_x_slider_label,8,4)
        layout3.addWidget(self.grid_x_slider,8,6)
        layout3.addWidget(self.grid_x_spinbox,8,5)
        
        layout3.addWidget(self.grid_y_slider_label,9,4)
        layout3.addWidget(self.grid_y_slider,9,6)
        layout3.addWidget(self.grid_y_spinbox,9,5)


        layout3.addWidget(mic_parameters,2,7, 1,2)
        layout3.addWidget(self.predict_button, 3,7, 1,2)
        layout3.addWidget(self.weights_label, 4,7)
        layout3.addWidget(self.weights_combobox, 4,8)
        layout3.addWidget(self.train_atlas_parameters, 5,7, 1,2)
        layout3.addWidget(self.predict_atlas_button, 6,7, 1,2)
        layout3.addWidget(self.atlas_weights_label, 7,7)
        layout3.addWidget(self.atlas_weights_combobox,7,8)

        layout3.addWidget(self.num_squares_label, 8,7)
        layout3.addWidget(self.num_squares_spinbox, 8,8)
        layout3.addWidget(self.progress_bar, 9,7, 1,2)
        layout3.addWidget(self.log_view,10, 5, 1,4)
        
        

        
        layout1.addLayout( layout3 , stretch = 4)

        
        # Reset the appearance of the GUI
        
        self.input_pix.setDisabled(True)
        self.input_res.setDisabled(True)
        self.input_length.setDisabled(True)

        
        
        self.sc.ax1.set_axis_off()
        self.Mic.ax1.set_axis_off()
        self.Mic.ax2.set_axis_off()
        self.Mic.draw()
        self.sc.draw()
        
        #Set default values for micrograph plotting options
        self.mic_params = {
            "binning_factor":  2,
            "FFT" : True,
            "pixel_size" : 1.23,
            "draw_scale" : False,
            "scale_length" : 100,
            "FFT_scale" : 10
        }

        #Actions

        self.plot_button.clicked.connect(self.plot_Data)
        update_button.clicked.connect(self.update_data)
        self.align_button.clicked.connect(self.realign)


        #self.predict_button.clicked.connect(self.predict_holes)
        self.predict_button.clicked.connect(self.start_prediction)
        self.ctf_button.clicked.connect(self.start_ctf_estimation)
        #train_atlas_button.clicked.connect(self.train_model_for_atlas_prediction)
        self.train_atlas_parameters.clicked.connect(self.open_atlas_training_dialog)
        mic_parameters.clicked.connect(self.open_mic_parameters_dialog)
        self.predict_atlas_button.clicked.connect(self.start_atlas_prediction)
        
        self.angle_slider.valueChanged.connect(self.realign)
        self.angle_spinbox.valueChanged.connect(self.realign)
        self.extend_slider.valueChanged.connect(self.realign)
        self.num_squares_spinbox.valueChanged.connect(self.update_num_squares)  
        
        self.offset_x_slider.valueChanged.connect(self.realign)
        self.offset_y_slider.valueChanged.connect(self.realign)
        self.Scale_box.clicked.connect(self.turn_on_pixel_input)
        self.input_squares.textChanged.connect(self.cluster_in_grid_squares)
        self.input_squares.textChanged.connect(self.mark_grid_square)
        self.grid_x_slider.valueChanged.connect(self.align_grid_square)
        self.grid_y_slider.valueChanged.connect(self.align_grid_square)
        self.squares_box.currentIndexChanged.connect(self.align_grid_square)
        self.squares_box.currentIndexChanged.connect(self.mark_grid_square)
        self.colormap.currentIndexChanged.connect(self.recolour)
        self.Atlas_resolution.currentIndexChanged.connect(self.recolour)
        
        self.sc.canvas.mpl_connect('button_press_event', self.onclick)
        self.load_button.clicked.connect(self.load_session)
        self.save_button.clicked.connect(self.save_session)
        self.browse_micrographs_button.clicked.connect(self.browse_micrographs)
        self.browse_atlas_button.clicked.connect(self.browse_atlas)        
        
        #Shortcuts
        
        self.plot_button.setShortcut("Ctrl+P")

        
        # Create a placeholder widget to hold the layout
        widget = QtWidgets.QWidget()
        widget.setLayout(layout1)
        self.setCentralWidget(widget)
        self.closeEvent = self.closeEvent   
        self.showMaximized() 

        self.show()
        
    def closeEvent(self, event):
        try:
            # Create filename with timestamp
            timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
            filename = f"log_{timestamp}.txt"

            # Optional: choose a directory
            log_dir = "./logs"
            os.makedirs(log_dir, exist_ok=True)
            filepath = os.path.join(log_dir, filename)

            # Get log text
            log_text = self.log_view.toPlainText()

            # --- Delete oldest logs if more than 10 ---
            log_files = [
                os.path.join(log_dir, f)
                for f in os.listdir(log_dir)
                if f.endswith(".txt")
            ]

            # Sort by modification time (oldest first)
            log_files.sort(key=os.path.getmtime)

            # Remove oldest files if exceeding limit
            MAX_LOGS = 10
            while len(log_files) >= MAX_LOGS:
                os.remove(log_files[0])
                log_files.pop(0)

                # Write to file
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(log_text)

        except Exception as e:
            print(f"Failed to save log: {e}")

        # Accept close event
        event.accept()

    def log(self, message):
        time_stamp = QtCore.QDateTime.currentDateTime().toString("HH:mm")
        self.log_view.appendPlainText(f"[{time_stamp}] {message}")

    def update_num_squares(self):
        self.num_squares = int(self.num_squares_spinbox.value())
        if hasattr(self, 'cluster_coords'):
            self.recolour()
        return self.num_squares

    def browse_micrographs(self):
        path = os.getcwd()
        directory = QFileDialog.getExistingDirectory(None, "Select Directory", path, QFileDialog.ShowDirsOnly)
        
        if directory:
            self.log("Selected directory:", directory)
            self.input_xml.setText(directory)
    
    def browse_atlas(self):
        qfd = QFileDialog()
        path =  os.getcwd()
        filter = "MRC files (*.mrc);;TIFF files (*.mrc,*.tif,*.tiff,*.TIF,*.TIFF);;All files (*)"
        f, _filter = QFileDialog.getOpenFileName(qfd, "Save session", path, filter)
        valid_extensions = [".tif", "tiff", ".mrc"]
        if f != "" and any(f.lower().endswith(ext) for ext in valid_extensions):
            print("Selected atlas file:", f)
            self.input_Atlas.setText(f)


    def save_session(self):
        qfd = QFileDialog()
        path = os.path.join(os.getcwd(), "sessions")
        filter = "csv(*.csv)"
        f, _filter = QFileDialog.getSaveFileName(qfd, "Save session", path, filter)
        #print(f)
        try:
            
            self.Locations_rot.to_csv(f)
        except Exception as e:
            self.log("saving unsuccesful")
            self.log(f"Error 12: {e}")
            print("Error occured while saving this dataframe:")
            print(self.Locations_rot)

        else:
            self.log(f"saved session to {f}")
        
    def load_session(self):
        qfd = QFileDialog()
        path = os.path.join(os.getcwd(), "sessions")
        filter = "csv(*.csv)"
        f, _filter = QFileDialog.getOpenFileName(qfd, "Load session", path, filter)
        if f == "":
            return
            
        success = True
        try:
            self.Locations_rot = pd.read_csv(f)
            x,y, df = self.Locations_rot["x"],self.Locations_rot["y"],self.Locations_rot["defocus"]
            Atlaspath = self.Locations_rot["atlas_path"][0]
            Micpath = self.Locations_rot["mic_path"][0]
            scale = self.Locations_rot["scale"][0]
            angle = self.Locations_rot["angle"][0]
            offset_x = self.Locations_rot["offset_x"][0]
            offset_y = self.Locations_rot["offset_y"][0]
            
        except Exception as e:
            self.log("Session loading was unsuccesfull.")
            self.log(f"Error 13: {e}")
            
            success = False
        try:
            #generate self.angle for the realign function
            self.offset_x = offset_x
            self.offset_y = offset_y
            self.scale = scale
            self.angle =  angle/360*2*np.pi #convert to radians
            p = Path(Micpath)
            if p.exists() == False:
                raise ValueError("Path to the micrographs does not exist.")

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
                self.log(f"Stitching an atlas from: {Atlaspath}")
                Atlas = stitch_atlas(Atlaspath)
        except Exception as e:
            self.log(f"Atlas loading from {Micpath} or {Atlaspath} was unsuccesfull")
            self.log(f"Error 14: {e}")
            
            success = False
        if success:
            #Set the initial phase to prevent recolouring and realignments.
            self.initial = True
            #Reset the values in the UI
            self.input_xml.setText(Micpath)
            self.input_Atlas.setText(Atlaspath)
            self.input_mm.setText(str(scale))
            self.offset_x_slider.setValue(int(self.offset_x))
            self.offset_y_slider.setValue(int(self.offset_y))
            self.angle_spinbox.setValue(angle)


        
            self.colormap.clear()
            self.colormap.addItem("applied defocus")
            if "score" in self.Locations_rot.columns:
                self.colormap.addItem("predicted score")
            if "ctf_estimate" in self.Locations_rot.columns:
                self.colormap.addItem("estimated powerspectrum signal")

            self.sc.ax1.cla()
            
            scale = float(self.input_mm.text())
            self.Atlas = Atlas
            self.Atlas_resolution.clear()
            self.Atlas_resolution.addItem("low resolution")
            if self.Atlas.shape[0] > 5000:
                self.Atlas_resolution.addItem("high resolution")
                small_atlas = rebin(self.Atlas, (int(self.Atlas.shape[0]/4), int(self.Atlas.shape[1]/4)))
                self.small_atlas = small_atlas
                self.sc.ax1.imshow(small_atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")
            else:
                self.small_atlas = self.Atlas
                self.sc.ax1.imshow(self.Atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")
            self.exposures = self.sc.ax1.scatter(x, y, c = df, s = 0.5, cmap = "GnBu")

            #self.current_hole = self.sc.ax1.scatter(scale,scale, c = "red", s = 0.8, alpha = 0.0)
            self.sc.ax1.set_axis_off()
            self.sc.draw()
            self.initial = False
            self.log(f"Successfully loaded session from {f} containing {len(self.Locations_rot)} exposures.")

            return self.Atlas, self.Locations_rot, self.angle, self.offset_x, self.offset_y, self.scale, self.small_atlas, self.initial
        
    def populate_weights_combobox(self):
        folder_path = os.path.join(os.getcwd(), 'weights')
        if os.path.isdir(folder_path):
            pth_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
            self.weights_combobox.clear()
            self.weights_combobox.addItems(pth_files)
        else:
            print(f"Folder not found: {folder_path}")
        
        folder_path = os.path.join(os.getcwd(), 'atlas_weights')
        if os.path.isdir(folder_path):
            pth_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
            self.atlas_weights_combobox.clear()
            self.atlas_weights_combobox.addItems(pth_files)
        else:
            print(f"Folder not found: {folder_path}")
        
    def start_atlas_prediction(self):
        """Start the training thread."""
        success = True
        try:
            Atlas = self.Atlas
        except Exception as e:
            success = False
            print("Atlas prediction was unsuccesful.")
            print(f"Error 15: {e}")
            self.log("Error: No data set loaded.")
            
            
        try:
            
            model_name = self.atlas_weights_combobox.currentText()
            
        except Exception as e:
            print("Atlas prediction was unsuccesful.")
            print(f"Error 16: {e}")
            success = False
            self.log("Error: No model trained.")
            
        if success:
            # Stop previous thread if running
            if hasattr(self, 'threada'):
                    print("Trying to stop the thread")
                    print(self.threada._is_running)
                    self.threada.stop()
                    print("Thread stopped")
                    print(self.threada._is_running)
            self.threada = Atlas_predictionThread(Atlas, model_name)
            self.threada.progress_signal.connect(self.progress_bar.setValue)
            self.threada.start()
            self.log("Atlas prediction in progress.")
            
            self.disable_alignment(True) #disable UI
            self.threada.data_signal.connect(self.color_after_atlas_prediction)

                        
    def start_ctf_estimation(self):
        success = True
        try:
            Locations_rot = self.Locations_rot
        except Exception as e:
            success = False
            self.log("Powerspectrum signal estimation was unsuccesful.")
            self.log(f"Error 17: {e}")
            

        if success:
            # Stop previous thread if running
            if hasattr(self, 'thread_ctf') and hasattr(self, 'batch_thread_ps'):
                    self.log("Trying to stop the running powerspectrum estimation thread")
                    print(self.thread_ctf._is_running)
                    self.thread_ctf.stop()
                    self.log(f"Thread stopped: {self.thread_ctf._is_running}")

                    self.log("Trying to stop the batch thread")
                    self.log(f"self.batch_thread._is_running: {self.batch_thread_ps._is_running}")
                    self.batch_thread_ps.stop()
                    self.log(f"self.batch_thread._is_running: {self.batch_thread_ps._is_running}")
  
            self.log(f"Powerspectrum signal estimation in progress. Estimated duration: {np.round(len(Locations_rot['JPG'].tolist()) / 60, 2)} minutes.")
            
            self.batch_queue_ps = queue.Queue(maxsize=4)
            # Start the batch generation thread
            size = 1024 #hardcoded for now, could be improved by storing the size used for prediction in the dataframe
            batch_size = det_batch_size(size)
            self.batch_thread_ps = BatchGenerationThread(Locations_rot["JPG"].tolist(), size, batch_size, self.batch_queue_ps, Fourier=True)
            self.batch_thread_ps.start()
            # add a delay to give the batch_thread a head start
            # Start the prediction thread


            
            self.thread_ctf = NyquistPredictionThread(Locations_rot, self.batch_queue_ps)
            self.thread_ctf.start()
            self.thread_ctf.progress_signal.connect(self.progress_bar.setValue)
            self.thread_ctf.data_signal.connect(self.color_after_ctf_estimation)

    def start_ctf_estimation_non_ai(self):
        success = True
        try:
            Locations_rot = self.Locations_rot
        except Exception as e:
            success = False
            self.log("Powerspectrum signal estimation was unsuccesful.")
            self.log(f"Error 18: {e}")
            self.log("Error: No data set loaded.")
            

        if success:
            # Stop previous thread if running
            if hasattr(self, 'thread_ctf'):
                    print("Trying to stop the thread")
                    print(self.thread_ctf._is_running)
                    self.thread_ctf.stop()
                    print("Thread stopped")
                    print(self.thread_ctf._is_running)
            self.thread_ctf = CTFEstimationThread(Locations_rot)
            self.thread_ctf.progress_signal.connect(self.progress_bar.setValue)
            self.thread_ctf.start()
            self.thread_ctf.data_signal.connect(self.color_after_ctf_estimation)

    def color_after_ctf_estimation(self, ctf_estimates):
        self.Locations_rot["ctf_estimate"] = ctf_estimates
        self.xlim = self.sc.ax1.get_xlim()
        self.ylim = self.sc.ax1.get_ylim()    
        
    
        self.exposures = self.sc.ax1.scatter(self.Locations_rot["x"], self.Locations_rot["y"], c = self.Locations_rot["ctf_estimate"], s = 0.5, cmap = "cool_r")
        self.sc.draw()
        if self.colormap.findText("estimated powerspectrum signal") == -1: #check if that is already in the combo box
            self.colormap.addItem("estimated powerspectrum signal")
        self.log("CTF estimation finished.")

    def update_score_prediction(self, new_Locations_rot):
        success = True

        # Check if data is loaded
        try:
            Locations_rot = new_Locations_rot
            test = self.Locations_rot["model"][0] #retrieving the model from the old data frame
            #self.Locations_rot = old_Locations_rot #storing the non-updated data frame
        except Exception as e:
            self.log(f"Update prediction failed: no dataset available. Error 19: {e}")
            self.log("Error: Cannot update scores without a data set.")
            
            return

        # Filter only entries with score == -1
        if "score" not in Locations_rot.columns:
            self.log("No 'score' column found. Aborting prediction update.")
            self.log("Error 20: No 'score' column found.")
            
            return

        Locations_to_update = Locations_rot[Locations_rot["score"] == -1]

        if Locations_to_update.empty:
            self.log("No scores to update.")
            self.log("All entries already scored.")
            return


        # Stop previous threads if running
        if hasattr(self, 'prediction_thread'):
            self.log("Stopping previous score prediction thread")
            self.prediction_thread.stop()
            self.prediction_thread.wait()
        if hasattr(self, 'batch_thread'):
            self.log("Stopping previous batch thread")
            self.batch_thread.stop()
            self.batch_thread.wait()

        try:
            # Load model weight and size
            #model_weights = "./weights/" + self.weights_combobox.currentText()
            model_weights = "./weights/" + self.Locations_rot["model"][0]

            clean_weights = model_weights.replace("./weights/", "")
            split_list = clean_weights.split("_")
            size = int(split_list[1])  # Extract image size from filename
        except Exception as e:
            self.log("Failed to load model weights.")
            self.log(f"Error 21: {e}")
            self.log("Error: Model weight loading failed.")
            
            return

        # Configure batch size based on model input
        batch_size = det_batch_size(size)
        
        old_Locations_rot = self.Locations_rot
        # Prepare queue and threads
        self.batch_queue = queue.Queue(maxsize=4)
        self.batch_thread = BatchGenerationThread(Locations_to_update["JPG"].tolist(), size, batch_size, self.batch_queue)
        self.batch_thread.start()

        self.prediction_thread = PredictionThread(Locations_to_update, model_weights, self.batch_queue)
        self.prediction_thread.start()
        self.prediction_thread.progress_signal.connect(self.progress_bar.setValue)
        #self.prediction_thread.data_signal.connect(self.color_after_prediction)

        self.log(f"Started score prediction thread. Estimated duration: {np.round(len(Locations_to_update['JPG'].tolist()) / 100, 2)} minutes.")


        self.prediction_thread.data_signal.connect(self.on_score_update_finished)
        self.Locations_rot["model"] = clean_weights
        if len(self.Locations_rot) > len(old_Locations_rot):
            self.color_after_prediction(self.Locations_rot)
        return 
    
    def on_score_update_finished(self, new_scores):
        print("Data received for score update:")
        print(self.Locations_rot[-5:])
        # Mask of rows that need updating
        mask = self.Locations_rot["score"] == -1

        # Safety check
        if mask.sum() != len(new_scores):
            raise ValueError(
                f"Number of new scores ({len(new_scores)}) does not match number of rows with score == -1 ({mask.sum()})"
            )

        # Assign scores in order
        self.Locations_rot.loc[mask, "score"] = new_scores

        self.log("Score prediction updated.")
        self.colormap.setCurrentText("predicted score")
        if not self.ctf_update_running:
            self.disable_alignment(False)
        self.score_update_running = False

        return self.Locations_rot

    def update_ctf_estimation(self, new_Locations_rot):
        success = True

        # Check if data is loaded
        try:
            Locations_rot = new_Locations_rot
        except Exception as e:
            self.log(f"Update prediction failed: no dataset available. Error 22: {e}")
            self.log("Error: Cannot update scores without a data set.")
            
            return

        # Filter only entries with score == -1
        if "ctf_estimate" not in Locations_rot.columns:
            self.log("No 'ctf estimate' column found. Aborting prediction update.")
            self.log("Error: No 'score' column found.")
            
            return

        Locations_to_update = Locations_rot[Locations_rot["ctf_estimate"] == -1]

        if Locations_to_update.empty:
            self.log("No scores to update.")
            self.log("All entries already ctf estimated.")
            return


        # Stop previous threads if running
        if hasattr(self, 'batch_thread_ps'):
            self.log("Stopping previous batch_thread_ps")
            self.batch_thread_ps.stop()
            self.batch_thread_ps.wait()
        if hasattr(self, 'thread_ctf'):
            self.log("Stopping previous thread_ctf")
            self.thread_ctf.stop()
            self.thread_ctf.wait()


        # Configure batch size based on model input
        size = 1024 #hardcoded for now, could be improved by storing the size used for prediction in the dataframe
        batch_size = det_batch_size(size)
        
        old_Locations_rot = self.Locations_rot
        # Prepare queue and threads

        self.batch_queue_ps = queue.Queue(maxsize=4)
        self.batch_thread_ps = BatchGenerationThread(Locations_to_update["JPG"].tolist(), size, batch_size, self.batch_queue_ps, Fourier=True)
        self.batch_thread_ps.start()


        # Start the prediction thread

        self.thread_ctf = NyquistPredictionThread(Locations_to_update, self.batch_queue_ps)
        self.thread_ctf.start()
        self.thread_ctf.progress_signal.connect(self.progress_bar.setValue)
        self.thread_ctf.data_signal.connect(self.on_ctf_update_finished)
        self.log(f" Started powerspectrum signal estimation thread. Estimated duration: {np.round(len(Locations_to_update['JPG'].tolist()) / 60, 2)} minutes.")

        return 
    
    def on_ctf_update_finished(self, new_ctf_estimate):
        print("Data received for ctf estimate update:")
        print(self.Locations_rot[-5:])
        # Mask of rows that need updating
        mask = self.Locations_rot["ctf_estimate"] == -1

        # Safety check
        if mask.sum() != len(new_ctf_estimate):
            raise ValueError(
                    f"Number of new ctf estimates ({len(new_ctf_estimate)}) does not match number of rows with ctf_estimate == -1 ({mask.sum()})"
                )

        # Assign ctf estimates in order
        self.Locations_rot.loc[mask, "ctf_estimate"] = new_ctf_estimate
        self.log("powerspectrum signal estimation updated.")
        self.colormap.setCurrentText("estimated powerspectrum signal")
        if not self.score_update_running:
            self.disable_alignment(False)
        self.ctf_update_running = False
        return self.Locations_rot
        
    def start_prediction(self):
        success = True
        try:
            Locations_rot = self.Locations_rot
        except Exception as e:
            self.log("Score prediction was unsuccesful.")
            self.log(f"Error 23: {e}")
            success = False
            self.log("Error: Cannot predict scores without a data set.")
            
        if success:
            # Stop previous thread if running
            if hasattr(self, 'prediction_thread'):
                self.log("Trying to stop the score prediction thread")
                self.prediction_thread.stop()
                self.prediction_thread.wait()  # Wait for the thread to finish
            if hasattr(self, 'batch_thread'):
                self.log("Trying to stop the batch thread")
                self.batch_thread.stop()
                self.batch_thread.wait()  # Wait for the thread to finish
            try:
                # Retrieve model weights and extract size
                model_weights = "./weights/" + self.weights_combobox.currentText()
                self.Locations_rot["model"] = self.weights_combobox.currentText()
                clean_weights = model_weights.replace("./weights/", "")
                split_list = clean_weights.split("_")
                size = int(split_list[1])  # Extract image size from model weights
            except Exception as e:
                success = False
                self.log("Loading the prediction model was unsuccesfull.")
                self.log(f"Error 24: {e}")
                
            if success:
                self.log(f"Score prediction running using model: {self.weights_combobox.currentText()}. Estimated duration: {np.round(len(Locations_rot['JPG'].tolist()) / 100, 2)} minutes.")
                
                # Create a queue for batch communication
                self.batch_queue = queue.Queue(maxsize=4)
                #Set the batch size according to the image size
                batch_size = det_batch_size(size)
                    

                # Start the batch generation thread
                self.batch_thread = BatchGenerationThread(Locations_rot["JPG"].tolist(), size, batch_size, self.batch_queue)
                self.batch_thread.start()
                # add a delay to give the batch_thread a head start
                # Start the prediction thread
                self.prediction_thread = PredictionThread(Locations_rot, model_weights, self.batch_queue)
                self.prediction_thread.start()
                self.prediction_thread.progress_signal.connect(self.progress_bar.setValue)
                self.prediction_thread.data_signal.connect(self.color_after_prediction)
            
    def color_after_prediction(self, predicted_scores):
        self.Locations_rot["score"] = predicted_scores
        self.xlim = self.sc.ax1.get_xlim()
        self.ylim = self.sc.ax1.get_ylim()    
        
    
        self.exposures = self.sc.ax1.scatter(self.Locations_rot["x"], self.Locations_rot["y"], c = self.Locations_rot["score"], s = 0.5, cmap = "viridis")
        self.sc.draw()
        if self.colormap.findText("predicted score") == -1: #check if that is already in the combo box
            self.colormap.addItem("predicted score")
        self.log(f"Score prediction using model {self.weights_combobox.currentText()} finished.")
            
            
    def color_after_atlas_prediction(self, array):
        #Collect all variables
        self.heatmap = array
        success = True
        self.cluster_coords = []
        num_squares = self.num_squares

        # 1. Label clusters: 0s are background, >0s are clusters
        structure = np.ones((3, 3))  # 8-connectivity
        labeled_array, num_features = label(self.heatmap > 0.4, structure=structure) #only consider squares with a score higher than 0.4
        if num_features < num_squares:
            labeled_array, num_features = label(self.heatmap > 0.3, structure=structure) 
            if num_features < num_squares:
                labeled_array, num_features = label(self.heatmap > 0.2, structure=structure) 
                if num_features < num_squares:
                    labeled_array, num_features = label(self.heatmap > 0.1, structure=structure)
                    if num_features < num_squares:
                        success = False
                        self.log("Could only find less clusters than grid squares.")
                        if num_features != 0:
                            self.log(f"Found {num_features} clusters, but there are {num_squares} grid squares.")
                            success = True #allow to continue, even if less clusters than squares.
                            num_squares = num_features
                    
        self.log(f"Labelled {num_features} clusters on the heat map. Success: {success}")            
            
        if success:
            # Get cluster indices (1 to num_features)
            cluster_ids = np.arange(1, num_features + 1)

            # Compute total sum of pixel intensities in each cluster
            cluster_sums = ndi_sum(self.heatmap, labeled_array, index=cluster_ids)

            # Compute area (number of pixels) of each cluster
            cluster_areas = ndi_sum(np.ones_like(self.heatmap), labeled_array, index=cluster_ids)

            # Compute mean as usual
            cluster_means = cluster_sums / cluster_areas

            # Peak intensity per cluster
            cluster_peaks = ndi_max(self.heatmap, labeled_array, index=cluster_ids)
            max_area = max(cluster_areas)
            # Custom score: adjust weights as needed
            # You can tune the weights: w1, w2, w3
            w1, w2, w3 = 1.0, 0/max_area, 1  # mean, area, peak weights, max_area normalizes the areas to the intervall 0,1. 
            cluster_scores = (w1 * cluster_means) + (w2* cluster_areas) + (w3 * cluster_peaks)

            arr = cluster_scores

            top_n = num_squares
            top_indices = np.argpartition(arr, -top_n)[-top_n:]
            top_indices = top_indices[np.argsort(arr[top_indices])[::-1]]
            top_cluster_labels = cluster_ids[top_indices]
            print(f"These are the best clusters: {top_cluster_labels }")
            # 4. Create a mask for the top clusters
            self.highlight_mask = np.isin(labeled_array, top_cluster_labels )

            # Get coordinates (center of mass or max position) of top clusters
            
            for rank, cluster_idx in enumerate(top_indices, start=1):
                label_value = cluster_ids[cluster_idx]
                coords = center_of_mass(self.heatmap, labels=labeled_array, index=label_value)
                coords = restore_coordinates(coords, self.Atlas.shape[0], self.Atlas.shape[0], self.scale)
                score = cluster_scores[cluster_idx]
                self.cluster_coords.append((rank, coords, score))


        else: 
            #Return an array only containing True, if there was no success.
            self.highlight_mask = np.full_like(self.heatmap, True, dtype=bool)
            self.log("Could not determine the best clusters. Probably the atlas prediction went wrong.")
        if self.colormap.findText("prediction heat-map") == -1: #check if that is already in the combo box
            self.colormap.addItem("prediction heat-map")
            
        self.colormap.setCurrentText("prediction heat-map")
        self.disable_alignment(False) #enable alignments again.
        self.log(f"Atlas prediction using model {self.atlas_weights_combobox.currentText()} finished.")
        
        #print(self.cluster_coords)
        return self.highlight_mask, self.cluster_coords

            
    def open_atlas_training_dialog(self):
        """Open the dialog window."""
        try:
            scale = float(self.input_mm.text())
            atlas = self.Atlas
            test = self.Locations_rot
        except Exception as e:
            self.log("Atlas training is not possible.")
            self.log(f"Error 25: {e}")
            self.log("Error: No data set loaded.")
            
        else:
            try:
                test = self.Locations_rot["score"]
                

            except Exception as e:
                try: 
                    test = self.Locations_rot["ctf_estimate"]
                except Exception as e:
                    self.log("Atlas training is not possible. Error: No powerspectrum signal or scores available.")
                else:
                    dialog = Atlas_training_Dialog(self.Atlas, scale, self.Locations_rot)
                    if dialog.exec_() == QDialog.Accepted:  # Check if user clicked "OK"
                        self.atlas_params = dialog.get_parameters()
                        self.populate_weights_combobox()
            else:

                dialog = Atlas_training_Dialog(self.Atlas, scale, self.Locations_rot)
                if dialog.exec_() == QDialog.Accepted:  # Check if user clicked "OK"
                    self.atlas_params = dialog.get_parameters()
                    self.populate_weights_combobox()
                    
    
    def open_mic_parameters_dialog(self):
        """Open the dialog window."""
        try:
            scale = float(self.input_mm.text())
            atlas = self.Atlas
            test = self.Locations_rot
        except Exception as e:
            print("Setting micrograph options is not possible")
            print(f"Error: {e}")
            self.log("Error 26: No data set loaded.")
            
        else:
            dialog = mic_options_dialog(self.mic_params)
            if dialog.exec_() == QDialog.Accepted:  # Check if user clicked "OK"
                self.mic_params = dialog.get_parameters()
                self.log(f"Micrograph parameters updated. Pixel size: {self.mic_params['pixel_size']} Å/pixel, Binning factor: {self.mic_params['binning_factor']}x.")

    def update_colormap_limits_widget(self, vmin, vmax, cmap_name):
        """Update the small widget that shows left / middle / right colours and numeric values."""
        try:
            mid = 0.5 * (vmin + vmax)
            cmap = cm.get_cmap(cmap_name)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cols = [cmap(norm(val)) for val in (vmin, mid, vmax)]
            hexcols = [colors.to_hex(c) for c in cols]

            # Build a small HTML snippet: three colored boxes with numeric labels
            html = (
                f'<div style="display:flex; align-items:center; justify-content:center; gap:6px;">'
                f'  <div style="text-align:center;">'
                f'    <div style="width:40px; height:18px; background:{hexcols[0]}; border:1px solid #000;"></div>'
                f'    <div style="font-size:10px;">{vmin:.3f}</div>'
                f'  </div>'
                f'  <div style="text-align:center;">'
                f'    <div style="width:40px; height:18px; background:{hexcols[1]}; border:1px solid #000;"></div>'
                f'    <div style="font-size:10px;">{mid:.3f}</div>'
                f'  </div>'
                f'  <div style="text-align:center;">'
                f'    <div style="width:40px; height:18px; background:{hexcols[2]}; border:1px solid #000;"></div>'
                f'    <div style="font-size:10px;">{vmax:.3f}</div>'
                f'  </div>'
                f'</div>'
            )
            self.colormap_limits.setText(html)
        except Exception as e:
            # on failure, clear widget
            self.colormap_limits.setText("")
            
    
    def recolour(self):
        if self.initial == False:
            try: 
                
                self.xlim = self.sc.ax1.get_xlim()
                self.ylim = self.sc.ax1.get_ylim()
                self.sc.ax1.cla()                
                self.scale = float(self.extend_slider.value())
                scale = self.scale
                if self.Atlas.shape[0] > 5000 and self.Atlas_resolution.currentText() == "low resolution":
                    print("... plotting a 4x-binned version of the atlas.")
                    self.sc.ax1.imshow(self.small_atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")
                else:
                    print(f"Plotting a {self.Atlas.shape[0]} pixel wide atlas.")                   
                    self.sc.ax1.imshow(self.Atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")
            except Exception as e:
                print("Recolouring is not possible.")
                print(f"Error: {e}")
                print("coloring failed at the atlas stage")
                self.log(f"Error 27: {e}")
                
            else:
                self.scale = float(self.extend_slider.value())
                scale = self.scale
                try:
                    if self.colormap.currentText() == "predicted score":
                        self.exposures = self.sc.ax1.scatter(
                            self.Locations_rot["x"], 
                            self.Locations_rot["y"],
                              c = self.Locations_rot["score"],
                                s = 0.5, cmap = "viridis", 
                                vmin = 0, vmax = 1
                                )
                        vmin, vmax, cmap_name = 0.0, 1.0, "viridis"
                    elif self.colormap.currentText() == "cluster":
                        self.exposures = self.sc.ax1.scatter(
                            self.Locations_rot["x"],
                            self.Locations_rot["y"],
                            c = self.Locations_rot["cluster"],
                            s = 0.5, cmap = "viridis"
                            )
                        vmin, vmax, cmap_name = float(self.Locations_rot["cluster"].min()), float(self.Locations_rot["cluster"].max()), "viridis"
                    elif self.colormap.currentText() == "applied defocus":
                        self.exposures = self.sc.ax1.scatter(
                            self.Locations_rot["x"],
                            self.Locations_rot["y"],
                            c = self.Locations_rot["defocus"],
                            s = 0.5, 
                            cmap = "GnBu"
                            )
                        vmin, vmax, cmap_name = float(self.Locations_rot["defocus"].min()), float(self.Locations_rot["defocus"].max()), "GnBu"
                    elif self.colormap.currentText() == "prediction heat-map":
                        self.highlight_mask, self.cluster_coords = self.color_after_atlas_prediction(self.heatmap)
                        if self.heatmap.shape[0] > 5000:
                            small_heatmap = rebin(self.heatmap, (int(self.heatmap.shape[0]/4), int(self.heatmap.shape[1]/4)))                
                            heat = self.sc.ax1.imshow(small_heatmap, alpha = 0.5, vmin=0, vmax = 1, extent=[-1*scale,scale,-1*scale,scale])
                        else:
                            heat = self.sc.ax1.imshow(self.heatmap, alpha = 0.5, vmin=0, vmax = 1, extent=[-1*scale,scale,-1*scale,scale])
                        #only colour by the predicted score if that is available.
                        try:
                            self.exposures = self.sc.ax1.scatter(self.Locations_rot["x"], self.Locations_rot["y"], c = self.Locations_rot["score"], s = 0.5, cmap = "viridis")
                        except:
                            self.exposures = self.sc.ax1.scatter(self.Locations_rot["x"], self.Locations_rot["y"], c = self.Locations_rot["defocus"], s = 0.5, cmap = "GnBu")
                        self.highlight = self.sc.ax1.contour(
                            self.highlight_mask,
                            colors='silver',
                            linewidths=1,
                            extent=[-1*scale,scale,scale,-1*scale]
                                  )
                        if self.cluster_coords != []:
                            for rank, coords, score in self.cluster_coords:
                                x, y = coords
                                self.sc.ax1.text(
                                    x, y,
                                    f"{rank}", 
                                    color='white',
                                    size = "xx-small",
                                    fontweight='bold',
                                    ha='center',
                                    va='center'
                                )
                        vmin, vmax, cmap_name = 0.0, 1.0, "viridis"
                    elif self.colormap.currentText() == "estimated powerspectrum signal":
                        self.exposures = self.sc.ax1.scatter(
                            self.Locations_rot["x"],
                            self.Locations_rot["y"],
                            c = self.Locations_rot["ctf_estimate"],
                            s = 0.5,
                            cmap = "cool_r")
                        vmin, vmax, cmap_name = float(self.Locations_rot["ctf_estimate"].min()), float(self.Locations_rot["ctf_estimate"].max()), "cool_r"
                        
                        self.sc.draw()
                        
                    try:
                        self.update_colormap_limits_widget(vmin, vmax, cmap_name)
                    except Exception:
                        # silent fallback if something unexpected happens
                        pass
                    self.sc.ax1.set_axis_off()
                    self.sc.ax1.set_xlim(self.xlim)
                    self.sc.ax1.set_ylim(self.ylim)
                    self.sc.draw()
                except Exception as e:
                    print("coloring failed at the exposure stage.")
                    self.log(f"Error 28: {e}")
                    
                   
                else:
                    print(f"The current display limits are: {self.xlim} and {self.ylim}")
                    return self.xlim, self.ylim
        else:
            print("no recolouring occuring in the initial phase.")
        
    


               
    def cluster_in_grid_squares(self):
        self.intial = True
        try: 
            num_clusters = int(self.input_squares.text())
            assert num_clusters > 0
            assert num_clusters < len(self.Locations_rot["x"])

        except:
            self.log("Error: Number of grid squares has to be an integer")
            
        else:
            
            self.squares_box.clear()
            for i in range(int(self.input_squares.text())):
                self.squares_box.addItem(f'{i+1}')
                
            self.Locations_rot, self.kmeans = perform_kmeans_clustering(self.Locations_rot, num_clusters)

            if self.colormap.findText("cluster") == -1: #check if that is already in the combo box
                self.colormap.addItem("cluster")
                
            self.colormap.setCurrentText("cluster") #should trigger the recolour function
            self.initial = False
            return self.Locations_rot, self.kmeans, self.initial
        self.initial = False
    
    def mark_grid_square(self):
        if self.initial == False:
            try:
                num_clusters = int(self.input_squares.text())
                assert num_clusters != 0
                assert num_clusters < len(self.Locations_rot["x"])
                
            except:
                self.log("Error: Number of grid squares has to be an integer")

            else:
                try: 
                    self.Locations_rot["cluster"]
                    test = self.kmeans
                    self.Cluster = int(self.squares_box.currentText())
                except:
                    print("wait a little")
                else:
                    self.Cluster = int(self.squares_box.currentText())-1 
                    self.xlim = self.sc.ax1.get_xlim()
                    self.ylim = self.sc.ax1.get_ylim()    
                    self.sc.ax1.cla()


                    self.scale = float(self.extend_slider.value())
                    scale = self.scale
                    self.sc.ax1.imshow(self.small_atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")
                    self.exposures = self.sc.ax1.scatter(self.Locations_rot["x"], self.Locations_rot["y"], c = self.Locations_rot["cluster"], s = 0.5, cmap = "viridis")
                    grid_center = self.sc.ax1.scatter(self.kmeans.cluster_centers_[self.Cluster, 0], self.kmeans.cluster_centers_[self.Cluster, 1], c='red', marker='s', label='Centroids', s = 1.68)
                    self.sc.ax1.set_axis_off()
                    self.sc.ax1.set_xlim(self.xlim)
                    self.sc.ax1.set_ylim(self.ylim)
                    self.sc.draw()
            return self.Cluster
        
    def align_grid_square(self):
        if self.initial == False:
            success = True
            try:
                num_clusters = int(self.input_squares.text())
                assert num_clusters > 0
                assert num_clusters < len(self.Locations_rot["x"])
            except:
                self.log("Error 1: Number of grid squares has to be an integer")
                
                success = False
            try:
                self.kmeans.cluster_centers_
                
            except:
                success = False
                print("kmeans-clustering still running.")
            try:
                test = int(self.squares_box.currentText())-1 #account for the case when there is no number in the squares_box
            except:
                success = False 
            try: 
                cluster_id = int(self.squares_box.currentText())-1
                mask = self.Locations_rot["cluster"] == cluster_id
            except Exception as e:
                print(f"Could not create a mask for the selected cluster. Error: {e}")
                self.log(f"Error 2: {e}")
                
                success = False
                
            if success:
                print(f"Working on cluster {self.Cluster}")
                if self.Cluster != int(self.squares_box.currentText())-1: #Reset offset when the is changed
                    self.grid_offset_x, self.grid_offset_y = 0,0
                    self.grid_x_spinbox.setValue(0)
                    self.grid_y_spinbox.setValue(0)
                self.Cluster = int(self.squares_box.currentText())-1
                print(f"Current cluster: {self.Cluster}")

                # Identify rows matching the target cluster
                mask = self.Locations_rot["cluster"] == self.Cluster

                # Remove old offset from matched rows
                self.Locations_rot.loc[mask, "x"] -= self.grid_offset_x
                self.Locations_rot.loc[mask, "y"] -= self.grid_offset_y
                
                self.Locations_rot.loc[mask, "cluster_offset_x"] -= self.grid_offset_x
                self.Locations_rot.loc[mask, "cluster_offset_y"] -= self.grid_offset_y

                # Update the offsets
                self.grid_offset_x = float(self.grid_x_spinbox.value())
                self.grid_offset_y = float(self.grid_y_spinbox.value())

                # Apply new offset to matched rows
                self.Locations_rot.loc[mask, "x"] += self.grid_offset_x
                self.Locations_rot.loc[mask, "y"] += self.grid_offset_y

                self.Locations_rot.loc[mask, "cluster_offset_x"] += self.grid_offset_x
                self.Locations_rot.loc[mask, "cluster_offset_y"] += self.grid_offset_y
                       
                self.xlim = self.sc.ax1.get_xlim()
                self.ylim = self.sc.ax1.get_ylim()    
                self.sc.ax1.cla()


                self.scale = float(self.extend_slider.value())
                scale = self.scale
                self.sc.ax1.imshow(self.small_atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")

                self.exposures = self.sc.ax1.scatter(self.Locations_rot["x"], self.Locations_rot["y"], c = self.Locations_rot["cluster"], s = 0.5, cmap = "viridis")
                grid_center = self.sc.ax1.scatter(self.kmeans.cluster_centers_[self.Cluster, 0], self.kmeans.cluster_centers_[self.Cluster, 1], c='red', marker='s', label='Centroids', s = 1.68)
                self.sc.ax1.set_axis_off()
                self.sc.ax1.set_xlim(self.xlim)
                self.sc.ax1.set_ylim(self.ylim)
                self.sc.draw()
                
                
                return self.Locations_rot, self.grid_offset_x, self.grid_offset_y, self.scale, self.xlim, self.ylim, self.Cluster
        else:
            print("No realignment taking place in the initial phase.")
            
            
            
    
    def plot_Data(self):
        #Reads the data set and plots it. Is triggered by pressing the plot button
        try:
            x, y, df, self.Locations_rot, Atlas = load_mic_data(offsetx = float(self.input_offsetx.text()),
                                                                offsety = float(self.input_offsety.text()),
                                                                angle = float(self.input_angle.text()),
                                                                Micpath = self.input_xml.text(),
                                                                Atlaspath = self.input_Atlas.text(),
                                                                TIFF = self.tif_button.isChecked(),
                                                                MDOC = self.mdoc_button.isChecked(), 
                                                                scale = float(self.input_mm.text())
                                                               )
            
        except Exception as e:
            self.log(f"Error 3: {e}")
            
            self.Locations_rot = []



        else:
            #Stop any atlas prediction thread
            if hasattr(self, 'threada'):
                    print("Trying to stop the thread")
                    print(self.threada._is_running)
                    self.threada.stop()
                    print("Thread stopped")
                    print(self.threada._is_running)
                    
            #Stop any score prediction thread
            if hasattr(self, 'prediction_thread'):
                self.log("Trying to stop the score prediction thread")
                self.prediction_thread.stop()
                self.prediction_thread.wait()  # Wait for the thread to finish
            if hasattr(self, 'batch_thread'):
                self.log("Trying to stop the batch thread")
                self.batch_thread.stop()
                self.batch_thread.wait()
            if hasattr(self, 'batch_thread_ps'):
                self.log("Trying to stop the batch thread")
                self.batch_thread_ps.stop()
                self.batch_thread_ps.wait()
            if hasattr(self, 'thread_ctf'):
                self.log("Trying to stop the prediction thread")
                self.thread_ctf.stop()
                self.thread_ctf.wait()

            x, y, df, self.Locations_rot, Atlas = load_mic_data(offsetx = float(self.input_offsetx.text()),
                                                                offsety = float(self.input_offsety.text()),
                                                                angle = float(self.input_angle.text()),
                                                                Micpath = self.input_xml.text(),
                                                                Atlaspath = self.input_Atlas.text(),
                                                                TIFF = self.tif_button.isChecked(),
                                                                MDOC = self.mdoc_button.isChecked(),
                                                                scale = float(self.input_mm.text())
                                                               )
            
            self.initial = True #flag to suppress the 
            
            
            self.colormap.clear()
            self.colormap.addItem("applied defocus")
            self.sc.ax1.cla()

            scale = float(self.input_mm.text())
    
            
            self.Atlas = Atlas
            self.Atlas_resolution.clear()
            self.Atlas_resolution.addItem("low resolution")
            if self.Atlas.shape[0] > 5000:
                self.Atlas_resolution.addItem("high resolution")
                small_atlas = rebin(self.Atlas, (int(self.Atlas.shape[0]/4), int(self.Atlas.shape[1]/4)))
                self.small_atlas = small_atlas
                self.sc.ax1.imshow(small_atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")
            else:
                self.small_atlas = self.Atlas
                self.sc.ax1.imshow(self.Atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")
            self.exposures = self.sc.ax1.scatter(x, y, c = df, s = 0.5, cmap = "GnBu")
            #self.cbar = self.sc.ax1.colorbar(exposures, orientation='vertical',location = "left",extend='both', shrink = 0.66, pad = 0.01, label = "defocus (µm)", cax = self.sc.ax2)
            #self.cbar.ax.tick_params(labelsize="xx-small")
            #self.cbar.set_label(label='defocus (µm)', size='xx-small', weight='bold')
            #self.current_hole = self.sc.ax1.scatter(scale,scale, c = "red", s = 2, alpha = 0.7)
            self.sc.ax1.set_axis_off()
            self.sc.draw()
            
            self.angle =  float(self.input_angle.text())/360*2*np.pi
            self.offset_x = float(self.input_offsetx.text())
            self.offset_y = float(self.input_offsety.text())
            self.df = df
            self.Locations_rot["atlas_path"]= self.input_Atlas.text()
            self.Locations_rot["mic_path"] = self.input_xml.text()
            self.Locations_rot["created with version"] = VERSION
            self.initial = False
            self.log(f"Loaded data from {self.input_xml.text() } and {self.input_Atlas.text()}. Plotted {len(self.Locations_rot)} exposures.")
            return self.Locations_rot, self.angle, self.offset_x, self.offset_y, self.df, self.Atlas, self.small_atlas, self.initial

    def disable_alignment(self, Flag=True):
        if Flag:
            self.log("Calculation running. Alignment and predictions are temporarily not available.")

            print("Score update in progress. Alignment and predictions are temporarily not available.")
            

        self.ctf_button.setDisabled(Flag)
        self.plot_button.setDisabled(Flag)
        self.load_button.setDisabled(Flag)

        self.angle_slider.setDisabled(Flag)
        self.angle_spinbox.setDisabled(Flag)
        
        self.extend_slider.setDisabled(Flag)
        self.extend_spinbox.setDisabled(Flag)
        
        self.offset_x_slider.setDisabled(Flag)
        self.offset_x_spinbox.setDisabled(Flag)
        
        self.offset_y_slider.setDisabled(Flag)
        self.offset_y_spinbox.setDisabled(Flag)
        
        self.Scale_box.setDisabled(Flag)
        
        self.input_squares.setDisabled(Flag)
        self.input_squares.setDisabled(Flag)
        self.grid_x_spinbox.setDisabled(Flag)
        self.grid_x_slider.setDisabled(Flag)
        self.grid_y_spinbox.setDisabled(Flag)
        self.grid_y_slider.setDisabled(Flag)
        
        self.squares_box.setDisabled(Flag)
        self.input_squares.setDisabled(Flag)
        self.align_button.setDisabled(Flag)
        
        self.predict_button.setDisabled(Flag)
        self.predict_atlas_button.setDisabled(Flag)
        self.train_atlas_parameters.setDisabled(Flag)
        
    def update_data(self):
        try:
            #Check the extensions of the currently loaded data.
            if self.Locations_rot["xml"][0].endswith(".xml"):
                MDOC = False
            else:
                MDOC = True
                
            if self.Locations_rot["JPG"][0].endswith(".mrc"):
                TIFF = False
            else:
                TIFF = True
                
            # Load new data
            x, y, df, new_Locations_rot, Atlas = load_mic_data(
                offsetx=self.Locations_rot["offset_x"][0],
                offsety=self.Locations_rot["offset_y"][0],
                angle=self.Locations_rot["angle"][0],
                Micpath=self.Locations_rot["mic_path"][0],
                Atlaspath=self.Locations_rot["atlas_path"][0],
                TIFF=TIFF,
                MDOC=MDOC,
                scale = self.Locations_rot["scale"][0]
            )

            
        except Exception as e:
            self.log("Error: Invalid path or atlas parameters")
            self.log(f"Error 4: {e}")
            return


        self.initial = True
        self.log("Looking for new exposures ...")

        if "cluster" in self.Locations_rot.columns:
            #Preserve clustering information if available
            self.log("Detected existing grid square alignment. Applying to new data ...")
            num_clusters = max(self.Locations_rot["cluster"])+1
            new_Locations_rot, self.kmeans = perform_kmeans_clustering(new_Locations_rot, num_clusters)
            
            #Reading the aligned offsets for the clusters from the existing data set
            cluster_offsets = self.Locations_rot.groupby("cluster")[["cluster_offset_x", "cluster_offset_y"]].mean().to_dict(orient="index")
            print(f"Cluster offsets to be applied: {cluster_offsets}") 

            #Apply the offsets
            for col in ["cluster_offset_x", "cluster_offset_y"]:
                new_Locations_rot[col] = new_Locations_rot["cluster"].map(lambda c: cluster_offsets.get(c, {}).get(col, 0))
           
            new_Locations_rot["x"] += new_Locations_rot["cluster_offset_x"]
            new_Locations_rot["y"] += new_Locations_rot["cluster_offset_y"]
            



        # Identify new entries based on the unique 'JPG' column
        existing_ids = set(self.Locations_rot["JPG"])
        new_rows = new_Locations_rot[~new_Locations_rot["JPG"].isin(existing_ids)]
                
        if not new_rows.empty:
            
            self.log(f"Found {len(new_rows)} new exposures.")
            if "score" in self.Locations_rot.columns:
                predict_scores = True
                self.log(f"Starting score prediction update.")
            else:
                predict_scores = False

            if "ctf_estimate" in self.Locations_rot.columns:
                predict_ctf = True
                self.log(f"Starting powerspectrum signal estimation update.")
            else:
                predict_ctf = False
                
            if predict_ctf or predict_scores:
                self.disable_alignment(True)  # prevent realignments and any action during score update
                print(f"self.Locations_rot.columns: {self.Locations_rot.columns}")
                print(f"new_Locations_rot.columns: {new_Locations_rot.columns}")
                # Perform a left merge to bring in the 'score' from Locations_rot
                if predict_scores:
                    self.score_update_running = True

                    merged = new_Locations_rot.merge(
                        self.Locations_rot[['JPG', 'score', "model"]],
                        on='JPG',
                        how='left'
                    )

                    # Fill missing scores with -1
                    merged['score'] = merged['score'].fillna(-1)

                    # Update new_Locations_rot with the merged score
                    new_Locations_rot['score'] = merged['score']
                    new_Locations_rot["model"] = self.Locations_rot["model"][0]
                    
                    
                    _ = self.update_score_prediction(new_Locations_rot) #Starts the prediction update. Calls on_update_finished after it is done, which handles merging the columns
                    

                if predict_ctf:
                    self.ctf_update_running = True
                    print(self.Locations_rot.columns)
                    print(new_Locations_rot.columns)
                    merged_ctf = new_Locations_rot.merge(
                        self.Locations_rot[['JPG', 'ctf_estimate', "defocus"]],
                        on='JPG',
                        how='left'
                    )
                    print(merged_ctf)

                    # Fill missing ctf_estimate with -1
                    merged_ctf['ctf_estimate'] = merged_ctf['ctf_estimate'].fillna(-1)

                    # Update new_Locations_rot with the merged ctf_estimate
                    new_Locations_rot['ctf_estimate'] = merged_ctf['ctf_estimate']

                    _ = self.update_ctf_estimation(new_Locations_rot) #Starts the prediction update. Calls on_update_finished after it is done, which handles merging the columns
        
                
                #Updating the Locations_rot will be handled in the on_update_finished function after the predictions are done
                self.Locations_rot = new_Locations_rot

                self.initial = False
                return self.Locations_rot
                
            else:
                
                self.log("No score or powerspectrum signal prediction necessary. Updating data directly.")

                self.Locations_rot = new_Locations_rot


                self.Locations_rot.reset_index(drop=True, inplace=True)

                self.sc.ax1.cla()

                scale = self.Locations_rot["scale"][0]

                self.Atlas = Atlas
                self.Atlas_resolution.clear()
                self.Atlas_resolution.addItem("low resolution")
                if self.Atlas.shape[0] > 5000:
                    self.Atlas_resolution.addItem("high resolution")
                    small_atlas = rebin(self.Atlas, (int(self.Atlas.shape[0]/4), int(self.Atlas.shape[1]/4)))
                    self.small_atlas = small_atlas
                    self.sc.ax1.imshow(small_atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")
                else:
                    self.small_atlas = self.Atlas
                    self.sc.ax1.imshow(self.Atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")
                self.exposures = self.sc.ax1.scatter(x, y, c = df, s = 0.5, cmap = "GnBu")

                #self.current_hole = self.sc.ax1.scatter(scale,scale, c = "red", s = 2, alpha = 0.7)
                self.sc.ax1.set_axis_off()
                self.sc.draw()
                self.initial = False        

        else:
            self.log("No new data to update.")
            
        self.initial = False
        
        return self.Locations_rot

    def realign(self):
        try:
            test = self.Atlas
            #Retrieve the x,y coordinates and the current offset
            x_rot = self.Locations_rot["x"]
            y_rot = self.Locations_rot["y"]
            self.offset_x = self.Locations_rot["offset_x"][0]
            self.offset_y = self.Locations_rot["offset_y"][0]
            self.angle = self.Locations_rot["angle"][0]/360*2*np.pi
            
        except:
            self.log("No data loaded. Realignment currently not possible.")
            
        if not self.initial: 
            
            #print(self.angle)



            #Calculate the unrotated and unshifted coordinates
            self.Locations_rot["x"] = np.cos(-1*self.angle) * x_rot - np.sin(-1*self.angle) * y_rot
            self.Locations_rot["y"] = np.sin(-1*self.angle) * x_rot + np.cos(-1*self.angle) * y_rot
            
            self.Locations_rot["x"] -= self.offset_x
            self.Locations_rot["y"] -= self.offset_y

            # Read the new shifts and angles

            self.offset_x = float(self.offset_x_slider.value())
            self.offset_y = float(self.offset_y_slider.value())
            self.angle = float(self.angle_spinbox.value())/360*2*np.pi

            # Add the new shifts

            self.Locations_rot["x"] += self.offset_x 
            self.Locations_rot["y"] += self.offset_y         
            self.Locations_rot["offset_x"] = self.offset_x
            self.Locations_rot["offset_y"] = self.offset_y
            
            #Calculate the rotated coordinates
            # x_rot = np.cos(angle) * x - np.sin(angle) * y
            # y_rot = np.sin(angle) * x + np.cos(angle) * y
            x_unrot = self.Locations_rot["x"]
            y_unrot = self.Locations_rot["y"]
            self.Locations_rot["x"] = np.cos(self.angle) * x_unrot - np.sin(self.angle) * y_unrot
            self.Locations_rot["y"] = np.sin(self.angle) * x_unrot + np.cos(self.angle) * y_unrot
            
            
            x,y = self.Locations_rot["x"], self.Locations_rot["y"]
            #Reset the drawing area

            #Update the current atlas extent (scale)
            self.scale = float(self.extend_slider.value())
            scale = self.scale
            #Store the current angle and scale values
            self.Locations_rot["scale"] = scale
            self.Locations_rot["angle"] = float(self.angle_spinbox.value()) 
            
            #Plot the realigned coordinates.
            
            self.xlim = self.sc.ax1.get_xlim()
            self.ylim = self.sc.ax1.get_ylim()
            
            self.sc.ax1.cla()
        
            self.sc.ax1.imshow(self.small_atlas, cmap ="gray",extent=[-1*scale,scale,-1*scale,scale], norm = "linear")
            self.exposures = self.sc.ax1.scatter(x, y, c = self.Locations_rot["defocus"], s = 0.5, cmap = "GnBu")
            #self.current_hole = self.sc.ax1.scatter(scale,scale, c = "red", s = 2, alpha = 0.7)
            self.sc.ax1.set_axis_off()
            self.sc.ax1.set_xlim(self.xlim)
            self.sc.ax1.set_ylim(self.ylim)
            self.sc.draw()


            return self.Locations_rot, self.offset_x, self.offset_y, self.angle, self.scale
        
    def turn_on_pixel_input(self):
        self.input_pix.setEnabled(self.Scale_box.isChecked())
        self.input_res.setEnabled(self.Scale_box.isChecked())
        self.input_length.setEnabled(self.Scale_box.isChecked())
        
    def turn_on_fft_input(self):
        self.input_bin.setEnabled(self.FFT_box.isChecked())
    
    def onclick(self, event):
        
        self.x = event.xdata
        self.y = event.ydata
        x1,y1 = float(self.x), float(self.y)
        dataframe = self.Locations_rot

        dataframe = dataframe.assign(distance = [calc_distance(x1,y1,x2,y2) for x2,y2 in zip(list(dataframe["x"]),list(dataframe["y"]))])

        minimum = min(dataframe.iloc[:,-1])

        
        hits = dataframe[dataframe.iloc[:,-1] == minimum]

        if len(hits) == 1:
            #self.current_hole.remove()
            #self.recolour()
            self.x_hole, self.y_hole = hits["x"].iloc[0],hits["y"].iloc[0]
            #self.current_hole = self.sc.ax1.scatter(self.x_hole,self.y_hole, c = "red", s =2, alpha = 0.7)
            if hasattr(self, 'current_hole') and self.current_hole in self.sc.ax1.collections:
                self.current_hole.set_offsets([[self.x_hole, self.y_hole]])
            else:
                self.current_hole = self.sc.ax1.scatter(self.x_hole, self.y_hole, c = "red", s =2, alpha = 0.7)
                
            self.Mic.ax1.cla()
            self.Mic.ax2.cla()
            try:
                if hits["JPG"].iloc[0].endswith(".mrc"):
                    with mrcfile.open(hits["JPG"].iloc[0]) as mrc:
                        Micrograph=mrc.data[:]
                else:
                    Micrograph = tifffile.imread(hits["JPG"].iloc[0])

                if len(np.shape(Micrograph)) > 2: #summing movies
                    Micrograph = np.sum(Micrograph, axis=np.argmin(Micrograph.shape)) #sums over the smallest axis, should be time
            except Exception as e:
                self.log("Error: Could not read micrograph")
                
                self.log(f"Error 5: {e}")
            else:
                print("Micrograph loaded successfully")

            try:
                bin_factor = self.mic_params["binning_factor"]
                Bin = rebin(Micrograph, (int(Micrograph.shape[0]/bin_factor), int(Micrograph.shape[1]/bin_factor)))
            except:
                self.log("Error: Invalid binning factor")
                
            else:
                

                bin_factor = self.mic_params["binning_factor"]
                Pix_x= Micrograph.shape[1]/bin_factor
                Pix_y= Micrograph.shape[0]/bin_factor
                Bin = rebin(Micrograph, (int(Micrograph.shape[0]/bin_factor), int(Micrograph.shape[1]/bin_factor)))
                

                if self.mic_params["FFT"]==True:


                        ft = np.fft.ifftshift(Bin)

                        ft = np.fft.fft2(ft)

                        #Thon = np.log(np.abs(np.fft.fftshift(ft)))
                        Thon = np.log(np.abs(np.fft.fftshift(ft)))
                        #Thon = rebin(Thon, (int(Thon.shape[0]/4), int(Thon.shape[1]/4)))
                        vmin,vmax = contrast_normalization(Thon)

                        self.Mic.ax2.imshow(Thon, cmap ="gray", extent=[-1,1,-1,1], filternorm= True, vmin=vmin, vmax=vmax)

                #Normalize using 0.001st and 99.999th percentile, ~160 pixels in a 4k image
                lo = np.percentile(Micrograph, 0.001)
                hi = np.percentile(Micrograph, 99.999)
                self.Mic.ax1.imshow(Bin, cmap ="gray", vmin=lo, vmax=hi)
                #self.Mic.ax1.imshow(Bin, cmap ="gray")

                self.Mic.ax1.text(
                    0.05*Pix_x,
                    0.05*Pix_y,
                    "appl. defocus {:.1f} µm".format(hits["defocus"].iloc[0]),
                    c ="white",
                    size = "xx-small",
                    horizontalalignment='left',
                    verticalalignment='bottom'
                )

                if self.mic_params["draw_scale"]==True:
                    try:
                        size = self.mic_params["scale_length"]
                        pixelsize = self.mic_params["pixel_size"]
 

                    except:
                        self.log("Invalid input for scale options")
                        

                    else:
                        
                        offset = Pix_x*0.05
                        offsety = Pix_y*0.05

                        #size = float(self.input_length.text())
                        size = self.mic_params["scale_length"]
                        #pixelsize = float(self.input_pix.text())*bin_factor
                        pixelsize = self.mic_params["pixel_size"]*bin_factor

                        x_marker = [Pix_x - size/pixelsize - offset,Pix_x - offset]
                        y_marker = [Pix_y-offsety,Pix_y-offsety]
                        self.Mic.ax1.plot(x_marker, y_marker, c = "white", linewidth = 1)
                        self.Mic.ax1.text(np.mean(x_marker), np.mean(y_marker)*0.99,"{:.0f} Å".format(round(size,0)), horizontalalignment='center', verticalalignment='bottom', c = "white", size = "xx-small")
                        
                        #if self.FFT_box.isChecked()==True:
                        if self.mic_params["FFT"]==True:
                            #radius = (2*pixelsize)/float(self.input_res.text())
                            FFT_resolution = self.mic_params["FFT_scale"]
                            radius = (2*pixelsize)/FFT_resolution
                            Res_ring = plt.Circle((0, 0), radius, color='w', fill=False, lw = 0.33)
                            self.Mic.ax2.add_patch(Res_ring)
                            self.Mic.ax2.text(0, radius*-1.03 ,"{:.1f} Å".format(FFT_resolution), c ="white", fontsize=4, horizontalalignment='center', verticalalignment='top')
            self.Mic.ax1.set_axis_off()
            self.Mic.ax2.set_axis_off()
            self.Mic.draw()
     
            #self.Mic.ax1.set_title(hits.iloc[0,0].rsplit('/', 1)[-1])
            #self.label_xy.setText("X:{:.1f} µm, Y: {:.1f} µm".format(hits.iloc[0,1],hits.iloc[0,2]))
            self.label_xy.setText("{}".format(hits["JPG"].iloc[0].rsplit('/', 1)[-1]))
            msg = f"Micrograph: {hits['JPG'].iloc[0].rsplit('/', 1)[-1]} recorded with a defocus of {hits['defocus'].iloc[0]:.1f} µm"
            

            if "score" in hits.columns:
                msg += f", predicted score: {hits['score'].iloc[0]:.3f}"
            if "ctf_estimate" in hits.columns:
                msg += f", estimated powerspectrum signal to: {hits['ctf_estimate'].iloc[0]:.3f} Nyquist"
            self.log(msg)
        else: 
            self.log("Something is wrong with the coordinates")

        return self.x, self.y, self.x_hole, self.y_hole
    

    
        
 


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
