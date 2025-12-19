import sys
import matplotlib
matplotlib.use('Qt5Agg')
from tqdm import tqdm
import os
import math
from Training import * # Import your ResNet model (adjust as needed)

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import mrcfile
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from scipy.spatial import distance
from PIL import Image
import random
from torch.utils.data import Dataset, random_split, DataLoader
import re
from sklearn.cluster import KMeans
from scipy.ndimage import uniform_filter1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def det_batch_size(size):
    """
    Returns an appropriate batch size based on the input image size.
    """
    if size < 257:
        batch_size = 8
    elif size < 513:
        batch_size = 8
    elif size < 2046:
        batch_size = 8
    else:
        batch_size = 8
    return batch_size

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

    
def preprocess_mrc(image_path, size, Fourier = False):
    """
    Load and preprocess an .mrc or .tif image: 
    - sums movies
    - Normalize between 0 and 1 (based on min-max normalization)
    - Convert to tensor and ensure correct shape
    """
    # Load image
    if image_path.lower().endswith("mrc"):
        with mrcfile.open(image_path, permissive=True) as mrc:
            image = mrc.data.astype(np.float32)
    elif image_path.lower().endswith(("tif", "tiff")):
        image = tifffile.imread(image_path).astype(np.float32)
    else:
        raise ValueError("Unsupported file format. Only .mrc, .tif, or .tiff are allowed.")

    # If it's a movie, sum across the smallest axis, most probably the frames
    if image.ndim == 3:
        np.sum(image, axis=np.argmin(image.shape))
        
    #Normalize using 1st and 99th percentile
    lo = np.percentile(image, 1)
    hi = np.percentile(image, 99)
    
    image = (image - lo) / (hi - lo)
    image = np.clip(image, 0, 1)  # ensure strictly in [0,1]

    # Convert to PyTorch tensor
    image = torch.from_numpy(image).to(device)
    
    if Fourier:
        transform = transforms.Compose([
            LogNormalizedPowerSpectrum(),
            transforms.Resize((size, size))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((size, size))
        ])

    # Ensure it has a single channel (C, H, W)
    if image.ndimension() == 2:
        image = image.unsqueeze(0)  # Add channel dimension: (1, H, W)


    image = transform(image)

    return image

class LogNormalizedPowerSpectrum:
    def __call__(self, img):
        if not torch.is_tensor(img):
            img = transforms.functional.to_tensor(img)

        if img.shape[0] > 1:
            img = img.mean(dim=0)

        fft = torch.fft.fft2(img)
        fft = torch.fft.fftshift(fft)
        power = torch.abs(fft) ** 2

        log_power = torch.log(power + 1e-8) # avoid log(0)
        log_power -= log_power.min()
        log_power /= log_power.max()

        return log_power
    
def load_model(model_path, device, model_name, dropout):
    """
    Load a ResNet model and its weights.
    """
    cls = {
        "ResNet4": ResNet4, "ResNet6": ResNet6, "ResNet8": ResNet8, "ResNet10": ResNet10, "ResNet12": ResNet12, "ResNet34": ResNet34, "ResNet50": ResNet50, "ResNet152": ResNet152, "CoordNet8": CoordNet8, "RCoordNet8": RCoordNet8
    }.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = cls(dropout)
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle checkpoints with or without a 'state_dict' key
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Check if 'module.' prefix exists in keys
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_patches(image, coord_list, patch_size=32, dark = True, light = True):
    """
    Extracts 32x32 patches centered at given coordinates with associated scores.

    :param image: Input image (NumPy array)
    :param coordinates: List of (x, y, score) tuples
    :param patch_size: Size of the square patches (default: 32)
    :return: (patches, scores) - List of patches and corresponding scores
    """
    height, width = image.shape[:2]
    half_size = patch_size // 2
    patches, scores = [], []
    if np.min(image) != np.max(image):
        image = (image - np.min(image))/(np.max(image)-np.min(image))
    for x, y, score in coord_list:
        x1, y1 = max(0, x - half_size), max(0, y - half_size)
        x2, y2 = min(width, x + half_size), min(height, y + half_size)
        
        # Extract and pad if necessary
        patch = image[y1:y2, x1:x2]
        #print(f"Extracting image with a shape of {patch.shape[0]}, {patch.shape[1]}.")
        if (
            isinstance(score, (int, float)) and 
            not math.isnan(score) and 
            0 < score < 1 and 
            patch.shape[0] == patch_size and 
            patch.shape[1] == patch_size
        ):

            patches.append(patch)
            scores.append(score)
        else:
            print(f"INFO: Skipping patch at ({x},{y}) with score {score} and shape {patch.shape}")
                
    # Add dark patches from random image locations (max `ten` patches)
    ten = len(patches) // 10 + 1
    added = 0
    max_random_trials = 10000
    if dark:
        for _ in range(max_random_trials):
            if added >= ten:
                break

            # Random coordinate (center point), ensure patch fits in image
            x = np.random.randint(half_size, width - half_size)
            y = np.random.randint(half_size, height - half_size)

            # Extract patch
            patch = image[y - half_size:y + half_size, x - half_size:x + half_size]
            patch_mean = patch.mean()

            if patch.shape == (patch_size, patch_size) and patch_mean < 0.03:
                patches.append(patch)
                scores.append(0)
                added += 1
    else:
        added = 0
    

    if light:
        # Step 1: Find 10,000 brightest pixels
        flattened_indices = np.argsort(image.ravel())[::-1]  # Brightest first
        bright_coords = np.column_stack(np.unravel_index(flattened_indices[:10000], image.shape))

        # Step 2: Filter coordinates that allow full patch extraction
        valid_coords_with_mean = []
        for y, x in bright_coords:
            if (x - half_size >= 0 and x + half_size <= width and
                y - half_size >= 0 and y + half_size <= height):
                patch = image[y - half_size:y + half_size, x - half_size:x + half_size]
                if patch.shape == (patch_size, patch_size):
                    patch_mean = patch.mean()
                    valid_coords_with_mean.append(((x, y), patch_mean))

        # Step 3: Sort patches by brightness and keep top 1,000
        valid_coords_with_mean.sort(key=lambda x: x[1], reverse=True)
        top_bright_coords = [coord for coord, mean in valid_coords_with_mean[:10000]]

        # Step 4: Select patches while enforcing distance constraint
        selected_coords = []
        for coord in top_bright_coords:
            if all(np.linalg.norm(np.subtract(coord, existing)) > half_size for existing in selected_coords):
                selected_coords.append(coord)
                if len(selected_coords) >= ten:
                    break

        # Step 5: Extract and store the selected light patches
        added_light = 0
        for x, y in selected_coords:
            patch = image[y - half_size:y + half_size, x - half_size:x + half_size]
            if patch.shape == (patch_size, patch_size):
                patches.append(patch)
                scores.append(0)
                added_light += 1
    else:
        added_light = 0

    print(f"Added {added} dark and {added_light} light background patches (from top 1,000 ranked patches")
    
    # Plot up to 100 background (score = 0) patches
    background_patches = [patch for patch, score in zip(patches, scores) if score == 0]
    
    num_to_plot = min(100, len(background_patches))
    random.shuffle(background_patches) #randomize the order of the images
    
    if num_to_plot > 0:
        grid_size = int(np.ceil(np.sqrt(num_to_plot)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12), squeeze=False)
        axes = axes.flatten()

        for i in range(num_to_plot):
            axes[i].imshow(background_patches[i], cmap='gray')
            axes[i].axis('off')

        # Hide any extra axes
        for j in range(num_to_plot, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig("./tmp/patches_junk.png", dpi=300)
        plt.close()
        
    # Plot up to 100 background (score = 0) patches
    weird_stuff_patches = [patch for patch, score in zip(patches, scores) if (score > 0 and score < 0.5)]
    num_to_plot = min(100, len(weird_stuff_patches))
    
    if num_to_plot > 0:
        grid_size = int(np.ceil(np.sqrt(num_to_plot)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12), squeeze=False)
        axes = axes.flatten()

        for i in range(num_to_plot):
            axes[i].imshow(weird_stuff_patches[i], cmap='gray')
            axes[i].axis('off')

        # Hide any extra axes
        for j in range(num_to_plot, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig("./tmp/patches_bad.png", dpi=300)
        plt.close()
    
        # Plot up to 100 background (score = 0) patches
    great_patches = [patch for patch, score in zip(patches, scores) if score > 0.8]
    num_to_plot = min(100, len(great_patches))
    
    if num_to_plot > 0:
        grid_size = int(np.ceil(np.sqrt(num_to_plot)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12), squeeze=False)
        axes = axes.flatten()

        for i in range(num_to_plot):
            axes[i].imshow(great_patches[i], cmap='gray')
            axes[i].axis('off')

        # Hide any extra axes
        for j in range(num_to_plot, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig("./tmp/patches_great.png", dpi=300)
        plt.close()

    return np.array(patches), np.array(scores)
    
def extract_overlapping_patches_generator(image, patch_size=32, step=4):
    height, width = image.shape[:2]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    for y in range(0, height - patch_size + 1, step):
        for x in range(0, width - patch_size + 1, step):
            patch = image[y:y+patch_size, x:x+patch_size]
            center = (x + patch_size // 2, y + patch_size // 2)
            yield patch, center
            

class AtlasScoreDataset(Dataset):
    def __init__(self, patches, scores):
        self.patches = patches
        #self.transform = transform
        self.scores = scores
        self.data = []
        
        # Read in the lists
        for patch, score in list(zip(patches, scores)):
            self.data.append((patch, score))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch, score = self.data[idx]
        #image = Image.fromarray(patch)  # Convert back to PIL
        image = torch.from_numpy(patch).to(torch.float32).to(device)


        # Ensure it has a single channel (C, H, W)
        if image.ndimension() == 2:
            image = image.unsqueeze(0)  # Add channel dimension: (1, H, W)


        return image, torch.tensor(score, dtype=torch.float32)



def recalculate_coordinates(coord_list, image_width, image_height, atlas_extent):
    """
    Recalculates coordinates from a center-origin system to an upper-left-origin system 
    and rescales them to a new size.

    :param coord_list: List of (x, y, score) tuples with original center-based coordinates.
    :param image_width: Width of the original image.
    :param image_height: Height of the original image.
    :param scale_factor: Scaling factor to adjust coordinate values.
    :return: List of transformed (x, y, score) tuples.
    """
    new_coords = []
    # Find center
    scale_factor = image_height/2/atlas_extent

    for x, y, score in coord_list:
        # Shift origin from center to upper left
        new_x = int((x + atlas_extent) * scale_factor)
        new_y = int((y - atlas_extent) *-1*scale_factor)
        
        new_coords.append((new_x, new_y, score))

    return new_coords

def restore_coordinates(coords, image_width, image_height, atlas_extent):
    """
    Converts coordinates from upper-left-origin pixel space back to center-origin, real-world scale.

    :param coords: Tuple of (x, y, score) tuples with pixel coordinates.
    :param image_width: Width of the image in pixels.
    :param image_height: Height of the image in pixels.
    :param atlas_extent: Real-world extent used during the forward transformation.
    :return: List of (x, y, score) tuples in center-origin, real-world coordinates.
    """
    # Must match scale_factor used in forward transformation
    scale_factor = image_height / 2 / atlas_extent
    x_px, y_px = coords
    # Reverse scaling and origin shift
    xt = x_px / scale_factor - atlas_extent
    yt = atlas_extent - y_px / scale_factor
    # Rotate 180 degrees
    y = -xt
    x = -yt
    return (x, y)

def preprocess_patch(image):

    image = torch.from_numpy(image).to(torch.float32).to(device)


    # Ensure it has a single channel (C, H, W)
    if image.ndimension() == 2:
        image = image.unsqueeze(0)  # Add channel dimension: (1, H, W)



    return image

    
def predict(model, image_tensor, device, batch = False):
    """
    Run inference on a single image.
    """
    if batch == False:
        image_tensor = image_tensor.unsqueeze(0).to(device) # Add batch dimension

        
        with torch.no_grad():
            score = model(image_tensor)  # Forward pass
        return score.item()
    
    else:
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            score = model(image_tensor)  # Forward pass
        return score
        
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def reflect_point(x, y, angle):
    """
    Reflects a 2D point across a line with the given angle to the x-axis.
    
    Parameters:
    - x: X coordinate of the point
    - y: Y coordinate of the point
    - angle: Angle of the line with the x-axis in radians
    
    Returns:
    - x_reflected: Reflected X coordinate as float
    - y_reflected: Reflected Y coordinate as float
    """
    angle = angle/360*2*np.pi
    # Calculate the reflection matrix
    cos_2theta = np.cos(2 * angle)
    sin_2theta = np.sin(2 * angle)
    reflection_matrix = np.array([
        [cos_2theta, sin_2theta],
        [sin_2theta, -cos_2theta]
    ])
    
    # Apply the reflection matrix
    point = np.array([x, y])
    reflected_point = reflection_matrix @ point
    
    return float(reflected_point[0]), float(reflected_point[1])

def get_xy_rotated(xml_file, offsetx, offsety, angle = 170, angle_s=0, mirror_angle = 45):
    """
    Extracts and rotates x, y coordinates and defocus from an XML or MDOC file.
    xml_file: Path to the XML or MDOC file
    angle is the rotation angle of the stage
    angle_s is the rotation angle of the beamshifts
    mirror_angle is the angle of the mirror line for the beamshifts with regard to the x-axis
    """
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
        #Beam shift in µm? Empirical factor is used to convert the beamshift given in the xml file
        scale_beamshift = 22.5 
        #Read the x and y coordinate from the xml file
        Start = String_list_xml[2]
        End = String_list_xml[3]
        
        x=float(meta[meta.index(Start)+len(Start):meta.index(End)])*1000*1000
        Start = String_list_xml[4]
        End = String_list_xml[5]
        x_shift=float(Beamshift[Beamshift.index(Start)+len(Start):Beamshift.index(End)])

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
        x_shift=float(Beamshift[Beamshift.index(Start)+len(Start):Beamshift.index(End)])*scale_beamshift

        Start = String_list_xml[6]
        End = String_list_xml[7]
        y=float(meta[meta.index(Start)+len(Start):meta.index(End)])*1000*1000
        
        Start = String_list_xml[8]
        End = String_list_xml[9]
        y_shift=float(Beamshift[Beamshift.index(Start)+len(Start):Beamshift.index(End)])*scale_beamshift

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


    angle_s = angle_s/360*2*np.pi
         
    x_r = np.cos(angle_s) * x_shift - np.sin(angle_s) * y_shift
    y_r = np.sin(angle_s) * x_shift + np.cos(angle_s) * y_shift
    x_shift, y_shift = x_r, y_r
    
    x_shift, y_shift = reflect_point(x_shift, y_shift, mirror_angle)
                      
 
    x += x_shift
    x += offsetx

    y += y_shift
    y += offsety
    
    #Calculate the rotated coordinates
    x_rot = np.cos(angle) * x - np.sin(angle) * y
    y_rot = np.sin(angle) * x + np.cos(angle) * y
    
    #Apply beam shift after rotation
    #y_rot += y_shift
    #x_rot += x_shift
 
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

import math

def stitch_tiles(tiles, tile_coords, pixel_size):
    if not tiles:
        raise ValueError("No tiles provided.")

    # 1) convert real-world → pixel coords
    pixel_coords = [(x / pixel_size *-1, y / pixel_size*-1) for (x, y) in tile_coords]

    # 2) get flat lists of X and Y
    
    
    ys = np.array([pt[0] for pt in pixel_coords])
    xs = np.array([pt[1] for pt in pixel_coords])
    #create empty list containing the rotated coordinates
    angle = 0/360*2*np.pi
    x_rots = []
    y_rots = []
    #print(x_rots)
    for x,y in list(zip(xs,ys)):
        x_rot = np.cos(angle) * x - np.sin(angle) * y
        y_rot = np.sin(angle) * x + np.cos(angle) * y
        x_rots.append(x_rot)
        y_rots.append(y_rot)
    #print(x_rots)

    # 3) cluster them into discrete rows/cols
    def cluster_axis(vals, tol):
        sv = np.sort(vals)
        centers = []
        group = [sv[0]]
        for v in sv[1:]:
            if abs(v - group[-1]) < tol:
                group.append(v)
            else:
                centers.append(np.mean(group))
                group = [v]
        centers.append(np.mean(group))
        return centers

    # use ~10% of tile size as grouping tolerance
    tile_h, tile_w = tiles[0].shape
    row_centers = cluster_axis(y_rots, tile_h * 0.1)
    col_centers = cluster_axis(x_rots, tile_w * 0.1)
    #print(row_centers)
    #print(col_centers)
    num_rows, num_cols = len(row_centers), len(col_centers)

    # 4) build an empty (row x col) grid
    tile_grid = [[None]*num_cols for _ in range(num_rows)]
    pixel_coords_rot = zip(x_rots,y_rots)
    # 5) assign each tile
    for idx, (tile, (x_pix, y_pix)) in enumerate(zip(tiles, pixel_coords_rot)):
        # *** CORRECTED axes! ***
        row_idx = np.argmin([abs(y_pix - rc) for rc in row_centers])
        col_idx = np.argmin([abs(x_pix - cc) for cc in col_centers])
        print(f"Tile #{idx}: x={x_pix:.1f}, y={y_pix:.1f} → row {row_idx}, col {col_idx}")
        if tile_grid[row_idx][col_idx] is not None:
            print(f"  ⚠️ collision at ({row_idx},{col_idx}) — overwriting!")
        tile_grid[row_idx][col_idx] = tile

    # 6) compose final image
    H = num_rows * tile_h
    W = num_cols * tile_w
    canvas = np.zeros((H, W), dtype=tiles[0].dtype)
    for r in range(num_rows):
        for c in range(num_cols):
            t = tile_grid[r][c]
            if t is not None:
                y0, x0 = r*tile_h, c*tile_w
                canvas[y0:y0+tile_h, x0:x0+tile_w] = t

    return canvas




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
        #print(tile_coords)
    
    
    # Stitch tiles together
    stitched_image = stitch_tiles(tiles, tile_coords, pixel_size)
    
    return stitched_image
    
def load_mic_data(angle, offsetx, offsety, Micpath,Atlaspath, angle_s=65, mirror_angle = 45, TIFF = False, MDOC = False, scale = 910):
    #Create an empty dictionary
    Data_rot = {}
    p = Path(Micpath)
    #Load the exposures should be located in a Data folder
    if MDOC == False:
        No_foilhole = [path for path in list(p.glob('**/*.xml')) if ("Data" in path.parts or "Images" in path.parts) and not p.name.endswith("_Fractions.xml")]
    else:
        No_foilhole = [path for path in list(p.glob('**/*.mdoc'))]
        
    for i in No_foilhole:
        Data_rot[str(i)]=get_xy_rotated(i, offsetx, offsety, angle, angle_s, mirror_angle)
    
    pd.set_option("display.precision", 4)
    #Rearange the data base file
    Locations_rot=pd.DataFrame(Data_rot).T
    Locations_rot.round(4)
    #Extract list of x and y for plotting purposed in the plot function
    x, y, df = list(zip(*Data_rot.values()))
    
    #Reorder the data base new column contains the path of the exposures
    Locations_rot = Locations_rot.reset_index().rename(columns={"index":"xml"})
    Locations_rot.columns.values[1:4] = ["x", "y", "defocus"]
    

    
    if MDOC == False:
        if TIFF == False:
            Locations_rot = Locations_rot.assign(JPG = [w.replace(".xml",".mrc") for w in list(Locations_rot.iloc[:,0])])
    
        else:
            Locations_rot = Locations_rot.assign(JPG = [w.replace(".xml",".tiff") for w in list(Locations_rot.iloc[:,0])])
    else:
        Locations_rot = Locations_rot.assign(JPG = [w.replace(".mdoc","") for w in list(Locations_rot.iloc[:,0])])

    all_files = [str(file) for file in list(p.glob('**/*'))]
    #Check if the first micrograph is present
    assert Locations_rot.loc[0,"JPG"] in all_files, ValueError("No micrograph found. Try another file format (tiff/mrc)")

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
        print(f"Stitching an atlas from: {Atlaspath}")
        Atlas = stitch_atlas(Atlaspath)
        #print(Atlas)
        
        
    #print(Atlas.shape)
    
    #Set the angle, data paths and offset values
    
    Locations_rot["atlas_path"] = Atlaspath
    Locations_rot["mic_path"] = Micpath
    Locations_rot["scale"] = scale
    Locations_rot["angle"] = angle
    Locations_rot["offset_x"] = offsetx
    Locations_rot["offset_y"] = offsety
    #print(Locations_rot)

    return x, y, df, Locations_rot, Atlas

def perform_kmeans_clustering(df, n_clusters):
    """
    Performs K-Means clustering on a DataFrame with 'x' and 'y' columns.
    Adds a 'cluster' column indicating the cluster assignment for each point.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'x' and 'y' columns.
    n_clusters (int): Number of clusters to form.
    
    Returns:
    pd.DataFrame: Updated DataFrame with 'cluster' column.
    """
    os.environ["OMP_NUM_THREADS"] = '1' #prevents memory leakage
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    df['cluster'] = kmeans.fit_predict(df[["x", "y"]])
    
    return df, kmeans

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


## Code for the CTF extent estimation

def load_image(path):
    if path.lower().endswith(".mrc"):
        with mrcfile.open(path, permissive=True) as mrc:
            data = np.array(mrc.data, dtype=np.float32)
    elif path.lower().endswith((".tif", ".tiff")):
        data = np.array(tifffile.imread(path), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    if data.shape[0] != data.shape[1]:
        smallest_axis = min([data.shape[0], data.shape[1]])
        data = data[0:smallest_axis, 0:smallest_axis]
        print(data.shape)
    if data.ndim == 3:
        data = np.sum(data, axis=np.argmin(data.shape))
    return data

def compute_power_spectrum(image):
    image = image - np.mean(image)
    ft = np.fft.ifftshift(image)
    ft = np.fft.fft2(ft)
    Thon = np.log(np.abs(np.fft.fftshift(ft)))
    return Thon

def radial_average(power_spectrum):
    ny, nx = power_spectrum.shape
    y, x = np.indices((ny, nx))

    center = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0])

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(np.int32)

    # Compute full radial mean first
    radial_mean_full = np.bincount(r.ravel(), power_spectrum.ravel()) \
                       / np.bincount(r.ravel())

    # Maximum valid radius = smallest half-dimension
    max_valid_radius = int(min(nx, ny) // 2)

    # Crop results
    radii = np.arange(max_valid_radius)
    radial_mean = radial_mean_full[:max_valid_radius]

    return radii, radial_mean

def estimate_ctf_extent(micrograph_path, relative_threshold = 0.05, scaling_factor=5, win = 10, central_peak = 12, plot = False, verbose = False , num_coefficients = 42):
    
    mic = load_image(micrograph_path)
    x_size = mic.shape[0]

    ps = compute_power_spectrum(mic)
    radius, radial_mean = radial_average(ps)

    #Exclude the central spot
    radius, radial_mean = radius[central_peak:], radial_mean[central_peak:]
    
    bg_coefficients = np.polyfit(radius,radial_mean, num_coefficients)
    bg = np.poly1d(bg_coefficients)

    bg_subtracted = (radial_mean - bg(radius))

    y = bg_subtracted.astype(float)
    env = np.sqrt(uniform_filter1d(y**2, size=win))
    threshold =  np.min(env) + relative_threshold * (np.max(env) - np.min(env))

    flat = np.where(np.abs(env) < threshold)[0]
    plateau_start = 1
    for idx in flat:
        if np.all(np.abs(env[idx:idx+win]) < threshold):
            plateau_start = radius[idx]
            break
    extent_Nq = plateau_start/(x_size/2)

    if verbose:
        print(f"maximum value: {ps.max()}, minimum value: {ps.min()}")
        print(f"radius length: {len(radius)}, radius mean: {len(radial_mean)}")
        print(f"maxium value: {np.max(env)}, minimum value of the envelope: {np.min(env)}, giving a threshold of {threshold}")
        print("Plateau begins at pixel:", plateau_start)
        print("Plateau begins at:", extent_A)
        print("Threshold:", threshold)

    if plot:

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        # --- Subplot 1: image (ps) ---
        im = axs[0].imshow(ps, vmax=p99, cmap = "grey")
        axs[0].set_title("PS Image")
        axs[0].axis("off")

        # --- Subplot 1: background_fit ---
        axs[1].plot(radius, radial_mean)
        axs[1].plot(radius, bg(radius))
        axs[1].set_ylim(radial_mean.min(), radial_mean.max())
        axs[1].set_title("Background fitting")
        axs[1].set_xlabel("Radius")
        axs[1].set_ylabel("Value")


        # --- Subplot 1: bg_subtracted ---
        axs[2].plot(radius, bg_subtracted)
        axs[2].set_ylim(bg_subtracted.min(), bg_outlier)
        axs[2].set_title("Background Subtracted")
        axs[2].set_xlabel("Radius")
        axs[2].set_ylabel("Value")



        # --- Subplot 3: masked plots ---
        axs[3].plot(radius, env, label='env')
        axs[3].plot(radius, y**2 * scaling_factor, label='y² × scaling')
        axs[3].axvline(x=plateau_start, color='red', linestyle='--', linewidth=1.5, label=f'signal limit at {round(extent_A,2)} A')
        axs[3].set_title("Enevolpe")
        axs[3].set_xlabel("Radius")
        axs[3].legend()

        plt.tight_layout()
        plt.show()
    return extent_Nq