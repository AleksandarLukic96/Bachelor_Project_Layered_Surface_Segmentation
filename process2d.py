import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from slgbuilder import GraphObject 
from slgbuilder import MaxflowBuilder
from skimage.io import imread

class Process2D:
    def __init__(self, data, n_layers = 1, delta = 1, min_margin = 10):
        # Load data of 2D image
        self.data = data
        
        # Set no. of layers
        self.n_layers = n_layers
        
        # Create list with n layers for segmentation
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(GraphObject(data))
        
        self.helper = MaxflowBuilder()
        self.helper.add_objects(self.layers)
        self.helper.add_layered_boundary_cost()
        self.helper.add_layered_smoothness(delta = delta)
        
        for i in range(len(self.layers)-1):
            self.helper.add_layered_containment(
                outer_object = self.layers[i], 
                inner_object = self.layers[i + 1], 
                min_margin = min_margin
            ) 
        
        self.flow = self.helper.solve()
        
        # Segmentation across edge 
        self.segmentations = [self.helper.what_segments(l).astype(np.int32) for l in self.layers] 
        self.segmentation_lines = [np.argmin(s, axis = 0) - 0.5 for s in self.segmentations]


if __name__ == "__main__":        
    
    choose = 1
    
    if choose == 0:
        # Get nifty file from directory
        current_dir = os.getcwd()
        path = "data"
        file = "canvas_cropout.nii" 
        file_path = os.path.join(current_dir, path, file)

        # Load data as Nifty1Image using the nibabel package
        data = nib.load(file_path)

        # Load fdata as numpy.memmap
        fdata = data.get_fdata().astype(np.float16)

        # Load slice as numpy.ndarray
        slice = fdata[:, fdata.shape[1]//2, :]
        
        data = slice
    
    elif choose == 1:
        path = "./data/slices_canvas_cropout/dim1_slice105.png"
        data = imread(path).astype(np.int32)

    proc2d = Process2D(data = data, n_layers = 3)
    
    print(np.shape(proc2d.segmentations))
    
    print(np.shape(proc2d.segmentation_lines))
    
    # Draw results.
    plt.figure(figsize = (10, 10))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(proc2d.data, cmap = "gray")

    ax = plt.subplot(1, 3, 2)
    ax.imshow(np.sum(proc2d.segmentations, axis = 0))

    ax = plt.subplot(1, 3, 3)
    ax.imshow(data, cmap = "gray")
    for line in proc2d.segmentation_lines:
        ax.plot(line)
    plt.show()