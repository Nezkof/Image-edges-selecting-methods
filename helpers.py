import numpy as np
from PIL import Image

import numpy as np

def process_layer(matrix, use_corners=False):
   new_matrix = matrix.copy()
    
   # Get shifted matrices for each Neumann neighborhood direction
   up = np.roll(matrix, -1, axis=0)
   down = np.roll(matrix, 1, axis=0)
   left = np.roll(matrix, -1, axis=1)
   right = np.roll(matrix, 1, axis=1)
    
   # Zero out edges for the rolled matrices (to avoid wrap-around)
   up[-1, :] = 0
   down[0, :] = 0
   left[:, -1] = 0
   right[:, 0] = 0
    
   # Initialize list of neighbors with only the direct neighbors
   neighbors = [up, right, down, left]

   if use_corners:
      # Get shifted matrices for each diagonal (corner) direction
      up_left = np.roll(np.roll(matrix, -1, axis=0), -1, axis=1)
      up_right = np.roll(np.roll(matrix, -1, axis=0), 1, axis=1)
      down_left = np.roll(np.roll(matrix, 1, axis=0), -1, axis=1)
      down_right = np.roll(np.roll(matrix, 1, axis=0), 1, axis=1)
        
      # Zero out edges for the diagonal rolled matrices
      up_left[-1, :] = 0
      up_left[:, -1] = 0
      up_right[-1, :] = 0
      up_right[:, 0] = 0
      down_left[0, :] = 0
      down_left[:, -1] = 0
      down_right[0, :] = 0
      down_right[:, 0] = 0

      # Add diagonal neighbors if `use_corners` is True
      neighbors.extend([up_left, up_right, down_left, down_right])

   # Stack all neighborhood layers based on `use_corners` flag
   neighbors = np.stack(neighbors, axis=2)
    
   # Apply the AND operation across all neighbors
   neighbor_and = np.all(neighbors, axis=2)
    
   # Set the center pixel to 0 where all selected neighbors are 1
   new_matrix[neighbor_and] = 0
    
   return new_matrix

   
def binary_rows_to_image(binary_rows, width, height, scale=1):
   binary_rows = np.array(binary_rows, dtype=np.uint8).reshape((height, width, 3))
   img = Image.fromarray(binary_rows, 'RGB')
    
   if scale > 1:
      img = img.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    
   return img
   

def AND(*args):
   return all(args)



