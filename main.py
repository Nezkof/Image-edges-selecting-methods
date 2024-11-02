import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from helpers import process_layer, binary_rows_to_image

def upload_image():
   Tk().withdraw()  
   filepath = filedialog.askopenfilename(
      filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
   )
   if not filepath:
      print("No file selected.")
      return None
   return filepath

filepath = upload_image()
if filepath:
   img = cv2.imread(filepath)
   img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   # Define operator kernels
   kernelx_roberts = np.array([[-1, 0], [0, 1]], dtype=int)
   kernely_roberts = np.array([[0, -1], [1, 0]], dtype=int)
   kernelx_prewitt = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
   kernely_prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

   # Apply operators
   x_roberts = cv2.filter2D(img_gray, cv2.CV_16S, kernelx_roberts)
   y_roberts = cv2.filter2D(img_gray, cv2.CV_16S, kernely_roberts)
   x_prewitt = cv2.filter2D(img_gray, cv2.CV_16S, kernelx_prewitt)
   y_prewitt = cv2.filter2D(img_gray, cv2.CV_16S, kernely_prewitt)
   x_sobel = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
   y_sobel = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)

   # Convert gradients to uint8
   absX_roberts = cv2.convertScaleAbs(x_roberts)
   absY_roberts = cv2.convertScaleAbs(y_roberts)
   absX_prewitt = cv2.convertScaleAbs(x_prewitt)
   absY_prewitt = cv2.convertScaleAbs(y_prewitt)
   absX_sobel = cv2.convertScaleAbs(x_sobel)
   absY_sobel = cv2.convertScaleAbs(y_sobel)

   # Combine gradients
   Roberts = cv2.addWeighted(absX_roberts, 0.5, absY_roberts, 0.5, 0)
   Prewitt = cv2.addWeighted(absX_prewitt, 0.5, absY_prewitt, 0.5, 0)
   Sobel = cv2.addWeighted(absX_sobel, 0.5, absY_sobel, 0.5, 0)

   # Binary bit layers processing
   b, g, r = cv2.split(img)
   height, width = b.shape
   channels = 24

   # Stack binary layers
   bit_sequence_b = np.unpackbits(b[:, :, None], axis=2)
   bit_sequence_g = np.unpackbits(g[:, :, None], axis=2)
   bit_sequence_r = np.unpackbits(r[:, :, None], axis=2)
   mapsNeiman = np.concatenate((bit_sequence_b, bit_sequence_g, bit_sequence_r), axis=2)
   mapsMoore = np.concatenate((bit_sequence_b, bit_sequence_g, bit_sequence_r), axis=2)

    # Process each layer
   for layer in range(channels):
      mapsNeiman[:, :, layer] = process_layer(mapsNeiman[:, :, layer])
      mapsMoore[:, :, layer] = process_layer(mapsMoore[:, :, layer])

   # Pack the binary layers back into bytes
   resultNeiman = np.packbits(mapsNeiman, axis=2)
   resultMoore = np.packbits(mapsMoore, axis=2)

   # Convert binary rows to an image
   resultNeiman = binary_rows_to_image(resultNeiman, width, height)
   resultMoore = binary_rows_to_image(resultNeiman, width, height)

   # Display images
   plt.figure(figsize=(12, 6))
   plt.subplot(231), plt.imshow(img_RGB), plt.title('Original Image'), plt.axis('off')
   plt.subplot(232), plt.imshow(Roberts, cmap='gray'), plt.title('Roberts Operator'), plt.axis('off')
   plt.subplot(233), plt.imshow(Prewitt, cmap='gray'), plt.title('Prewitt Operator'), plt.axis('off')
   plt.subplot(234), plt.imshow(Sobel, cmap='gray'), plt.title('Sobel Operator'), plt.axis('off')
   plt.subplot(235), plt.imshow(resultNeiman, cmap='gray'), plt.title('Neiman Operator'), plt.axis('off')
   plt.subplot(236), plt.imshow(resultMoore, cmap='gray'), plt.title('Moore Operator'), plt.axis('off')

   plt.show()
else:
   print("Image upload was canceled.")
