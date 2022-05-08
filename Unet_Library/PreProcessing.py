import cv2
import os
import nibabel as nib
import numpy as np


## FUNCTION ::   Convert png 2 array_set
def png2set(Input_dimension, path):
  Img_width, Img_height = Input_dimension
  file_list = os.listdir(path)
  X_img = []
  X_name = []
  # Extract name from file
  for name in file_list:
    if name.find('.png') is not -1:
      X_name.append(name)
  X_name = sorted(X_name)
  # Build Train Set
  for i in range(len(X_name)):
    img_tempo = cv2.imread(path + X_name[i], cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img_tempo)
    nimg = img / 255
    X_img.append(nimg)
  # Reshape Train Set Dimension
  X_train = np.concatenate(X_img, axis=0)
  X_train = X_train.reshape(-1, Img_width, Img_height, 1)
  return X_train


## FUNCTION ::   Convert nif 2 array_set
def nif2set(Input_dimension, path):
  Img_width, Img_height = Input_dimension
  nif = nib.load(path)
  nif_arr = nif.get_fdata()
  y_train = np.transpose(nif_arr, (2,1,0))
  y_train = y_train.reshape(-1, Img_width, Img_height, 1)
  y_train = y_train.astype('bool')
  return y_train