from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np

def Predictor(model, Test_img):
    # Test image Preprocessing
    img_test = cv2.imread(Test_img, cv2.IMREAD_GRAYSCALE)
    Img_width, Img_height = img_test.shape()
    img = cv2.equalizeHist(img_test) / 255
    img_resized = img.reshape(-1, Img_width, Img_height, 1)

    # Segmentation result
    result = model.predict(img_resized)
    result_r = result.reshape(Img_width, Img_height, 2)
    result_new = np.transpose(result_r, (2, 0, 1))

    scaler = MinMaxScaler()
    normalres1 = scaler.fit_transform(result_new[0])
    After_seg1 = img * normalres1   # Target
    #normalres2 = scaler.fit_transform(result_new[1])
    #After_seg2 = img * normalres2   # rests
    return After_seg1