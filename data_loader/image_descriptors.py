# aplicação de descritores de imagens
# e outras operações

import numpy as np
import math

import cv2
import skimage.feature as feature

class ImageDescriptors():
    @staticmethod
    def lbp(image,
            num_points=8,
            radius=2,
            eps=1e-7):

        # se a imagem é rgb, torna grayscale
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
  
        lbp_img = feature.local_binary_pattern(
            image, num_points, radius, method='uniform'
        )
      
        (hist, _) = np.histogram(
            lbp_img.ravel(),
            bins=np.arange(0, num_points+3), range=(0, num_points + 2))
  
        # normalize the histogram
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)
    
        return np.array(hist), lbp_img

    @staticmethod
    def hu_moments(image, threshold=128):

        # se a imagem é rgb, torna grayscale
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 

        _, img_threshold = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        # calculate moments 
        moments = cv2.moments(img_threshold) 
        
        # calculate hu moments 
        hu_moments = cv2.HuMoments(moments)
    
        # log scale hu moments 
        for i in range(0, len(hu_moments)):
          if hu_moments[i] != 0:        
            hu_moments[i] = -1* math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))
    
        return hu_moments.reshape(hu_moments.shape[0]), img_threshold

    @staticmethod
    def gabor(image, ksize=31, n_filters=8):
        
        # se a imagem é rgb, torna grayscale
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        gabor_filters = []
           
        for theta in np.arange(0, np.pi, np.pi/n_filters):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            gabor_filters.append(kern)
    
        gabor_img = np.zeros_like(image)
        
        for kern in gabor_filters:
            f_im = cv2.filter2D(image, cv2.CV_8UC3, kern)
            np.maximum(gabor_img, f_im, gabor_img)
        
        return np.array(gabor_filters), gabor_img

    @staticmethod
    def hog(image):
        # se a imagem é rgb, torna grayscale
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 

        hog_features, hog_image = feature.hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True)

        return np.array(hog_features), hog_image

    @staticmethod
    def image_to_gray(image):
        # aplicação de uma bateria de filtros
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def calc_hist(img):
        uniques, counts = np.unique(img, return_counts=True)
        counts = np.array(counts)
        missing = np.array(list(filter(lambda i: i not in uniques, range(255+1))))

        # inserindo bins nulos no histograma
        for i in missing:
            counts = np.insert(counts, i, 0)

        return np.array(counts)