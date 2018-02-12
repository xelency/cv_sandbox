import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bingo.jpg',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
scharrx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
scharry = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(scharrx,cmap = 'gray')
plt.title('Scharr  X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(scharry,cmap = 'gray')
plt.title('Scharr  Y'), plt.xticks([]), plt.yticks([])

plt.show()
