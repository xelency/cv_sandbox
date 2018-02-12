import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load original image and template
img = cv2.imread('origin.jpg',0)
template = cv2.imread('template.jpg',0)
w, h = template.shape[::-1]
method = eval('cv2.TM_CCOEFF_NORMED') # Use Template Matching Normalized Coffeicient

# Apply template Matching
res = cv2.matchTemplate(img,template,method)

# Get square coords
min_val, max_val, min_loc, top_left = cv2.minMaxLoc(res)
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img,top_left, bottom_right, 255, 2) # Draw square on original image

# matplot
plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([]) # Look for the white dot!
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle('cv2.TM_CCOEFF_NORMED')
plt.show()
