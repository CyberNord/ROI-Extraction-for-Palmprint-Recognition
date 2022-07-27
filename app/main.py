import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# path
path = r'db\test_color.jpg'

# Read in pic & show
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
plt.imshow(cv2.imread(path))    # BGR
plt.show()
plt.imshow(image)               # RGB
plt.show()
print(image)                    # array


# RGB to YCrCb
ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
plt.imshow(ycrcb)
plt.show()
print(image)  # array