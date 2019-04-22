import numpy as np
import cv2
from matplotlib import pyplot as plt

def bgr2rgb(bgr):
    b, g, r = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]
    # gray = 0.3333 * r + 0.3333 * g + 0.3333 * 
    bgr[:,:,0]=r;
    bgr[:,:,1]=b;
    bgr[:,:,2]=g;
    return bgr;

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.3333 * r + 0.3333 * g + 0.3333 * b
    return gray

def sobelOperator(img):
    sobGradmap = np.copy(img)
    size = sobGradmap.shape
    print(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            sobGradmap[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return sobGradmap

def edges(img):
	edgemap = np.copy(img)
	size = img.shape
	print(size)
	for i in range(1, size[0] - 1):
		for j in range(1, size[1] - 1):
			if(edgemap[i][j]<0.95*255):
				edgemap[i][j] = 0;
			else:
				edgemap[i][j] = 255;
	return edgemap




image = cv2.imread("cameraman.jpg")
image = bgr2rgb(image)
plt.imshow(image)
plt.show()
# image = cv2.imread("cameraman.jpg")
image = rgb2gray(image) 
img = sobelOperator(image)
print(img)
img = edges(img)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
plt.imshow(img,cmap='gray', vmin=0, vmax=255)
plt.show()