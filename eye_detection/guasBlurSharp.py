import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('blur.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def corr(img, mask):
	(row, col) = img.shape[:2]
	(m, n) = mask.shape[:2]
	new = np.zeros((row+m-1, col+n-1))
	n = n//2
	m = m//2
	filtered_img = np.zeros(img.shape)
	new[m:new.shape[0]-m, n:new.shape[1]-n] = img
	for i in range(m, new.shape[0]-m):
		for j in range(n, new.shape[1]-n):
			temp = new[i-m:i+m+1, j-m:j+m+1]
			result = temp*mask
			filtered_img[i-m, j-n] = result.sum()
	return filtered_img

def gaussian(m, n, sigma):
	gaussian = np.zeros((m,n))
	m = m//2
	n = n//2
	for x in range(-m, m+1):
		for y in range(-n, n+1):
			x1 = sigma*(2*np.pi)**2
			x2 = np.exp(-(x**2+y**2)/(2*sigma**2))
			gaussian[x+m, y+n] = (1/x1)*x2
	return gaussian

g = gaussian(5, 5, 2)
Ig1 = corr(img, g)
g = gaussian(5, 5, 5)
Ig2 = corr(img, g)
edg = (Ig1-Ig2)*30
#alpha = 30
sharped = cv2.add(img, edg)
cv2.imshow("sharp", img)
cv2.waitKey(0)
cv2.imshow("sharp", sharped)
cv2.waitKey(0)
cv2.destroyAllWindows()
