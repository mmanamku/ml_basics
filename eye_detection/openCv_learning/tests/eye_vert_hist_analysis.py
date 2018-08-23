import numpy as np
import cv2
import cv2.cv as cv
#import argparse
import matplotlib.pyplot as plt

#parser = argparse.ArgumentParser()
#parser.add_argument('-f', '--folder_number', type=int, default=1,
#    help='Path folder number')
#parser.add_argument('-n', '--num', type=int, default=10,
#    help='Number of files (default 10)')
#args = vars(parser.parse_args())
#folder_number = args['folder_number']
#count = args['num']
#for i in range(1,count):
    #img = cv2.imread('Datasets/eye/data_set_'+ str(folder_number) +'_' + str(count) + '/'+ str(i) +'.png')
    #img = cv2.resize(img,(500,500))
img = cv2.imread('fdas.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,gray_t = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)

cv2.imwrite('fdas_i.png', gray_t)
gray_t = 255-gray_t
cv2.imshow("gray", gray_t)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_row_sum = np.sum(gray_t,axis=1).tolist()
plt.plot(img_row_sum)
plt.show()
