import numpy as np
import cv2
import cv2.cv as cv
import argparse
import matplotlib.pyplot as plt

eye_cascade = cv2.CascadeClassifier('/haar/haarcascade_eye_tree_eyeglasses.xml')
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',
    help='Path to file')
parser.add_argument('-f', '--format',
    help='Format of file')
parser.add_argument('-e', '--ext',
    help='File extention')
parser.add_argument('-n', '--num', type=int, default=1,
    help='Number of files (default 1)')
args = vars(parser.parse_args())
path = args['path']
fmt = args['format']
ext = args['ext']
count = args['num']
for i in range(1,count):  
    print(path + fmt + str(format(i, '04')) + ext + '\n')
    img = cv2.imread(path + fmt + str(format(i, '04')) + ext)
    #img = cv2.imread('Datasets/face/BioID-FaceDatabase-V1.2/BioID_'+  +'.pgm')
    #img = cv2.resize(img,(500,500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,gray_t = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
    gray_t = 255-gray_t
    plt.subplot(1,2,1),plt.imshow(gray_t,'gray')
    plt.title('gray')
    plt.xticks([]),plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(gray_t,'gray')
    plt.title('gray')
    plt.xticks([]),plt.yticks([])
    plt.show()
    #cv2.imshow("gray", gray_t)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #img_row_sum = np.sum(gray_t,axis=1).tolist()
    #plt.plot(img_row_sum)
    #plt.show()
