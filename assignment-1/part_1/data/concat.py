import numpy as np
import cv2

image = cv2.imread('20_square.jpg')
print image.shape

new_image = np.zeros((256, 256))

for x in range(256):
    for y in range(256):
        bit_24 = ''
        for c in range(3):
            bit_24 += '{0:08b}'.format(image[x][y][c])
        print bit_24
        print len(bit_24)
        print int(bit_24, 2)
        exit()


