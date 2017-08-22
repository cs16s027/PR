import sys
import numpy as np
import cv2

def bittify24(image, order):

    row, col, _ = image.shape
    image_24bit = np.zeros((row, col))

    for x in range(row):
        for y in range(col):
            bit_24 = ''
            for c in order:
                bit_24 += '{0:08b}'.format(image[x][y][c])
            image_24bit[x][y] = int(bit_24, 2)

    return np.float32(image_24bit)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage : python concat.py <input> <order> <output>\n'
        print '<input> : path to an input image'
        print '<order> : comma separated order for concatenating channels, example 1,3,2'
        print '<output>: path to write the numpy file, example 24bit_square.npy\n'
        exit()

    _, image, order, output = sys.argv
    
    order = [int(word) - 1 for word in order.split(',')]
    image = cv2.imread(image)
    image_24bit = bittify24(image, order)
    np.save(output, image_24bit)

