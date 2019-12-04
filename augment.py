from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
from skimage.transform import resize
import shutil

def augmentdata(input_path):
    if not os.path.exists(input_path):
        os.mkdir(input_path)

    train_imgs = os.listdir(input_path)
    for img in train_imgs:
        if img.startswith('.'):
            continue
        id = img.split('.')[0]
        img = cv2.imread(input_path + '/' + img)

        # random flip
        flip = np.random.randint(-1, 1)
        flip_img = cv2.flip(img, flip)
        cv2.imwrite(input_path + '/' + id + '_flip.jpg', flip_img)

        # random rotation
        rotation = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
        random_rotation = rotation[np.random.randint(0, 2)]
        rotate_img = cv2.rotate(img, random_rotation)
        cv2.imwrite(input_path + '/' + id + '_rotation.jpg', rotate_img)

        # random noises
        random_noise = np.random.randint(5, size=img.shape, dtype='uint8')
        noise_img = img + random_noise
        cv2.imwrite(input_path + '/' + '_noised.jpg', noise_img)

        # Some affine transformations
        sample1 = np.float32([[50, 50], [200, 50], [50, 200]])
        sample2 = np.float32([[10, 100], [200, 50], [100, 250]])
        transform_matrix = cv2.getAffineTransform(sample1, sample2)
        affine_img = cv2.warpAffine(img, transform_matrix, (img.shape[1], img.shape[0]))
        cv2.imwrite(input_path + '/' + id + '_affinetrans.jpg', affine_img)

        # salt and pepper noise
        i, j, k = img.shape
        output = np.copy(img)
        salt = 0.05 * img.size * 0.5
        coords = [np.random.randint(0, i - 1, int(salt)) for i in img.shape]
        for i in range(int(salt)):
            output[coords[0][i], coords[1][i], coords[2][i]] = 255
        pepper = 0.05 * img.size * 0.5
        coords = [np.random.randint(0, i - 1, int(pepper)) for i in img.shape]
        for i in range(int(pepper)):
            output[coords[0][i], coords[1][i], coords[2][i]] = 0
        cv2.imwrite(input_path + '/' + id + '_saltandpepper.jpg', output)

        # image sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(input_path + '/' + id + '_sharpen.jpg', sharpened)

        # image smoothing
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite(input_path + '/' + id + '_blur.jpg', blur)

if __name__ == '__main__':
    guns = './train/guns'
    knives = './train/knives'
    augmentdata(guns)
    augmentdata(knives)
