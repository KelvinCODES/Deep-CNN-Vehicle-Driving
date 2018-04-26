import cv2, os
import numpy as np
import matplotlib.image as mpimg

IM_HEIGHT, IM_WIDTH, IM_CHANNELS = 66, 200, 3
IM_SHAPE = (IM_HEIGHT, IM_WIDTH, IM_CHANNELS)

def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def preprocess(image):
    image = image[60:-25, :, :]
    image = cv2.resize(image, (IM_WIDTH, IM_HEIGHT), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def choose_image(data_dir, center, left, right, steer_angle):
    coin_flip = np.random.randint(1,4,1)
    if coin_flip == 1:
        return load_image(data_dir, left), steer_angle + 0.2
    elif coin_flip == 2:
        return load_image(data_dir, right), steer_angle - 0.2
    return load_image(data_dir, center), steer_angle

def random_flip(image, steer_angle):
    coin_flip = np.random.rand()
    if coin_flip < 0.5:
        return cv2.flip(image, 1), -steer_angle
    else:
        return image, steer_angle

def random_trans(image, steer_angle, range_x, range_y):
    trans_x = range_x * np.random.rand()
    trans_y = range_y * np.random.rand()
    steer_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steer_angle

def augment(data_dir, center, left, right, steer_angle, range_x=50, range_y=20):
    image, steer_angle = choose_image(data_dir, center, left, right, steer_angle)
    image, steer_angle = random_flip(image, steer_angle)
    image, steer_angle = random_trans(image, steer_angle, range_x, range_y)
    return image, steer_angle

def batch_generator(data_dir, img_paths, steer_angles, batch_size, is_training):
    images = np.empty([batch_size, IM_HEIGHT, IM_WIDTH, IM_CHANNELS])
    steers = np.empty(batch_size)
    while 1:
        i = 0
        for index in np.random.permutation(img_paths.shape[0]):
            if i < batch_size:
                center, left, right = img_paths[index]
                steer_angle = steer_angles[index]
                if is_training and np.random.rand() < 0.5:
                    image, steer_angle = augment(data_dir, center, left, right, steer_angle)
                else:
                    image = load_image(data_dir, center)
                images[i] = preprocess(image)
                steers[i] = steer_angle
                i += 1
            
        yield images, steers






