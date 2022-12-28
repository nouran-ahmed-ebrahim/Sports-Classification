import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
import csv
from sklearn.model_selection import train_test_split
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = 'Train'
TEST_DIR = 'Test'
IMG_SIZE = 200
NUM_OF_CHANNELS = 3
LR = 0.001
MODEL_NAME = 'sport_classification'
sports = {"Basketball": [1, 0, 0, 0, 0, 0], "Football": [0, 1, 0, 0, 0, 0], "Rowing": [0, 0, 1, 0, 0, 0]
    ,"Swimming": [0, 0, 0, 1, 0, 0], "Tennis": [0, 0, 0, 0, 1, 0], "Yoga": [0, 0, 0, 0, 0, 1]}


def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    sport_name = image_name.split('_')[0]
    sport_label = sports[sport_name]
    return np.array(sport_label)


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, 1)
#         img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_label = create_label(img)
       # training_data.append([np.array(img_data), img_label])
        imgs = preprocessing(img_data)
        for image in imgs:
            training_data.append([np.array(image), img_label])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def preprocessing(img_data):
    rotate_image = cv2.rotate(img_data, cv2.ROTATE_180)
    norm_image = (img_data-np.min(img_data)) / (np.max(img_data)- np.min(img_data))
    return [norm_image]  # rotate_image


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 1)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data = preprocessing(img_data)[0]
        testing_data.append([np.array(img_data), img])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


def predict_test_result():
    predictions = model.predict(X_test)
    test_labels = np.argmax(predictions, axis=1)
    sample_idx = 0
    test_prediction = []
    for img_name in test_imgs_names:
        sample_prediction = test_labels[sample_idx]
        test_prediction.append([img_name,sample_prediction])
        sample_idx += 1
    return test_prediction


if os.path.exists('train_data.npy'):  # If you have already created the dataset:
    train_data = np.load('train_data.npy', allow_pickle=True)
    # train_data = create_train_data()
else:  # If dataset is not created:
    train_data = create_train_data()

# print(train_data.shape)
if os.path.exists('test_data.npy'):
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    test_data = create_test_data()


train = train_data
test = test_data

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, NUM_OF_CHANNELS)
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, NUM_OF_CHANNELS)
test_imgs_names = [i[1] for i in test]


tf.compat.v1.reset_default_graph()

conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, NUM_OF_CHANNELS], name='input')
pool0 = max_pool_2d(conv_input, 2, strides=2)

# Block1
conv1 = conv_2d(conv_input, 16, 3, activation='relu', padding='same')
conv2 = conv_2d(conv1, 16, 3, activation='relu', padding='same')
pool1 = max_pool_2d(conv2, 2, strides=2)

# Block2
conv3 = conv_2d(pool1, 32, 3, activation='relu', padding='same')
conv4 = conv_2d(conv3, 32, 3, activation='relu', padding='same')
pool2 = max_pool_2d(conv4, 2, strides=2)

# Block3
conv5 = conv_2d(pool2, 64, 3, activation='relu', padding='same')
conv6 = conv_2d(conv5, 64, 3, activation='relu', padding='same')
conv7 = conv_2d(conv6, 64, 3, activation='relu', padding='same')
conv8 = conv_2d(conv7, 64, 3, activation='relu', padding='same')
pool3 = max_pool_2d(conv8, 2, strides=2)

# Block4
conv9 = conv_2d(pool3, 128, 3, activation='relu', padding='same')
conv10 = conv_2d(conv9, 128, 3, activation='relu', padding='same')
conv11 = conv_2d(conv10, 128, 3, activation='relu', padding='same')
conv12 = conv_2d(conv11, 128, 3, activation='relu', padding='same')
pool4 = max_pool_2d(conv12, 2, strides=2)

# Block5
conv13 = conv_2d(pool4, 128, 3, activation='relu', padding='same')
conv14 = conv_2d(conv13, 128, 3, activation='relu', padding='same')
conv15 = conv_2d(conv14, 128, 3, activation='relu', padding='same')
conv16 = conv_2d(conv15, 128, 3, activation='relu', padding='same')
pool5 = max_pool_2d(conv16, 2, strides=2)


fully_layer = fully_connected(pool4, 4096, activation='relu')
fully_layer = fully_connected(fully_layer, 4096, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)


if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=50,
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')


filename = 'submission.csv'
fields = ['image_name', 'label']
test_prediction = predict_test_result()
# writing to csv file
with open(filename, 'w',  newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(test_prediction)

