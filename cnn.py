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
IMG_SIZE = 225
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
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_label = create_label(img)
        training_data.append([np.array(img_data), img_label])
        imgs = preprocessing(img_data)
        for image in imgs:
            training_data.append([np.array(image), img_label])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def preprocessing(img_data):
    norm_image = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    rotate_image = cv2.rotate(img_data, cv2.ROTATE_180)

    return norm_image, rotate_image#, flip_img


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data)])
    np.save('test_data.npy', testing_data)
    return testing_data


def predict_test_result():
    predictions = model.predict(X_test)
    test_labels = np.argmax(predictions, axis=1)
    sample_idx = 0
    test_prediction = []
    for img_name in tqdm(os.listdir(TEST_DIR)):
        sample_prediction = predictions[sample_idx]
        test_prediction.append([img_name, sample_prediction])
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

# print(train_data.shape)
# print(test_data.shape)
train = train_data
test = test_data

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_test = np.array(test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

tf.compat.v1.reset_default_graph()

conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 3, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 3, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)

fully_layer = fully_connected(pool3, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
print(X_train.shape)


if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
              validation_set=({'input': X_val}, {'targets': y_val}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')


filename = 'Test Prediction'
fields = ['image_name', 'label']
test_prediction = predict_test_result()
# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(test_prediction)
plt.show()
