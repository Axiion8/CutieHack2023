import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random

DATADIR = "Datasets"
CATEGORIES = ["Human", "Not Human"]


for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap ="gray")
        break
    break

IMG_SIZE = 150
test_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(test_array, cmap = 'gray')
plt.show()


training_data = []

def trainingDataFunc():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        classNum = CATEGORIES.index(category) #0 for human 1 for non human
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                test_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #Resizing operation
                training_data.append([test_array, classNum])
            except Exception as e: #in case some of the pics are broken
                pass

trainingDataFunc()

print(len(training_data))

import random
random.shuffle(training_data)

       
for sample in training_data[:10]:
    print(sample[1])

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #since working with grayscale

import pickle

pickle_out = open("x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", rb)
X = pickle.load(pickle_in)