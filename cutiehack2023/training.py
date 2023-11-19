#Libraries/Modules used
import cv2
import imghdr
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import random
import matplotlib.pyplot as plt
import numpy as np
import os




#Filtering through weird/not relevant unages

DIRECTORY = 'Datasets' #Folder containing sub folders/categories
os.listdir(DIRECTORY)
IMG_EXTENSIONS = ['jpg', 'png', 'webp', 'jpeg'] #Valid extensions
print(os.listdir(DIRECTORY))

#Looping through every folder inside directory
for image_class in os.listdir(DIRECTORY):
    for image in os.listdir(os.path.join(DIRECTORY,image_class)): #Print every image within the directories
        image_path = os.path.join(DIRECTORY,image_class,image) #Grabbing image storing inside variable image_path
    try:
        img = cv2.imread(image_path) #Opens up image
            #Ex ^^^ : cv2.imread(os.path.join('Datasets', 'Human', 's465_Sir-Winston-Churchill.jpg')) // Reads this as numpy ARRAY
        img = imghdr.what(image_path)
        if tip not in IMG_EXTENSIONS:
            print("Image not valid")
            os.remove(image_path) #Remove the invalid image
    except Exception as e:
        print("Removing invalid image")



#Loading Data files 

data = tf.keras.utils.image_dataset_from_directory('Datasets', batch_size = 2) #Builds image data set, meaning you need not build classes or labels. 
 #Essentially building data pipeline. Automatically reshapes images so its consistent sizes. 

data_iterator = data.as_numpy_iterator() #acesss generator from our data pipeline. Loops through and pulls data continously

batch = data_iterator.next() #Grabbing batch should be a length of 2. One part is the images, Second part is the labels
#batch[0] are images in numpy arrays #batch[1] represents the different lables (Human, Non Human, etc)

#check which class is assigned to what image

#This section uses matplot to display images and labels
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:2]): #Should display 2 pictures according to dataset
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

plt.show()
#After having data loaded now its time to Preprocess to generate better results.
#Ideally we want our values to small as possible for optimazation so lets divide batch by /255 so you get values 0 and 1
newData = data.map(lambda x,y: (x/255,y)) #data.map allows you to preform the transformation in pipline of dividing by 255.
#x are our images, y are our labels. x/255 is our scaling!!
#tf.data.Dataset gives you a lot of functions that allow you to do this. For our case we are use .map function

newIterator = data.as_numpy_iterator()
#data.as_numpy_iterator().next() #Gives access to iterator and grabs next batch. Note you have to shuffle or else data will not change

batch = newIterator.next() # .next() iterates and gives new batch


fig, ax = plt.subplots(ncols=4, figsize=(10,10))
for idx, img in enumerate(batch[0][:3]): #Should display 2 pictures according to dataset
    ax[idx].imshow(img) #Make sure its not set to integers since values are between 0 to 1. Ex: 0.2 evaluates to 0 which is not what we want 
    ax[idx].title.set_text(batch[1][idx])

#To ensure our data does not overflow or overfit we did to partition the data.
#To c heck how many batches you have do len(data)

#Lets partition the data to determine how much data goes into each Training/validation/test 
train_size = int(len(data)*.7) #This data trains our model
val_size = int(len(data)*.2) #validation Evaluates our model whilst training 
test_size = int(len(data)*.1)#This waits untul final evaluation set. Used post training to do the evaluation.

#Make sure data is shuffled before this is executed.
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#DEEP LEARNING STATE !!!!!!!!!!!!!!

#Flatten allows us to go from Conv2D and changes it make to a format in which Dense can take.

model = Sequential()

#Adding layers

model.add(Conv2D(16, (3,3), 1, activation ='relu', input_shape=(256,256,3))) 
#Conv2D(16 -> 16 filters 3x3 in size, with a stride of 1 meaning moves 1 at a time)
#acitvation 'relu' converts any negative value to 0. So it accounts for non-linear patterns.
#sigmoid takes output from layer to modify what the output my look like. Ex: extremly large numbers are reshaped.
#256 pixels height and width through 3 channels
model.add(MaxPooling2D()) #Takes maximum value and returns that value

model.add(Conv2D(32, (3,3), 1, activation ='relu')) #New set of layers
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation ='relu')) #New set of layers
model.add(MaxPooling2D())

model.add(Flatten()) #We want everything into a single value this is what Flatten() is for.

model.add(Dense(256, activation = 'relu')) #256 neurons
model.add(Dense(1, activation = 'sigmoid')) #Single dense layer. Single output either 0 or 1 (Because of sigmoid activation)
#Sigmoid takes any output and converts it to 0 and 1. 

#compiling model
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy']) #Tracking accuracy or how well our model computs 0 or 1

model.summary() #Shows how the model transforms data

#Training
logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir) #Checks to see how model preforms

#Fit our model:
history = model.fit(train, epochs=5, validation_data=val, callbacks=[tensorboard_callback]) #Size of training data must be larger than batch size
#Epochs tells us how long to train our data, validaiton_data tells us how our data is preforming
#callbacks allow us to track our information to check history or errors.


#Plotting performance
fig = plt.figure()
plt.plot(histroy.history['loss'], color = 'red', label = 'loss')
plt.plot(histroy.history['accuracy'], color = 'green', label = 'accuracy')
fig.suptitle('Changes over time', fontsize = 20)
plt.legend(loc = "upper left")
plt.show()


#Preformance metrics
precision = Precision()
recall = Recall()
binaryAcc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x = batch
    y = batch
    yPredict = model.predict(x)
    precision.update_state(y, yPredict)
    recall.update_state(y, yPpredict)
    binaryAcc.update_state(y, yPredict)

#print('Precision' {precision.result().numpy()})

img = cv2.imread() #Read NEW image that model hs not scene
plt.imshow(cv2.cvtColr(img, cv2.COLOR_BGR2RGB)) #converts to normal picture color
#plt.show()

resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numoy().astpye(int))
plt.show()

resize.shape
np.expand_dims(resize, 0)
yPreict = model.predict(np.expand_dims(resize/255,0))
print(yPredict) # Predicts new image class

if(yPredict > 0.5):
    print("Not Human")
else:
    print("Human")

#saving models
from tensorflow.keras.models import load_model

model.save(os.path.join('models', 'humanOrNot.h5')) #saves new models each run

new_model = load_model(os.path.join('models','humanOrNot.h5')) #loads model 

#if you want to pass data to it you get a prediciton again: 
newPrediction2 = new_model.predict(np.expand_dims(resize/255,0))