from PIL import Image
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from datetime import datetime
from keras.preprocessing.image import img_to_array, array_to_img
from imgaug import augmenters as iaa


#variables
image_directory = 'data/train/'

dim_x = 168
dim_y = 96

input_shape = (dim_x, dim_y, 1)
batch_size = 32
epochs = 10
validation_pct = 0.03

np.random.seed(20)

#set the number of whales to classify
num_classes = 20
last_layer = 1000

y_base = [0] * num_classes

#how many new whales to include
nw_sample = 400

#function to return image as array 
def load_image_array(image_file):     
    #open file
    im = Image.open(image_file)
    im = treat_image(im)
    return(im)

def treat_image(im):
    size = dim_x, dim_y
    #convert to greyscale
    im = im.convert('L')
    #resize
    im = im.resize(size,Image.LANCZOS) #if your images are not already the size you want    
    im = np.asarray(im)/255
    return(im)

#function to return a bunch of augmented images for each image
#input = image file name
#output = array of augmented images
def augment_image2(image_file):
    im = Image.open(image_file)
    im_array = img_to_array(im)
    #function returns an array of images (including the first one)
    output_images = []
    output_images.append(im_array)
    #darken image
    #aug_dark = iaa.Add(-30)
    #aug_image = aug_dark.augment_images([im_array])[0]
    #output_images.append(aug_image)
    #lighten image
    #aug_light = iaa.Add(30)
    #aug_image = aug_light.augment_images([im_array])[0]
   # output_images.append(aug_image)
    
   
    #flip the images
    aug_flip = iaa.Fliplr(1)
    for i in range(len(output_images)):
        img_arr = output_images[i]
        aug = aug_flip
        aug_image = aug.augment_images([img_arr])[0]
        output_images.append(aug_image)
    
    #minor afine - all
    aug_affine = iaa.PiecewiseAffine(scale=0.01)
    for i in range(len(output_images)):
        img_arr = output_images[i]    
        aug = aug_affine
        aug_image = aug.augment_images([img_arr])[0]
        output_images.append(aug_image)
    
    #some perspective transformations
    aug_pt1 = iaa.PerspectiveTransform(0.04, True)
    for i in range(len(output_images)):
        img_arr = output_images[i]    
        aug = aug_pt1
        aug_image = aug.augment_images([img_arr])[0]
        output_images.append(aug_image)
    return(output_images)

def augment_image(image_file):
    im = Image.open(image_file)
    im_array = img_to_array(im)
    #function returns an array of images (including the first one)
    output_images = []
    output_images.append(im_array)
    #darken image
    #aug_dark = iaa.Add(-30)
    #aug_image = aug_dark.augment_images([im_array])[0]
    #output_images.append(aug_image)
    #lighten image
    #aug_light = iaa.Add(30)
    #aug_image = aug_light.augment_images([im_array])[0]
   # output_images.append(aug_image)
    augmentations = [
        iaa.Fliplr(1),
        iaa.PerspectiveTransform(0.03, True),
        iaa.PerspectiveTransform(0.04, True),
        iaa.PerspectiveTransform(0.045, True),
        iaa.Add(30, per_channel=True),
        iaa.Add(-20, per_channel=True),
        iaa.PiecewiseAffine(scale=0.01),
        iaa.AdditiveGaussianNoise(scale=10),
        iaa.MedianBlur(k=(3, 3)),
        iaa.PerspectiveTransform(0.02, True),
        iaa.PerspectiveTransform(0.021, True),
        iaa.PerspectiveTransform(0.022, True),
        iaa.AddElementwise(20),
        iaa.Affine(rotate=3),
        iaa.Affine(rotate=-5),
        iaa.Affine(rotate=4)
    ]
    #do this 3 times
    base = 4
    for i in range(7):
    #for i in range(1):
        #get between 2 and 5 augs
        aug = iaa.SomeOf(i + base, augmentations)
        #produce more images where there are more augs
        for j in range(4):
            this_image = aug.augment_images([im_array])[0]
            output_images.append(this_image)

    return(output_images)


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (40,3)
def showImagesHorizontally(list_of_files):
    plt.rcParams["figure.figsize"] = (40,3)
    n_files = len(list_of_files)
    f,ax = plt.subplots(1,round(n_files/1))
    for i in range(n_files):
        ax[i].imshow(array_to_img(list_of_files[i]),cmap='Greys_r')
    plt.show()


#image_file = image_directory + 'f8396d08.jpg'
#test = augment_image(image_file)
#showImagesHorizontally(test)

#array_to_img(test[0])
#array_to_img(test[2])
#array_to_img(test[4])
#array_to_img(test[6])

#function to shuffle input arrays
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_training_input(td_subset):
    #now create training datasets for the subset
    x_train = []
    y_train = []
    
    #now iterate through dataframe making test datasets
    for i in range(len(td_subset)):
        print(i)
        this_row = td_subset.iloc[i]
        if this_row.Id == 'new_whale':
            this_y = np.asarray(y_base)
        else:
            this_y = keras.utils.to_categorical(int(top_whales.loc[[this_row.Id]].idx),num_classes)
        
        image_file = image_directory + this_row.Image
    
        image_arrays = augment_image(image_file)
        for i in range(len(image_arrays)):
            img_arr = image_arrays[i]
            im = array_to_img(img_arr)
            this_x = treat_image(im)
            x_train.append(this_x)
            y_train.append(this_y)
    return({
        'x':x_train,
        'y':y_train
    })

#for a POC, we'll just classify the top X most common whales
training_data = pd.read_csv('data/train.csv')

#check how many types of whale there are and how frequently they occur
whale_count =  training_data.Id.value_counts()

#filter out the new whales, the most common group
whale_count = whale_count[1:]
top_whales = whale_count[0:num_classes]
top_whales = list(top_whales.index) 
td_subset = training_data[training_data.Id.isin(top_whales)]

#add some new whales
new_whales = training_data[training_data.Id == 'new_whale']
new_whales = new_whales.sample(nw_sample)
td_subset = pd.concat([td_subset,new_whales])

#create classification df
top_whales = pd.DataFrame(data={'Id':top_whales, 'idx':range(len(top_whales))})
top_whales = top_whales.set_index('Id')

#split the data into train and validation
td_validation = td_subset.sample(frac = validation_pct)
td_subset = td_subset.drop(td_validation.index)

#get training inputs
train_input = get_training_input(td_subset)

x_train = train_input.get('x')
y_train = train_input.get('y')

x_train = np.asarray(x_train)
x_train = x_train.reshape(x_train.shape[0], dim_x, dim_y, 1)
y_train = np.asarray(y_train)
#y_train = keras.utils.to_categorical(y_train, num_classes)

#get validation inputs
validation_input = get_training_input(td_validation)

x_valid = validation_input.get('x')
y_valid = validation_input.get('y')

x_valid = np.asarray(x_valid)
x_valid = x_valid.reshape(x_valid.shape[0], dim_x, dim_y, 1)
y_valid = np.asarray(y_valid)

#y_valid = keras.utils.to_categorical(y_valid, num_classes)

#add some new whales
#new_whales = training_data[training_data.Id == 'new_whale']
#new_whales = new_whales.sample(frac = validation_pct)

#splint into train and validation
#should also run the new whales through the image thing
#new_whales_validation = new_whales.sample(frac = validation_pct)
#new_whales = new_whales.drop(new_whales_validation.index)

#shuffle arrays
#x_train, y_train = unison_shuffled_copies ( x_train, y_train )

np.save('x_train_20',x_train)
np.save('y_train_20',y_train)
np.save('x_valid_20',x_valid)
np.save('y_valid_20',y_valid)

x_train = np.load('x_train_20.npy')
y_train = np.load('y_train_20.npy')
x_valid = np.load('x_valid_20.npy')
y_valid = np.load('y_valid_20.npy')


#now build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Conv2D(64, (7, 7), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (7, 7), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(last_layer, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
 
history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size, 
          epochs=epochs,
          verbose=1,
          callbacks=[history],
          #validation_split = 0.05
          validation_data = (x_valid,y_valid)
)

model_file = 'whale-subset-model' + str.replace(datetime.now().isoformat(),':','') + '.h5'

model.save(model_file)
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#plt.plot(range(1, 11), history.acc)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()