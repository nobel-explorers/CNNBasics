import os
import cv2
import shutil
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def display_image(img_data_rgb):
	fix_colors = cv2.cvtColor(img_data_rgb, cv2.COLOR_BGR2RGB)
	plt.imshow(fix_colors)
	plt.show()

def UB_data_setup(data_dir):

	#assumptions:
	# the original dataset is mostly even across data classes
	# the folder is made up of folders, each which hold a specific 
	#class of data that the data should be labeled as
	old_path = data_dir
	new_path = old_path + '_ub' #just add ub to the original file name
	os.makedirs(new_path)
	data_dir = pathlib.Path(data_dir)
	folders = [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]


	if len(folders) > 3: 
		#step through classes and select a couple evenly spread out
		folders = folders[::(len(folders)//3)]
		#shorten to 3 or fewer classes
		folders = folders[:3]

	findex = 0

	for folder in folders:

		copy_from = os.path.join(old_path, folder)
		copy_to = os.path.join(new_path, folder)
		os.makedirs(copy_to)
		copy_cap = len(os.listdir(copy_from))

		if findex > 0: #only copy over part of files if it's not the first folder
		#if there are 3 folders now, reduce the second 2 by 90% (multiply by .1)
		#if there are 2 folders now, reduce the second by 80% (multiply by .2)
		#didn't want to unbalance the files too much in the case of 2 classes
			copy_cap = copy_cap * .2 / (len(folders) - 1)

		imindex = 0

		for img in os.listdir(copy_from):

			source = os.path.join(copy_from, img)
			destination = os.path.join(copy_to, img)
			shutil.copyfile(source, destination)
			imindex += 1

			if imindex > copy_cap: #we've copied over the amount we want to
				break

		findex += 1 #move on to next file

	return pathlib.Path(new_path)


def data_preprocess(data_dir, IMG_SIZE, CLASS_NAMES, coloring):

    training_data = []
    testing_data = []
    distribution = []

    #use the folder names to generate a file path for each folder
    for class_name in CLASS_NAMES:

        class_count = 0
        path = os.path.join(data_dir,class_name)  # create path to flowers
        class_num = CLASS_NAMES.index(class_name)  # get the classification 0-4
        class_len = len(os.listdir(path))
        class_split = class_len * 0.8

        #check everything in the folder
        for img in os.listdir(path):  
           
            #img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
            img_array = cv2.imread(os.path.join(path,img) ,coloring)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size

            if class_count < class_split:
                training_data.append([new_array, class_num])
                # add this to our training data

            else:
                testing_data.append([new_array, class_num])
                  # add this to our testing data

            class_count+=1

        distribution.append(class_count)
        
    return training_data, testing_data, distribution


def restructure_data(training_data):

    X = []
    y = []

    #split up data	
    for image, label in training_data:
        X.append(image)
        y.append(label)
    
    X = np.array(X)

    #flip X so that the format of the array fits the model
    X_shape = (-1,) + X.shape[1:]
    X.reshape(X_shape)

    return X,y


def build_model(X, y, output_size, coloring):

    model = Sequential()
    
    input_shape = X.shape[1:]
    if coloring == 0:
        input_shape = (input_shape+(1,))

    print(input_shape)

    model.add(Conv2D(256, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))

    model.add(Dense(output_size))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
    

