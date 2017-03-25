import csv
import cv2
import sys
import numpy as np
import tensorflow as tf
import argparse
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Lambda, Cropping2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
import keras.backend.tensorflow_backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import optimizers

# This is the list of recorded data sets which I collected
img_sets = ['../data',                # right
            '../data/clean-backward/', # left
            '../data/bridge/',         # balanced
            '../data/roii_track1_forwards/', # left
            '../data/roii_track1_backwards/', #balanced
            '../data/roii_track2_forwards/',  # right
            '../data/some_more/'
            ]


def read_csv_files():
    '''
    Read the steering angle records from the CSV measurements file
    '''
    lines = []
    for img_set in img_sets:
        csv_file = img_set + '/driving_log.csv'
        print('Reading stats from ', csv_file)
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
        print("lines = ", len(lines))
    return lines

    
''' 
CSV format:
    column 0: center camera image file path
    column 1: left camera image file path
    column 2: right camera image file path
    column 3: steering wheel angle applied at time of frame capture
'''
center_camera = 0    
left_camera = 1
right_camera = 2
steering_angle = 3

'''
Steering angle corrections for center camera, 
left camera, right camera
'''
angle_correction = [0, -0.20, 0.20]

def read_images_and_steering():
    '''
    Create a list of images and their corresponding steering
    angle measurements.
    1. Read the image files names from the CSV files.
    2. Add to the list, the images from the center, right and left cameras.
    3. Correct the steering angle according to the camera vantage point, and
       add to the steer_angles measurements list.
    '''
    import random
    bool = [True, False]
    
    images, steer_angles = [], []
    for img_set in img_sets:
        img_set += "/IMG"
        print("Reading images from ", img_set)
        for line in lines:
            for column in [center_camera, left_camera, right_camera]:
                source_path = line[column]
                source_path = source_path.replace('\\', '/') # for captures from Windows 
                filename = source_path.split('/')[-1]
                current_path = img_set +'/' + filename
                image = cv2.imread(current_path)
                if image is not None: #and column!=left_camera:
                    # correct the steering angle, based on the camera view-point
                    steer_angle = float(line[steering_angle]) + angle_correction[column]
                    images.append(image)
                    steer_angles.append(steer_angle)
                    
                    # Augment data - flip horizontaly
                    # There's an equal chance of flipping as of not flipping 
                    # so the number of left and right turns should be about equal
                    flip = random.choice(bool)
                    if flip: # column==center_camera:
                        images.append(cv2.flip(image,1))
                        steer_angles.append(-1.0 * steer_angle)

        print("Total {} images".format(len(images)))
    return images, steer_angles


def plot_angle_distribution(angles):
    '''
    Plot the distribution of the steering angle measurements.
    We want to make sure that it is fairly balanced between
    right and left.
    '''
    hist, bins = np.histogram(angles, bins=20)
    width = 0.7*(bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.ioff()
    plt.clf()

    plt.bar(center, hist, align = 'center', width = width)
    plt.title('Steering angle distribution')
    plt.ylabel('Frequency')
    plt.xlabel('Angle (Radians)')
    #plt.xlim([-40,40])
    plt.savefig('angles.png', bbox_inches='tight')
    plt.close()
    
from keras.backend import tf as ktf
'''
Steering angle prediction model (based on Nvidia's model)
'''
def get_model():
    drop_rate = 0.25
    in_width, in_height, in_depth = 320, 160, 3
    with tf.device('/gpu:0'):
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))        
        model = Sequential()
        '''
        # Center around zero and normalize to values between -1 to +1
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(in_height,in_width,in_depth), name='Normalize'))
        # Crop ((top_crop, bottom_crop), (left_crop, right_crop))
        model.add(Cropping2D(cropping=((40,25),(0,0)), name='Crop'))

        model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding="same", name='C1_5x5'))
        model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu", padding="same" , name='C2_5x5'))
        model.add(Dropout(drop_rate, name='Drop1'))
        
        model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu", padding="same", name='C3_5x5'))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name='C4_3x3'))
        #model.add(Dropout(drop_rate, name='Drop2'))

        model.add(Conv2D(76, (3, 3), activation="relu", padding="same", name='C5_3x3'))
        model.add(Flatten())
        model.add(Dropout(drop_rate, name='Drop3'))
        
        model.add(Dense(1000, activation="relu", name='FC1'))
        model.add(Dropout(drop_rate, name='Drop4'))
        model.add(Dense(100, activation="relu", name='FC2'))
        model.add(Dense(50, activation="relu", name='FC3'))

        model.add(Dense(1, name='FC4'))
        '''       
        # Center around zero and normalize to values between -1 to +1
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3), name='Normalize'))
        
        #Lambda(lambda image: ktf.image.resize_images(image, (128, 128)))(inp)
        # Crop ((top_crop, bottom_crop), (left_crop, right_crop))
        model.add(Cropping2D(cropping=((70,25),(0,0)), name='Crop'))

        model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(160,320,3), activation="relu", name='C1_5x5'))
        #model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu", name='C2_5x5'))
        model.add(Dropout(0.5, name='Drop1'))
        
        model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu", name='C3_5x5'))
        model.add(Conv2D(64, (3, 3), activation="relu", name='C4_3x3'))
        model.add(Dropout(0.5, name='Drop2'))

        model.add(Conv2D(48, (3, 3), activation="relu", name='C5_3x3'))
        model.add(Dropout(0.5, name='Drop3'))
       
        model.add(Flatten())
        
        model.add(Dense(1000, activation="relu", name='FC1'))
        model.add(Dense(100, activation="relu", name='FC2'))
        model.add(Dense(50, activation="relu", name='FC3'))
        model.add(Dropout(0.5, name='Drop4'))

        model.add(Dense(1, name='FC4'))
        model.summary()
        return model    

def save_model(model):
    model.save('model.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights("model_weights.h5")
        
def plot_history(history):
    # summarize history for loss
    plt.ioff()
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('loss.png', bbox_inches='tight')
    plt.close()
    
def print_model():
    '''
    Print the layers: names and IFM/OFM shapes
    '''
    #for layer in model.layers:
    #    print("{}: {} {}".format(layer.name, layer.input_shape, #layer.output_shape))
    model.summary()
    
if __name__ == "__main__":
    print('-------------Steering Angle Model Training------------------')
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--learn_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-m', '--plot_model', action='store_true', default=False, help='Plot the model')
    parser.add_argument('-a', '--plot_angles', action='store_true', default=False, help='Print the steering angle distribution')
    args = parser.parse_args()

    if args.plot_model:
        model = get_model()
        print_model()
        sys.exit()

    print('-----------------Reading Stats from CSV---------------------')
    lines = read_csv_files()    
    print('---------------------Reading Images-------------------------')
    images, steer_angles = read_images_and_steering()
    assert( len(images)>0 and len(steer_angles)>0 )
    plot_angle_distribution(steer_angles)
    
    if args.plot_angles:
        sys.exit()

    print('------------------------Training----------------------------')
    y_train = np.array(steer_angles)
    X_train = np.array(images, dtype=np.float32)
    model = get_model()
    adam = optimizers.Adam(lr=args.learn_rate)
    model.compile(loss='mse', optimizer=adam)#, metrics=['accuracy'])

    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(X_train, y_train, 
                        epochs=args.epochs, batch_size=args.batch_size, 
                        validation_split=0.2,
                        callbacks=[early_stopping], shuffle=True)
    save_model(model)
    plot_history(history)
    print('--------------------------Done------------------------------')