import sys
import random
import numpy as np
import cv2
import skimage as skimage
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import tensorflow as tf
from keras import backend as K
import my_cv
from keras.models import load_model
import pickle


img_rows = 80
img_cols = 80



class DQNAgent:

    def __init__(self, state_size, action_size):


        self.t = 0
        self.max_Q = 0
        self.train = False
        self.lane_detection = True # Set to True to train on images with segmented lane lines

        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if (self.train):
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000

        # Create replay memory using deque
        self.memory = deque(maxlen=10000)

        # Create main model and target model
        self.model = self.build_model()
        #self.model = load_model('save_model.h5')
        self.target_model = self.build_model()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    def build_model(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(80,80,4)))  #80*80*4
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))

        # 15 categorical bins for Steering angles
        model.add(Dense(15, activation="linear")) 

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        print("We finished building the model")

        return model

    def return_model(self):

        return self.model

    def process_image(self, image):

        if not model.lane_detection:

            obs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            obs = cv2.resize(obs, (img_rows, img_cols))
            #obs = skimage.color.rgb2gray(obs)
            #obs = skimage.transform.resize(obs, (img_rows, img_cols))

            return obs          
        else:
            lower1 = np.uint8([0, 0, 0])
            upper1 = np.uint8([255, 255, 255])
            white_mask = cv2.inRange(image, lower1, upper1)

            lower2 = np.uint8([0, 0, 0])
            upper2 = np.uint8([255, 255, 255])
            yellow_mask = cv2.inRange(image, lower2, upper2)
            mask = cv2.bitwise_or(white_mask, yellow_mask)
            masked = cv2.bitwise_and(image, image, mask = mask)

            masked=cv2.Canny(masked[60:,10:], 300, 500)

            _, contours, hierarchy = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:

                (x, y, w, h) = cv2.boundingRect(contour)
            # if h<500: # check the length of verical lines 
                if w<30: # check the length of horizontal lines
                        #if w<15: # check the length of horizontal lines
                    cv2.fillConvexPoly(masked,contour,0) #fill the contour with black color

            edges = cv2.Canny(masked[:,:],100,200,apertureSize = 3)

            minLineLength=masked.shape[1]-300   
            lines = cv2.HoughLinesP(image=edges,rho=0.02,theta=np.pi/500, 
                                    threshold=1,lines=np.array([]), minLineLength=minLineLength,maxLineGap=100) 
            if lines is not None:   
                a,b,c = lines.shape
                for i in range(a):
                    cv2.line(masked, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255,255, 255), 2, cv2.LINE_AA)    

                    

            masked2 = skimage.transform.resize(masked, (img_rows, img_cols))

            return masked2,masked
        

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            #print("Return Random Value")
            #return random.randrange(self.action_size)
            return np.random.uniform(-1,1)
        else:
            #print("Return Max Q Prediction")
            q_value = self.model.predict(s_t)
            # Convert q array to steering value
            return linear_unbin(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            #self.epsilon *= self.epsilon_decay
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore


    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])

        self.model.train_on_batch(state_t, targets)

    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


###### Utils Functions #######

def linear_bin(a):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float
        A value between -1 and 1

    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    """
    Convert a categorical array to value.

    See Also
    --------
    linear_bin
    """
    if not len(arr) == 15:
        raise ValueError('Illegal array length, must be 15')
    b = np.argmax(arr)
    a = b * (2 / 14) - 1
    return a


##############################################


state_size = (80, 80, 4)

action_size = 2

model = DQNAgent(state_size, action_size)

image_arr = []

class PilotKeras:

    def __init__(self):

        self.current_steering = 0
        self.last_steering = 0
        self.diff = 0

    def run(self, img_arr):

        

        x_t,b= self.model.process_image(img_arr)

        image_arr.append(b)

        with open("img_new_process.pkl", "wb") as f:
            pickle.dump(image_arr, f)


        s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
    
        # In Keras, need to reshape
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*80*80*4
        #obs = skimage.color.rgb2gray(x_t)
        #s_t = skimage.transform.resize(obs, (80, 80)) 

     

        # This model currently assumes that the features of the model are just the images. Feel free to change this.
        steering = self.model.get_action(s_t)
        throttle = 0.65

        
        self.current_steering = steering * 1.5

        
        # The driving model currently just outputs a constant throttle. Feel free to edit this.

        if(self.current_steering >= 0 and self.last_steering >= 0 and abs(self.current_steering - self.last_steering) > 0.4):

            print("1111111111")

            self.diff = self.current_steering - self.last_steering

            if(self.diff > 0):

                self.current_steering = self.current_steering - (self.diff - 0.4)

            else:

                self.current_steering = self.current_steering + (-self.diff - 0.4)


        elif(self.current_steering <= 0 and self.last_steering <= 0 and abs(self.current_steering - self.last_steering) > 0.4):

            print("222222222")

            self.diff = self.current_steering - self.last_steering

            if(self.diff < 0):

                self.current_steering = self.current_steering + (-self.diff - 0.4)

            else:
                self.current_steering = self.current_steering - (self.diff - 0.4)


        elif(self.current_steering > 0 and self.last_steering < 0):

            print("33333333333")


            self.diff = self.current_steering + abs(self.last_steering)

            self.current_steering = self.current_steering - (self.diff - 0.4)

        elif(self.current_steering < 0 and self.last_steering > 0):

            print("444444444444")

            self.diff = abs(self.current_steering) + self.last_steering

            self.current_steering = self.current_steering + (self.diff - 0.4)


        print(self.current_steering, self.last_steering, "TTTTTTTTT")

        steering = self.current_steering
        self.last_steering = self.current_steering
        
        
        
        print(steering , throttle)

        return steering, throttle

    def load(self, model_json_file):
        

        state_size = (80, 80, 4)

        action_size = 2

        model = DQNAgent(state_size, action_size)
        model.train = False

        episodes = []

        if not model.train:
            print("Now we load the saved model")
            model.load_model("save_model.h5")


        self.model = model

