# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, Point
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Rectangle


import random
from robot import Robot
from sweeper import Sweeper
from dfs_sweeper import DFSSweeper


# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

import distanceLib as dl


# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9)
action2rotation = [0,20,-20]
last_reward = 0
scores = []

button_Size=0

size=12

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global last_tm
    global time_elapsed
    global time_max
    global g_painter
    global rects
    sand = np.zeros((longueur,largeur))
    goal_x = 20+button_Size
    goal_y = largeur - 20
    first_update = False
    last_tm = 0
    time_elapsed = 0
    time_max = 1000000

    rects= None 
    #rects= [[Line(rectangle=(x, y, 4,4), width=4) for x in range(l1)] for y in range(l2)] 

# Initializing the last distance
last_distance = 0

from utils import sin, cos

def random_matrix(no_rows, no_cols, no_obs):
    arr = []
    for i in range(no_rows * no_cols):
        if i < no_obs:
            arr.append(1)
        else:
            arr.append(0)

    random.shuffle(arr)

    start_position = {'x': 0, 'y': 0}
    rand_pos = random.randint(0, no_rows * no_cols - no_obs - 1)

    matrix = []
    count = 0
    for i in range(no_rows):
        row = []
        for j in range(no_cols):
            row.append(arr[i * no_cols + j])
            if arr[j] == 0:
                if count == rand_pos:
                    start_position = {'x': j, 'y': i}
                count += 1
        matrix.append(row)
    return matrix, start_position

class MyRobot(object):

    def __init__(self, matrix, start_position, start_direction):
        self.matrix = matrix
        self.current_position = {'x': start_position['x'], 'y': start_position['y']}
        self.current_direction = start_direction
        self.__visited_position = {str(start_position['x']) + '_' + str(start_position['y']): 1}
        self.move_count = 0
        self.turn_count = 0
        self.loggable = False

    def turn_left(self):
        """turn 90 degree counter-clockwise"""
        #print("Left")
        self.current_direction = (self.current_direction + 1) % 4
        self.turn_count += 1
        return self

    def turn_right(self):
        """turn 90 degree clockwise"""
        self.current_direction = (self.current_direction + 3) % 4
        self.turn_count += 1
        return self

    def move(self):
        """move ahead"""
        next_pos_x = self.current_position['x'] + cos(self.current_direction)
        next_pos_y = self.current_position['y'] - sin(self.current_direction)
        if not self.__can_move(next_pos_x, next_pos_y):
            self.__visited_position[str(next_pos_x) + "_" + str(next_pos_y)] = -1
            return False
        self.move_count += 1
        self.current_position['x'] = next_pos_x
        self.current_position['y'] = next_pos_y
        self.__visited_position[str(next_pos_x) + "_" + str(next_pos_y)] = 1
        if self.loggable:
            self.log()
        return True
    
    def update_data(self, mat):
        self.matrix=mat
        
    def __can_move(self, next_pos_x, next_pos_y):
        next_pos_x=int(next_pos_x/(size*2))
        next_pos_y=int(next_pos_y/(size*2))
        print('%d, %d , xMax : %d, ymax : %d' % (next_pos_x, next_pos_y, len(self.matrix) ,len(self.matrix[0])))
        if next_pos_x < 0 or next_pos_y < 0:
            return False
        if next_pos_y >= len(self.matrix):
            return False
        if next_pos_x >= len(self.matrix[0]):
            return False
        next_pos_y=int(next_pos_y)
        next_pos_x=int(next_pos_x)
       # print(self.matrix[next_pos_y][next_pos_x] == 0)
        
        return self.matrix[next_pos_y][next_pos_x] == 0

    def log(self):
        for i in range(len(self.matrix)):
            text = ""
            for j in range(len(self.matrix[i])):
                if i == self.current_position['y'] and j == self.current_position['x']:
                    if self.current_direction == 0:
                        text += '>'
                    elif self.current_direction == 1:
                        text += '^'
                    elif self.current_direction == 2:
                        text += '<'
                    else:
                        text += 'v'
                elif self.__visited_position.get(str(j) + "_" + str(i), None) == 1:
                    text += '*'
                elif self.matrix[i][j] == 0:
                    text += '.'
                else:
                    text += '|'
            print(text)
        print('')





# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        if rotation <1:
            self.pos = Vector(*self.velocity).rotate(self.angle) + self.pos
        #self.pos = Vector(*self.velocity) + self.pos
        
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos


class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

lst = list()



class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

  
    def clear_canvas(self):
        global longueur
        global largeur
        self.canvas.clear()
        self.serve_car()
        sand = np.zeros((longueur,largeur))
        start_position={'x': self.car.x, 'y': self.car.y}
        start_direction = 0#random.randint(0, 3)

        # run with dfs
        self.robot = MyRobot(sand, start_position, start_direction)
        # robot.log()
        self.sweeper = Sweeper(self.robot)
        self.sweeper.loggable = False
        self.robot.loggable = False
        
    def serve_car(self):
        global longueur
        global largeur
        self.car.center = self.center
        self.car.velocity = Vector(1,0)
        self.car.angle=0

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global last_tm
        global time_elapsed
        global time_max
        global rects
            

        if first_update:
            longueur = int(self.width)
            largeur = int(self.height)
            longueur=int(longueur/size)
            largeur=int(largeur/size)
            init()
            no_obs=int(longueur*largeur/5)
            matrix, start_position = random_matrix(longueur,largeur, no_obs)

            sand = matrix
            print(sand)
            start_position={'x': self.car.x, 'y': self.car.y}
            start_direction = 0#random.randint(0, 3)
    
            # run with dfs
            self.robot = MyRobot(sand, start_position, start_direction)
            # robot.log()
            self.sweeper = Sweeper(self.robot)
            self.sweeper.loggable = False
            self.robot.loggable = False
       # print("rotation")    
        rotation = self.sweeper.get_move()
        if rotation==-1 or rotation==None:
            #self.car.move(0)
            pass
            #pass #self.car.velocity= Vector(1, 0).rotate(self.car.angle)*0 # Stop the car
        else:
           self.car.move(rotation)
        print(rotation)
        #print(rotation)  
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
 


# Adding the painting tools

class MyPaintWidget(Widget):
    pass
        
"""        
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

            
            

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y
"""
# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        global longueur
        global largeur

        parent = Game()
        longueur = int(parent.width/4)
        largeur = int(parent.height/4)
        self.p=parent
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()

        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        saveMapbtn = Button(text = 'saveMap', pos = (3 * parent.width, 0))
        loadMapbtn = Button(text = 'loadMap', pos = (4 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        saveMapbtn.bind(on_release = self.saveMap)
        loadMapbtn.bind(on_release = self.loadMap)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        parent.add_widget(saveMapbtn)
        parent.add_widget(loadMapbtn)        
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))
        self.p.serve_car()
        

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()
        
    def saveMap(self, obj):
        print("saving Map...")
        global sand, indxLst
        numrows = len(sand)    # 3 rows in your example
        numcols = len(sand[0]) # 2 columns in your example
        indxLst=np.array([])
        for i in range(0,numrows):
            for j in range(0,numcols):
                if(sand[i][j]==1):
                    indxLst = np.hstack((indxLst, i,j))
                    
        np.savetxt('sandMap.txt', indxLst)
        
    def loadMap(self, obj):
        print("loading Map...")
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))     
        pnts = ((np.loadtxt('sandMap.txt')).astype(int)).tolist() # read from txt file
        pnts = np.array(pnts).reshape(int(len(pnts)/2), 2).tolist()
        sorted_list = dl.sort_pt_new(pnts)
        ptlinedraw = np.array(sorted_list).ravel().tolist()
        sand.put(ptlinedraw,1)
        with self.painter.canvas:
            Color(0.8,0.0,0.2)
            Line(points = ptlinedraw)
            
            
            
# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
