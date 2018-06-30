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


# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9)
action2rotation = [0,20,-20]
last_reward = 0
scores = []

button_Size=0

move_car_now=False
#size=12

# Initializing the map
first_update = True
def init():
    global sand
    global first_update
    global move_car_now
    global step_size, nrow, ncol
    global lineList
    sand = np.zeros((nrow, ncol))
    first_update = False
    move_car_now=False
    

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

    def __init__(self, matrix, start_position, start_direction, car):
        self.matrix = matrix
        self.current_position = {'x': start_position['x'], 'y': start_position['y']}
        self.current_direction = start_direction
        self.__visited_position = {str(start_position['x']) + '_' + str(start_position['y']): 1}
        self.move_count = 0
        self.turn_count = 0
        self.loggable = False
        self.car = car

    def turn_left(self):
        """turn 90 degree counter-clockwise"""
        #print("Left")
        self.current_direction = (self.current_direction + 1) % 4
        self.turn_count += 1
        #self.car.rotate(90)
        time.sleep(0.1)
        return self

    def turn_right(self):
        """turn 90 degree clockwise"""
        #print("Right")
        self.current_direction = (self.current_direction + 3) % 4
        self.turn_count += 1
        #self.car.rotate(-90)
        time.sleep(0.1)
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
        #self.car.rotate(90*self.current_direction)
            
        #print("Move")
        #print('%d, %d' % (next_pos_x, next_pos_y))
        self.car.move_step_grid(Vector(next_pos_x, next_pos_y))
        time.sleep(0.1)
        return True
    
    def update_data(self, mat):
        self.matrix=mat
        
    def __can_move(self, next_pos_x, next_pos_y):
        global ncol, nrow, sand
        #print('%d, %d , xMax : %d, ymax : %d' % (next_pos_x, next_pos_y, len(self.matrix) ,len(self.matrix[0])))
        if next_pos_x < 0 or next_pos_y < 0:
            return False
        if next_pos_y >= ncol : #len(self.matrix):
            return False
        if next_pos_x >= nrow :#len(self.matrix[0]):
            return False
        #print('sand %d' %(sand[next_pos_x][next_pos_y]))
        
        return sand[next_pos_x][next_pos_y] == 0#self.matrix[next_pos_y][next_pos_x] == 0

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

    def rotate(self, angle):
        self.velocity = Vector(*self.velocity).rotate(self.angle)
        self.rotation = angle
        self.angle= angle
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
    
    def move_step_grid(self, position_in_grid):        
        global nrow, ncol
        srX,srY=self.grid_to_screen(position_in_grid.x,position_in_grid.y, nrow, ncol)
        self.move_step(Vector(srX, srY))
        
        
    def move_step(self , position):
        global step_size
        global lineList
        #print('%f, %f' %(self.center[0]-position[0],self.center[1]-position[1]))
        xFlag,yFlag=self.center[0]-position[0],self.center[1]-position[1]
        
        if xFlag>0:
            self.angle=180
        elif xFlag< 0:
            self.angle=0
        elif yFlag<0:
            self.angle=90
        elif yFlag>0:
            self.angle=-90
        else:
            return
        #print(Vector(*self.velocity).rotate(self.angle)*50 + self.pos)
        
        self.pos = Vector(*self.velocity).rotate(self.angle)*step_size + self.pos
        if self.center != position:
            print("wrong")
            #self.center = position
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        with self.canvas:
            Color(0.8,0.8,0.8,0.9)
            lineList.append(Line(rectangle = (position.x-step_size/4, position.y-step_size/4,step_size/2,step_size/2), width=8))

        
    def move(self, rotation):
        self.velocity = Vector(*self.velocity).rotate(self.angle)
        if rotation !=-1:
            self.pos = Vector(*self.velocity) + self.pos

        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        
    def move_init(self):
        self.velocity = Vector(*self.velocity).rotate(self.angle)

        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos    


    def grid_to_screen(self, x, y, nrow, ncol):    
        global step_size
        sX = (y+2)*step_size + (float(step_size/2))
        sY = (ncol-x)*step_size + (float(step_size/2))
        return sX,sY
    
    def screen_to_grid(self, x):
        global step_size
        return float((x-(float(step_size/2)))/step_size)-(2*step_size)

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class


class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def move_car(self, position):
        self.car.center = position
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
    
    def rotate_car(self, angle):
        self.car.move(angle)
        
    def clear_canvas(self):
        global step_size, nrow, ncol
        global sand

        self.serve_car()
        sand = np.zeros((nrow,ncol))
        self.init_start()
        
    def serve_car(self):
        global step_size, nrow, ncol
        self.car.center = self.center
        self.car.velocity = Vector(1,0)
        self.car.angle=0
        
    
    def init_start(self):
        global step_size, nrow, ncol
        global sand, lineList
        nrow = 10
        ncol=10
        if self.width<self.height:
            step_size = self.width
        else:
            step_size = self.height
                    
        step_size=int (step_size/1.2)
        step_size = int(step_size/ncol)
        
        init()
        no_obs=20
        matrix, start_position = random_matrix(nrow, ncol, no_obs)
        
                
        for ln in lineList:
            self.canvas.remove(ln)
        lineList=[]    
        #print(start_position)
        with self.canvas:
            for i in range(nrow):
                for j in range(ncol):
                    wdt=1
                    if matrix[i][j] != 0 :
                        Color(1.0,0.0,0.1)
                        wdt=3
                    else:
                        Color(0.0,1.0,0.1,0.7)
                    scrX,scrY = self.car.grid_to_screen(i,j, nrow, ncol)
                    lineList.append(Line(rectangle = (scrX-step_size/2, scrY-step_size/2,step_size,step_size),width=wdt))

                        
        sand = matrix.copy()
        matrix=[[]]
        
        srX,srY=self.car.grid_to_screen(start_position['x'],start_position['y'], nrow, ncol)
    
        self.move_car(Vector(srX,srY))
        self.car.move_init()
        print(sand)
        #start_position={'x': self.car.x, 'y': self.car.y}
        start_direction = 0

        # run with dfs
        self.robot = MyRobot(matrix, start_position, start_direction, self.car)
        # robot.log()
        self.sweeper = Sweeper(self.robot)
        self.sweeper.loggable = False
        self.robot.loggable = False
            
    def update(self, dt):
        global step_size, nrow, ncol   
        global move_car_now
        global lineList
        if first_update:
            lineList=[]
            self.init_start()
   
        #if move_car_now == True:
            #self.sweeper.get_move()
            #move_car_now=False
        #print(step_size)
        self.sweeper.get_move()
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        #time.sleep(0.1)


# Adding the painting tools

class MyPaintWidget(Widget):
    pass

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        global move_car_now
        move_car_now=True
        parent = Game()
        self.p=parent
        parent.serve_car()
        self.evnt = Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()

        clearbtn = Button(text = 'clear')
        clearbtn.bind(on_release = self.clear_canvas)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        #movebtn = Button(text = 'move')
        #movebtn.bind(on_release = self.move_car)
        #parent.add_widget(movebtn)
        return parent

    def clear_canvas(self, obj):
        global first_update
        first_update =True
        self.p.canvas.clear()
        Clock.unschedule(self.evnt)
        App.get_running_app().stop()

        
    def move_car(self, obj):
        global move_car_now
        move_car_now=True

            
            
            
# Running the whole thing
if __name__ == '__main__':
    while True:
        CarApp().run()
    
