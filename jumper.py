#!/usr/bin/env python
# encoding: utf=8

"""
longwalk.py
"""

import numpy as np
from optparse import OptionParser
import os
from datetime import datetime as dt
from time import time

from echonest.remix.audio import LocalAudioFile

import pygame
from pygame import *
import pickle
import time
from random import randint
from pylab import * 
import math

'''
dt_int = np.zeros([381,381], dtype=int)
dt_int[:,:] = distance[:,:]
dt = dt + (-1 * np.min(dt))
np.fill_diagonal(dt, np.max(dt))
pickle.dump(dt, open('the88-distance.pkl', 'wb'))

beats = track.analysis.beats
beat_mat = np.zeros([len(beats),2], dtype='float32')
for beat_index in range(len(beats)):
    beat_mat[beat_index,0] = beats[beat_index].start
    beat_mat[beat_index,1] = beats[beat_index].duration
pickle.dump(beat_mat, open('the88-beats.pkl','wb'))

from pylab import * 
threshold = THRESHOLD_RATIO * np.ptp(distance) + np.min(distance)
vis_mat = distance >= threshold
# colormap
cmap = cm.jet
# set NaN values as white
cmap.set_bad('w')
im = imshow(distance, cmap=cmap, interpolation='nearest')
colorbar()
show()

--> 110-120
--> 295-315

'''

THRESHOLD_RATIO = .15 # jumps must be within X of the ptp distance for all distances to be considered
HARD_THRESHOLD = 25
JUMP_LIKELIHOOD = 5 # jump proceeds 10-N out of 10 times
JUMP_BARRIER = 4 # jumps cannot be closer than N beats
JUMP_TIME_THRESHOLD = 4
def test():
    pygame.mixer.init()
    pygame.mixer.music.load('the88.ogg')
    pygame.mixer.music.play()
    start = dt.now()
    while 1:
        print "real duration is %s" % (dt.now() - start)
        print "mixer position is %s" % pygame.mixer.music.get_pos()
        time.sleep(3)

def vis_mat(matrix):
    threshold = THRESHOLD_RATIO * np.ptp(distance) + np.min(distance)
    # colormap
    cmap = cm.jet
    # set NaN values as white
    cmap.set_bad('w')
    im = imshow(matrix, cmap=cmap, interpolation='nearest')
    colorbar()
    show()
    

def generate_jump_matrix(distance):
    #threshold = THRESHOLD_RATIO * np.ptp(distance) + np.min(distance)
    dist_bool = distance <= HARD_THRESHOLD
    # is this the best way to get the jumps? is the axis right?
    jump_matrix = np.multiply(distance.argsort(axis=0), dist_bool)
    return jump_matrix

def get_potential_jumps(jump_vector, distance_vector):
    jump_points = []
    threshold_pass = np.where(jump_vector > 0)[0].tolist()
    distance_debug = []
    if len(threshold_pass) > 0:
        for point in threshold_pass:
            distance_debug.append(distance_vector[point])
        jump_points.append(threshold_pass[np.where(distance_debug == np.min(distance_debug))[0][0]])
    return jump_points
    
    
def get_jump_decision(beat_index, jump_points, last_jump_time):
    jump_decision = False
    jump_chance = randint(1,10)
    if jump_chance > JUMP_LIKELIHOOD and time.time() - last_jump_time > JUMP_TIME_THRESHOLD:
        #possible_jump = jump_points[randint(0,len(jump_points)-1)]
        possible_jump = jump_points[0]
        if abs(possible_jump - beat_index) > JUMP_BARRIER:
            jump_decision = possible_jump  - 1
            print "next beat index was %s, possible jump that passed was to %s" %(beat_index, jump_decision + 1)
            COUNT_SINCE_LAST = 0
            #print "DEBUG: POSSIBLE JUMP IS %s" % jump_decision
        else:
            jump_decision = False
            #print "DEBUG: TOO CLOSE, JUMP WOULD HAVE BEEN TO %s" % possible_jump
    #else:
        #print "DEBUG: JUMP CHANCE WAS ONLY %s" % jump_chance 
    return jump_decision

def play_beat_list(beatlist, beats):
    for target in beatlist:
        #print "ABOUT TO PLAY BEAT %s" % target 
        start = beats[target, 0]
        duration = 0.0
        for x in range(1):
            duration += beats[target+x, 1]
        pygame.mixer.music.play(1, start)
        while 1:
            get_pos = pygame.mixer.music.get_pos()
            if float(get_pos)/1000.0 > duration:
                pygame.mixer.music.stop()
                break

def get_servo_pos(current_position, current_angle, max_position):
    angle = math.ceil(float(current_position) / float(max_position)  * 144) * 2.5
    if angle != current_angle:
        print "current angle: %s" % angle
    return angle


def find_nearest_beat(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
        
if __name__ == "__main__":    
    beats = pickle.load(open('callme-beats.pkl', 'rb'))
    distance = pickle.load(open('callme-distance.pkl', 'rb'))
    COUNT_SINCE_LAST = 0
    jump_matrix = generate_jump_matrix(distance)
    
    pygame.mixer.init()
    pygame.mixer.music.load('callme.ogg')
    pygame.mixer.music.play(0,0)
    start = dt.now()
    
    last_jump_time= time.time()
    max_position = beats[-1][0] + beats[-1][1]
    
    sleep_time = np.mean(beats[:,1]) / 2.0

    offset = 0.0
    current_angle = 0.0
    vis_mat(distance)
    while 1:
        get_pos = pygame.mixer.music.get_pos()
        current_position = float(get_pos)/1000. + offset
        current_angle = get_servo_pos(current_position, current_angle, max_position)
        calculated_beat_index = find_nearest_beat(beats[:,0], current_position)
        next_beat_index = calculated_beat_index + 1
        if next_beat_index > len(beats)-1:
            next_beat_index -= 1
        jump_points = get_potential_jumps(jump_matrix[:,next_beat_index], distance[:,next_beat_index])
        #print "current position is %s" % current_position
        #print "calculated beat_index is %s" % calculated_beat_index
        if len(jump_points) > 0:
            jump_decision = get_jump_decision(next_beat_index, jump_points, last_jump_time) 
            if jump_decision != False:
                #print "calculated beat_index is %s" % calculated_beat_index
                #print "next beat would have been %s" % next_beat_index
                #print "jumping to " + str(jump_decision)
                pygame.mixer.music.play(0, beats[jump_decision,0])
                offset = beats[jump_decision,0]
                last_jump_time = time.time() 
                #print "=========================="
        time.sleep(sleep_time)

    '''
    pygame.mixer.music.stop()
    for calculated_beat_index in range(len(beats))[20:]:
        jump_points = get_potential_jumps(jump_matrix[:,calculated_beat_index])
        if len(jump_points) > 0:
            print "ORIGINAL BEAT: %s" % calculated_beat_index
            pygame.mixer.music.play(1, beats[calculated_beat_index,0])
            while 1:
                   next_point = calculated_beat_index + 1
                   if pygame.mixer.music.get_pos() >= beats[next_point, 1] * 1000:
                       pygame.mixer.music.stop()
                       break
            time.sleep(1)
            print "GET READY TO HEAR %s BEATS" % len(jump_points)
            for jump_point in jump_points:
                print "ABOUT TO PLAY THE SIMILAR BEAT %s" % jump_point
                time.sleep(1)
                pygame.mixer.music.play(1, beats[jump_point,0])
                while 1:
                       next_point = jump_point + 1
                       if pygame.mixer.music.get_pos() >= beats[jump_point, 1] * 1000:
                           pygame.mixer.music.stop()
                           break
                time.sleep(1)
            print "=========================="
    '''
