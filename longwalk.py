#!/usr/bin/env python
# encoding: utf=8

"""
longwalk.py
"""

from copy import deepcopy
import numpy as np
from numpy.matlib import repmat, repeat
from optparse import OptionParser
from numpy import sqrt
import operator
import os
from datetime import datetime as dt
import sys
from time import sleep

from pyechonest import config
from echonest.remix.audio import LocalAudioFile

from longwalk_support import evaluate_distance, matrix_whiten, resample_features_from_point, resample_features_from_vector
#from earworm_support import evaluate_distance, timbre_whiten, resample_features
from utils import rows, tuples, flatten

import pygame
from pygame import *


DEF_DUR = 600
MAX_SIZE = 800
MIN_RANGE = 16
MIN_JUMP = 16
MIN_ALIGN = 16
MAX_EDGES = 8
FADE_OUT = 3
RATE = 'beats'


#def make_similarity_matrix(matrix, size=MIN_ALIGN):
def make_similarity_matrix_from_matrix(matrix):
    singles = matrix.tolist()
    points = [flatten(t) for t in tuples(singles, 1)]    
    numPoints = len(points)
    distMat = sqrt(np.sum((repmat(points, numPoints, 1) - repeat(points, numPoints, axis=0))**2, axis=1, dtype=np.float32))
    return distMat.reshape((numPoints, numPoints))

def make_similarity_matrix_from_vector(vector):
    vector1x, vector2x = np.ix_(vector, vector)
    vector_similarity_matrix = sqrt((vector1x - vector2x)**2)
    return vector_similarity_matrix


def get_ss_matrices(track):
    ''' TIME TRIAL
    got timbre matrix, took 0:00:00.069210
    got pitch matrix, took 0:00:00.135894
    got duration vector, took 0:00:00.136310
    got loudness vector, took 0:00:00.427258
    got confidence vector, took 0:00:00.427841
    '''
    start = dt.now()
    timbre = resample_features_from_vector(track, feature='timbre')
    end = dt.now() - start
    #print "got timbre matrix, took %s" % end
    pitch = resample_features_from_vector(track, feature='pitches')
    #whiten pitch
    end = dt.now() - start
    #print "got pitch matrix, took %s" % end
    duration = resample_features_from_point(track, 'duration')
    end = dt.now() - start
    #print "got duration vector, took %s" % end
    loudness = resample_features_from_point(track, 'loudness')
    end = dt.now() - start
    #print "got loudness vector, took %s" % end
    confidence = resample_features_from_point(track, 'confidence')
    end = dt.now() - start
    #print "got confidence vector, took %s" % end 
    
    ss_timbre = make_similarity_matrix_from_matrix(timbre['matrix'])
    end = dt.now() - start
    #print "got timbre ss matrix, took %s" % end
    ss_pitch = make_similarity_matrix_from_matrix(pitch['matrix'])
    end = dt.now() - start
    #print "got pitch ss matrix, took %s" % end
    ss_duration = make_similarity_matrix_from_vector(duration)
    end = dt.now() - start
    #print "got duration ss matrix, took %s" % end
    ss_loudness = make_similarity_matrix_from_vector(loudness)
    end = dt.now() - start
    #print "got loudness ss matrix, took %s" % end
    ss_confidence = make_similarity_matrix_from_vector(confidence)
    end = dt.now() - start
    #print "got confidence ss matrix, took %s" % end
    
    return ss_timbre, ss_pitch, ss_loudness, ss_duration, ss_confidence

def normalize_matrix(matrix):
    matrix = (matrix - np.mean(matrix)) / np.std(matrix)
    return matrix

def play_beat(beat_index, track):
    start = track.analysis.beats[beat_index].start 
    print 'actual start is %s ' % start
    beat_duration = track.analysis.beats[beat_index].duration 
    print 'target duration is %s'% beat_duration
    pygame.mixer.music.play(1,start)
    while 1:
        if pygame.mixer.music.get_pos()*.85 >= beat_duration * 1000:
            print 'mixer get pos was %s' % (pygame.mixer.music.get_pos())
            pygame.mixer.music.stop()
            break    

def play_three_beats(beat_index, track, distance_matrix, threshold):
    original_beat = track.analysis.beats[beat_index]
    distance_vector = distance_matrix[:,beat_index].argsort()
    closest = [distance_vector[1], distance_matrix[distance_vector[1],beat_index]]
    second = [distance_vector[2], distance_matrix[distance_vector[2],beat_index]]
    third = [distance_vector[3], distance_matrix[distance_vector[3],beat_index]]
    fourth = [distance_vector[4], distance_matrix[distance_vector[4],beat_index]]
    fifth = [distance_vector[5], distance_matrix[distance_vector[5],beat_index]]
    for item in [closest, second, third, fourth, fifth]:
        if item[1] <= threshold and abs(item[0] - beat_index) > .10*len(track.analysis.beats):  
            print "get ready for first beat"
            play_beat(beat_index, track)
            sleep(1)
            print "get ready for second beat"
            play_beat(item[0], track)
            print beat_index, item
            sleep(3)

'''
def get_jump_decision(jump_probability, count_since_last_jump, current_beat):
## DECIDE IF JUMP SHOULD HAPPEN
## FUNCTION OF PROBABILITY (USER CONTROLLED), TIME SINCE LAST JUMP
## IF AT FINAL POSSIBLE JUMP POINT, MAKE A BACKWARDS JUMP
    return jump_decision, new_count
    
def which_beat_next(current_beat):
## USE JUMP DECISION TO FIGURE OUT IF WE SHOULD JUMP
## IF YES, GET A NEW VALUE TO JUMP TO
## IF NO, INCREMENT BEAT
    return next_beat
'''   

def test(track):
    duration = 0
    duration += track.analysis.beats[20].duration
    print 'expected start is %s' %track.analysis.beats[20].start
    print 'expected duration is %s' % duration
    start_test = dt.now()
    for beat in range(20,21):
        play_beat(beat, track)
    end = dt.now() - start_test 
    print 'actual duration was %s' % end

def play_song(track, distance):
    threshold = .1*np.ptp(distance) + np.min(distance)
    
    for beat_index in range(0,len(track.analysis.beats)):
        play_three_beats(beat_index, track, distance, threshold)
        
if __name__ == "__main__":    
    
    weights = {
        'timbre':10,
        'duration':1,
        'pitch':5,
        'loudness':5,
        'confidence':1,
    }
    
    factors = weights.keys()
    
    config.ECHO_NEST_API_KEY = 'BY9A5IP5MQWF4EQKX'
    usage = "usage: %s [options] <one_single_mp3>" % sys.argv[0]
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()
    start = dt.now()
    track = LocalAudioFile(args[0])
    
    end = dt.now() - start
    #print "GOT FILE, TOOK %s" % end
    # this is where the work takes place
    timbre, pitch, loudness, duration, confidence = get_ss_matrices(track)
    #print "GOT SS MATRICES, CURRENTLY AT %s" % (dt.now() - start)
    timbre = normalize_matrix(timbre)
    pitch = normalize_matrix(pitch)
    loudness = normalize_matrix(loudness)
    duration = normalize_matrix(duration)
    confidence = normalize_matrix(confidence)
    #print "GOT NORMALIZED MATRICES, CURRENTLY AT %s" % (dt.now() - start)

    distance = np.zeros(timbre.shape, dtype = 'float32')
    distance = timbre * weights['timbre'] + duration * weights['duration'] + pitch * weights['pitch'] + loudness * weights['loudness'] + confidence * weights['confidence']

    pygame.mixer.init()
    
    pygame.mixer.music.load(args[0])
    print "GOT DISTANCE MATRICES AND LOADED SONG, CURRENTLY AT %s" % (dt.now() - start)
    
    play_song(track, distance)
    #test(track)   
#    for item in list(np.where(distance[:,100].argsort() <= 8))[0]:
#       ....:     play_beat(item, track)



