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
import pickle


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
#####
def build_beat_segs(track_beats, track_segments):
    beat_segs = []
    for beat_idx in range(len(track_beats)):
        beat_segs_list = []
        if beat_idx < len(track_beats) - 1:
            for segment_idx in range(len(track_segments)):
                if track_beats[beat_idx].start < track_segments[segment_idx].start and track_beats[beat_idx+1].start > track_segments[segment_idx].start:
                    beat_segs_list.append(segment_idx)
        else:
            for segment_idx in range(len(track_segments)):
                if track_beats[beat_idx].start < track_segments[segment_idx].start:
                    beat_segs[beat_idx].append(segment_idx)
        beat_segs.append(beat_segs_list)
    return beat_segs

def segment_distance(segment_1, segment_2):
    distance = 0    
    # for now, just timbre
    a = np.array(segment_1.timbre)
    b = np.array(segment_2.timbre)
    dist = np.linalg.norm(a-b)
    return dist

def get_beat_distance

      
if __name__ == "__main__":    
    
    weights = {
        'timbre':1,
        'duration':100,
        'pitch':10,
        'loudness':1,
        'confidence':1,
    }
    
    factors = weights.keys()
    
    config.ECHO_NEST_API_KEY = 'BY9A5IP5MQWF4EQKX'
    start = dt.now()
    track = LocalAudioFile('callme.ogg')
    
    timbre, pitch, loudness, duration, confidence = get_ss_matrices(track)
    print "GOT SS MATRICES, CURRENTLY AT %s" % (dt.now() - start)
    #timbre = normalize_matrix(timbre)
    #pitch = normalize_matrix(pitch)
    loudness = normalize_matrix(loudness)
    duration = normalize_matrix(duration)
    confidence = normalize_matrix(confidence)
    print "GOT NORMALIZED MATRICES, CURRENTLY AT %s" % (dt.now() - start)

    distance = np.zeros(timbre.shape, dtype = 'float32')
    distance = timbre * weights['timbre'] + duration * weights['duration'] + pitch * weights['pitch'] + loudness * weights['loudness'] + confidence * weights['confidence']
    distance = distance + (-1*np.min(distance))
    np.fill_diagonal(distance, np.max(distance))
    dist_32 = np.zeros(distance.shape, dtype='float32')
    dist_32[:,:] = distance[:,:]
    print "GOT DISTANCE MATRIX, ABOUT TO PICKLE %s" % (dt.now() - start)
    pickle.dump(dist_32, open('callme-distance.pkl','wb'))
    
    beats = track.analysis.beats
    beat_mat = np.zeros([len(beats),2], dtype='float32')
    for beat_index in range(len(beats)):
        beat_mat[beat_index,0] = beats[beat_index].start
        beat_mat[beat_index,1] = beats[beat_index].duration
    pickle.dump(beat_mat, open('callme-beats.pkl','wb'))
###
    beat_segs = build_beat_segs(track.analysis.beats, track.analysis.segments)
    

    