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
from echonest.remix.action import Playback, Jump, Fadeout, render, display_actions
from echonest.remix.audio import LocalAudioFile

#from longwalk_support import evaluate_distance, matrix_whiten, resample_features_from_point, resample_features_from_vector
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


FUSION_INTERVAL = .06   # This is what we use in the analyzer
AVG_PEAK_OFFSET = 0.025 # Estimated time between onset and peak of segment.


def evaluate_distance(mat1, mat2):
    return np.linalg.norm(mat1.flatten() - mat2.flatten())

def timbre_whiten(mat):
    if rows(mat) < 2: return mat
    m = np.zeros((rows(mat), 12), dtype=np.float32)
    m[:,0] = mat[:,0] - np.mean(mat[:,0],0)
    m[:,0] = m[:,0] / np.std(m[:,0],0)
    m[:,1:] = mat[:,1:] - np.mean(mat[:,1:].flatten(),0)
    m[:,1:] = m[:,1:] / np.std(m[:,1:].flatten(),0) # use this!
    return m


def get_central(analysis, member='segments'):
    """ Returns a tuple: 
        1) copy of the members (e.g. segments) between end_of_fade_in and start_of_fade_out.
        2) the index of the first retained member.
    """
    def central(s):
        return analysis.end_of_fade_in <= s.start and (s.start + s.duration) < analysis.start_of_fade_out
    
    members = getattr(analysis, member)
    ret = filter(central, members[:]) 
    index = members.index(ret[0]) if ret else 0
    
    return ret, index


def get_mean_offset(segments, markers):
    if segments == markers:
        return 0
    
    index = 0
    offsets = []
    try:
        for marker in markers:
            while segments[index].start < marker.start + FUSION_INTERVAL:
                offset = abs(marker.start - segments[index].start)
                if offset < FUSION_INTERVAL:
                    offsets.append(offset)
                index += 1
    except IndexError, e:
        pass
    
    return np.average(offsets) if offsets else AVG_PEAK_OFFSET



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
    print "got timbre matrix, took %s" % end
    pitch = resample_features_from_vector(track, feature='pitches')
    #whiten pitch
    end = dt.now() - start
    print "got pitch matrix, took %s" % end
    duration = resample_features_from_point(track, 'duration')
    end = dt.now() - start
    print "got duration vector, took %s" % end
    loudness = resample_features_from_point(track, 'loudness')
    end = dt.now() - start
    print "got loudness vector, took %s" % end
    confidence = resample_features_from_point(track, 'confidence')
    end = dt.now() - start
    print "got confidence vector, took %s" % end 
    
    ss_timbre = make_similarity_matrix_from_matrix(timbre['matrix'])
    end = dt.now() - start
    print "got timbre ss matrix, took %s" % end
    ss_pitch = make_similarity_matrix_from_matrix(pitch['matrix'])
    end = dt.now() - start
    print "got pitch ss matrix, took %s" % end
    ss_duration = make_similarity_matrix_from_vector(duration)
    end = dt.now() - start
    print "got duration ss matrix, took %s" % end
    ss_loudness = make_similarity_matrix_from_vector(loudness)
    end = dt.now() - start
    print "got loudness ss matrix, took %s" % end
    ss_confidence = make_similarity_matrix_from_vector(confidence)
    end = dt.now() - start
    print "got confidence ss matrix, took %s" % end
    
    return ss_timbre, ss_pitch, ss_loudness, ss_duration, ss_confidence

def normalize_matrix(matrix):
    matrix = (matrix - np.mean(matrix)) / np.std(matrix)
    return matrix

def play_beat(beat_index, track):
    start = track.analysis.beats[beat_index].start *.9
    beat_duration = track.analysis.beats[beat_index].duration *.9
    pygame.mixer.music.play(1,start)
    while 1:
        if pygame.mixer.music.get_pos() >= beat_duration * 1000:
            pygame.mixer.music.stop()
            break    

def play_three_beats(beat_index, track, distance_matrix):
    print "get ready"
    sleep(2)
    original_beat = track.analysis.beats[beat_index]
    distance_vector = distance_matrix[:,beat_index].argsort()
    closest = distance_vector[1]
    second = distance_vector[2]
    third = distance_vector[3]
    print beat_index, closest, second    
    play_beat(beat_index, track)
    play_beat(closest, track)
    play_beat(second, track)


#def get_beat_jumps(beat):
    # first.argsort

if __name__ == "__main__":    
    
    weights = {
        'timbre':10,
        'duration':1,
        'pitch':1,
        'loudness':1,
        'confidence':1,
    }
    
    factors = weights.keys()
    
    config.ECHO_NEST_API_KEY = 'BY9A5IP5MQWF4EQKX'
    usage = "usage: %s [options] <one_single_mp3>" % sys.argv[0]
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()
    start = dt.now()
    track = LocalAudioFile(args[0])

    rate = 'beats'
    data = track
    feature = 'timbre'
    ret = {'rate': rate, 'index': 0, 'cursor': 0, 'matrix': np.zeros((1, 12), dtype=np.float32)}
    segments, ind = get_central(data.analysis, 'segments')
    markers, ret['index'] = get_central(data.analysis, rate)

    # Find the optimal attack offset
    meanOffset = get_mean_offset(segments, markers)
    # Make a copy for local use
    tmp_markers = deepcopy(markers)

    # Apply the offset
    for m in tmp_markers:
        m.start -= meanOffset
        if m.start < 0: m.start = 0

    # Allocate output matrix, give it alias mat for convenience.
    mat = ret['matrix'] = np.zeros((len(tmp_markers)-1, 12), dtype=np.float32)

    # Find the index of the segment that corresponds to the first marker
    f = lambda x: tmp_markers[0].start < x.start + x.duration
    index = (i for i,x in enumerate(segments) if f(x)).next()

    # Do the resampling
    try:
        for (i, m) in enumerate(tmp_markers):
            while segments[index].start + segments[index].duration < m.start + m.duration:
                dur = segments[index].duration
                if segments[index].start < m.start:
                    dur -= m.start - segments[index].start

                C = min(dur / m.duration, 1)

                mat[i, 0:12] += C * np.array(getattr(segments[index], feature))
                index += 1

            C = min( (m.duration + m.start - segments[index].start) / m.duration, 1)
            mat[i, 0:12] += C * np.array(getattr(segments[index], feature))
    except IndexError, e:
        pass # avoid breaking with index > len(segments)





#    timbre = normalize_matrix(timbre)
#    pitch = normalize_matrix(pitch)
#    loudness = normalize_matrix(loudness)
#    duration = normalize_matrix(duration)
#    confidence = normalize_matrix(confidence)
#    print "GOT NORMALIZED MATRICES, CURRENTLY AT %s" % (dt.now() - start)

#    distance = np.zeros(timbre.shape, dtype = 'float32')
#    distance = timbre * weights['timbre'] + duration * weights['duration'] + pitch * weights['pitch'] + loudness * weights['loudness'] + confidence * weights['confidence']

   # pygame.mixer.init()
    
#    pygame.mixer.music.load('the88.ogg')
#    print "GOT DISTANCE MATRICES AND LOADED SONG, CURRENTLY AT %s" % (dt.now() - start)
    
#    for beat_index in range(10,len(track.analysis.beats),100):
#        play_three_beats(beat_index, track, distance)
        
#    for item in list(np.where(distance[:,100].argsort() <= 8))[0]:
#       ....:     play_beat(item, track)



