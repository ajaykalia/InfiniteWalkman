#!/usr/bin/env python
# encoding: utf=8

"""
longwalk.py
"""

import numpy as np
from optparse import OptionParser
from numpy import sqrt
import operator
import os
from datetime import datetime as dt
import sys
from time import sleep

from pyechonest import config
from echonest.remix.audio import LocalAudioFile


import pickle

#def make_similarity_matrix(matrix, size=MIN_ALIGN):

def make_similarity_matrix_from_vector(vector):
    vector1x, vector2x = np.ix_(vector, vector)
    vector_similarity_matrix = sqrt((vector1x - vector2x)**2)
    return vector_similarity_matrix

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
                    beat_segs_list.append(segment_idx)
        beat_segs.append(beat_segs_list)
    return beat_segs

def segment_distance(segment_1, segment_2):
    # for now, just timbre
    a = np.array(segment_1.timbre)
    b = np.array(segment_2.timbre)
    dist = np.linalg.norm(a-b)
    #a = segment_1.duration
    #b = segment_2.duration
    #dist += abs(a-b) * 100
    return dist

def get_beat_distance(beat_1_idx, beat_2_idx):
    beat_dist = 0
    beat_1_segments = beat_segs[beat_1_idx]
    beat_2_segments = beat_segs[beat_2_idx]
    if len(beat_1_segments) > 0:
        if len(beat_2_segments) > 0:
            i = 0
            while i <= len(beat_1_segments):
                try:
                   beat_dist += segment_distance(segments[beat_1_segments[i]], segments[beat_2_segments[i]])
                   i += 1
                except:
                    break
        else:
            # fix this
            beat_dist = 0
    else:
        beat_dist = 0
    return beat_dist

      
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
    track = LocalAudioFile('the88.ogg')
    
    start = dt.now()
    print "starting analysis now"
    beats = track.analysis.beats
    segments = track.analysis.segments

    beat_segs = build_beat_segs(beats, segments)
    
    distance_mat = np.zeros([len(beats), len(beats)], dtype='float32')
    
    for i in range(len(beats)):
        for j in range(len(beats)):
            distance_mat[i,j] = get_beat_distance(i,j)
    np.fill_diagonal(distance_mat, np.max(distance_mat))
    distance_mat[np.where(distance_mat == 0)] = np.max(distance_mat)
    print "GOT DISTANCE MATRIX, ABOUT TO PICKLE %s" % (dt.now() - start)
    pickle.dump(distance_mat, open('the88-distance.pkl','wb'))
    
    beats = track.analysis.beats
    beat_mat = np.zeros([len(beats),2], dtype='float32')
    for beat_index in range(len(beats)):
        beat_mat[beat_index,0] = beats[beat_index].start
        beat_mat[beat_index,1] = beats[beat_index].duration
    pickle.dump(beat_mat, open('the88-beats.pkl','wb'))
###
    

    