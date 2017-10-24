# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:21:50 2017

@author: Anna
"""

from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import pandas as pd


reader = pd.read_csv('GPS Trajectory/go_track_trackspoints.csv')
mast_dict = defaultdict(list)
for index, row in reader.iterrows():
    mast_dict[row.track_id].append([float(row.latitude), float(row.longitude)])

measurements = np.asarray(mast_dict[4])  # 3 observat

initial_state_mean = [measurements[0, 0],
                      0,
                      measurements[0, 1],
                      0]

transition_matrix = [[1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]]

observation_matrix = [[1, 0, 0, 0],
                      [0, 0, 1, 0]]

kf1 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean)

kf1 = kf1.em(measurements, n_iter=5)
(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)


kf2 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean,
                  observation_covariance = 10*kf1.observation_covariance,
                  em_vars=['transition_covariance', 'initial_state_covariance'])

kf2 = kf2.em(measurements, n_iter=5)
(smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)


plt.plot(measurements[:, 1], measurements[:, 0], 'bo',
         smoothed_state_means[:, 2], smoothed_state_means[:, 0], 'r--',)
plt.show()