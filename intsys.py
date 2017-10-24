from pykalman import KalmanFilter
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

def plot_path(points):
    plt.scatter(*zip(*points))
    plt.ylabel('some numbers')
    plt.show()

reader = pd.read_csv('GPS Trajectory/go_track_trackspoints.csv')
mast_dict = defaultdict(list)
for index, row in reader.iterrows():
    mast_dict[row.track_id].append([float(row.latitude), float(row.longitude)])
#print(mast_dict)

kf = KalmanFilter(transition_matrices=[[1, 1], [0, 1]], observation_matrices=[[0.1, 0.5], [-0.3, 0.0]])
#kf = KalmanFilter(n_dim_obs=2)
#measurements = np.asarray([[1, 0], [0, 0], [0, 1]])  # 3 observations


measurements = np.asarray(mast_dict[4])  # 3 observations
plot_path(measurements)
# measurements = [np.reshape(x, (1, 1)) for x in measurements]
#print(measurements)
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
#plt.plot(filtered_state_means, filtered_state_covariances)
#plt.ylabel('some numbers')
#plt.show()
#(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
print ((filtered_state_means, filtered_state_covariances))#, (smoothed_state_means, smoothed_state_covariances))


