import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanFilterModule:
    def __init__(self, dim, process_variance=1e-5, measurement_variance=1e-3):

        self.kf = KalmanFilter(dim_x=dim, dim_z=dim)
        
        self.kf.x = np.zeros((dim, 1))
        
        self.kf.P *= 1000.
        
        self.kf.F = np.eye(dim)
        self.kf.H = np.eye(dim)
        
        self.kf.R = measurement_variance * np.eye(dim)
        
        self.kf.Q = process_variance * np.eye(dim)

    def update(self, measurement):

        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x.flatten()