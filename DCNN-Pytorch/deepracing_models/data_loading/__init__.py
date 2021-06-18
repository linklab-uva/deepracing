import bisect
import numpy as np
import torch
from typing import Union


class TimeIndex:
    def __init__(self, time_array : Union[np.ndarray, torch.Tensor], data_values : Union[np.ndarray, torch.Tensor]):
        if torch.any(torch.as_tensor(time_array[1:]-time_array[:-1])<0):
            raise ValueError("time_array must be everywhere non-decreasing")
        self.time_array = time_array
        self.data_values = data_values
    def sample(self, tmin : float, tmax : float):
        if tmin>=tmax:
            raise ValueError("tmin (%f) cannot be greater-equal to tmax (%f)" % (tmin, tmax))
        if tmin < self.time_array[0]:
            raise ValueError("tmin (%f) cannot be outside range of time values [%f, %f]" %(tmin, self.time_array[0], self.time_array[-1]))
        if tmax > self.time_array[-1]:
            raise ValueError("tmax (%f) cannot be outside range of time values [%f, %f]" %(tmax, self.time_array[0], self.time_array[-1]))
        leftbisect = max(bisect.bisect_left(self.data_values, tmin) - 1, 0)
        rightbisect = min(bisect.bisect_right(self.data_values, tmax) + 1, self.time_array.shape[0]-1)

        return (leftbisect, rightbisect),  self.data_values[leftbisect:rightbisect]
