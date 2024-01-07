import numpy as np

class Strategy:
    def __init__(self, data,series1, series2=None):
        self.data = data
        self.series1 = series1
        self.series2 = series2

    def crossover(self, series1, series2, direction):
        if direction == "above":
            self.data["indicatorabove"] = np.where(series1 > series2, 1, -1)
        elif direction == "below":
            self.data["indicatorbelow"] = np.where(series1 < series2, 1, -1)
        else:
            raise ValueError("Invalid direction. Use 'above' or 'below'.")
