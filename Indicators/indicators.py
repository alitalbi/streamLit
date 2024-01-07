import pandas as pd
import numpy as np


class Indicator:
    def __init__(self, data):
        self.data = data

    def simple_moving_average(self, window):
        self.data["sma"+str(window)] = self.data['Close'].rolling(window=window).mean()

    def exponential_moving_average(self, span):
        self.data["ema"+str(span)] =  self.data['Close'].ewm(span=span, adjust=False).mean()

    def hull_moving_average(self, window):
        weighted_moving_avg = self.data['Close'].rolling(window=window).mean()
        half_window = int(window / 2)
        sqrt_window = int(np.sqrt(window))
        wma_sqrt_window = weighted_moving_avg.rolling(window=sqrt_window).mean()
        hull_moving_avg = wma_sqrt_window.rolling(window=sqrt_window).mean().shift(-half_window)
        self.data["hma"+str(window)] = hull_moving_avg


    def relative_strength_index(self, window):
        price_diff = self.data['Close'].diff(1)
        gain = price_diff.where(price_diff > 0, 0)
        loss = -price_diff.where(price_diff < 0, 0)

        average_gain = gain.rolling(window=window).mean()
        average_loss = loss.rolling(window=window).mean()

        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))

        self.data["rsi"+str(window)] = rsi

# Example usage:
# Assuming 'data' is a DataFrame with 'Date' and 'Close' columns
# Replace this with your actual data
