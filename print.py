import numpy as np
import pandas as pd

# convert your array into a dataframe
data = np.load('results.npy')
data2 = np.load('results (1).npy')

# Daten anzeigen
print("First run:", data, "\n")
print("Second run:", data2)
