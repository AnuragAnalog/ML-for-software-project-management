#!/usr/bin/python3

import pandas as pd
from scipy.io.arff import loadarff

def load_cocomo81():
    raw_data = loadarff('./cocomo81.arff')
    data = pd.DataFrame(raw_data[0])
    data.to_csv('cocomo81.csv', index=False)

    return data

if __name__ == "__main__":
    cocomo = load_cocomo81()
    print(cocomo)