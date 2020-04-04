import numpy as np


def mbcg(m500):
    x = np.log10(m500) - 14.5
    y = 0.39*x+12.15
    return 10**y


def msat(m500):
    x = np.log10(m500) - 14.5
    y = 0.87*x+12.42
    return 10**y


def rbcg(r200):
    x = np.log10(r200) - 1.0
    y = 0.95*x-0.3
    return 10**y