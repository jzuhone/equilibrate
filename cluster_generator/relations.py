import numpy as np


def f_gas(M500, hubble=0.7):
    return ((0.72/hubble)**1.5)*(0.125+0.037*np.log10(M500*1.0e-15))


def m_bcg(m500):
    x = np.log10(m500) - 14.5
    y = 0.39*x+12.15
    return 10**y


def m_sat(m500):
    x = np.log10(m500) - 14.5
    y = 0.87*x+12.42
    return 10**y


def r_bcg(r200):
    x = np.log10(r200) - 1.0
    y = 0.95*x-0.3
    return 10**y