# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:12:42 2021

@author: oscar
"""
import numpy as np
from scipy.constants import astronomical_unit as UA


DictC = {
    'sol': {
        'masa': 1.9891E30,
        'x_0': [0., 0, 0],
        'v_0': [0., 0, 0.],
        'color': 'yellow'},

    'mercurio': {
        'masa': 3.285e23,
        'x_0': [5.791e10*np.cos(np.radians(7)), 0, 5.791e10*np.sin(np.radians(7))],
        'v_0': [0., 4.7e4, 0.],
        'color': 'silver'},

    'venus': {
        'masa': 4.87e24,
        'x_0': [1.082e11*np.cos(np.radians(3.4)), 0, 1.082e11*np.sin(np.radians(3.4))],
        'v_0': [0., 3.5e4, 0.],
        'color': 'bisque'},

    'tierra_S': {
        'masa': 5.972E24,
        'x_0': [UA, 0, 0],
        'v_0': [0, 2.9e4, 0.],
        'color': 'blue'},

    'tierra_T': {
        'masa': 5.972E24,
        'x_0': [0, 0, 0],
        'v_0': [0, 0, 0],
        'color': 'blue'},

    'luna_S': {
        'masa': 7.35e22,
        'x_0': [UA, 0, 0],
        'v_0': [1e3, 2.9E4, 0],
        'color': 'white'},

    'luna_T': {
        'masa': 7.35e22,
        'x_0': [0, 3.57e8, 0],
        'v_0': [1e3, 0, 0],
        'color': 'white'},

    'marte': {
        'masa': 6.4e23,
        'x_0': [0., 2.279e11*np.cos(np.radians(1.85)), 2.279e11*np.sin(np.radians(1.85))],
        'v_0': [-2.4e4, 0, 0.],
        'color': 'red'},

    'jupiter': {
        'masa': 1.898E27,
        'x_0': [7.7833e11*np.cos(np.radians(1.3)), 0., 7.7833e11*np.sin(np.radians(1.3))],
        'v_0': [0., 1.3E4, 0.],
        'color': 'orange'},

    'saturno': {
        'masa': 5.68e26,
        'x_0': [1.429e12*np.cos(np.radians(2.49)), 0, 1.429e12*np.sin(np.radians(2.49))],
        'v_0': [0., 9.6e3, 0.],
        'color': 'tab:olive'},

    'urano': {
        'masa': 8.68e25,
        'x_0': [2.871e12*np.cos(np.radians(0.77)), 0, 2.871e12*np.sin(np.radians(0.77))],
        'v_0': [0., 6.8e3, 0.],
        'color': 'c'},

    'neptuno': {
        'masa': 1.024e26,
        'x_0': [4.504e12*np.cos(np.radians(1.77)), 0, 4.504e12*np.sin(np.radians(1.77))],
        'v_0': [0., 5.4e3, 0.],
        'color': 'lightblue'},
}
