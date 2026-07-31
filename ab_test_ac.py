import sys
import os

sys.argv = ["multispan_net.py", "--ripple"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural_nets"))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

import multispan_net as M