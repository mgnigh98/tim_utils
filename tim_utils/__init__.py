import numpy as np
import pandas as pd
import scipy
import sklearn 
import torch as T
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objs as go

import os
import time
import re
import subprocess
import warnings
import sys
from functools import partial
import types
from types import SimpleNamespace

import pyarrow.parquet as pq

from . import utils as tim 
from .syn_data import alpha_beta_prec_recall

# from .scipy import patch_beta_fitstart
# patch_beta_fitstart()