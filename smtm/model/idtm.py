
"""
Infinite Dynamic Topic Model (iDTM)
"""

#######################
### Imports
#######################

## Standard Library
import os
import sys
from datetime import datetime
from collections import Counter

## External Libraries
import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.special import logsumexp