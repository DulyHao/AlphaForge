from typing import List, Union, Optional, Tuple, Dict
from enum import IntEnum
import numpy as np
import pandas as pd
import torch

from stock_data import StockData

class StockData_df(StockData):
    def 