import os
os.chdir('/content')

from argparse import Namespace
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from datasets import augmentations
#from utils.common import tensor2im, log_input_image
#from models.psp import pSp

#from google.colab import drive
#drive.mount('/content/drive')