# import tensorflow as tf
# import tensorflow.keras as keras
from utils import *
import numpy as np


def process(file_data):
    a = np.load(file_data)[:3000]
   
    score = np.empty(len(a))

    for i in range(len(a)):
        score[i] = np.mean(a[i] == a[i][0])

    return score

if __name__ == '__main__':
    
    AUROC_Score(process("../../dmlab_wanet.npy"),
               process("../../dmlab_modify_benign.npy"), "dmlab")
  
