import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd 



num_epochs=100
total_series_length=50000
truncated_backdrop_length=15
state_size=4
num_classes=2
echo_step=3
batch_size=5
num_batches=total_series_length//batch_size//truncated_backdrop_length
# print(20/2/2)
# 5

def gen_data():
    #returning a series length of 'total_series_length' comprised of 1 or 0
    x=np.array(np.random.choice(2,total_series_length,[0.5,0.5]))
    print(x)
    y=np.roll(x,echo_step)          #shifting by echo_steps
    print(y)
    x=x.reshape((batch_size,-1))    #creating 'batch_size' mini-batches
    y=y.reshape((batch_size,-1))    #from the sequence of 0s and 1s generated
    print(y)
    return (x,y)










