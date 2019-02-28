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

# def gen_data():
x=np.array(np.random.choice(4,1,p=[0.25,0.25,0.25,0.25]))
print(x)
