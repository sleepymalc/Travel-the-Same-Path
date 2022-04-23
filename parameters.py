"""
This is the parameters module which is typically fixed under 
our experiment setting.

Specifically, the fixed seed is used everywhere to reproduce the 
experiments. 

And all other parameters is for data generation, since this is 
expected to be done only once, so we place them here as a fixed 
number and this is easier for future reference.
"""
# default parameters
seed = 545
time_limit = 3600

# 02_dataset
train_size = 100000
valid_size = 20000
test_size = 1000