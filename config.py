# define the variables for train the model
# output directory, this work if you pass an image for the program as if you train a new model.
OUTPUT_PATH = 'output'

# model settings
# initialize the input shape and number of classes
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 2
# compiling settings
METRICS = ["accuracy"]
LOSS = "binary_crossentropy"
OPTIMIZER = 'adam'

EPOCHS = 20
BS = 32

# if keras auto tunning
EARLY_STOPPING_PATIENCE = 5

# TUNER = 'random'
# TUNER = 'hyperband'
TUNER = 'bayesian'

# save the model
MODEL_SAVE = True

DATASET = '../../datasets/SMILEs'
