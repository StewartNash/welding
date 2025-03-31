import pandas as pd
import numpy as np

TOTAL_ROWS = 10000

COLUMNS = [
    'electrode force',
    'electrode contact surface diameter',
    'squeeze time',
    'weld time',
    'hold time',
    'weld current',
    'leak rate',
    'explosive force',
    'leaking',
    'explosion'
]

INPUT_COLUMNS = [
    'electrode force',
    'electrode contact surface diameter',
    'squeeze time',
    'weld time',
    'hold time',
    'weld current'    
]

OUTPUT_COLUMNS = [
    'leak rate',
    'explosive force',
    'leaking',
    'explosion'
]

#def generate_row(length):
#    return np.random.rand(length)

def generate_coefficients(input_length, output_length):
    return np.random.rand(length)

#def generate_row(input_length, output_length):
def generate_row(coefficients, length):
    input_row = np.random.rand(len(coefficients))

my_dataframe = pd.Dataframe(columns=COLUMNS)
coefficients = generate_coefficients(len(INPUT_COLUMNS))
for i in range(TOTAL_ROWS):
    #my_dataframe.loc[len(my_dataframe)] = generate_row(len(my_dataframe.columns))
    my_dataframe.loc[len(my_dataframe)] = generate_row(coefficients, len(OUTPUT_COLUMNS))
