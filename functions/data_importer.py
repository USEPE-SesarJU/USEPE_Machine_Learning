import pandas as pd
from pandas import read_csv
import os

# file_name = 'TEST_LOGGER_logger_20220123_12-57-27.log'
header_line = 1

### Data importer function

def data(file_name):
    #Opening the file
    file = open(os.path.dirname(__file__) + '/../data/' + file_name)
    
    #Creates the dataframe
    df=pd.read_csv(file, header = header_line)

    return(df)

# d = data(file_name)
# print(d)
    