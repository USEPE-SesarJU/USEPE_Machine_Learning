import pandas as pd
from pandas import read_csv
import os

file_name = 'TEST_LOGGER_logger_20220123_12-57-27.log'
header_line = 1

### Data importer function
def data_import(file_name, header_line):
    #Opening the file
    file = open(os.path.dirname(__file__) + '/../data/' + file_name)    
    #Creates the dataframe
    d=pd.read_csv(file, header = header_line)
    return(d)

# d = data_import(file_name)
# print(d)


### Data exporter function
def data_export(file):
    file_name = (file[:-4] + '.csv')
    f = os.path.join(os.path.dirname(__file__), '..', 'data', file_name)
    return(f)