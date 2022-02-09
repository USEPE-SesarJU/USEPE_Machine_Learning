import pandas as pd
from pandas import read_csv
import os

from pyparsing import line

### Data importer function
def data_import(file_name, header_line):
    #Opening the file
    file = open(os.path.dirname(__file__) + '/../data/' + file_name)    
    #Creates the dataframe
    d=pd.read_csv(file, header = header_line)
    return(d)

# d = data_import(file_name)
# print(d)


### Exporter function for ML result
def data_export(file):
    file_name = (file[:-4] + '.csv')
    f = os.path.join(os.path.dirname(__file__), '..', 'data', file_name)
    return(f)

### Exporter function for map
def export_png(file):
    file_name = (file[:-4] + '.png')
    f = os.path.join(os.path.dirname(__file__), '..', 'data', file_name)
    return(f)


# file = 'TEST_LOGGER_logger_20220124_20-39-04.log'
# result_file_name = (file[:-4] + '.csv')
# result_file = data_export(result_file_name)
# file = open(os.path.dirname(__file__) + '/../data/' + file)
# df = pd.read_csv(file, header = 1)
# file.close();


def search_string_in_file(file_name, string_to_search):
    line_number = 0
    list_of_results = []
    file = open(os.path.dirname(__file__) + '/../data/' + file_name)
    with file as read_obj:
        for line in read_obj:
            line_number += 1
            if string_to_search in line:
                list_of_results.append((line_number, line.rstrip()))
    return list_of_results


matched_lines = search_string_in_file('TEST_LOGGER_logger_20220124_20-39-04.log', 'sim')
print('Total Matched lines : ', len(matched_lines))
for elem in matched_lines:
    print('Line Number = ', elem[0], ' :: Line = ', elem[1])