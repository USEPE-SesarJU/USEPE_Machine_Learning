import pandas as pd
import plotly.express as px
from data_importer import * 

file = 'TEST_LOGGER_logger_20220124_20-39-04.csv'
d = data(file)
print(d)

fig = px.line_mapbox(d, lat=" lat", lon=" lon", color=" Data Type", zoom=100, height=800)
fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=11, mapbox_center_lat = 52.45, margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
figure_name = (file[:-4] + '.png')
#fig.write_image(figure_name)
