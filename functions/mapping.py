import pandas as pd
import os
import plotly.express as px
from data import data_import
from data import data_export
from data import export_png

def on_map(file):
    d = data_import(file, 0)
    fig = px.line_mapbox(d, lat=" lat", lon=" lon", color=" Data Type", zoom=100, height=800)
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=11, mapbox_center_lat = 52.45, margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    fig.write_image(export_png(file))