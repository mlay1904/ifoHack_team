import pandas as pd
import streamlit as st
import requests
from PIL import Image
import json

import folium
import geopandas as gpd
from streamlit_folium import st_folium


if "load_state" not in st.session_state:
     st.session_state.load_state = False

st.set_page_config(
    page_title="geogeoapp",
    layout="wide"
)

def parse_output(json_str):
    # Parse the JSON string
    json_obj = json.loads(json_str)

    # Extract the list of integer values inside the brackets
    values = json_obj["predictions"]

    return values


def request_prediction(data):

    host = 'iris_model_api'
    port = '8080'

    url = f'http://{host}:{port}/invocations'

    headers = {
        'Content-Type': 'application/json',
    }

    r = requests.post(url=url, headers=headers, data=data)

    return r.text


def user_input_features():
    with st.sidebar.form("my_form"):
        f1 = st.slider(label='Robbery', min_value=0., max_value=600.1, step=2.5)
        f2 = st.slider(label='bp_weight', min_value=0., max_value=25.1, step=2.5)
        f3 = st.slider(label='restaurant', min_value=0., max_value=300.1, step=2.5)
    #    f4 = st.slider('petal width (cm)', 0., 10., 2.5)
        submitted = st.form_submit_button("Submit")

    data = {'Robbery (count)': f1,
            'bp_weight (count1000)': f2,
            'restaurant': f3
            }
#
    df = pd.DataFrame(data, index=[0])
    # Convert DataFrame to dictionary
    data_dict = df.to_dict(orient='list')
#
    # Create the new dictionary with the required structure
    new_dict = {}
    new_dict["dataframe_split"] = {}
    new_dict["dataframe_split"]["columns"] = list(data_dict.keys())
    new_dict["dataframe_split"]["data"] = [
        list(x) for x in zip(*data_dict.values())]
#
    # Convert the new dictionary to JSON
    new_json = json.dumps(new_dict)
    return new_json


#target = {0.: "Predicted as Iris-Setosa",
#          1.: "Predicted as Iris-Versicolour",
#          2.: "Predicted as Iris-Virginica"}
#target_image = {0.: "Iris-Setosa.jpg",
#                1.: "Iris-Versicolour.jpg",
#                2.: "Iris-Virginica.jpg"}




def run_onetime():
    if st.session_state.load_state == False:
        berlin_gpd = gpd.read_file("crime_pp_cafes.gpkg")

    #    st.session_state.robbery =
    #    st.session_state.robbery =
    #    st.session_state.robbery =

        m = berlin_gpd.explore(
            column="Land_Value",
            tooltip="Land_Value", # show "BoroName" value in tooltip (on hover)
            popup=True, # show all values in popup (on click)
            tiles="CartoDB positron", # use "CartoDB positron" tiles
            cmap="viridis", # use "Set1" matplotlib colormap
            style_kwds=dict(color="black"), # use black outline
            name = "land"
        )

        at = berlin_gpd.explore(
            m=m,
            column = "Area_Types",
            #categories= categories,
            categorical = True,
            legend = True,
            tooltip="Area_Types", # show "BoroName" value in tooltip (on hover)
            popup=True, # show all values in popup (on click)
            tiles="CartoDB positron", # use "CartoDB positron" tiles
            cmap="Set1", # use "Set1" matplotlib colormap
            style_kwds=dict(color="black"), # use black outline
            name="area_type" # show all values in popup (on click)
        )

        ro = berlin_gpd.explore(
            m=at,
            column = "Robbery",
            #categories= categories,
            #categorical = True,
            legend = True,
            tooltip="Robbery", # show "BoroName" value in tooltip (on hover)
            popup=True, # show all values in popup (on click)
            tiles="CartoDB positron", # use "CartoDB positron" tiles
            cmap="Blues", # use "Set1" matplotlib colormap
            style_kwds=dict(color="black"), # use black outline
            name="Robbery" # show all values in popup (on click)
        )

        bg=berlin_gpd.explore(
            m=ro,
            column = "bp_weight",
            #categories= categories,
            #categorical = True,
            legend = True,
            tooltip="bp_weight", # show "BoroName" value in tooltip (on hover)
            popup=True, # show all values in popup (on click)
            tiles="CartoDB positron", # use "CartoDB positron" tiles
            cmap="Reds", # use "Set1" matplotlib colormap
            style_kwds=dict(color="black"), # use black outline
            name="bp_weight" # show all values in popup (on click)
        )

        re= berlin_gpd.explore(
            m=bg,
            column = "restaurant",
            #categories= categories,
            #categorical = True,
            legend = True,
            tooltip="restaurant", # show "BoroName" value in tooltip (on hover)
            popup=True, # show all values in popup (on click)
            tiles="CartoDB positron", # use "CartoDB positron" tiles
            cmap="Greens", # use "Set1" matplotlib colormap
            style_kwds=dict(color="black"), # use black outline
            name="restaurant" # show all values in popup (on click)
        )

        folium.TileLayer('Stamen Toner', control=True).add_to(re)  # use folium to add alternative tiles
        folium.LayerControl('topleft', collapsed=False).add_to(re)
        
        st.session_state.load_state = re
#    else:
#        st.stop()
    st.subheader('Map')
    st_folium(st.session_state.load_state, width='100%', height=1000, returned_objects=[])
    return

run_onetime()
data = user_input_features()

output = request_prediction(data)
output = parse_output(output)
output = output[0]

#st_data = 

#st.stop()
st.write(output)

#image = Image.open(target_image[output])
#st.image(image, caption=target_image[output])
