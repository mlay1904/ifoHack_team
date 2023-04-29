import pandas as pd
import streamlit as st
import requests
import folium
import streamlit_folium
import json

# Probably some geo stuff still needed

# Get the predictions from the model (?)
def parse_output(json_str):
    # Parse the JSON string
    json_obj = json.loads(json_str)

    # Extract the list of integer values inside the brackets
    values = json_obj["predictions"]

    return values

def request_prediction(data):

    host = 'land_price_model_api'
    port = '8080'

    url = f'http://{host}:{port}/invocations'

    headers = {
        'Content-Type': 'application/json',
    }

    r = requests.post(url=url, headers=headers, data=data)

    return r.text

# Put features for the slider here 
def user_input_features():
    with st.sidebar.form("my_form"):
        f1 = st.slider('Area_Types_num', 1, 9, 1)
        f2 = st.slider('Area Count', 0, 50, 2)
        submitted = st.form_submit_button("Submit")

    data = {'Area_Types_num': f1,
            'Area Count': f2
            }


# TO DO:
# 1. Display a static map
# 2. Map the predicted land prices on the respective neighborhoods
# 3. Add colour to the predicted neighborhoods
# 4. Possibly bonus: hovering over neighborhoods