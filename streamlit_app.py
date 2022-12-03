
import streamlit as st
import pandas as pd
import time
import datetime
import base64
import numpy as np
from sklearn.preprocessing import PowerTransformer
import pydeck
from pydeck.types import String
import geopandas
import statsmodels.api as sm
from fuzzywuzzy import fuzz, process

### Best practices I've found so far:
    # Use an empty container first so you can blank it out if you need to later
    # Create a style tag at the top to fill with CSS
    # Use CSS to arrange your containers on the page
    # It can sometimes be more useful to use a st.markdown with "<p></p>" in it, than to just use the st.write function.

### Set Page title, add page icon
st.set_page_config(page_title="Washington, D.C. Housing Price Predictor", page_icon="", layout='wide')

### Place for CSS markup
style ="""
    <!-- changing font size of title -->
    <style>
        #residential-dc-sale-price-predictor > div > span {
            font-size: 34px;
        }
    </style>
    <!-- changing drop down to be darker and bigger -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > .streamlit-expander > li > div {
            font-size: 24px;
            font-weight: 700;
            background-color: whitesmoke;
            border-style: outset;
        }
    </style>
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > .streamlit-expander > li > .streamlit-expanderHeader > div > p {
            font-size: 24px;
            font-weight: 700;
        }
    </style>
    <!-- Move down the feature selection area -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div > div.streamlit-expanderContent > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div {
            margin-top: 15px;
        }
    </style>
    <!-- Fill in feature selection from left to right and remove margins -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div > div {
            display: flex;
            flex-direction: row;
            gap: 0px;
            margin-top: 2px;
        }
    </style>
    <!-- remove the margin on the label for the drop downs -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div > div > div > div {
            margin: 0px;
            width: 100% !important;
        }
    </style>
    <!-- hide the label on the drop downs -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div > div > div > div > label {
            height: 0px;
            margin: 0px;
            min-height: 0px;
        }
    </style>
    <!-- making sure output expands with the page -->
    <style>
        .e1tzin5v0 {
            position: inherit !important;
        }
    </style>
    <!-- changing fill direction and adding padding on top and bottom containers -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > div {
            display: flex;
            flex-direction: row;
            padding-top: 15px;
        }
    </style>
    <!-- making sure output containers dont overfill -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > div > div > div ,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > div > div > div > div ,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > div > div > div > div > div {
            width: 100% !important;
        }
    </style>
    <!-- changing first text line of each output container to bold bigger font -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > div > div > div > div:nth-child(1) > div > div > p {
            font-weight: 700;
            font-size: 20px;
        }
    </style>
    <!-- changing border of each output container -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > div > div {
            border-style: outset;
            padding: 5px;
            width: 0px;
            background-color: whitesmoke;
        }
    </style>
    <!-- centering the mid estimate -->
    <style>
        div.mid-estimate {
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: center;
            font-weight: 900;
        }
    </style>
    <!-- left and right aligning the low and high estimates -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > div > div > div > div > div > div > div > div {
            margin: 0px 0px 0px;
        }
        div.low-high-estimates {
            display: flex;
            flex-direction: row;
        }
        div.low-estimate {
            padding-left: 24px;
        }
        div.high-estimate {
            position: absolute;
            right: 24px;
        }
    </style>
    <!-- making the important features info fill in on one line each and padding -->
    <style>
        div.important-features {
            display: flex;
            flex-direction: row;
            padding-left: 12px;
            padding-right: 12px;
        }
    </style>
    <!-- making the address lookup input fill in on one line each and padding -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(1),
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(1) > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(1) > div > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(1) > div > div > div > div ,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(1) > div > div > div > div > div {
            width: 100% !important;
        }
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(1) > div > div > div > div > div > div ,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(1) > div > div > div > div > div > div > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(1) > div > div > div > div > div > div > div > div {
            display: flex;
            flex-direction: row;
            width: 100% !important;
        }
    </style>
    <!-- make all the feature descriptions span the entire width and not wrap -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div > div > div > div > div > div > div > div > div.e1tzin5v0 > div > div > label,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div > div > div > div > div > div > div > div > div > label,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div > div.streamlit-expanderContent > div > div > div > div > div > div > div > div > div > div > div > div > div > label {
            min-width: max-content;
            padding-right: 10px;
            margin-bottom: 0px;
            font-size: 16px;
            font-weight: 600;
        }
    </style>
    <!-- makeing the internal feature sub section headers bigger and bolder -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div > div > div > div > div > div > div > div > div > div > div > div > div > div > p,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div > div > div > div > div > div > div > div > div.e1tzin5v0 > div > p,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div > div > div > div > div > div > div > div > p {
            font-weight: 700;
            font-size: 20px;
        }
    </style>
    <!-- making the feature selection section fill in on one line each and padding -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2),
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div {
            width: 100% !important;
        }
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div > div > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div ,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div ,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div > div > div {
            display: flex;
            flex-direction: row;
            width: 100% !important;
        }
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div > div > div.e1tzin5v0 ,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div > div > div.e1tzin5v0 > div,
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div > div > div.e1tzin5v0 > div > div {
            display: flex;
            flex-direction: row;
            width: 100% !important;
        }
    </style>
    <!-- making the "OR" sections in the size and location sections stay fitted -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div > div > div.e1tzin5v0 > div:nth-child(2) > div > div {
            margin: 0px 0px 0px;
            align-items: center;
            justify-content: center;
            vertical-align: middle;
            display: flex;
        }
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div > div > div.e1tzin5v0 > div:nth-child(2) > div > div > p {
            margin: 0px 12px 0px;
            font-weight: 900;
            font-size: 20px;
        }
    </style>
    <!-- making the important features info fill in on one line each and padding -->
    <style>
        #root > div > div.withScreencast > div > div > div > section > div > div > div > div > div > div > ul > li > div.streamlit-expanderContent > div > div > div > div > div.epcbefy1:nth-child(2) > div > div > div > div > div > div > div > div.e1tzin5v0 > div:nth-child(2) {
            max-width: max-content;
        }
    </style>

"""
st.markdown(style, unsafe_allow_html=True)

### Header area
st.title("Home Sale Price Predictor for the Washington, D.C. Area")

### Feature selector area
features = ['Number of Half-Bathrooms','Number of Times Previously Sold','Year Built','Year Last Remodeled','Fireplaces','Land Area (Lot Sq.Ft.)','Interior Walls Material','Structure Type','Condition','Grade (Quality of Construction)']
feature_types = ['dropdown','dropdown','input','input','dropdown','input','dropdown','dropdown','dropdown','dropdown']
feature_options = [['0','1','2','3','4','5','6','7','8 or more'],['0','1','2','3','4','5','6','7','8 or more'],[],[],
                    ['0','1','2','3','4','5 or more'],[],
                    ['Hardwood','Other'],
                    ['Internal Apartment (Not an End Unit)', 'Other'],
                    ['Average','Good','Very Good','Other'],
                    ['Average','Above Average','Good Quality','Very Good','Excellent']]
feature_container = st.empty()
with feature_container.container():
    if 'openclose' not in st.session_state:
        openclose = True
        st.session_state["openclose"] = True
    else:
        openclose = st.session_state["openclose"]

    if 'lookup_message' not in st.session_state:
        lookup_message = ""
        fillin_squarefootage = ""
        fillin_zipcode = ""
        fillin_halfbath = ""
        fillin_timessold = ""
        fillin_yearbuilt = ""
        fillin_lastremodel = ""
        fillin_fireplaces = ""
        fillin_landarea = ""
        fillin_intwall = ""
        fillin_struct = ""
        fillin_cond = ""
        fillin_grade = ""
        st.session_state["lookup_message"] = lookup_message
        st.session_state["fillin_squarefootage"] = lookup_message
        st.session_state["fillin_zipcode"] = lookup_message
        st.session_state["fillin_halfbath"] = lookup_message
        st.session_state["fillin_timessold"] = lookup_message
        st.session_state["fillin_yearbuilt"] = lookup_message
        st.session_state["fillin_lastremodel"] = lookup_message
        st.session_state["fillin_fireplaces"] = lookup_message
        st.session_state["fillin_landarea"] = lookup_message
        st.session_state["fillin_intwall"] = lookup_message
        st.session_state["fillin_struct"] = lookup_message
        st.session_state["fillin_cond"] = lookup_message
        st.session_state["fillin_grade"] = lookup_message
    else:
        lookup_message = st.session_state["lookup_message"]
        fillin_squarefootage = st.session_state["fillin_squarefootage"]
        fillin_zipcode = st.session_state["fillin_zipcode"]
        fillin_halfbath = st.session_state["fillin_halfbath"]
        fillin_timessold = st.session_state["fillin_timessold"]
        fillin_yearbuilt = st.session_state["fillin_yearbuilt"]
        fillin_lastremodel = st.session_state["fillin_lastremodel"]
        fillin_fireplaces = st.session_state["fillin_fireplaces"]
        fillin_landarea = st.session_state["fillin_landarea"]
        fillin_intwall = st.session_state["fillin_intwall"]
        fillin_struct = st.session_state["fillin_struct"]
        fillin_cond = st.session_state["fillin_cond"]
        fillin_grade = st.session_state["fillin_grade"]

    feature_fillins = [fillin_halfbath,fillin_timessold,fillin_yearbuilt,fillin_lastremodel,fillin_fireplaces,fillin_landarea,fillin_intwall,fillin_struct,fillin_cond,fillin_grade]

    my_expander = st.expander("Input your Home's Features", openclose)
    featureselect_container = my_expander.empty()
    with featureselect_container.container():
        #Address lookup container
        address_container = st.empty()
        with address_container.form("address_container",True):
            addresstemp_container = st.empty()
            with addresstemp_container.container():
                st.write("Look up an existing address:")
                addressint_container = st.empty()
                with addressint_container.container():
                    input_address = st.text_input("Street Address: ", )
                    input_addresszip = st.text_input("Zip Code: ", )
                address_button = st.form_submit_button("Fill in Address Features")
                st.write(lookup_message)
        #Square Footage container
        feature_form = st.empty()
        with feature_form.form("feature_form"):
            size_container = st.empty()
            with size_container.container():
                st.write('Input Home Size: ')
                sizeint_container = st.empty()
                with sizeint_container.container():
                    sizeint2_container = st.container()
                    with sizeint2_container:
                        input_gba = st.text_input("Square Footage: ", fillin_squarefootage)
                        st.write("OR")
                        input_rooms = st.selectbox("Number of Rooms: ", ["0","1","2","3","4","5","6","7","8 or more"], key="room_select")
            #Location container
            location_container = st.empty()
            with location_container.container():
                st.write('Input Home Location: ')
                locationint_container = st.empty()
                with locationint_container.container():
                    locationint2_container = st.container()
                    with locationint2_container:
                        input_ward = st.selectbox("Ward: ", ["Ward 1","Ward 2","Ward 3","Ward 4","Ward 5","Ward 6","Ward 7","Ward 8"], key="ward_select")
                        st.write("OR")
                        input_zipcode = st.text_input("Zip Code: ", fillin_zipcode)
            #Other required features container
            other_container = st.empty()
            with other_container.container():
                st.write('Other Required Attributes: ')
                f = 0
                features_inputs = []
                while f < len(features):
                    temp_container = st.empty()
                    with temp_container.container():
                        if feature_types[f] == 'dropdown':
                            if len([i for i,n in enumerate(feature_options[f]) if str(n) == str(feature_fillins[f])]) > 0:
                                temp_input = [i for i,n in enumerate(feature_options[f]) if str(n) == str(feature_fillins[f])][0]
                            else:
                                temp_input = 0
                            features_inputs.append(st.selectbox(features[f]+":", feature_options[f], key='feat'+str(f), index=temp_input))
                        elif feature_types[f] == 'input':
                            features_inputs.append(st.text_input(features[f]+":", feature_fillins[f]))
                        if f < len(features)-1:
                            if feature_types[f+1] == 'dropdown':
                                if len([i for i,n in enumerate(feature_options[f+1]) if str(n) == str(feature_fillins[f+1])]) > 0:
                                    temp_input = [i for i,n in enumerate(feature_options[f+1]) if str(n) == str(feature_fillins[f+1])][0]
                                else:
                                    temp_input = 0
                                features_inputs.append(st.selectbox(features[f+1]+":", feature_options[f+1], key='feat'+str(f+1), index=temp_input))
                            elif feature_types[f+1] == 'input':
                                features_inputs.append(st.text_input(features[f+1]+":", feature_fillins[f+1]))
                    f = f + 2
            gobutton_container = st.empty()
            with gobutton_container.container():
                predict_button = st.form_submit_button("Predict Price!")

###Output area
file_ = open("line.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

output_container = st.empty()
with output_container.container():
    ### Top half container = Price prediction and Most important features
    top_container = st.empty()
    with top_container.container():
        ### Price prediction container
        price_container = st.empty()
        with price_container.container():
            st.write("Estimated Sale Price")
            if 'midest' not in st.session_state:
                midestimate = "0"
                st.session_state["midest"] = "0"
            else:
                midestimate = st.session_state["midest"]
            if 'lowest' not in st.session_state:
                lowestimate = "0"
                st.session_state["lowest"] = "0"
            else:
                lowestimate = st.session_state["lowest"]
            if 'highest' not in st.session_state:
                highestimate = "0"
                st.session_state["highest"] = "0"
            else:
                highestimate = st.session_state["highest"]
            st.markdown(f"""
            <div class="mid-estimate" style="font-size: 36px;">${midestimate}</div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <img src="data:image/gif;base64,{data_url}" alt="line" style="height: 100%; width: 90%; margin-left: 5%; margin-right: 5%; margin-top: -1rem; margin-bottom: -1rem;">
            """, unsafe_allow_html=True)
            ci_container = st.empty()
            with ci_container.container():
                st.markdown(f"""
                    <div class="low-high-estimates" >
                        <div class="low-estimate" style="font-size: 30px;">${lowestimate}</div>
                        <div class="high-estimate" style="font-size: 30px;">${highestimate}</div>
                    </div>
                """, unsafe_allow_html=True)

        ### Most important features container
        bestft_container = st.empty()
        with bestft_container.container():
            if 'bestft_list' not in st.session_state:
                bestft_list = """   <div class="important-features">&check;&emsp;Square Footage<div style="color: green;">&emsp;&emsp;Increase by 1 ⟶ Price increase by $100</div></div>
                                    <div class="important-features">&check;&emsp;Number of Times Sold<div style="color: red;">&emsp;&emsp;Increase by 1 ⟶ Price increase by $100</div></div>"""
                st.session_state["bestft_list"] = bestft_list
            else:
                bestft_list = st.session_state["bestft_list"]
            st.write("Most Important Features")
            st.markdown(f"""
                <div class="feature-list">
                    {bestft_list}
                </div>
            """, unsafe_allow_html=True)

    ### Bottom half container = Map and Time series
    bottom_container = st.empty()
    with bottom_container.container():
        ### Map container
        map_container = st.empty()
        with map_container.container():
            st.write("Estimated Price By Ward")
            File = "Wards.geojson"
            df = geopandas.read_file(File)
            if 'ChosenWard' not in st.session_state:
                ChosenWard = "Ward 0"
                estimate_ward = []
                st.session_state["ChosenWard"] = ChosenWard
                st.session_state["est_ward1"] = ""
                st.session_state["est_ward2"] = ""
                st.session_state["est_ward3"] = ""
                st.session_state["est_ward4"] = ""
                st.session_state["est_ward5"] = ""
                st.session_state["est_ward6"] = ""
                st.session_state["est_ward7"] = ""
                st.session_state["est_ward8"] = ""
            else:
                ChosenWard = st.session_state["ChosenWard"]

            ###### ----- ADDED CODE ----- #####

            data = [["Ward 1", 38.92552593, -77.03142317],
                    ["Ward 2", 38.89323277, -77.04330384],
                    ["Ward 3", 38.93637463, -77.07898659],
                    ["Ward 4", 38.96383619, -77.0341463],
                    ["Ward 5", 38.92543737, -76.98547552],
                    ["Ward 6", 38.88680702, -77.00277129],
                    ["Ward 7", 38.88698898, -76.94784183],
                    ["Ward 8", 38.84020937, -77.00658791]]

            WardLabels_DF = pd.DataFrame(data, columns=["Ward", "Lat", "Long"])
            WardLabels_DF["Price"] = np.nan

            ### This is where the predicted price would be assigned to each ward
            WardLabels_DF.loc[WardLabels_DF.Ward == 'Ward 1', 'Price'] = st.session_state["est_ward1"]
            WardLabels_DF.loc[WardLabels_DF.Ward == 'Ward 2', 'Price'] = st.session_state["est_ward2"]
            WardLabels_DF.loc[WardLabels_DF.Ward == 'Ward 3', 'Price'] = st.session_state["est_ward3"]
            WardLabels_DF.loc[WardLabels_DF.Ward == 'Ward 4', 'Price'] = st.session_state["est_ward4"]
            WardLabels_DF.loc[WardLabels_DF.Ward == 'Ward 5', 'Price'] = st.session_state["est_ward5"]
            WardLabels_DF.loc[WardLabels_DF.Ward == 'Ward 6', 'Price'] = st.session_state["est_ward6"]
            WardLabels_DF.loc[WardLabels_DF.Ward == 'Ward 7', 'Price'] = st.session_state["est_ward7"]
            WardLabels_DF.loc[WardLabels_DF.Ward == 'Ward 8', 'Price'] = st.session_state["est_ward8"]
            ###### ----- ADDED CODE ----- #####

            Initial_View_State = pydeck.ViewState(
              latitude=38.9072,
              longitude=-77.0369,
              zoom=10.5,
              max_zoom=16
            )

            geojson = pydeck.Layer(
                'GeoJsonLayer',
                data=df,
                opacity=0.6,
                get_fill_color= "NAME == '" + ChosenWard + "' ? [0, 0, 255] : [100, 100, 100]",
                get_line_color=[255, 255, 255],
                getLineWidth=15
            )

            ###### ----- ADDED CODE ----- #####
            layer = pydeck.Layer(
                "TextLayer",
                WardLabels_DF,
                pickable=True,
                get_position=["Long", "Lat"],
                get_text="Price",
                get_size=28,
                get_color=[255, 255, 255],
                get_angle=0,
                get_text_anchor=String("middle"),
                get_alignment_baseline=String("center")
            )
            ###### ----- ADDED CODE ----- #####

            Deck = pydeck.Deck(
                layers=[geojson, layer], ### Updated
                initial_view_state=Initial_View_State)

            st.pydeck_chart(Deck)

        ### Time series container
        time_container = st.empty()
        with time_container.container():
            TS = pd.read_csv("TS_Prediction_Full.csv")
            TS = pd.DataFrame(TS)
            TS['Locale'] = 'Full Region'
            TS_W12 = pd.read_csv("TS_Prediction_W_1_2.csv")
            TS_W12 = pd.DataFrame(TS_W12)
            TS_W12['Locale'] = 'Wards 1 & 2'
            TS_W34 = pd.read_csv("TS_Prediction_W_3_4.csv")
            TS_W34 = pd.DataFrame(TS_W34)
            TS_W34['Locale'] = 'Wards 3 & 4'
            TS_W56 = pd.read_csv("TS_Prediction_W_5_6.csv")
            TS_W56 = pd.DataFrame(TS_W56)
            TS_W56['Locale'] = 'Wards 5 & 6'
            TS_W78 = pd.read_csv("TS_Prediction_W_7_8.csv")
            TS_W78 = pd.DataFrame(TS_W78)
            TS_W78['Locale'] = 'Wards 7 & 8'

            frames = [TS, TS_W12, TS_W34, TS_W56, TS_W78]
            TS_Total = pd.concat(frames)

            #input drop down for ward or all
            if st.session_state["ChosenWard"] == "Ward 0":
                time_title = "Average Sale Price Over Time for All Properties Across All Wards"
                selected_locale = "Full Region"
                selected_date = "2022-11"
                file_ = open("1_Full Region Time Series Plot.png", "rb")
                contents = file_.read()
                time_image = base64.b64encode(contents).decode("utf-8")
                file_.close()
            elif st.session_state["ChosenWard"] == "Ward 1" or st.session_state["ChosenWard"] == "Ward 2":
                time_title = "Average Sale Price Over Time for All Properties Across Wards 1 & 2"
                selected_locale = "Wards 1 & 2"
                file_ = open("1_Wards 1_2 Time Series Plot.png", "rb")
                contents = file_.read()
                time_image = base64.b64encode(contents).decode("utf-8")
                file_.close()
            elif st.session_state["ChosenWard"] == "Ward 3" or st.session_state["ChosenWard"] == "Ward 4":
                time_title = "Average Sale Price Over Time for All Properties Across Wards 3 & 4"
                selected_locale = "Wards 3 & 4"
                file_ = open("1_Wards 3_4 Time Series Plot.png", "rb")
                contents = file_.read()
                time_image = base64.b64encode(contents).decode("utf-8")
                file_.close()
            elif st.session_state["ChosenWard"] == "Ward 5" or st.session_state["ChosenWard"] == "Ward 6":
                time_title = "Average Sale Price Over Time for All Properties Across Wards 5 & 6"
                selected_locale = "Wards 5 & 6"
                file_ = open("1_Wards 5_6 Time Series Plot.png", "rb")
                contents = file_.read()
                time_image = base64.b64encode(contents).decode("utf-8")
                file_.close()
            elif st.session_state["ChosenWard"] == "Ward 7" or st.session_state["ChosenWard"] == "Ward 8":
                time_title = "Average Sale Price Over Time for All Properties Across Wards 7 & 8"
                selected_locale = "Wards 7 & 8"
                file_ = open("1_Wards 7_8 Time Series Plot.png", "rb")
                contents = file_.read()
                time_image = base64.b64encode(contents).decode("utf-8")
                file_.close()


            st.write(time_title)
            selected_date = st.selectbox('Pick a future date for prediction', TS_Total.Date.drop_duplicates(), 4)
            selected_date_price = TS_Total.loc[(TS_Total["Locale"] == selected_locale) & (TS_Total["Date"] == selected_date), 'Average Home Sale Price'].iloc[0]
            selected_date_price = "{:,}".format(abs(int(round(selected_date_price, 0))))
            st.write(f'Average Home Price: ${selected_date_price}')
            st.markdown(f"""
            <img src="data:image/gif;base64,{time_image}" alt="line" style="height: 70%; width: 70%; margin-bottom: 5%; margin-left: 15%; margin-right: 15%;">
            """, unsafe_allow_html=True)



if address_button:
    add_features = pd.read_csv("ResidentialSalesWithAddress_Small.csv")
    if input_addresszip == "":
        lookup_message = "Error: Please input zip code."
        st.session_state["lookup_message"] = lookup_message
    elif len(add_features[add_features["ZIPCODE"] == int(input_addresszip)]) > 0:
        #print(process.extractOne(input_address, add_features[add_features["ZIPCODE"] == int(input_addresszip)]["FULLADDRESS"].tolist()))
        found_address = str(process.extractOne(input_address, add_features[add_features["ZIPCODE"] == int(input_addresszip)]["FULLADDRESS"])[0])
        lookup_message = found_address
        st.session_state["lookup_message"] = "Found:  " + lookup_message + ", WASHINGTON, DC " + str(input_addresszip)
        st.session_state["fillin_squarefootage"] = add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["GBA"].tolist()[0]
        st.session_state["fillin_zipcode"] = input_addresszip
        st.session_state["fillin_halfbath"] = int(add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["HF_BATHRM"].tolist()[0])
        st.session_state["fillin_timessold"] = int(add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["SALE_NUM"].tolist()[0])-1
        st.session_state["fillin_yearbuilt"] = int(add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["AYB"].tolist()[0])
        st.session_state["fillin_lastremodel"] = str(add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["YR_RMDL"].tolist()[0])
        if st.session_state["fillin_lastremodel"] == "nan":
            st.session_state["fillin_lastremodel"] = ""
        else:
            st.session_state["fillin_lastremodel"] = int(add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["YR_RMDL"].tolist()[0])
        st.session_state["fillin_fireplaces"] = int(add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["FIREPLACES"].tolist()[0])
        st.session_state["fillin_landarea"] = add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["LANDAREA"].tolist()[0]
        st.session_state["fillin_intwall"] = int(add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["INTWALL"].tolist()[0])
        if st.session_state["fillin_intwall"] == 6:
            st.session_state["fillin_intwall"] = "Hardwood"
        else:
            st.session_state["fillin_intwall"] = "Other"
        st.session_state["fillin_struct"] = int(add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["STRUCT"].tolist()[0])
        if st.session_state["fillin_struct"] == 7:
            st.session_state["fillin_struct"] = "Internal Apartment (Not an End Unit)"
        else:
            st.session_state["fillin_struct"] = "Other"
        st.session_state["fillin_cond"] = int(add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["CNDTN"].tolist()[0])
        if st.session_state["fillin_cond"] == 3:
            st.session_state["fillin_cond"] = "Average"
        elif st.session_state["fillin_cond"] == 4:
            st.session_state["fillin_cond"] = "Good"
        elif st.session_state["fillin_cond"] == 4:
            st.session_state["fillin_cond"] = "Very Good"
        else:
            st.session_state["fillin_cond"] = "Other"
        st.session_state["fillin_grade"] = int(add_features[(add_features["ZIPCODE"] == int(input_addresszip)) & (add_features["FULLADDRESS"] == found_address)]["GRADE"].tolist()[0])
        if st.session_state["fillin_grade"] == 3:
            st.session_state["fillin_grade"] = "Average"
        elif st.session_state["fillin_grade"] == 4:
            st.session_state["fillin_grade"] = "Above Average"
        elif st.session_state["fillin_grade"] == 4:
            st.session_state["fillin_grade"] = "Good Quality"
        elif st.session_state["fillin_grade"] == 4:
            st.session_state["fillin_grade"] = "Very Good"
        else:
            st.session_state["fillin_grade"] = "Excellent"
    else:
        lookup_message = "Error: No matching address found."
        st.session_state["lookup_message"] = lookup_message

    st.experimental_rerun()


if predict_button:

    if features_inputs[0] == "8 or more":
        hf_bathrm = 8
    else:
        hf_bathrm = int(features_inputs[0])
    if features_inputs[1] == "8 or more":
        sale_num = 8+1
    else:
        sale_num = int(features_inputs[1])+1
    if features_inputs[4] == "5 or more":
        fireplaces = 8
    else:
        fireplaces = int(features_inputs[4])
    if features_inputs[7] == "Internal Apartment (Not an End Unit)":
        struct7 = 1
    else:
        struct7 = 0
    grade3 = 0
    grade4 = 0
    grade5 = 0
    grade6 = 0
    grade999 = 0
    grade_color = "green"
    grade_change = "increase"
    grade_change_amt = 0.5334
    if features_inputs[9] == "Average":
        grade3 = 1
        grade_color = "green"
        grade_change = "increase"
        grade_change_amt = 0.8625
    elif features_inputs[9] == "Above Average":
        grade4 = 1
        grade_color = "green"
        grade_change = "increase"
        grade_change_amt = 0.6982
    elif features_inputs[9] == "Good Quality":
        grade5 = 1
        grade_color = "green"
        grade_change = "increase"
        grade_change_amt = 0.5334
    elif features_inputs[9] == "Very Good":
        grade6 = 1
        grade_color = "green"
        grade_change = "increase"
        grade_change_amt = 0.3309
    elif features_inputs[9] == "Excellent":
        grade999 = 1
        grade_color = "red"
        grade_change = "decrease"
        grade_change_amt = -0.3309
    cndtn3 = 0
    cndtn4 = 0
    cndtn5 = 0
    cndtn999 = 0
    cndtn_color = "green"
    cndtn_change = "increase"
    cndtn_change_amt = 0.1853
    if features_inputs[8] == "Average":
        cndtn3 = 1
        cndtn_color = "green"
        cndtn_change = "increase"
        cndtn_change_amt = 0.8078
    elif features_inputs[8] == "Good":
        cndtn4 = 1
        cndtn_color = "green"
        cndtn_change = "increase"
        cndtn_change_amt = 0.4902
    elif features_inputs[8] == "Very Good":
        cndtn5 = 1
        cndtn_color = "green"
        cndtn_change = "increase"
        cndtn_change_amt = 0.1853
    elif features_inputs[8] == "Other":
        cndtn999 = 1
        cndtn_color = "red"
        cndtn_change = "decrease"
        cndtn_change_amt = -0.1853
    if features_inputs[6] == "Hardwood":
        intwall6 = 1
    else:
        intwall6 = 0
    ward1 = 0
    ward2 = 0
    ward3 = 0
    ward4 = 0
    ward5 = 0
    ward6 = 0
    ward7 = 0
    ward8 = 0
    ward_color = "green"
    ward_change = "increase"
    ward_change_amt = 0.3048
    if input_zipcode != "":
        zipdict = pd.read_csv("ZipcodeWardLookup.csv")
        new_ward = zipdict[zipdict["Zipcode"] == int(input_zipcode)]["Ward"].tolist()[0].strip()
    else:
        new_ward = input_ward
    if new_ward == "Ward 1":
        ward1 = 1
    elif new_ward == "Ward 2":
        ward2 = 1
        ward_color = "red"
        ward_change = "decrease"
        ward_change_amt = -0.0084
    elif new_ward == "Ward 3":
        ward3 = 1
        ward_color = "green"
        ward_change = "increase"
        ward_change_amt = 0.0084
    elif new_ward == "Ward 4":
        ward4 = 1
    elif new_ward == "Ward 5":
        ward5 = 1
        ward_color = "green"
        ward_change = "increase"
        ward_change_amt = 0.5603
    elif new_ward == "Ward 6":
        ward6 = 1
    elif new_ward == "Ward 7":
        ward7 = 1
        ward_color = "green"
        ward_change = "increase"
        ward_change_amt = 0.7745
    elif new_ward == "Ward 8":
        ward8 = 1
        ward_color = "green"
        ward_change = "increase"
        ward_change_amt = 1.0052

    landarea = int(features_inputs[5])
    built = 2022-int(features_inputs[2])
    if features_inputs[3] == "":
        last_remodel = 2022-int(features_inputs[2])
    else:
        last_remodel = 2022-int(features_inputs[3])

    if input_gba == "":
        if input_rooms == "8 or more":
            numrooms = 8
        else:
            numrooms = int(input_rooms)
        y_gba_train_log = pd.read_csv("y_gba_train_log.csv")
        X_rooms_train = pd.read_csv("X_rooms_train.csv")
        X_rooms_train = np.array(X_rooms_train).reshape(-1, 1)
        X_rooms_OLS_train_log = sm.add_constant(X_rooms_train)
        model_OLS_ygba_log = sm.OLS(y_gba_train_log, X_rooms_OLS_train_log)
        result_OLS_ygba_log = model_OLS_ygba_log.fit()
        gba = int(np.exp(result_OLS_ygba_log.predict(np.array([1,numrooms]).reshape(1, -1))))
    else:
        gba = int(input_gba)



    y_train = pd.read_csv("y_train.csv")
    #print(y_train.head())
    X_all_train = pd.read_csv("X_all_train.csv")
    #print(y_train.head())
    power = PowerTransformer(method='yeo-johnson', standardize=True)
    features = ["STORIES", "KITCHENS", "ROOMS", "BEDRM", "NUM_UNITS", "AYB_AGE", "YR_RMDL_AGE", "EYB_AGE", "FIREPLACES", "LANDAREA", "GBA"]
    X_all_train[features] = power.fit_transform(X_all_train[features])

    collected = [0, 0, 0, 0, 0, built, last_remodel, 0, fireplaces, landarea, gba]
    collected = power.transform(np.array(collected).reshape(1, -1))
    collectedgba = [0, 0, 0, 0, 0, built, last_remodel, 0, fireplaces, landarea, gba+100]
    collectedgba = power.transform(np.array(collectedgba).reshape(1, -1))

    X_features2 = ['HF_BATHRM', 'SALE_NUM', 'GBA', 'FIREPLACES', 'LANDAREA', 'AYB_AGE', 'YR_RMDL_AGE', 'STRUCT_7.0', 'GRADE_3.0', 'GRADE_4.0', 'GRADE_5.0', 'GRADE_6.0', 'CNDTN_3.0', 'CNDTN_4.0', 'CNDTN_5.0', 'INTWALL_6.0', 'WARD_2', 'WARD_3', 'WARD_5', 'WARD_7', 'WARD_8']
    X_selected2_train = X_all_train[X_features2]
    y_train_trans = power.fit_transform(np.array(y_train).reshape(-1, 1))
    X_selected2_OLS_train = sm.add_constant(X_selected2_train)
    model_OLS_ytrans2 = sm.OLS(y_train_trans, X_selected2_OLS_train)
    result_OLS_ytrans2 = model_OLS_ytrans2.fit()


    #midestimate = -0.0125 + 0.0515*hf_bathrm + 0.1536*sale_num + 0.2743*collected[0][10] + 0.0658*collected[0][8] + 0.0149*collected[0][9] + 0.0858*collected[0][5] + 0.0619*collected[0][6] + 0.0846*struct7 - 0.3292*grade3 - 0.1649*grade4 + 0.2024*grade6 + 0.5334*grade999 - 0.6224*cndtn3 - 0.3049*cndtn4 + 0.1853*cndtn999 + 0.0669*intwall6 + 0.3048*ward2 + 0.2964*ward3 - 0.2555*ward5 - 0.4697*ward7 - 0.7004*ward8
    midestimate = result_OLS_ytrans2.predict(np.array([1,hf_bathrm,sale_num,collected[0][10],collected[0][8],collected[0][9],collected[0][5],collected[0][6],struct7,grade3,grade4,grade5,grade6,cndtn3,cndtn4,cndtn5,intwall6,ward2,ward3,ward5,ward7,ward8]).reshape(1, -1))

    gbaestimate = result_OLS_ytrans2.predict(np.array([1,hf_bathrm,sale_num,collectedgba[0][10],collected[0][8],collected[0][9],collected[0][5],collected[0][6],struct7,grade3,grade4,grade5,grade6,cndtn3,cndtn4,cndtn5,intwall6,ward2,ward3,ward5,ward7,ward8]).reshape(1, -1))

    gba_change_amt = "{:,}".format(abs(int(round((power.inverse_transform(np.array(gbaestimate).reshape(1, -1))[0][0] - power.inverse_transform(np.array(midestimate).reshape(1, -1))[0][0]), 0))))
    numsales_change_amt = "{:,}".format(abs(int(round(power.inverse_transform(np.array(0.1536 + midestimate).reshape(1, -1))[0][0] - power.inverse_transform(np.array(midestimate).reshape(1, -1))[0][0], 0))))
    cndtn_change_amt = "{:,}".format(abs(int(round(power.inverse_transform(np.array(cndtn_change_amt + midestimate).reshape(1, -1))[0][0] - power.inverse_transform(np.array(midestimate).reshape(1, -1))[0][0], 0))))
    grade_change_amt = "{:,}".format(abs(int(round(power.inverse_transform(np.array(grade_change_amt + midestimate).reshape(1, -1))[0][0] - power.inverse_transform(np.array(midestimate).reshape(1, -1))[0][0], 0))))
    ward_change_amt = "{:,}".format(abs(int(round(power.inverse_transform(np.array(ward_change_amt + midestimate).reshape(1, -1))[0][0] - power.inverse_transform(np.array(midestimate).reshape(1, -1))[0][0], 0))))
    bestft_list = f"""<div class="important-features">&check;&emsp;Square Footage<div style="color: green;">&emsp;&emsp;Increase by 100 ⟶ Price increase by ${gba_change_amt}</div></div>
                        <div class="important-features">&check;&emsp;Number of Times Sold<div style="color: green;">&emsp;&emsp;Increase by 1 ⟶ Price increase by ${numsales_change_amt}</div></div>
                        <div class="important-features">&check;&emsp;Property Condition<div style="color: {cndtn_color};">&emsp;&emsp;Change ⟶ Price {cndtn_change} by ${cndtn_change_amt}</div></div>
                        <div class="important-features">&check;&emsp;Construction Material Grade<div style="color: {grade_color};">&emsp;&emsp;Change ⟶ Price {grade_change} by ${grade_change_amt}</div></div>
                        <div class="important-features">&check;&emsp;Location<div style="color: {ward_color};">&emsp;&emsp;Change ⟶ Price {ward_change} by ${ward_change_amt}</div></div>
                    """

    midestimate = power.inverse_transform(np.array(midestimate).reshape(1, -1))[0][0]
    midestimate = "{:,}".format(abs(int(round(midestimate, 0))))
    lowestimate = result_OLS_ytrans2.get_prediction(np.array([1,hf_bathrm,sale_num,collected[0][10],collected[0][8],collected[0][9],collected[0][5],collected[0][6],struct7,grade3,grade4,grade5,grade6,cndtn3,cndtn4,cndtn5,intwall6,ward2,ward3,ward5,ward7,ward8]).reshape(1, -1)).conf_int()[:,0]
    lowestimate = power.inverse_transform(np.array(lowestimate).reshape(1, -1))[0][0]
    lowestimate = "{:,}".format(abs(int(round(lowestimate, 0))))
    highestimate = result_OLS_ytrans2.get_prediction(np.array([1,hf_bathrm,sale_num,collected[0][10],collected[0][8],collected[0][9],collected[0][5],collected[0][6],struct7,grade3,grade4,grade5,grade6,cndtn3,cndtn4,cndtn5,intwall6,ward2,ward3,ward5,ward7,ward8]).reshape(1, -1)).conf_int()[:,1]
    highestimate = power.inverse_transform(np.array(highestimate).reshape(1, -1))[0][0]
    highestimate = "{:,}".format(abs(int(round(highestimate, 0))))

    estimate_ward = []
    tempward = 0
    for w in range(1,9):
        w1=0
        w2=0
        w3=0
        w4=0
        w5=0
        w6=0
        w7=0
        w8=0
        if w == 1:
            w1=1
        elif w == 2:
            w2=1
        elif w == 3:
            w3=1
        elif w == 4:
            w4=1
        elif w == 5:
            w5=1
        elif w == 6:
            w6=1
        elif w == 7:
            w7=1
        elif w == 8:
            w8=1
        tempward = result_OLS_ytrans2.predict(np.array([1,hf_bathrm,sale_num,collected[0][10],collected[0][8],collected[0][9],collected[0][5],collected[0][6],struct7,grade3,grade4,grade5,grade6,cndtn3,cndtn4,cndtn5,intwall6,w2,w3,w5,w7,w8]).reshape(1, -1))
        tempward = power.inverse_transform(np.array(tempward).reshape(1, -1))[0][0]
        estimate_ward.append("{:,}".format(int(round(tempward, 0))))

    openclose = False

    st.session_state["midest"] = midestimate
    st.session_state["lowest"] = lowestimate
    st.session_state["highest"] = highestimate
    st.session_state["est_ward1"] = "$" + estimate_ward[0]
    st.session_state["est_ward2"] = "$" + estimate_ward[1]
    st.session_state["est_ward3"] = "$" + estimate_ward[2]
    st.session_state["est_ward4"] = "$" + estimate_ward[3]
    st.session_state["est_ward5"] = "$" + estimate_ward[4]
    st.session_state["est_ward6"] = "$" + estimate_ward[5]
    st.session_state["est_ward7"] = "$" + estimate_ward[6]
    st.session_state["est_ward8"] = "$" + estimate_ward[7]
    st.session_state["bestft_list"] = bestft_list
    st.session_state["ChosenWard"] = new_ward
    st.session_state["openclose"] = openclose
    st.experimental_rerun()
