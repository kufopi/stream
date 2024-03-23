import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image
import random
import sklearn.neighbors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk

#use an intor image
frontimg = Image.open('ban.png')
st.image(frontimg, use_column_width=True, caption='Google map showing part of AU campus')

st.title('Dare Aremu Project Code')
st.markdown('''
This web app is bla bla bla sheep
* **Python Libraries** : sklearn, numpy,seaborn, matplotlib, pandas, streamlit and altair
* **Image Map Source** : [Openstreetmap](https://www.openstreetmap.org/#map=17/7.69244/4.42085)
* ** Google Map Link** : [Google Map](https://www.google.com/maps/@7.69244,4.42085,17z)
''')
st.header('Simulating GPS Data of the Index Person and others within AU campus')
st.write('Using longitude: 4.42085 and latitude: 7.69244 as reference points')

long = 4.42085
lat = 7.69244
fname= 'datatest.csv'
students =['Bose','Henry','Callistus','Yahaya','Dokubo']
filename='others.csv'

def generate_random(long,lat,num_rows,filna):
    with open(filna, 'w') as file:
        for _ in range(num_rows):
            #uniqueid = '%012x' % random.randrange(16**12) # 12 char random string
            uniqueid = 'POI'
            rand_lat = random.random()/100
            rand_long = random.random()/100
            file.write(f"{uniqueid.lower()},{long+rand_long:.6f},{lat+rand_lat:.6f}\n")


def generate_others_coord(long, lat, num_rows, filename):
    with open(filename, 'w') as file:
        for stdt in students:
            for _ in range(num_rows):
                uniqueid = stdt
                rand_lat = random.random() / 100
                rand_long = random.random() / 100
                file.write(f"{uniqueid.lower()},{long + rand_long:.6f},{lat + rand_lat:.6f}\n")


st.sidebar.header('Controls')
nrows = st.sidebar.slider('Number of rows',min_value=100, max_value=500)
# Generate the  gps points
generate_random( long,lat,nrows,fname)
generate_others_coord(long,lat,nrows,filename)
#palying with merge the dataframes
mdf_others =pd.read_csv('others.csv', names= ('pple','longitude','latitude'))
mdf = pd.read_csv('datatest.csv', names= ('pple','longitude','latitude'))

#concat
appenddf = pd.concat([mdf_others,mdf],axis=0)
st.subheader("Sample of appended People's GPS Dataframe")
st.dataframe(appenddf.sample(20))
st.write(f'Just to confirm the shape of dataframe is correct: {appenddf.shape}')

#read & display the data
df_others =pd.read_csv('others.csv', names= ('pple','longitude','latitude'))
st.subheader("Sample of People's GPS Dataframe")
st.dataframe(df_others.sample(9))
st.write(f'Just to confirm the shape of dataframe is correct: {df_others.shape}')

st.subheader("Sample of Index Person's Dataframe")
df = pd.read_csv('datatest.csv', names= ('uniqueid','longitude','latitude'))
st.dataframe(df.sample(5))
st.write(f'Just to confirm the shape of dataframe is correct: {df.shape}')

st.header('Data Manipulation')
st.markdown('Additon of columns of longitude and latitude in **radians** using np.radians')

df[['long_radians', 'lat_radians']] = (np.radians(df.loc[:,['longitude','latitude']]))
df_others[['long_radians', 'lat_radians']] = (np.radians(df_others.loc[:,['longitude','latitude']]))

st.write('Sample of New Dataframe for df_others')
st.dataframe(df_others.sample(5))

st.write('Sample of New Dataframe for df')
st.dataframe(df.sample(5))

st.markdown('''
* Next create a pairwise matrix involving both dataframes
* Multiply by 6371 to covert to kilometers to get the distance between each individual using
[Haversine Formula](https://en.wikipedia.org/wiki/Haversine_formula#:~:text=The%20haversine%20formula%20determines%20the,given%20their%20longitudes%20and%20latitudes.&text=These%20names%20follow%20from%20the,sin2(%CE%B82).)

''')

dist = sklearn.neighbors.DistanceMetric.get_metric('haversine')
distance_matrix = (dist.pairwise(df[['long_radians','lat_radians']],
                               df_others[['long_radians','lat_radians']]) * 6371)

df_distance_matrix = ( pd.DataFrame(distance_matrix, index= df['uniqueid'], columns= df_others['pple']))

st.subheader('Shape of the pairwise matrix')
st.write(df_distance_matrix.shape)

# Unpivot the dataframe from wide format to long format
df_dist_km_long = (pd.melt(df_distance_matrix.reset_index(),id_vars='uniqueid'))
df_dist_km_long = df_dist_km_long.rename(columns={'value':'Kilometres'})

st.subheader('Sample of the manipulated Data')
st.dataframe(df_dist_km_long.sample(5))

# Filtering out people of interest based on distance
dist = st.sidebar.slider('Filter contacts based on distance (feet)', min_value=1, max_value=10)
def dist_converter(dist):
    km = float(dist)*0.000305
    return km
st.subheader(f"Filtering out people who may have had contact with the index case based on distance (ie distance < {dist}ft)")

filter_df = df_dist_km_long.loc[df_dist_km_long['Kilometres'] < dist_converter(dist)]
persons_df = filter_df.rename(columns={'pple':'Person of Interest'})

st.dataframe(persons_df)

st.header('Data Visualization')

Bbox= ((df.longitude.max(), df.latitude.max()),
       (df.longitude.min(), df.latitude.min()))
st.markdown("""
Obtain the boundaries of the dataframe:
```python 
Bbox= ((df.longitude.max(), df.latitude.max()),
 (df.longitude.min(), df.latitude.min()))
```
Plotting the GPS coordinates of the Index person (green dots) and other individuals on campus (red dots)
""")

# Mapping the movements

fig,ax = plt.subplots(figsize=(20,16))
Bbox= (df.longitude.min(), df.longitude.max(),
       df.latitude.min(), df.latitude.max())
ax.scatter(df.longitude, df.latitude, zorder=1, alpha=0.7, c='g', s=10)
ax.scatter(df_others.longitude, df_others.latitude, zorder=1, alpha=0.2, c='r', s=10,marker='x')
ax.set_title('Movement of Index person(green) and others(red)')
ax.set_xlim(Bbox[0],Bbox[1])
ax.set_ylim(Bbox[2],Bbox[3])

#ax.imshow(map, zorder=0, extent=Bbox, aspect='equal')
MAPBOX_API_KEY = 'pk.eyJ1Ijoia3Vmb3BvIiwiYSI6ImNrc3ZoZnZ2djFwdzUzMm9kc2tpMXF3NnoifQ.htE3W1FN0-rj9iqKUBFRdg'
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    api_keys={'mapbox':MAPBOX_API_KEY},
    initial_view_state=pdk.ViewState(
        latitude=lat,
        longitude=long,
        zoom=13,
        pitch=50

    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[longitude,latitude]',
            get_color='[100,130,50]',
            get_radius=4,
        ),

        pdk.Layer(
            'ScatterplotLayer',
            data=df_others,
            get_position='[longitude,latitude]',
            get_color='[200,70,50]',
            get_radius=2,
            pickable=True,
            onClick=True,
        ),

        pdk.Layer(
            type='TextLayer',
            data=df_others,
            get_position='[longitude,latitude]',

            pickable=False,
        ),

    ],
    tooltip={
        "html":"<b>Person</b> {pple}</br>",
        "style":{
            "backgroundColor": "steelblue",
            "color":"black",
}
    },
))

