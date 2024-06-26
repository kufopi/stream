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
import random
import string
import os.path

student_names = [
        'Emma', 'Noah', 'Olivia', 'Olufemi', 'Idriss', 'William', 'Sophia', 'Yusuf',
        'Isabella', 'Bako', 'Amaka', 'Bayowa', 'Amara', 'Emeka', 'Kafayat',
        'Tijani', 'Aliu', 'Gbolahan', 'Chinasa', 'Hauwa'
    ]
    
departments = [
        'Computer Science', 'Mechanical Engineering', 'Physics', 'Chemistry',
        'Biology', 'Mathematics', 'History', 'English', 'Nursing', 'Economics'
    ]

# Function to get student name from identifier
def get_student_name(identifier, dataframe):
        student_row = dataframe[dataframe['Identifier'] == identifier]
        if not student_row.empty:
            return student_row['Student Name'].values[0]
        else:
            return "Identifier not found."
                

if os.path.exists('students.csv'):
    student_pop_df = pd.read_csv('students.csv')
else:
    
    
    
    # Function to generate a random alphanumeric identifier
    def generate_identifier(length):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def generate_phone_number():
        return f"+234-{random.randint(800, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    
    # Sample student names
   
    
    # Generate the DataFrame
    student_pop = pd.DataFrame({
        'Student Name': student_names,
        'Identifier': [generate_identifier(7) for _ in range(20)],
        'Department': [random.choice(departments) for _ in range(20)],
        'Phone Number': [generate_phone_number() for _ in range(20)]
    })
    student_pop.to_csv('students.csv', index=False)
    
    


student_pop_df = pd.read_csv('students.csv')
#print(student_pop_df)




#use an intor image
frontimg = Image.open('ban.png')
st.image(frontimg, use_column_width=True, caption='Google map showing part of AU campus')

st.title('Development of a Secure Contact Tracing Model for Controlling the Spread of Infectious Diseases')
st.subheader('By Oluwadamilare A. Aremu - _AUPG/22/0062_')
st.markdown('''

* **Python Libraries** : [Sklearn](https://scikit-learn.org/), [Numpy](https://numpy.org/), [Seaborn](https://seaborn.pydata.org/), [matplotlib](https://matplotlib.org/), [Pandas](https://pandas.pydata.org/), [Streamlit](https://streamlit.io/) and altair
* **Image Map Source** : [Openstreetmap](https://www.openstreetmap.org/#map=17/7.69244/4.42085)
* ** Google Map Link** : [Google Map](https://www.google.com/maps/@7.69244,4.42085,17z)
''')
st.header('Simulating Decrypted GPS Data of the Index Person and others within AU campus using [CryptPandas](https://pypi.org/project/CryptPandas/)')
st.markdown(''' * Using longitude: 4.42085 and latitude: 7.69244 as reference points as our [geo-fence](https://en.wikipedia.org/wiki/Geo-fence#:~:text=A%20geofence%20is%20a%20virtual,school%20zones%20or%20neighborhood%20boundaries) ''')


st.write('Simulated student population database')
st.dataframe(student_pop_df)

long = 4.42085
lat = 7.69244
fname= 'datatest.csv'
students =['Bose','Henry','Callistus','Yahaya','Dokubo']
filename='others.csv'
chosen = random.choice(student_pop_df['Identifier'].tolist())
version2 = student_names.copy()
remanat_df = student_pop_df[student_pop_df['Identifier']!=chosen]
st.write(f'Let us assume the student with identifier {chosen} = {get_student_name(chosen,student_pop_df)} is our person of interest POI whose body temperature exceeds normal ')
st.write(f'Remnant population database excluding {chosen}- {get_student_name(chosen,student_pop_df)}')
st.dataframe(remanat_df)
black_sheep =get_student_name(chosen,student_pop_df)
version2.remove(black_sheep)
st.write(f'The left over students: {version2}')

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
        for stdt in version2:
            for _ in range(num_rows):
                uniqueid = stdt
                rand_lat = random.random() / 100
                rand_long = random.random() / 100
                file.write(f"{uniqueid.lower()},{long + rand_long:.6f},{lat + rand_lat:.6f}\n")


st.sidebar.header('Controls')
nrows = st.sidebar.slider('Number of rows',min_value=150, max_value=500)
# Generate the  gps points
generate_random( long,lat,nrows,fname)
generate_others_coord(long,lat,nrows,filename)
#palying with merge the dataframes
mdf_others =pd.read_csv('others.csv', names= ('pple','longitude','latitude'))
mdf = pd.read_csv('datatest.csv', names= ('pple','longitude','latitude'))

#concat
appenddf = pd.concat([mdf_others,mdf],axis=0)
st.subheader("A Sample of all Student's GPS Dataframe")
st.dataframe(appenddf.sample(20))
st.write(f'Just to confirm the shape of dataframe is correct: {appenddf.shape}')

#read & display the data
df_others =pd.read_csv('others.csv', names= ('pple','longitude','latitude'))
st.subheader("A Sample of other students GPS coordinates Dataframe except that of the POI")
st.dataframe(df_others.sample(10))
#st.write(f'Just to confirm the shape of dataframe is correct: {df_others.shape}')

st.subheader("A Sample of POI's GPS Coordinates Dataframe")
df = pd.read_csv('datatest.csv', names= ('uniqueid','longitude','latitude'))
st.dataframe(df.sample(10))
#st.write(f'Just to confirm the shape of dataframe is correct: {df.shape}')

st.header('Data Manipulation')
st.markdown('Addition of columns of longitude and latitude in **radians** using np.radians')

df[['long_radians', 'lat_radians']] = (np.radians(df.loc[:,['longitude','latitude']]))
df_others[['long_radians', 'lat_radians']] = (np.radians(df_others.loc[:,['longitude','latitude']]))

st.write('A Sample of New Dataframe for the Students')
st.dataframe(df_others.sample(10))

st.write('A Sample of New Dataframe for the POI')
st.dataframe(df.sample(10))

st.markdown('''
* Next create a pairwise matrix involving both dataframes using sklearn libraries
* Multiply by 6371 to covert to kilometers to get the distance between each individual using
[Haversine Formula](https://en.wikipedia.org/wiki/Haversine_formula#:~:text=The%20haversine%20formula%20determines%20the,given%20their%20longitudes%20and%20latitudes.&text=These%20names%20follow%20from%20the,sin2(%CE%B82).)
* [Sklearn.neighbors.DistanceMetric](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html)

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
dist = st.sidebar.slider('Filter contacts based on distance (feet)', min_value=3, max_value=10)
def dist_converter(dist):
    km = float(dist)*0.000305
    return km
st.subheader(f"Filtering out people who may have had contact with the index case based on distance (ie distance < {dist}ft)")

filter_df = df_dist_km_long.loc[df_dist_km_long['Kilometres'] < dist_converter(dist)]
persons_df = filter_df.rename(columns={'pple':'Potential Contact Person'})

st.dataframe(persons_df)

st.header('Data Visualization using Pydeck Library')

Bbox= ((df.longitude.max(), df.latitude.max()),
       (df.longitude.min(), df.latitude.min()))
st.markdown("""
Obtain the boundaries of the dataframe:
```python 
Bbox= ((df.longitude.max(), df.latitude.max()),
 (df.longitude.min(), df.latitude.min()))
```
""")
st.markdown(f"""
Plotting the GPS coordinates of the Index person -_{get_student_name(chosen,student_pop_df)}_ (green dots) and other individuals on campus (red dots)
[zoom in]
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
