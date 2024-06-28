import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import random
import matplotlib.pyplot as plt
from sklearn.metrics import DistanceMetric
from streamlit_gsheets import GSheetsConnection
from sklearn.metrics.pairwise import haversine_distances

st.image('ban.png')
url= "https://docs.google.com/spreadsheets/d/1tkP-WCCoGMcUZky80OKHpDcEkwuMcC8CRRyAthBUgRY/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)

# Function to get student name from identifier
def get_student_name(identifier, dataframe):
        student_row = dataframe[dataframe['Matric'] == identifier]
        if not student_row.empty:
            return student_row['Student_name'].values[0]
        else:
            return "Identifier not found."



# data.to_csv('students.csv', index=False)
# data = pd.read_csv('students.csv')

st.title(' A Secure Contact Tracing Model for Controlling the Spread of Infectious Diseases')
st.subheader('By Oluwadamilare A. Aremu - _AUPG/22/0062_')
st.markdown('''

* **Python Libraries** : [Sklearn](https://scikit-learn.org/), [Numpy](https://numpy.org/), [Seaborn](https://seaborn.pydata.org/), [matplotlib](https://matplotlib.org/), [Pandas](https://pandas.pydata.org/), [Pydeck](https://deckgl.readthedocs.io/en/latest/), [Streamlit](https://streamlit.io/) and [Cryptpandas](https://medium.com/@lucamingarelli/encrypted-pandas-dataframes-for-secure-storage-and-sharing-in-python-a714f441d7fa)
* **Image Map Source** : [Openstreetmap](https://www.openstreetmap.org/#map=17/7.69244/4.42085)
* ** Google Map Link** : [Google Map](https://www.google.com/maps/@7.69244,4.42085,17z)
''')
st.subheader('1. Simulating Student Data within AU Campus ')
st.warning(''' * Using longitude: 4.42085 and latitude: 7.69244 {representing GPS coordinates of Faculty of Science} as reference points ''')
st.write(" :point_left: Adjust/Select Parameters")

st.subheader('1.1 Simulated Student Population Database')
st.caption('Click to see fullview/sort/download')
data = conn.read(spreadsheet=url)

st.dataframe(data)

long = 4.42085
lat = 7.69244
fname= 'datatest.csv'
filename='others.csv'


st.sidebar.header('Controls')
affected = st.sidebar.selectbox("Select an individual",
                     data['Matric'])
pix = data.loc[data['Matric']==affected,'Picture'].iloc[0]
st.sidebar.header(f'Image: {pix}')
nrows = st.sidebar.slider('Number of rows',min_value=150, max_value=500)
dista = st.sidebar.slider('Filter contacts based on distance (feet)', min_value=3, max_value=10)


chosen = affected #random.choice(data['Matric'].tolist())
version2 = data['Student_name'].tolist()
remanat_df = data[data['Matric']!=chosen]



def program_run():



    st.success(f'The chosen student with Matric {chosen} = {get_student_name(chosen,data)} is our person of interest POI whose body temperature exceeds normal ')
    st.info(f'1.2 Remnant population database excluding {chosen}- {get_student_name(chosen,data)}',icon="🚨")
    st.dataframe(remanat_df)


    def generate_random(long,lat,num_rows,filna):
        with open(filna, 'w') as file:
            for _ in range(num_rows):
                #uniqueid = '%012x' % random.randrange(16**12) # 12 char random string
                uniqueid = get_student_name(chosen,data) #'POI'
                rand_lat = random.random()/100
                rand_long = random.random()/100
                file.write(f"{uniqueid.lower()},{long+rand_long:.6f},{lat+rand_lat:.6f}\n")


    def generate_others_coord(long, lat, num_rows, filename):
        with open(filename, 'w') as file:
            for stdt in remanat_df['Student_name'].tolist():
                for _ in range(num_rows):
                    uniqueid = stdt
                    rand_lat = random.random() / 100
                    rand_long = random.random() / 100
                    file.write(f"{uniqueid.lower()},{long + rand_long:.6f},{lat + rand_lat:.6f}\n")




    # Generate the  gps points
    generate_random( long,lat,nrows,fname)
    generate_others_coord(long,lat,nrows,filename)

    #palying with merge the dataframes
    mdf_others =pd.read_csv('others.csv', names= ('pple','longitude','latitude'))
    mdf = pd.read_csv('datatest.csv', names= ('pple','longitude','latitude'))

    #concat
    appenddf = pd.concat([mdf_others,mdf],axis=0)
    st.subheader("1.3 A Snippet of  Student's GPS Coordinates Data")
    st.dataframe(appenddf.sample(20))
    st.write(f'Just to confirm the shape of dataframe is correct: {appenddf.shape}')

    #read & display the data
    df_others =pd.read_csv('others.csv', names= ('pple','longitude','latitude'))
    st.header('2. Separating the data')
    st.subheader(f"2.1 A Snippet of other students GPS coordinates Data **except** that of the {get_student_name(chosen,data)}")
    st.dataframe(df_others.sample(10))
    #st.write(f'Just to confirm the shape of dataframe is correct: {df_others.shape}')

    st.subheader(f"2.2 A Sample of {get_student_name(chosen,data)}'s GPS Coordinates Dataframe")
    df = pd.read_csv('datatest.csv', names= ('uniqueid','longitude','latitude'))
    st.dataframe(df.sample(10))

    st.subheader('3. Data Manipulation')
    st.markdown('Addition of columns of longitude and latitude in **radians** using np.radians')
    st.code('''
        df[['long_radians', 'lat_radians']] = (np.radians(df.loc[:,['longitude','latitude']]))
        df_others[['long_radians', 'lat_radians']] = (np.radians(df_others.loc[:,['longitude','latitude']]))
    ''')

    df[['long_radians', 'lat_radians']] = (np.radians(df.loc[:,['longitude','latitude']]))
    df_others[['long_radians', 'lat_radians']] = (np.radians(df_others.loc[:,['longitude','latitude']]))

    st.write('3.1.  A Snippet of GPS Data for the Students after Conversion into Radians')
    st.dataframe(df_others.sample(20))


    st.write(f'3.2 A Snippeet of GPS Data for the {get_student_name(chosen,data)}')
    st.dataframe(df.sample(20))

    st.warning('''
    * Next create a pairwise matrix involving both dataframes using sklearn libraries
    * Multiply by 6371 to covert to kilometers to get the distance between each individual using
    [Haversine Formula](https://en.wikipedia.org/wiki/Haversine_formula#:~:text=The%20haversine%20formula%20determines%20the,given%20their%20longitudes%20and%20latitudes.&text=These%20names%20follow%20from%20the,sin2(%CE%B82).)
    * [Sklearn.neighbors.DistanceMetric](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html)
    
    ''')
    st.latex(r'''
    [ haversine(\theta) = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1) \cdot \cos(\phi_2) \cdot \sin^2\left(\frac{\Delta\lambda}{2}\right) ]
    ''')

    # distances= haversine_distances(df[['long_radians','lat_radians']], df_others[['long_radians','lat_radians']]) * 6371  # Earth radius in kilometers
    # # st.write(distances)
    # distance_df = (pd.DataFrame(distances, index=df['uniqueid'],columns=df_others['pple']))
    # # distance_df.index = df['uniqueid']
    # st.dataframe(distance_df.sample(5))
    # st.write(distance_df.shape)
    #
    # # df_dist_km_long = (pd.melt(distance_df.reset_index(),id_vars='uniqueid'))
    # # df_dist_km_long = df_dist_km_long.rename(columns={'value':'Kilometres'})
    # # st.dataframe(df_dist_km_long.sample(5))

    st.code("""
        dist = DistanceMetric.get_metric('haversine')
    
        distance_matrix = (dist.pairwise(df[['long_radians','lat_radians']],
                                   df_others[['long_radians','lat_radians']]) * 6371)

        df_distance_matrix = ( pd.DataFrame(distance_matrix, index= df['uniqueid'], columns= df_others['pple']))
    """)
    dist = DistanceMetric.get_metric('haversine')

    distance_matrix = (dist.pairwise(df[['long_radians','lat_radians']],
                                   df_others[['long_radians','lat_radians']]) * 6371)

    df_distance_matrix = ( pd.DataFrame(distance_matrix, index= df['uniqueid'], columns= df_others['pple']))

    st.write('3.3.  Shape of the pairwise matrix')
    st.write(df_distance_matrix.shape)

    # Unpivot the dataframe from wide format to long format
    st.info('3.4  Unpivot the dataframe from wide format to long format')
    st.code("""
        df_dist_km_long = (pd.melt(df_distance_matrix.reset_index(),id_vars='uniqueid'))
        df_dist_km_long = df_dist_km_long.rename(columns={'value':'Kilometres'})
    
    """)
    df_dist_km_long = (pd.melt(df_distance_matrix.reset_index(),id_vars='uniqueid'))
    df_dist_km_long = df_dist_km_long.rename(columns={'value':'Kilometres'})

    st.subheader('3.5. Sample of the manipulated Data')
    st.dataframe(df_dist_km_long.sample(5))
    #
    # Filtering out people of interest based on distance
    # dist = st.sidebar.slider('Filter contacts based on distance (feet)', min_value=3, max_value=10)


    def dist_converter(dista):
        km = float(dista)*0.000305
        return km
    st.subheader(f"3.6. Filtering out people who may have had contact with the index case based on distance (ie distance < {dista}ft)")
    st.code("""
        def dist_converter(dist):
            km = float(dist)*0.000305
            return km
        
        filter_df = df_dist_km_long.loc[df_dist_km_long['Kilometres'] < dist_converter(dist)]
        persons_df = filter_df.rename(columns={'pple':'Potential Contact Person'})
    """)

    filter_df = df_dist_km_long.loc[df_dist_km_long['Kilometres'] < dist_converter(dista)]
    persons_df = filter_df.rename(columns={'pple':'Student_name'})
    ppdf = persons_df.copy()
    ppdf.reset_index(drop=True, inplace=True)
    ppdf['Student_name'] =ppdf['Student_name'].apply(str.title)
    merged_df = ppdf.merge(data,on='Student_name',how='inner')
    id_map = dict(zip(data['Student_name'], data['Picture']))
    persons_df['Pix'] = persons_df['Student_name'].apply(str.title).map(id_map)    
    

    
    st.dataframe(merged_df)
    # print(type(persons_df))
      
    st.write(persons_df)   
        

    st.subheader('4. Data Visualization using Pydeck Library')

    Bbox= ((df.longitude.max(), df.latitude.max()),
           (df.longitude.min(), df.latitude.min()))
    st.info("""
    Obtain the boundaries of the dataframe:
    ```python
    Bbox= ((df.longitude.max(), df.latitude.max()),
     (df.longitude.min(), df.latitude.min()))
    ```
    """)
    st.markdown(f""" 4.1 
    Plotting the GPS coordinates of the Index person -_{get_student_name(chosen,data)}_ 
    """)
    st.caption(f":green[{get_student_name(chosen,data)}] in (green dots) and other :red[individuals] on campus (red dots) [zoom in]")

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

if st.button("Run Program"):
    program_run()
