import pandas as pd
import streamlit as st
import numpy as np
import pydeck as pdk
import random
import string
import os.path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.neighbors import DistanceMetric
from datetime import datetime, timedelta

# Constants and configurations
@dataclass
class Config:
    REFERENCE_LONGITUDE = 4.42085
    REFERENCE_LATITUDE = 7.69244
    MAPBOX_API_KEY = 'pk.eyJ1Ijoia3Vmb3BvIiwiYSI6ImNrc3ZoZnZ2djFwdzUzMm9kc2tpMXF3NnoifQ.htE3W1FN0-rj9iqKUBFRdg'
    DEFAULT_ROWS = 200
    MIN_DISTANCE_FT = 3
    MAX_DISTANCE_FT = 10
    
    STUDENT_NAMES = [
        'Emma', 'Noah', 'Olivia', 'Olufemi', 'Idriss', 'William', 'Sophia', 'Yusuf',
        'Isabella', 'Bako', 'Amaka', 'Bayowa', 'Amara', 'Emeka', 'Kafayat',
        'Tijani', 'Aliu', 'Gbolahan', 'Chinasa', 'Hauwa'
    ]
    
    DEPARTMENTS = [
        'Computer Science', 'Mechanical Engineering', 'Physics', 'Chemistry',
        'Biology', 'Mathematics', 'History', 'English', 'Nursing', 'Economics'
    ]

class DataGenerator:
    @staticmethod
    def generate_identifier(length: int = 7) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def generate_phone_number() -> str:
        return f"+234-{random.randint(800, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    
    @classmethod
    def create_student_database(cls) -> pd.DataFrame:
        return pd.DataFrame({
            'Student Name': Config.STUDENT_NAMES,
            'Identifier': [cls.generate_identifier() for _ in range(len(Config.STUDENT_NAMES))],
            'Department': [random.choice(Config.DEPARTMENTS) for _ in range(len(Config.STUDENT_NAMES))],
            'Phone Number': [cls.generate_phone_number() for _ in range(len(Config.STUDENT_NAMES))]
        })
    
    @staticmethod
    def generate_gps_data(base_long: float, base_lat: float, num_points: int, 
                         identifier: str) -> pd.DataFrame:
        data = []
        for _ in range(num_points):
            rand_lat = random.random() / 100
            rand_long = random.random() / 100
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 360))
            data.append({
                'identifier': identifier,
                'longitude': base_long + rand_long,
                'latitude': base_lat + rand_lat,
                'timestamp': timestamp
            })
        return pd.DataFrame(data)

class ContactTracer:
    def __init__(self, reference_point: Tuple[float, float]):
        self.reference_point = reference_point
        self.dist_metric = DistanceMetric.get_metric('haversine')
    
    def calculate_distances(self, poi_data: pd.DataFrame, others_data: pd.DataFrame) -> pd.DataFrame:
        poi_coords = np.radians(poi_data[['longitude', 'latitude']])
        others_coords = np.radians(others_data[['longitude', 'latitude']])
        
        distances = self.dist_metric.pairwise(poi_coords, others_coords) * 6371  # km
        
        return pd.DataFrame(
            distances,
            index=poi_data['identifier'],
            columns=others_data['identifier']
        )
    
    def identify_close_contacts(self, distances: pd.DataFrame, threshold_ft: float) -> pd.DataFrame:
        threshold_km = threshold_ft * 0.0003048  # convert feet to km
        return distances.melt(ignore_index=False).query('value < @threshold_km')

class Visualization:
    @staticmethod
    def create_map(poi_data: pd.DataFrame, others_data: pd.DataFrame, 
                  close_contacts: Optional[List[str]] = None) -> pdk.Deck:
        if close_contacts is not None:
            others_data['contact_type'] = np.where(
                others_data['identifier'].isin(close_contacts),
                'close_contact',
                'others'
            )
        
        return pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=Config.REFERENCE_LATITUDE,
                longitude=Config.REFERENCE_LONGITUDE,
                zoom=15,
                pitch=50
            ),
            layers=[
                # POI Layer
                pdk.Layer(
                    'ScatterplotLayer',
                    data=poi_data,
                    get_position='[longitude, latitude]',
                    get_color=[255, 0, 0],  # Red for POI
                    get_radius=6,
                ),
                # Others Layer
                pdk.Layer(
                    'ScatterplotLayer',
                    data=others_data,
                    get_position='[longitude, latitude]',
                    get_color=[0, 255, 0],  # Green for others
                    get_radius=4,
                    pickable=True
                )
            ]
        )

def main():
    st.set_page_config(page_title="Contact Tracing System", layout="wide")
    
    # Application title and description
    st.title('Secure Contact Tracing Model for Infectious Disease Control')
    st.subheader('AU Campus Monitoring System')
    
    # Load or create student database
    if os.path.exists('students.csv'):
        student_db = pd.read_csv('students.csv')
    else:
        student_db = DataGenerator.create_student_database()
        student_db.to_csv('students.csv', index=False)
    
    # Sidebar controls
    st.sidebar.header('Controls')
    num_points = st.sidebar.slider('Number of GPS points', 
                                 min_value=100, 
                                 max_value=500, 
                                 value=Config.DEFAULT_ROWS)
    
    distance_threshold = st.sidebar.slider('Contact distance threshold (feet)',
                                         min_value=Config.MIN_DISTANCE_FT,
                                         max_value=Config.MAX_DISTANCE_FT,
                                         value=6)
    
    # Select POI
    poi_identifier = random.choice(student_db['Identifier'].tolist())
    poi_name = student_db[student_db['Identifier'] == poi_identifier]['Student Name'].iloc[0]
    
    st.write(f"Person of Interest: {poi_name} (ID: {poi_identifier})")
    
    # Generate GPS data
    poi_gps = DataGenerator.generate_gps_data(
        Config.REFERENCE_LONGITUDE,
        Config.REFERENCE_LATITUDE,
        num_points,
        poi_identifier
    )
    
    others_gps = pd.concat([
        DataGenerator.generate_gps_data(
            Config.REFERENCE_LONGITUDE,
            Config.REFERENCE_LATITUDE,
            num_points,
            identifier
        )
        for identifier in student_db[student_db['Identifier'] != poi_identifier]['Identifier']
    ])
    
    # Contact tracing analysis
    tracer = ContactTracer((Config.REFERENCE_LONGITUDE, Config.REFERENCE_LATITUDE))
    distances = tracer.calculate_distances(poi_gps, others_gps)
    close_contacts = tracer.identify_close_contacts(distances, distance_threshold)
    
    # Display results
    st.subheader("Contact Tracing Results")
    if not close_contacts.empty:
        contact_names = student_db[student_db['Identifier'].isin(close_contacts['variable'])]
        st.write(f"Found {len(contact_names)} potential close contacts:")
        st.dataframe(contact_names)
    else:
        st.write("No close contacts identified.")
    
    # Visualization
    st.subheader("Movement Visualization")
    st.pydeck_chart(
        Visualization.create_map(
            poi_gps,
            others_gps,
            close_contacts['variable'].unique() if not close_contacts.empty else None
        )
    )
    
    # Additional statistics
    st.subheader("Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total GPS Points", len(poi_gps) + len(others_gps))
    with col2:
        st.metric("Average Distance (km)", 
                 f"{distances.values.mean():.2f}")

if __name__ == "__main__":
    main()
