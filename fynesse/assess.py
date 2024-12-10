from .config import *

from . import access

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from rapidfuzz import process, fuzz
"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def get_buildings_with_address(center_point):
    latitude_offset = 0.009
    longitude_offset = 0.015

    north = center_point[0] + latitude_offset
    south = center_point[0] - latitude_offset
    east = center_point[1] + longitude_offset
    west = center_point[1] - longitude_offset
    # Define the tags for buildings
    tags = {'building': True}

    # Retrieve building geometries within the bounding box
    buildings = ox.geometries_from_bbox(north, south, east, west, tags)

    # Filter only buildings that have a polygon geometry (i.e., actual building footprints)
    buildings = buildings[buildings.geometry.type == 'Polygon']

    # Calculate area for each building in square meters
    buildings = buildings.to_crs(epsg=3857)  # Project to a metric system (meters)
    buildings['area_sqm'] = buildings['geometry'].area

    # Reproject back to EPSG 4326 for plotting with latitude/longitude
    buildings = buildings.to_crs(epsg=4326)

    # Filter for buildings with full address (contains 'addr:housenumber', 'addr:street', 'addr:postcode')
    address_columns = ['addr:housenumber', 'addr:street', 'addr:postcode']
    return buildings,buildings.dropna(subset=address_columns)

def get_address_matching(pp_data,buildings_with_address):
    # For PP Data: Combine primary and secondary address fields to create a standardized housenumber
    pp_data['housenumber'] = pp_data['primary_addressable_object_name'].fillna('') + ' ' + pp_data['secondary_addressable_object_name'].fillna('')
    pp_data['housenumber'] = pp_data['housenumber'].str.strip()  # Remove extra whitespace
    pp_data['address_key'] = pp_data['housenumber'] + ' ' + pp_data['street'] + ' ' + pp_data['postcode']
    pp_data['address_key'] = pp_data['address_key'].str.lower().str.replace(r'\s+', ' ', regex=True)

    # For OSM Data: Combine address fields for a standardized address key
    buildings_with_address['address_key'] = buildings_with_address['addr:housenumber'].fillna('') + ' ' + buildings_with_address['addr:street'].fillna('') + ' ' + buildings_with_address['addr:postcode'].fillna('')
    buildings_with_address['address_key'] = buildings_with_address['address_key'].str.lower().str.replace(r'\s+', ' ', regex=True)

    # Merge DataFrames on address_key for exact matches
    merged_exact = pd.merge(pp_data, buildings_with_address, on='address_key', how='inner', suffixes=('_pp', '_osm'))
    non_matched_pp = pp_data[~pp_data['address_key'].isin(merged_exact['address_key'])]
    joined_data = non_matched_pp.merge(buildings_with_address, left_on='postcode', right_on='addr:postcode', suffixes=('_pp', '_osm'))
    fuzzy_matches = fuzzy_match_on_joined_data(joined_data)
    final_merged = pd.concat([merged_exact, fuzzy_matches], ignore_index=True)
    return final_merged

# Function to perform fuzzy matching on joined data
def fuzzy_match_on_joined_data(df, threshold=95):
    results = []
    for _, row in df.iterrows():
        # Use rapidfuzz to find the best fuzzy match for address_key within the joined subset
        match = process.extractOne(row['address_key_pp'], df['address_key_osm'], scorer=fuzz.token_sort_ratio)
        
        # Check if the match meets the similarity threshold
        if match and match[1] >= threshold:
            matched_row = df[df['address_key_osm'] == match[0]].copy()
            matched_row['fuzzy_match_score'] = match[1]  # Track the match score
            matched_row['original_row_index'] = row.name  # Track original row index for later merging
            matched_row['pp_data_row'] = row  # Store the original pp_data row information
            results.append(matched_row)
    
    # Combine all matched rows into a single DataFrame
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
