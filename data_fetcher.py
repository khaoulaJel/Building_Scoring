import pandas as pd
import requests
from random import uniform, randint
import time

def get_city_bounding_box(city, country="Morocco"):
    """
    Get the bounding box (lat/lon coordinates) for a given city using Nominatim API.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': f"{city}, {country}",
        'format': 'json',
        'limit': 1
    }
    
    headers = {
        'User-Agent': 'BuildingScoringApp/1.0'  
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data:
            bounding_box = data[0]['boundingbox']
            return {
                'min_lat': float(bounding_box[0]),
                'max_lat': float(bounding_box[1]),
                'min_lon': float(bounding_box[2]),
                'max_lon': float(bounding_box[3])
            }
        else:
            print(f"No bounding box found for {city}, {country}")
            return None
    
    except Exception as e:
        print(f"Error fetching bounding box for {city}: {str(e)}")
        return None

def fetch_buildings_from_osm(city, country="Morocco", limit=1000):
    """
    Fetch building data from OpenStreetMap for a given city using Overpass API.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Get the bounding box for the city
    bounding_box = get_city_bounding_box(city, country)
    if not bounding_box:
        return pd.DataFrame()
    
    # Overpass query to fetch buildings within the city's bounding box
    overpass_query = f"""
    [out:json][timeout:60];
    (
        node["building"]({bounding_box['min_lat']},{bounding_box['min_lon']},{bounding_box['max_lat']},{bounding_box['max_lon']});
        way["building"]({bounding_box['min_lat']},{bounding_box['min_lon']},{bounding_box['max_lat']},{bounding_box['max_lon']});
        relation["building"]({bounding_box['min_lat']},{bounding_box['min_lon']},{bounding_box['max_lat']},{bounding_box['max_lon']});
    );
    out center body qt;
    """
    
    try:
        response = requests.post(overpass_url, data=overpass_query)
        response.raise_for_status()
        data = response.json()
        
        buildings = []
        
        for element in data.get('elements', []):
            if 'center' in element:
                lat = element['center']['lat']
                lon = element['center']['lon']
            elif 'lat' in element and 'lon' in element:
                lat = element['lat']
                lon = element['lon']
            else:
                continue
                
            tags = element.get('tags', {})
            building_name = tags.get('name', f"Building-{element['id']}")
            
            energy_consumption = randint(400000, 1000000)
            carbon_footprint = round(energy_consumption * uniform(0.0004, 0.0006), 1)
            water_usage = round(energy_consumption * uniform(0.02, 0.04), 0)
            
            buildings.append({
                'city': city,
                'building_name': building_name,
                'latitude': lat,
                'longitude': lon,
                'energy_consumption_kwh': energy_consumption,
                'carbon_footprint_tco2e': carbon_footprint,
                'water_usage_m3': water_usage
            })
            
            # Limit the number of buildings fetched
            if len(buildings) >= limit:
                break
        
        return pd.DataFrame(buildings)
    
    except Exception as e:
        print(f"Error fetching data for {city}: {str(e)}")
        return pd.DataFrame()