import pandas as pd
import numpy as np
import os
import osmnx as ox

def load_osm_data(city_name, force_refresh=False):
    cache_path = f"data/cache/{city_name.lower().replace(' ', '_')}.csv"

    if not force_refresh and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    gdf = ox.features_from_place(city_name, tags={"building": True})
    gdf = gdf[gdf.geometry.type == "Polygon"]

    n = len(gdf)
    gdf["CO2_Usage"] = np.random.uniform(50, 500, n)
    gdf["Water_Usage"] = np.random.uniform(1000, 10000, n)
    gdf["Energy_Consumption"] = np.random.uniform(500, 5000, n)
    gdf["height"] = np.random.uniform(10, 100, n)
    gdf["latitude"] = gdf.geometry.centroid.y
    gdf["longitude"] = gdf.geometry.centroid.x
    gdf["polygon"] = gdf.geometry.apply(lambda geom: [[list(coord) for coord in geom.exterior.coords]])

    df = pd.DataFrame({
        'building_id': range(n),
        'CO2_Usage': gdf["CO2_Usage"],
        'Water_Usage': gdf["Water_Usage"],
        'Energy_Consumption': gdf["Energy_Consumption"],
        'height': gdf["height"],
        'latitude': gdf["latitude"],
        'longitude': gdf["longitude"],
        'polygon': gdf["polygon"]
    })

    os.makedirs("data/cache", exist_ok=True)
    df.to_csv(cache_path, index=False)

    return df
