from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from data.data_loader import load_osm_data
from models.mahalanobis import classify_mahalanobis
from models.pca import classify_pca
from models.manhattan import classify_manhattan
from models.weighted import classify_weighted
from models.bayesian import classify_bayesian

app = Flask(__name__)
CORS(app)  # This allows your Vue.js frontend to make requests to this API

# Cache to store loaded data for different cities
city_data_cache = {}

@app.route('/api/cities/<city_name>', methods=['GET'])
def get_city_data(city_name):
    """Load data for a specific city"""
    try:
        # Check if data is already in cache
        if city_name in city_data_cache:
            df = city_data_cache[city_name]
        else:
            df = load_osm_data(city_name)
            if df.empty:
                return jsonify({"error": f"No data available for {city_name}"}), 404
            # Store in cache for future requests
            city_data_cache[city_name] = df
        
        # Convert DataFrame to dict for JSON response
        result = df.to_dict(orient='records')
        return jsonify({
            "city": city_name,
            "count": len(result),
            "data": result,
            "min_max": {
                "CO2_Usage": [float(df["CO2_Usage"].min()), float(df["CO2_Usage"].max())],
                "Water_Usage": [float(df["Water_Usage"].min()), float(df["Water_Usage"].max())],
                "Energy_Consumption": [float(df["Energy_Consumption"].min()), float(df["Energy_Consumption"].max())],
                "height": [float(df["height"].min()), float(df["height"].max())]
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify_buildings():
    """Apply classification to building data"""
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        city_name = data.get('city_name')
        classification_method = data.get('classification_method')
        
        # Check if we have data for this city
        if city_name not in city_data_cache:
            df = load_osm_data(city_name)
            if df.empty:
                return jsonify({"error": f"No data available for {city_name}"}), 404
            city_data_cache[city_name] = df
        
        # Get the data
        df = city_data_cache[city_name].copy()
        
        # Apply classification
        if classification_method == "manhattan":
            df = classify_manhattan(df)
        elif classification_method == "mahalanobis":
            df = classify_mahalanobis(df)
        elif classification_method == "pca":
            df = classify_pca(df)
        elif classification_method == "weighted":
            df = classify_weighted(df)
        elif classification_method == "bayesian":
            df = classify_bayesian(df)
        else:
            return jsonify({"error": f"Invalid classification method: {classification_method}"}), 400
        
        # Return the classified data
        result = df.to_dict(orient='records')
        return jsonify({
            "city": city_name,
            "classification_method": classification_method,
            "count": len(result),
            "data": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/filter', methods=['POST'])
def filter_buildings():
    """Filter building data based on provided parameters"""
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        city_name = data.get('city_name')
        filters = data.get('filters', {})
        
        # Check if we have data for this city
        if city_name not in city_data_cache:
            return jsonify({"error": f"No data loaded for {city_name}. Load city data first."}), 404
        
        # Get the data
        df = city_data_cache[city_name].copy()
        
        # Apply filters
        if 'co2_range' in filters:
            co2_min, co2_max = filters['co2_range']
            df = df[(df["CO2_Usage"] >= co2_min) & (df["CO2_Usage"] <= co2_max)]
            
        if 'water_range' in filters:
            water_min, water_max = filters['water_range']
            df = df[(df["Water_Usage"] >= water_min) & (df["Water_Usage"] <= water_max)]
            
        if 'energy_range' in filters:
            energy_min, energy_max = filters['energy_range']
            df = df[(df["Energy_Consumption"] >= energy_min) & (df["Energy_Consumption"] <= energy_max)]
            
        if 'height_range' in filters:
            height_min, height_max = filters['height_range']
            df = df[(df["height"] >= height_min) & (df["height"] <= height_max)]
            
        if 'classes' in filters:
            selected_classes = filters['classes']
            df = df[df["class_label"].isin(selected_classes)]
        
        # Return the filtered data
        result = df.to_dict(orient='records')
        return jsonify({
            "city": city_name,
            "count": len(result),
            "total_count": len(city_data_cache[city_name]),
            "data": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics', methods=['POST'])
def get_metrics():
    """Get metrics overview for a filtered dataset"""
    try:
        # Get request data
        data = request.json
        city_name = data.get('city_name')
        filtered_data = data.get('data')
        
        if filtered_data:
            # If filtered data is provided, convert to DataFrame
            df = pd.DataFrame(filtered_data)
        elif city_name in city_data_cache:
            # Otherwise use all data for the city
            df = city_data_cache[city_name].copy()
        else:
            return jsonify({"error": "No data provided or city not loaded"}), 400
        
        # Calculate metrics (example - modify based on your existing metrics.py functionality)
        metrics = {
            "building_count": len(df),
            "class_distribution": df["class_label"].value_counts().to_dict(),
            "avg_co2": float(df["CO2_Usage"].mean()),
            "avg_water": float(df["Water_Usage"].mean()),
            "avg_energy": float(df["Energy_Consumption"].mean()),
            "avg_height": float(df["height"].mean())
        }
        
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export_data():
    """Export filtered data (simplified version - expand based on your export.py)"""
    try:
        # Get request data
        data = request.json
        filtered_data = data.get('data')
        
        if not filtered_data:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(filtered_data)
        
        # Return export URL or data depending on your implementation
        # This is simplified - you may need to implement actual file saving/generation
        return jsonify({
            "success": True,
            "count": len(df),
            "export_url": "/downloads/export.csv"  # This would be handled by your frontend
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/benchmark', methods=['POST'])
def benchmark_comparison():
    """Benchmark comparison (simplified - expand based on your actual implementation)"""
    try:
        # Get request data
        data = request.json
        filtered_data = data.get('data')
        
        if not filtered_data:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(filtered_data)
        
        # Placeholder for benchmark comparison
        # Replace with actual logic from your add_benchmark_comparison function
        benchmark = {
            "average_co2_vs_national": df["CO2_Usage"].mean() / 100,  # Example comparison
            "average_energy_vs_national": df["Energy_Consumption"].mean() / 200,
            "average_water_vs_national": df["Water_Usage"].mean() / 150,
            "building_efficiency_score": 85  # Example score
        }
        
        return jsonify(benchmark)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)