import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db_connection():
    """
    Create and return a SQLAlchemy database engine using environment variables
    """
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("root", "")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "buildings_db")
    
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    return create_engine(connection_string)

def load_dataset(city="Lyon", year=None):
    """
    Load building data from PostgreSQL database
    
    Parameters:
    -----------
    city : str
        City name to filter buildings
    year : int, optional
        Year to filter buildings, if None, returns all years
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the building data
    """
    engine = get_db_connection()
    
    # Base query to select all buildings
    query = """
    SELECT * FROM buildings 
    WHERE city = %(city)s
    """
    
    # Add year filter if specified
    if year is not None:
        query += " AND year = %(year)s"
    
    # Execute query with parameters
    return pd.read_sql(query, engine, params={"city": city, "year": year})

def get_available_cities():
    """
    Get list of available cities in the database
    
    Returns:
    --------
    list
        List of city names
    """
    engine = get_db_connection()
    query = "SELECT DISTINCT city FROM buildings ORDER BY city"
    return pd.read_sql(query, engine)["city"].tolist()

def get_available_years():
    """
    Get list of available years in the database
    
    Returns:
    --------
    list
        List of years
    """
    engine = get_db_connection()
    query = "SELECT DISTINCT year FROM buildings ORDER BY year"
    return pd.read_sql(query, engine)["year"].tolist()

def save_benchmark_data(benchmark_data, name):
    """
    Save benchmark data to database
    
    Parameters:
    -----------
    benchmark_data : pandas.DataFrame
        DataFrame containing benchmark data
    name : str
        Name of the benchmark
    """
    engine = get_db_connection()
    benchmark_data["benchmark_name"] = name
    benchmark_data.to_sql("benchmarks", engine, if_exists="append", index=False)

def load_benchmark_data(name=None):
    """
    Load benchmark data from database
    
    Parameters:
    -----------
    name : str, optional
        Name of the benchmark, if None, returns all benchmarks
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the benchmark data
    """
    engine = get_db_connection()
    
    if name is None:
        query = "SELECT DISTINCT benchmark_name FROM benchmarks"
        return pd.read_sql(query, engine)["benchmark_name"].tolist()
    else:
        query = "SELECT * FROM benchmarks WHERE benchmark_name = %(name)s"
        return pd.read_sql(query, engine, params={"name": name})