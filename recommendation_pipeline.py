import pandas as pd
import numpy as np
import sqlite3
from scipy.sparse import csr_matrix
import pickle
import os
from typing import Tuple, Optional
import logging
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AnimeRecommender:
    def __init__(self, factors: int = 30, iterations: int = 30, regularization: float = 0.1):
        """Initialize the recommender with PySpark ALS.
        
        Args:
            factors: Number of latent factors
            iterations: Number of ALS iterations
            regularization: Regularization parameter
        """
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.model = None
        self.spark = None
        self.user_mapping = None
        self.anime_mapping = None
        self.reverse_user_mapping = None
        self.reverse_anime_mapping = None
        
        # Initialize Spark session
        self.init_spark()

    def init_spark(self):
        """Initialize Spark session with appropriate configuration."""
        self.spark = (SparkSession.builder
            .appName("AnimeRecommender")
            .config("spark.sql.shuffle.partitions", "32")
            .config("spark.executor.memory", "16g")
            .config("spark.driver.memory", "8g")
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow -Xss4m")
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow -Xss4m")
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.kryoserializer.buffer.max", "1g")
            .master("local[*]")
            .getOrCreate())
        
        # Set log level to reduce verbosity
        self.spark.sparkContext.setLogLevel("ERROR")

    def extract_data(self, db_path: str, min_rating_count: int = 1000) -> pd.DataFrame:
        """Extract ratings data from SQLite database.
        
        Args:
            db_path: Path to SQLite database
            min_rating_count: Minimum number of ratings required for an anime to be included
            
        Returns:
            DataFrame with columns (username, anime_id, rating)
        """
        logging.info('Extracting data from database...')
        
        # First, get anime IDs that meet the minimum rating count
        query = '''
            SELECT anime_id, COUNT(*) as rating_count
            FROM anime_ratings
            GROUP BY anime_id
            HAVING rating_count >= ?
        '''
        
        with sqlite3.connect(db_path) as conn:
            popular_anime = pd.read_sql_query(query, conn, params=(min_rating_count,))
            
        if popular_anime.empty:
            raise ValueError(f"No anime found with at least {min_rating_count} ratings")
            
        logging.info(f'Found {len(popular_anime)} anime with at least {min_rating_count} ratings')
        
        # Then get all ratings for these anime
        anime_ids = tuple(popular_anime['anime_id'].tolist())
        ratings_query = f'''
            SELECT username, anime_id, rating
            FROM anime_ratings
            WHERE rating > 0 AND anime_id IN ({','.join('?' * len(anime_ids))})
        '''
        
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(ratings_query, conn, params=anime_ids)
        
        logging.info(f'Extracted {len(df)} ratings for {len(anime_ids)} anime')
        return df

    def create_mappings(self, df: pd.DataFrame) -> Tuple[dict, dict]:
        """Create mappings between user/anime IDs and matrix indices."""
        unique_users = df['username'].unique()
        unique_anime = df['anime_id'].unique()
        
        user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        anime_mapping = {anime: idx for idx, anime in enumerate(unique_anime)}
        
        self.user_mapping = user_mapping
        self.anime_mapping = anime_mapping
        self.reverse_user_mapping = {v: k for k, v in user_mapping.items()}
        self.reverse_anime_mapping = {v: k for k, v in anime_mapping.items()}
        
        return user_mapping, anime_mapping

    def prepare_and_save_data(self, df: pd.DataFrame, parquet_path: str = 'ratings.parquet'):
        """Prepare ratings data and save to parquet and mappings.
        
        Args:
            df: DataFrame with columns (username, anime_id, rating)
            parquet_path: Path to save the parquet file
        """
        logging.info('Preparing ratings data...')
        
        # Create mappings if they don't exist
        if self.user_mapping is None:
            self.create_mappings(df)
        
        # Save mappings first
        model_dir = os.path.dirname(parquet_path)
        mappings = {
            'user_mapping': self.user_mapping,
            'anime_mapping': self.anime_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_anime_mapping': self.reverse_anime_mapping,
            'factors': self.factors,
            'regularization': self.regularization,
            'iterations': self.iterations
        }
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'mappings.pkl'), 'wb') as f:
            pickle.dump(mappings, f)
        logging.info(f'Saved mappings with {len(self.anime_mapping)} anime to {model_dir}')
        
        # Convert usernames and anime_ids to indices
        df_mapped = df.copy()
        df_mapped['user'] = df['username'].map(self.user_mapping).astype('int32')
        df_mapped['item'] = df['anime_id'].map(self.anime_mapping).astype('int32')
        df_mapped['rating'] = df['rating'].astype('float32')
        
        # Select only needed columns
        ratings_df = df_mapped[['user', 'item', 'rating']]
        
        # Save to parquet
        ratings_df.to_parquet(parquet_path, index=False)
        logging.info(f'Saved ratings to {parquet_path}')
        
        # Free memory
        del df
        del df_mapped
        del ratings_df

    def train(self, parquet_path: str = 'ratings.parquet'):
        """Train the PySpark ALS model from parquet file.
        
        Args:
            parquet_path: Path to the parquet file containing ratings
        """
        logging.info('Training PySpark ALS model...')
        
        # Read parquet file into Spark
        spark_df = self.spark.read.parquet(parquet_path)
        
        # Initialize and train ALS model
        als = ALS(
            maxIter=self.iterations,
            regParam=self.regularization,
            rank=self.factors,
            userCol="user",
            itemCol="item",
            ratingCol="rating",
            coldStartStrategy="drop",
            implicitPrefs=False  # We're using explicit ratings
        )
        
        self.model = als.fit(spark_df)
        
        # Free Spark DataFrame
        spark_df.unpersist()
        logging.info('Training completed')

    def get_latent_factors(self):
        """Extract user and item latent factors from the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Get the user factors
        user_factors = self.model.userFactors.toPandas()
        user_factors = user_factors.sort_values('id')
        user_factors_array = np.array([row['features'] for _, row in user_factors.iterrows()])
        
        # Get the item factors
        item_factors = self.model.itemFactors.toPandas()
        item_factors = item_factors.sort_values('id')
        item_factors_array = np.array([row['features'] for _, row in item_factors.iterrows()])
        
        return user_factors_array, item_factors_array

    def save_model(self, directory: str):
        """Save the latent factors.
        
        Args:
            directory: Directory to save the model files
        """
        os.makedirs(directory, exist_ok=True)
        
        # Extract and save the latent factors
        user_factors, item_factors = self.get_latent_factors()
        np.save(os.path.join(directory, "user_factors.npy"), user_factors)
        np.save(os.path.join(directory, "item_factors.npy"), item_factors)
        
        logging.info(f'Latent factors saved to {directory}')

    def load_model(self, directory: str):
        """Load the latent factors and mappings."""
        # Load the mappings
        with open(os.path.join(directory, 'mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
            self.user_mapping = mappings['user_mapping']
            self.anime_mapping = mappings['anime_mapping']
            self.reverse_user_mapping = mappings['reverse_user_mapping']
            self.reverse_anime_mapping = mappings['reverse_anime_mapping']
            self.factors = mappings['factors']
            self.regularization = mappings['regularization']
            self.iterations = mappings['iterations']
        
        # Load the latent factors
        user_factors = np.load(os.path.join(directory, "user_factors.npy"))
        item_factors = np.load(os.path.join(directory, "item_factors.npy"))
        
        # Create a dummy model with the loaded factors
        user_factors_df = pd.DataFrame({
            'id': range(len(user_factors)),
            'features': list(user_factors)
        })
        item_factors_df = pd.DataFrame({
            'id': range(len(item_factors)),
            'features': list(item_factors)
        })
        
        # Convert to Spark DataFrames
        user_factors_spark = self.spark.createDataFrame(user_factors_df)
        item_factors_spark = self.spark.createDataFrame(item_factors_df)
        
        # Create a dummy ALS model with the loaded factors
        self.model = ALS(
            maxIter=self.iterations,
            regParam=self.regularization,
            rank=self.factors,
            userCol="user",
            itemCol="item",
            ratingCol="rating",
            coldStartStrategy="drop",
            implicitPrefs=False
        ).fit(self.spark.createDataFrame([], schema=["user", "item", "rating"]))
        
        # Set the factors in the model
        self.model.userFactors = user_factors_spark
        self.model.itemFactors = item_factors_spark
        
        logging.info('Latent factors and mappings loaded successfully')

    def __del__(self):
        """Clean up Spark session on object deletion."""
        if self.spark is not None:
            self.spark.stop()
            self.spark = None