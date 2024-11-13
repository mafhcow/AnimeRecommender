import argparse
import requests
import json
import logging
import numpy as np
import os
import pickle
import sqlite3
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RecommenderModel:
    def __init__(self, model_dir: str):
        """Initialize the recommender model with pre-trained latent factors.
        
        Args:
            model_dir: Directory containing the model files
        """
        self.model_dir = model_dir
        self.anime_mapping = None
        self.reverse_anime_mapping = None
        self.item_factors = None
        self.regularization = None
        
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('MAL_API_KEY')
        if not self.api_key:
            raise ValueError("MAL_API_KEY not found in .env file")
            
        self.base_url = 'https://api.myanimelist.net/v2'
        self.load_model()
    
    def load_model(self):
        """Load the latent factors and mappings from files."""
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"Model directory '{self.model_dir}' does not exist")
            
            # Check if required files exist
            mappings_path = os.path.join(self.model_dir, 'mappings.pkl')
            item_factors_path = os.path.join(self.model_dir, 'item_factors.npy')
            
            if not os.path.exists(mappings_path):
                raise FileNotFoundError(f"Mappings file not found at '{mappings_path}'")
            if not os.path.exists(item_factors_path):
                raise FileNotFoundError(f"Item factors file not found at '{item_factors_path}'")
            
            # Load mappings
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                
                # Verify mappings structure
                required_keys = ['anime_mapping', 'reverse_anime_mapping', 'regularization']
                missing_keys = [key for key in required_keys if key not in mappings]
                if missing_keys:
                    raise ValueError(f"Missing required keys in mappings file: {missing_keys}")
                
                self.anime_mapping = mappings['anime_mapping']
                self.reverse_anime_mapping = mappings['reverse_anime_mapping']
                self.regularization = mappings['regularization']
                
                # Verify mappings are not empty
                if not self.anime_mapping or not self.reverse_anime_mapping:
                    raise ValueError("Anime mappings are empty")
            
            # Load latent factors
            self.item_factors = np.load(item_factors_path)
            
            # Verify dimensions match
            if len(self.anime_mapping) != len(self.item_factors):
                raise ValueError(f"Mismatch between anime mapping size ({len(self.anime_mapping)}) "
                               f"and item factors size ({len(self.item_factors)})")
            
            logging.info('Model loaded successfully')
            logging.info(f'Number of anime in model: {len(self.anime_mapping)}')
            
        except Exception as e:
            logging.error(f'Error loading model: {e}')
            raise

    def get_anime_info(self, anime_ids: List[int]) -> Dict[int, Dict]:
        """Fetch anime information from the database and MAL API.
        
        Args:
            anime_ids: List of anime IDs to fetch information for
            
        Returns:
            Dictionary mapping anime_id to anime information
        """
        logging.info(f"Fetching info for {len(anime_ids)} anime IDs")
        # First get information from our database
        conn = sqlite3.connect('anime_averages.db')
        try:
            # Create the IN clause with the actual values
            placeholders = ','.join([str(anime_id) for anime_id in anime_ids])
            query = f"""
                SELECT anime_id, title, average_rating, num_ratings
                FROM anime_averages
                WHERE anime_id IN ({placeholders})
            """
            
            cursor = conn.cursor()
            cursor.execute(query)  # No parameters needed since values are in the query
            db_results = cursor.fetchall()
            logging.info(f"Found {len(db_results)} results in database")
            
            # Create initial results dictionary with database info
            results = {}
            for row in db_results:
                anime_id, title, avg_rating, num_ratings = row
                results[anime_id] = {
                    'title': title,
                    'mean': float(avg_rating) if avg_rating is not None else None,
                    'num_ratings': num_ratings
                }
            
            # If no results found, add placeholders
            if not results:
                logging.warning("No anime found in database, using placeholders")
                for anime_id in anime_ids:
                    results[anime_id] = {
                        'title': f'Anime {anime_id}',
                        'mean': None,
                        'num_ratings': 0
                    }
            
            return results
            
        finally:
            conn.close()
    
    def get_recommendations(self, user_vector: np.ndarray, n: int = 10) -> List[Tuple[int, float]]:
        """Get anime recommendations for a user vector.
        
        Args:
            user_vector: Vector of user ratings (same length as anime_mapping)
            n: Number of recommendations to return
            
        Returns:
            List of (anime_id, score) tuples
        """
        # Get indices of rated items
        rated_indices = np.nonzero(user_vector)[0]
        num_ratings = len(rated_indices)
        
        if len(rated_indices) == 0:
            raise ValueError("No rated items found in user vector")
            
        # Get rated item factors and ratings
        rated_factors = self.item_factors[rated_indices]
        rated_ratings = user_vector[rated_indices]
        
        # Compute user factors using regularized ALS update rule:
        # user_factors = (X^T * X + λI)^(-1) * X^T * ratings
        # where X is the item factors matrix for rated items
        XtX = rated_factors.T @ rated_factors
        num_factors = rated_factors.shape[1]
        
        # Add regularization term (λI)
        reg_matrix = num_ratings * self.regularization * np.eye(num_factors)
        
        # Solve the regularized normal equation
        A = XtX + reg_matrix
        b = rated_factors.T @ rated_ratings
        user_factors = np.linalg.solve(A, b)
        
        # Calculate predicted ratings for all items
        predictions = self.item_factors @ user_factors
        
        # Set rated items to negative infinity to exclude them
        predictions[rated_indices] = float('-inf')
        
        # Get top N recommendations
        top_indices = np.argpartition(predictions, -n)[-n:]
        top_indices = top_indices[np.argsort(-predictions[top_indices])]
        
        # Convert to list of (anime_id, score) tuples
        return [(self.reverse_anime_mapping[idx], float(predictions[idx])) 
                for idx in top_indices]

    def get_similar_anime(self, anime_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Get similar anime based on latent factor cosine similarity.
        
        Args:
            anime_id: ID of the anime to find similar titles for
            n: Number of similar anime to return
            
        Returns:
            List of (anime_id, similarity_score) tuples
        """
        try:
            # Get the anime's index in our item factors
            anime_idx = self.anime_mapping.get(anime_id)
            if anime_idx is None:
                raise ValueError(f"Anime ID {anime_id} not found in model")
            
            # Get the anime's latent factors
            anime_factors = self.item_factors[anime_idx]
            
            # Calculate cosine similarity with all other anime
            # Normalize the vectors for cosine similarity
            norms = np.linalg.norm(self.item_factors, axis=1)
            normalized_factors = self.item_factors / norms[:, np.newaxis]
            normalized_anime_factors = anime_factors / np.linalg.norm(anime_factors)
            
            # Calculate similarities
            similarities = normalized_factors @ normalized_anime_factors
            
            # Get indices of top N similar anime (excluding the input anime)
            similarities[anime_idx] = -1  # Exclude the input anime
            top_indices = np.argpartition(similarities, -n)[-n:]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
            
            # Convert to list of (anime_id, similarity) tuples
            similar_anime = []
            for idx in top_indices:
                similar_anime_id = self.reverse_anime_mapping[idx]
                similarity_score = float(similarities[idx])
                similar_anime.append((similar_anime_id, similarity_score))
            
            return similar_anime
            
        except Exception as e:
            logging.error(f"Error finding similar anime: {e}")
            raise

def get_user_anime_list(username: str) -> list:
    """Fetch a user's anime list from MAL API."""
    headers = {
        'X-MAL-CLIENT-ID': os.getenv('MAL_API_KEY')
    }
    
    params = {
        'fields': 'list_status',
        'limit': 1000,
        'nsfw': True  # Include NSFW titles to match database
    }

    all_anime = []
    next_url = f'https://api.myanimelist.net/v2/users/{username}/animelist'

    while next_url:
        try:
            response = requests.get(next_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract anime entries
            anime_entries = data['data']
            for entry in anime_entries:
                anime_id = entry['node']['id']
                rating = entry['list_status'].get('score', 0)
                if rating > 0:  # Only include rated anime
                    all_anime.append((anime_id, rating))

            # Check for next page
            next_url = data.get('paging', {}).get('next')
            if next_url:
                params = {}  # Parameters are included in the next_url

        except requests.RequestException as e:
            logging.error(f'Error fetching anime list: {e}')
            break

    logging.info(f'Found {len(all_anime)} rated anime for user {username}')
    return all_anime

def create_user_rating_vector(anime_mapping: dict, user_ratings: list) -> np.ndarray:
    """Create a rating vector for the user based on their ratings."""
    vector = np.zeros(len(anime_mapping))
    for anime_id, rating in user_ratings:
        if anime_id in anime_mapping:
            idx = anime_mapping[anime_id]
            vector[idx] = rating
    return vector