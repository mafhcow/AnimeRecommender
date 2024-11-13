import argparse
import logging
from recommendation_pipeline import AnimeRecommender
import gc
import os
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_model_directory(model_dir: str):
    """Remove all existing model files from the directory."""
    if os.path.exists(model_dir):
        logging.info(f'Cleaning model directory: {model_dir}')
        for filename in os.listdir(model_dir):
            file_path = os.path.join(model_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logging.info(f'Deleted: {filename}')
            except Exception as e:
                logging.error(f'Error deleting {filename}: {e}')

def main():
    parser = argparse.ArgumentParser(description='Train the anime recommendation model')
    parser.add_argument('--db-path', default='mal_users.db', help='Path to the SQLite database')
    parser.add_argument('--model-dir', default='model', help='Directory to save the model')
    parser.add_argument('--factors', type=int, default=30, help='Number of latent factors')
    parser.add_argument('--iterations', type=int, default=30, help='Number of ALS iterations')
    parser.add_argument('--regularization', type=float, default=0.1, help='Regularization parameter')
    parser.add_argument('--min-ratings', type=int, default=1000, 
                       help='Minimum number of ratings required for an anime to be included')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuilding parquet even if it exists')
    args = parser.parse_args()

    try:
        # Clean model directory if force rebuild
        if args.force_rebuild:
            clean_model_directory(args.model_dir)

        # Initialize recommender
        recommender = AnimeRecommender(
            factors=args.factors,
            iterations=args.iterations,
            regularization=args.regularization
        )

        # Extract data from database
        logging.info('Extracting data from database...')
        df = recommender.extract_data(args.db_path, args.min_ratings)
        
        if df.empty:
            raise ValueError("No ratings found in database")
        
        logging.info(f'Found {len(df)} ratings from {len(df.username.unique())} users')
        
        # Prepare data files
        parquet_path = os.path.join(args.model_dir, 'ratings.parquet')
        if not os.path.exists(parquet_path) or args.force_rebuild:
            logging.info('Preparing and saving data (including mappings)...')
            recommender.prepare_and_save_data(df, parquet_path)
            gc.collect()
        else:
            logging.info(f'Using existing parquet file: {parquet_path}')
            # Load existing mappings
            with open(os.path.join(args.model_dir, 'mappings.pkl'), 'rb') as f:
                import pickle
                mappings = pickle.load(f)
                recommender.anime_mapping = mappings['anime_mapping']
                recommender.user_mapping = mappings['user_mapping']
                recommender.reverse_anime_mapping = mappings['reverse_anime_mapping']
                recommender.reverse_user_mapping = mappings['reverse_user_mapping']
        
        # Train model
        logging.info('Training model...')
        recommender.train(parquet_path)
        
        # Verify model training
        if recommender.model is None:
            raise ValueError("Model training failed")

        # Save model factors
        logging.info('Saving model factors...')
        recommender.save_model(args.model_dir)
        
        logging.info('Training completed successfully!')
        logging.info(f'Model saved to {args.model_dir}')
        logging.info(f'Number of anime in model: {len(recommender.anime_mapping)}')

    except Exception as e:
        logging.error(f'Error during training: {e}')
        raise

if __name__ == '__main__':
    main()