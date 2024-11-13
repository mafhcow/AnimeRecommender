from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from get_recommendations import RecommenderModel, get_user_anime_list, create_user_rating_vector
import numpy as np
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,  # Rate limit by IP address
    storage_uri="memory://",  # Use in-memory storage
    strategy="fixed-window"  # Use fixed window strategy
)

# Configure custom error handler for rate limiting
@app.errorhandler(429)  # Too Many Requests
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description),
        "retry_after": e.retry_after
    }), 429

# Initialize the recommender model (load once at startup)
try:
    model = RecommenderModel('model')
    logging.info('Model loaded successfully')
except Exception as e:
    logging.error(f'Error loading model: {e}')
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/recommendations', methods=['POST'])
@limiter.limit("30 per hour")  # Stricter limit for recommendations
def get_recommendations():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        username = data.get('username')
        
        if not username:
            return jsonify({'error': 'Username is required'}), 400
        
        # Get user's anime list
        user_ratings = get_user_anime_list(username)
        if not user_ratings:
            return jsonify({'error': 'No rated anime found for user'}), 404
        
        # Create user rating vector
        user_vector = create_user_rating_vector(model.anime_mapping, user_ratings)
        
        # Get 100 recommendations
        recommendations = model.get_recommendations(user_vector, n=100)
        
        # Convert numpy types to native Python types
        recommendations = [(int(aid), float(score)) for aid, score in recommendations]
        
        # Get anime info in batch
        anime_ids = [anime_id for anime_id, _ in recommendations]
        anime_info_batch = model.get_anime_info(anime_ids)
        
        # Format recommendations
        formatted_recommendations = []
        for anime_id, score in recommendations:
            try:
                anime_info = anime_info_batch.get(anime_id)
                if anime_info:
                    mal_score = anime_info.get('mean')
                    # Convert MAL score to float if it exists
                    if mal_score is not None:
                        mal_score = float(mal_score)
                        
                    formatted_recommendations.append({
                        'title': anime_info['title'],
                        'predicted_score': round(score, 2),
                        'mal_score': mal_score if mal_score is not None else 'N/A',
                        'mal_id': anime_id,
                        'num_ratings': anime_info.get('num_ratings', 0)
                    })
            except Exception as e:
                logging.error(f'Error formatting recommendation for anime {anime_id}: {e}')
        
        return jsonify({
            'username': username,
            'recommendations': formatted_recommendations,
            'total_items': len(formatted_recommendations)
        })
        
    except Exception as e:
        logging.error(f'Error getting recommendations: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/similar', methods=['POST'])
@limiter.limit("60 per hour")  # Medium limit for similar anime requests
def get_similar():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        anime_id = data.get('anime_id')
        
        if not anime_id:
            return jsonify({'error': 'Anime ID is required'}), 400
        
        try:
            anime_id = int(anime_id)
        except ValueError:
            return jsonify({'error': 'Invalid anime ID'}), 400
        
        # Get similar anime
        similar_anime = model.get_similar_anime(anime_id, n=20)
        
        # Convert numpy types to native Python types
        similar_anime = [(int(aid), float(score)) for aid, score in similar_anime]
        
        # Get info for all anime at once
        anime_ids = [anime_id] + [aid for aid, _ in similar_anime]
        anime_info_batch = model.get_anime_info(anime_ids)
        
        # Format the response
        source_anime = anime_info_batch.get(anime_id)
        if not source_anime:
            return jsonify({'error': 'Anime not found'}), 404
            
        formatted_similar = []
        for similar_id, similarity in similar_anime:
            similar_info = anime_info_batch.get(similar_id)
            if similar_info:
                formatted_similar.append({
                    'title': similar_info['title'],
                    'similarity': round(similarity, 3),
                    'mal_score': similar_info.get('mean'),
                    'mal_id': similar_id,
                    'num_ratings': similar_info.get('num_ratings', 0)
                })
        
        return jsonify({
            'source_anime': {
                'title': source_anime['title'],
                'mal_id': anime_id,
                'mal_score': source_anime.get('mean'),
                'num_ratings': source_anime.get('num_ratings', 0)
            },
            'similar_anime': formatted_similar
        })
        
    except Exception as e:
        logging.error(f'Error finding similar anime: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/search_anime', methods=['GET'])
@limiter.limit("1000 per hour")  # More lenient limit for search queries
def search_anime():
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify([])
            
        # Connect to database
        conn = sqlite3.connect('anime_averages.db')
        try:
            cursor = conn.cursor()
            # Use LIKE with wildcards for substring matching
            cursor.execute("""
                SELECT anime_id, title, average_rating, num_ratings 
                FROM anime_averages 
                WHERE LOWER(title) LIKE '%' || LOWER(?) || '%'
                ORDER BY 
                    CASE 
                        WHEN LOWER(title) LIKE LOWER(?) || '%' THEN 1  -- Exact prefix match
                        WHEN LOWER(title) LIKE '% ' || LOWER(?) || '%' THEN 2  -- Word boundary match
                        ELSE 3  -- Substring match
                    END,
                    num_ratings DESC
                LIMIT 10
            """, (query, query, query))
            
            results = cursor.fetchall()
            return jsonify([{
                'id': row[0],
                'title': row[1],
                'rating': row[2],
                'num_ratings': row[3]
            } for row in results])
            
        finally:
            conn.close()
            
    except Exception as e:
        logging.error(f'Error searching anime: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)