# MAL Anime Recommender

A machine learning-based anime recommendation system that uses MyAnimeList (MAL) data to provide personalized recommendations and find similar anime.
This was written using [Windsurf](https://windsurf.ai/), I wrote almost none of the code and Cascade generated the rest. Give it a try today for free!
The ratings database is available [here](https://storage.cloud.google.com/anime_rating_db/mal_users.db), consisting of approximately 550,000 users and 90 million ratings, which was used to train a recommendation model. I was able to train the model in under 10 minutes on an 8 core 16 GB RAM laptop.

## Features

- **Personal Recommendations**: Get personalized anime recommendations based on your MyAnimeList ratings
- **Similar Anime Search**: Find anime similar to your favorites using collaborative filtering
- **Smart Search**: Search for anime with intelligent autocomplete and fuzzy matching

## How It Works

### Personal Recommendations

1. Enter your MyAnimeList username
2. The system fetches your anime ratings from MAL
3. Using collaborative filtering with Alternating Least Squares (ALS), it predicts scores for anime you haven't watched
4. Returns the top recommendations sorted by predicted score

### Similar Anime

1. Search for an anime using the autocomplete search
2. The system uses the latent factors learned during model training to find anime with similar patterns in user ratings
3. Returns similar anime ranked by cosine similarity

## Technical Details

### Components

- `main.py`: Flask web application and API endpoints
- `get_recommendations.py`: Core recommendation logic and model inference
- `recommendation_pipeline.py`: Model training pipeline using PySpark ALS

### Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, JavaScript, Bootstrap
- **Machine Learning**: PySpark ML (ALS)
- **Database**: SQLite
- **APIs**: MyAnimeList API v2

## Setup

1. Install Python requirements:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your MyAnimeList API key:
```bash
MAL_API_KEY=your_api_key_here
```
   - You can obtain an API key from the [MyAnimeList API](https://myanimelist.net/apiconfig) page

3. Train the model:
```bash
python train_model.py
```

4. Run the web application:
```bash
python main.py
```

The application will be available at `http://localhost:5000`

## Configuration

- Model parameters can be adjusted in `train_model.py`:
  - Number of latent factors
  - Number of iterations
  - Regularization parameter
  - Minimum ratings threshold

## API Endpoints

- `POST /api/recommendations`: Get personalized recommendations
- `POST /api/similar`: Find similar anime
- `GET /api/search_anime`: Search anime with autocomplete

## License

MIT License