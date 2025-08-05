from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional

# Load the trained model
model = joblib.load("data/pickleball_model.joblib")

# Load player data for lookups
player_data = pd.read_csv("data/pickleball_matches.csv")

# Create player stats lookup dictionaries
def create_player_lookups():
    player_stats = {}
    
    # Get stats for each player from both p1 and p2 positions
    for _, row in player_data.iterrows():
        # Player 1 stats
        if row['p1_id'] not in player_stats:
            player_stats[row['p1_id']] = {
                'rank': row['p1_rank'], 
                'win_rate': row['p1_win_rate']
            }
        
        # Player 2 stats
        if row['p2_id'] not in player_stats:
            player_stats[row['p2_id']] = {
                'rank': row['p2_rank'],
                'win_rate': row['p2_win_rate']
            }
    
    return player_stats

player_stats = create_player_lookups()

app = FastAPI(title="Pickleball Model API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: List[float]
    
class BatchPredictionRequest(BaseModel):
    features: List[List[float]]

class PlayerPredictionRequest(BaseModel):
    p1_id: str  # Player 1 ID (e.g., "P1001")
    p2_id: str  # Player 2 ID (e.g., "P1002") 
    court_type: str = "Outdoor"  # "Indoor" or "Outdoor"
    weather: str = "Sunny"  # "Sunny", "Windy", "Cloudy", or "N/A"
    match_duration_minutes: Optional[int] = 50  # Expected match duration

class PredictionResponse(BaseModel):
    prediction: float  # 0 = Player 1 wins, 1 = Player 2 wins
    probability: Optional[float] = None
    winner: str  # Human-readable winner
    
class BatchPredictionResponse(BaseModel):
    predictions: List[float]  # 0 = Player 1 wins, 1 = Player 2 wins
    probabilities: Optional[List[float]] = None
    winners: List[str]  # Human-readable winners

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

@app.get("/")
async def root():
    """Serve the main HTML interface"""
    return FileResponse('static/index.html')

@app.get("/api", response_model=dict)
async def api_root():
    """API root endpoint"""
    return {"message": "Pickleball Model API", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction - 0 = Player 1 wins, 1 = Player 2 wins"""
    try:
        # Convert to numpy array and reshape for prediction
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Determine winner
        winner = "Player 1" if prediction == 0 else "Player 2"
        
        # Try to get probability if model supports it
        probability = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(features)[0]
                probability = float(max(proba))
            except:
                pass
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=probability,
            winner=winner
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions - 0 = Player 1 wins, 1 = Player 2 wins"""
    try:
        # Convert to numpy array
        features = np.array(request.features)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Determine winners
        winners = ["Player 1" if pred == 0 else "Player 2" for pred in predictions]
        
        # Try to get probabilities if model supports it
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(features)
                probabilities = [float(max(proba)) for proba in probas]
            except:
                pass
        
        return BatchPredictionResponse(
            predictions=[float(pred) for pred in predictions],
            probabilities=probabilities,
            winners=winners
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/players", response_model=PredictionResponse)
async def predict_by_player_ids(request: PlayerPredictionRequest):
    """Make prediction using player IDs - looks up player stats automatically"""
    try:
        # Look up player stats
        if request.p1_id not in player_stats:
            raise HTTPException(status_code=404, detail=f"Player {request.p1_id} not found")
        if request.p2_id not in player_stats:
            raise HTTPException(status_code=404, detail=f"Player {request.p2_id} not found")
        
        p1_stats = player_stats[request.p1_id]
        p2_stats = player_stats[request.p2_id]
        
        # Build feature vector
        features = [
            p1_stats['rank'],           # p1_rank
            p2_stats['rank'],           # p2_rank  
            p1_stats['win_rate'],       # p1_win_rate
            p2_stats['win_rate'],       # p2_win_rate
            request.match_duration_minutes,  # match_duration_minutes
            1 if request.court_type == "Indoor" else 0,   # court_Indoor
            1 if request.court_type == "Outdoor" else 0,  # court_Outdoor
            1 if request.weather == "Sunny" else 0,       # weather_Sunny
            1 if request.weather == "Windy" else 0,       # weather_Windy
            1 if request.weather == "Cloudy" else 0,      # weather_Cloudy
        ]
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        winner = "Player 1" if prediction == 0 else "Player 2"
        
        # Get probability
        probability = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(features_array)[0]
                probability = float(max(proba))
            except:
                pass
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=probability,
            winner=winner
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Player prediction error: {str(e)}")

@app.get("/players")
async def get_players():
    """Get list of all available players"""
    return {
        "total_players": len(player_stats),
        "players": list(player_stats.keys()),
        "sample_player_stats": {
            player_id: {
                "rank": int(stats['rank']) if not pd.isna(stats['rank']) else None,
                "win_rate": float(stats['win_rate']) if not pd.isna(stats['win_rate']) else None
            } for player_id, stats in list(player_stats.items())[:5]
        }
    }

@app.get("/players/{player_id}")
async def get_player_stats(player_id: str):
    """Get detailed stats for a specific player"""
    if player_id not in player_stats:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    
    # Get all matches for this player
    player_matches = player_data[
        (player_data['p1_id'] == player_id) | (player_data['p2_id'] == player_id)
    ]
    
    wins = 0
    total_matches = len(player_matches)
    
    for _, match in player_matches.iterrows():
        if match['p1_id'] == player_id and match['winner'] == 'Player 1':
            wins += 1
        elif match['p2_id'] == player_id and match['winner'] == 'Player 2':
            wins += 1
    
    actual_win_rate = wins / total_matches if total_matches > 0 else 0.0
    
    # Handle NaN values to prevent JSON serialization errors
    rank = player_stats[player_id]['rank']
    win_rate = player_stats[player_id]['win_rate']
    
    return {
        "player_id": player_id,
        "rank": int(rank) if not pd.isna(rank) else None,
        "win_rate": float(win_rate) if not pd.isna(win_rate) else None,
        "total_matches": total_matches,
        "wins": wins,
        "losses": total_matches - wins,
        "actual_win_rate": float(actual_win_rate),
        "recent_matches": player_matches.head(5).fillna("N/A").to_dict('records')
    }

@app.get("/court-types")
async def get_court_types():
    """Get all available court types and their usage statistics"""
    court_stats = player_data['court_type'].value_counts().to_dict()
    return {
        "available_court_types": list(court_stats.keys()),
        "usage_statistics": court_stats,
        "total_matches": len(player_data)
    }

@app.get("/weather-conditions")
async def get_weather_conditions():
    """Get all available weather conditions and their usage statistics"""
    weather_stats = player_data['weather'].value_counts().to_dict()
    return {
        "available_weather_conditions": list(weather_stats.keys()),
        "usage_statistics": weather_stats,
        "total_matches": len(player_data)
    }

@app.get("/match-duration-stats")
async def get_match_duration_stats():
    """Get match duration statistics"""
    durations = player_data['match_duration_minutes']
    return {
        "min_duration": int(durations.min()),
        "max_duration": int(durations.max()),
        "average_duration": float(durations.mean()),
        "median_duration": float(durations.median()),
        "std_duration": float(durations.std()),
        "duration_distribution": {
            "under_30_min": len(durations[durations < 30]),
            "30_45_min": len(durations[(durations >= 30) & (durations < 45)]),
            "45_60_min": len(durations[(durations >= 45) & (durations < 60)]),
            "over_60_min": len(durations[durations >= 60])
        }
    }

@app.get("/rankings")
async def get_ranking_stats():
    """Get ranking statistics for all players"""
    all_ranks = []
    for stats in player_stats.values():
        all_ranks.append(stats['rank'])
    
    ranks_series = pd.Series(all_ranks)
    
    # Top and bottom players
    sorted_players = sorted(player_stats.items(), key=lambda x: x[1]['rank'])
    
    return {
        "total_players": len(player_stats),
        "best_rank": int(ranks_series.min()),
        "worst_rank": int(ranks_series.max()),
        "average_rank": float(ranks_series.mean()),
        "median_rank": float(ranks_series.median()),
        "top_5_players": [
            {"player_id": pid, "rank": stats['rank'], "win_rate": stats['win_rate']} 
            for pid, stats in sorted_players[:5]
        ],
        "bottom_5_players": [
            {"player_id": pid, "rank": stats['rank'], "win_rate": stats['win_rate']} 
            for pid, stats in sorted_players[-5:]
        ]
    }

@app.get("/win-rates")
async def get_win_rate_stats():
    """Get win rate statistics for all players"""
    all_win_rates = []
    for stats in player_stats.values():
        all_win_rates.append(stats['win_rate'])
    
    win_rates_series = pd.Series(all_win_rates)
    
    # Sort players by win rate
    sorted_by_winrate = sorted(player_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
    
    return {
        "total_players": len(player_stats),
        "highest_win_rate": float(win_rates_series.max()),
        "lowest_win_rate": float(win_rates_series.min()),
        "average_win_rate": float(win_rates_series.mean()),
        "median_win_rate": float(win_rates_series.median()),
        "win_rate_distribution": {
            "under_30%": len(win_rates_series[win_rates_series < 0.3]),
            "30-50%": len(win_rates_series[(win_rates_series >= 0.3) & (win_rates_series < 0.5)]),
            "50-70%": len(win_rates_series[(win_rates_series >= 0.5) & (win_rates_series < 0.7)]),
            "over_70%": len(win_rates_series[win_rates_series >= 0.7])
        },
        "top_5_by_winrate": [
            {"player_id": pid, "rank": stats['rank'], "win_rate": stats['win_rate']} 
            for pid, stats in sorted_by_winrate[:5]
        ],
        "bottom_5_by_winrate": [
            {"player_id": pid, "rank": stats['rank'], "win_rate": stats['win_rate']} 
            for pid, stats in sorted_by_winrate[-5:]
        ]
    }

@app.get("/dataset-summary")
async def get_dataset_summary():
    """Get complete dataset summary"""
    return {
        "total_matches": len(player_data),
        "total_players": len(player_stats),
        "court_types": player_data['court_type'].value_counts().to_dict(),
        "weather_conditions": player_data['weather'].value_counts().to_dict(),
        "match_duration": {
            "min": int(player_data['match_duration_minutes'].min()),
            "max": int(player_data['match_duration_minutes'].max()),
            "avg": float(player_data['match_duration_minutes'].mean())
        },
        "winner_distribution": player_data['winner'].value_counts().to_dict(),
        "features_for_model": [
            "p1_rank", "p2_rank", "p1_win_rate", "p2_win_rate", 
            "match_duration_minutes", "court_Indoor", "court_Outdoor",
            "weather_Sunny", "weather_Windy", "weather_Cloudy"
        ]
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    try:
        info = {
            "model_type": type(model).__name__,
            "has_predict": hasattr(model, 'predict'),
            "has_predict_proba": hasattr(model, 'predict_proba'),
        }
        
        # Try to get feature count
        if hasattr(model, 'n_features_in_'):
            info["n_features"] = model.n_features_in_
        elif hasattr(model, 'feature_count_'):
            info["n_features"] = model.feature_count_
            
        return info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
