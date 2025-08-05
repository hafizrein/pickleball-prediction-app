# Pickleball Match Predictor

A machine learning web app that predicts pickleball match outcomes based on player stats and match conditions.

## What it does

This app uses a trained logistic regression model to predict who will win a pickleball match. You select two players, set the match conditions (court type, weather, duration), and get a prediction with confidence score.

The model was trained on approximately 2,000 matches and considers:
- Player rankings and win rates
- Court type (indoor vs outdoor)
- Weather conditions  
- Expected match duration

## Running locally

You'll need Python 3.12+ installed.

### With uv (strongly recommended, I'm begging you to try it. It's life changing T_T)

Seriously, if you haven't tried [uv](https://docs.astral.sh/uv/) yet, you're missing out. It's absurdly fast and will change how you think about Python package management.


```bash
# Install uv if you don't have it (It's okay to install it on your base Python installation, trust me....)
pip install uv

# Option 1: Run directly (handles everything automatically)
uv sync
uv run app.py

# Option 2: Traditional approach
uv sync
python app.py
```


### With pip (if you must)


```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Unix/macOS:
source .venv/bin/activate
# Windows (Command Prompt):
# .venv\Scripts\activate.bat
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Install dependencies (much slower)
pip install -r requirements.txt

# Start the server
python app.py
```


Open http://localhost:7860 in your browser.

## Project structure

```
├── src/pickleball/
│   └── main.py              # FastAPI backend
├── data/
│   ├── pickleball_matches.csv
│   └── pickleball_model.joblib
├── notebooks/               # Data exploration and model training
├── static/
│   └── index.html          # Frontend
└── app.py                  # Entry point
```


## API endpoints

The backend provides a REST API:

- `GET /` - Web interface
- `POST /predict/players` - Get match prediction
- `GET /players` - List all players
- `GET /players/{id}` - Get player details
- `GET /health` - Health check

Full API docs available at `/docs` when running.

## Model performance

The logistic regression model achieves ~85% accuracy on the test set, which is solid performance for this type of sports prediction task.

## Dataset

The training dataset (`data/pickleball_matches.csv`) contains approximately 2,000 pickleball matches with the following features:

**Player Features:**
- Player rankings (p1_rank, p2_rank)
- Historical win rates (p1_win_rate, p2_win_rate)
- Player IDs for tracking individual performance

**Match Conditions:**
- Court type: Indoor vs Outdoor
- Weather conditions: Sunny, Cloudy, Windy, or N/A (for indoor)
- Expected match duration in minutes
- Match outcome (winner: Player 1 or Player 2)

**Feature Engineering:**
- Rank difference (p1_rank - p2_rank)
- Win rate difference (p1_win_rate - p2_win_rate)
- Categorical encoding for court type and weather
- Match duration normalization

The dataset is synthetic data generated in `notebooks/pickleball_dataset.ipynb` to simulate realistic match scenarios with varied player skill levels and conditions.

## Model Training

The machine learning pipeline is implemented in `notebooks/model_building.ipynb` and follows these steps:

**1. Data Preprocessing:**
- Handle missing values (some players lack complete historical data)
- Encode categorical variables (court_type, weather)
- Feature scaling and normalization
- Train/validation/test split (70/15/15)

**2. Model Selection:**
- Tested multiple algorithms: Logistic Regression, Random Forest, XGBoost
- Logistic Regression performed best with cross-validation
- Hyperparameter tuning using GridSearchCV

**3. Feature Importance:**
- Player rank difference: Most predictive feature
- Win rate difference: Second most important
- Court type: Moderate impact (some players prefer indoor/outdoor)
- Weather conditions: Minor but measurable effect

**4. Model Evaluation:**

```
Accuracy: 85.2%
Precision: 0.87 (Player 1), 0.83 (Player 2)  
Recall: 0.84 (Player 1), 0.86 (Player 2)
F1-Score: 0.85 (Player 1), 0.84 (Player 2)
```


**5. Model Persistence:**
The trained model is saved as `data/pickleball_model.joblib` for fast loading in production.

**Reproducing Results:**

```bash
# Generate new synthetic dataset (optional)
jupyter notebook notebooks/pickleball_dataset.ipynb

# Train the model from scratch
jupyter notebook notebooks/model_building.ipynb
```


## Deployment


### Other platforms
The app should work on most Python hosting platforms (Railway, Render, etc.) since it's just a standard FastAPI app with minimal dependencies.

## Technical details

**Backend**: FastAPI with scikit-learn for ML inference  
**Frontend**: Vanilla HTML/CSS/JavaScript  
**Model**: Logistic regression trained with scikit-learn  
**Data**: CSV files, model stored as joblib pickle  

## Future improvements

- More training data would definitely help accuracy
- Could add more sophisticated features (head-to-head records, recent form, etc.)
- Real-time match tracking would be cool
- Mobile-responsive design needs work


## Architecture

Pretty straightforward setup - the frontend calls the FastAPI backend, which loads the trained model and returns predictions. Player data and match history are stored in CSV files for simplicity.