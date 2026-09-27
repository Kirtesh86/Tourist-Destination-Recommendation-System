# Tourist Destination Recommendation System Using KNN

A small, beginner-friendly machine learning project that recommends Indian destinations from a traveler's budget, trip duration, and interests. The interface is built with Streamlit; recommendations use scikit-learn's K-Nearest Neighbors search.

The app uses curated Unsplash photos and Google Fonts for its visual styling. These are optional: the recommendation interface remains usable without them, with local color and system-font fallbacks.

## Project objective

Help a traveler explore five Indian destinations whose sample trip profiles are closest to their preferences. This is a similarity-based recommendation demo, not a live travel booking or price service.

## Dataset

`tourist_destinations.csv` contains 35 Indian destinations. Each row has a sample estimated trip budget (in rupees), suggested number of days, project-assigned interest scores from 1 to 10, and state/region labels for filtering. Scores are illustrative inputs for the model, not official ratings; budgets are not live quotes.

## Features

- `Destination`: place name
- `Budget`: estimated trip budget in INR
- `Days`: suggested trip duration
- `Nature`: nature and scenery interest score
- `Beach`: beach interest score
- `Adventure`: adventure interest score
- `Culture`: culture and history interest score
- `State` and `Region`: location metadata used by the region filter and result cards; these are not KNN model features

The app asks for budget, days, and four interest ratings (each 1–10).

## KNN algorithm

K-Nearest Neighbors (KNN) finds the destination profiles closest to the traveler in feature space. This project uses `NearestNeighbors` with Euclidean distance. It is a similarity search, rather than a model that predicts a destination label.

### How recommendations work

1. Load the CSV with Pandas and select `Budget`, `Days`, and the four interest scores.
2. Fit `StandardScaler` to destination features and transform them. This prevents the larger rupee values from dominating distance calculations.
3. Convert the submitted user preferences to the same feature order and transform them with that already-fitted scaler.
4. Optionally limit the candidates to a selected region. KNN still uses the scaler fitted on the full dataset, so regional results use the same feature scale.
5. Query KNN for up to five nearest destination rows in the chosen region.
6. Show each place, its state, sample budget, suggested days, and Euclidean distance. Expanded details compare each user interest score with that destination's score. A smaller KNN distance indicates a closer overall match.

## Install dependencies

Use Python 3.12 (the current Streamlit Community Cloud default) or another version supported by the pinned packages. From the project directory, create and activate a virtual environment if desired, then run:

```bash
python -m pip install -r requirements.txt
```

## Run the project

From the directory containing `app.py` and the CSV:

```bash
streamlit run app.py
```

Streamlit opens the app in your browser. Enter your preferences and select **Find my destinations**.

## Deploy on Streamlit Community Cloud

The app is ready to deploy from GitHub. Include `app.py`, `tourist_destinations.csv`, `requirements.txt`, and `.streamlit/config.toml` in the repository. The `.gitignore` keeps the local rollback copy, Python cache, and any Streamlit secrets out of Git.

1. Push this project to a GitHub repository that you have admin access to.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Select **Create app**, choose the repository and branch, and set the app file to `app.py`.
4. Deploy. Community Cloud installs the pinned packages from `requirements.txt`.

This project has no API keys or secrets. The destination dataset and app source will be visible to anyone with access to the GitHub repository. Community Cloud supports public repositories by default; private repositories require granting it additional GitHub access. See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) for current instructions.

## Project files

```text
app.py
tourist_destinations.csv
requirements.txt
README.md
.streamlit/config.toml
.gitignore
```
