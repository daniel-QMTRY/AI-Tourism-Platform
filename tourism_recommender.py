# Capstone Project: Indonesian Tourism Recommendation System

# =============================================================================
# 1. SETUP & IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import os

print("Libraries imported successfully.")

# =============================================================================
# 2. DATA LOADING & INITIAL INSPECTION (ROBUST METHOD)
# =============================================================================

# --- Create absolute paths to guarantee files are found ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# Define file paths
RATING_PATH = os.path.join(script_dir, 'data_sets', 'tourism_rating.csv')
# CORRECTED to point to the .xlsx file
TOURISM_PATH = os.path.join(script_dir, 'data_sets', 'tourism_with_id.xlsx') 
USER_PATH = os.path.join(script_dir, 'data_sets', 'user.csv')
# --- End of robust path creation ---

# Create a directory to save plots
plots_dir = os.path.join(script_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    
print("Loading datasets from the following paths:")
print(f"Ratings: {RATING_PATH}")
print(f"Tourism: {TOURISM_PATH}")
print(f"Users: {USER_PATH}")

try:
    ratings_df = pd.read_csv(RATING_PATH)
    # CORRECTED to use pd.read_excel for the .xlsx file
    tourism_df = pd.read_excel(TOURISM_PATH) 
    users_df = pd.read_csv(USER_PATH)
    print("\nDatasets loaded successfully.")
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: A file was not found. Please check the paths and filenames carefully.")
    print(e)
    exit()

# --- Initial Inspection ---
print("\n--- Initial Data Inspection ---")
print("\n[Ratings DataFrame]")
print(ratings_df.head())
print(ratings_df.info())

print("\n[Tourism DataFrame]")
print(tourism_df.head())
# Rename the unnamed column from excel import if it exists
if 'Unnamed: 0' in tourism_df.columns:
    tourism_df.rename(columns={'Unnamed: 0': 'Place_Id_Excel'}, inplace=True)
print(tourism_df.info())

print("\n[Users DataFrame]")
print(users_df.head())
print(users_df.info())


# =============================================================================
# 3. ETL (EXTRACT, TRANSFORM, LOAD) / DATA CLEANING
# =============================================================================
print("\n--- Starting Data Cleaning and Preprocessing ---")

# --- Tourism Data Cleaning ---
# Drop the extra unnamed column if it exists
if 'Unnamed: 11' in tourism_df.columns:
    tourism_df = tourism_df.drop('Unnamed: 11', axis=1)

# Fill missing Time_Minutes with the median value of the column
median_time = tourism_df['Time_Minutes'].median()
tourism_df['Time_Minutes'].fillna(median_time, inplace=True)
print(f"Filled missing 'Time_Minutes' with median value: {median_time}")

# --- Ratings Data Cleaning ---
# Check for duplicates
print(f"Number of duplicate ratings: {ratings_df.duplicated().sum()}")
ratings_df.drop_duplicates(inplace=True)
print("Duplicate ratings removed.")

# --- Merging DataFrames for EDA ---
# Merge ratings with tourism info
df = pd.merge(ratings_df, tourism_df, on='Place_Id')
# Merge the result with user info
df = pd.merge(df, users_df, on='User_Id')

print("\n--- Final Merged DataFrame for EDA ---")
print(df.head())
print(df.info())
print(f"Total rows in merged DataFrame: {len(df)}")


# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n--- Starting Exploratory Data Analysis ---")

# --- 4.1. User Demographics ---
# Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of User Ages', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.grid(axis='y', alpha=0.5)
plt.savefig(os.path.join(plots_dir, 'age_distribution.png'))
plt.show()
print("Saved plot: age_distribution.png")

# User Locations
top_10_locations = df['Location'].value_counts().nlargest(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_10_locations.values, y=top_10_locations.index, palette='viridis')
plt.title('Top 10 User Locations', fontsize=16)
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('Location', fontsize=12)
plt.savefig(os.path.join(plots_dir, 'top_user_locations.png'))
plt.show()
print("Saved plot: top_user_locations.png")


# --- 4.2. Tourism Spot Analysis ---
# Categories of Tourist Spots
plt.figure(figsize=(12, 7))
sns.countplot(y=df['Category'], order=df['Category'].value_counts().index, palette='crest')
plt.title('Number of Ratings per Tourism Category', fontsize=16)
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.savefig(os.path.join(plots_dir, 'ratings_per_category.png'))
plt.show()
print("Saved plot: ratings_per_category.png")

# Most Loved Tourist Spots (Top 10 by average rating)
# To be fair, only consider places with a significant number of ratings (e.g., > 15)
rating_counts = df.groupby('Place_Name')['Place_Ratings'].count()
places_with_enough_ratings = rating_counts[rating_counts > 15].index
df_filtered = df[df['Place_Name'].isin(places_with_enough_ratings)]

avg_ratings = df_filtered.groupby('Place_Name')['Place_Ratings'].mean().sort_values(ascending=False).nlargest(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=avg_ratings.values, y=avg_ratings.index, palette='magma')
plt.title('Top 10 Most Loved Tourist Spots (with >15 ratings)', fontsize=16)
plt.xlabel('Average Rating', fontsize=12)
plt.ylabel('Place Name', fontsize=12)
plt.xlim(4, 5)
plt.savefig(os.path.join(plots_dir, 'top_rated_places.png'))
plt.show()
print("Saved plot: top_rated_places.png")


# --- 4.3. Geospatial Analysis ---
# Which city has the most loved tourist spots?
city_avg_rating = df_filtered.groupby('City')['Place_Ratings'].mean().sort_values(ascending=False).nlargest(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=city_avg_rating.values, y=city_avg_rating.index, palette='plasma')
plt.title('Top 10 Cities with Highest Average Ratings', fontsize=16)
plt.xlabel('Average Rating', fontsize=12)
plt.ylabel('City', fontsize=12)
plt.savefig(os.path.join(plots_dir, 'top_rated_cities.png'))
plt.show()
print("Saved plot: top_rated_cities.png")


# Which city is best for a nature enthusiast?
nature_df = df[df['Category'] == 'Taman Hiburan'] # Assuming 'Taman Hiburan' (Amusement Park) is a proxy for Nature/Recreation
nature_city_counts = nature_df['City'].value_counts().nlargest(5)
print("\nTop cities for Nature Enthusiasts (based on 'Taman Hiburan' category):")
print(nature_city_counts)


# =============================================================================
# 5. RECOMMENDER SYSTEM - COLLABORATIVE FILTERING
# =============================================================================
print("\n--- Building the Recommendation System ---")

# Prepare data for Surprise library
# The reader object needs to know the rating scale
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['User_Id', 'Place_Id', 'Place_Ratings']], reader)

# Split data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
print("Data split into 80% training and 20% testing sets.")

# Use the SVD (Singular Value Decomposition) algorithm
model = SVD(n_factors=100, n_epochs=20, random_state=42)
print("Training the SVD model...")

# Train the algorithm on the trainset
model.fit(trainset)
print("Model training complete.")

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)
print(f"Model evaluation complete. RMSE: {rmse}")


# =============================================================================
# 6. GENERATE TOP N RECOMMENDATIONS
# =============================================================================

def get_top_n_recommendations(user_id, n=10):
    """
    Returns the top N recommendations for a given user.
    """
    # Get a list of all place IDs
    all_place_ids = ratings_df['Place_Id'].unique()
    
    # Get a list of place IDs the user has already rated
    rated_place_ids = ratings_df[ratings_df['User_Id'] == user_id]['Place_Id']
    
    # Get place IDs the user has NOT rated
    unrated_place_ids = [place_id for place_id in all_place_ids if place_id not in rated_place_ids]
    
    # Predict ratings for the unrated places
    testset_for_user = [[user_id, place_id, 4.] for place_id in unrated_place_ids]
    user_predictions = model.test(testset_for_user)
    
    # Sort predictions by estimated rating
    user_predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get the top N recommended place IDs
    top_n_predictions = user_predictions[:n]
    top_n_place_ids = [pred.iid for pred in top_n_predictions]
    
    # Get place names and categories from the tourism dataframe
    top_n_recommendations = tourism_df[tourism_df['Place_Id'].isin(top_n_place_ids)][['Place_Id', 'Place_Name', 'Category', 'City']]
    
    return top_n_recommendations


# --- Example Usage ---
TARGET_USER_ID = 100
print(f"\n--- Generating Top 10 Recommendations for User ID: {TARGET_USER_ID} ---")

# Get existing ratings for this user to see their preferences
user_100_ratings = df[df['User_Id'] == TARGET_USER_ID].sort_values(by='Place_Ratings', ascending=False)
print(f"\nUser {TARGET_USER_ID}'s Top 5 Rated Places:")
print(user_100_ratings[['Place_Name', 'Category', 'Place_Ratings']].head())

# Generate and display new recommendations
recommendations = get_top_n_recommendations(TARGET_USER_ID, n=10)
print(f"\nRecommended places for User {TARGET_USER_ID}:")
print(recommendations)

# =============================================================================
# 7. CONCLUSION
# =============================================================================
print("\n--- Project Conclusion ---")
print("This project successfully performed data cleaning, exploratory data analysis, and built a collaborative filtering recommendation system.")
print("Key insights from EDA include understanding user demographics and identifying popular tourism categories and locations.")
print(f"The SVD model achieved an RMSE of {rmse:.4f} on the test set, indicating a reasonably accurate prediction model.")
print("The recommendation function can now provide personalized tourism suggestions for any user in the dataset.")
print("\n--- END OF SCRIPT ---")

