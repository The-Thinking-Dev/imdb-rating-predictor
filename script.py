import pandas as pd
from sklearn.linear_model import LinearRegression

def load_dataset(filepath='movies.csv'):
    """
    Load the top 100 films dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print("Columns in dataset:", df.columns.tolist())  # Debug line
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

def display_films(df):
    """
    Print the list of films with their indices and IMDb ratings.
    """
    print("Top 100 Films:")
    for idx, row in df.iterrows():
        print(f"{idx}: {row['Title']} (Rating: {row['Rating']})")
    print("\nPlease rate at least 5 films from the list (ratings should be between 0 and 5).")

def get_user_ratings(df, min_ratings=5):
    """
    Prompt the user to rate films until at least min_ratings have been provided.
    Returns a dictionary mapping film indices to user ratings.
    """
    user_ratings = {}
    while len(user_ratings) < min_ratings:
        try:
            film_index = int(input("Enter film index: "))
            if film_index < 0 or film_index >= len(df):
                print("Invalid index, try again.")
                continue
            if film_index in user_ratings:
                print("You have already rated that film.")
                continue

            rating = float(input("Enter your rating (0-5): "))
            if rating < 0 or rating > 5:
                print("Rating must be between 0 and 5. Try again.")
                continue

            user_ratings[film_index] = rating
        except ValueError:
            print("Invalid input, please try again.")
    return user_ratings

def train_rating_model(df, user_ratings):
    """
    Train a Linear Regression model using IMDb_Rating as feature and user ratings as the target.
    """
    rated_indices = list(user_ratings.keys())
    rated_df = df.loc[rated_indices].copy()
    # Map the user ratings to a new column in the DataFrame
    rated_df['User_Rating'] = rated_df.index.map(user_ratings)
    
    X_train = rated_df[['Rating']]
    y_train = rated_df['User_Rating']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_unrated_films(df, user_ratings, model):
    """
    Use the trained model to predict ratings for films not yet rated by the user.
    Returns a DataFrame of unrated films with predicted ratings.
    """
    # Drop films that the user has rated
    unrated_df = df.drop(user_ratings.keys())
    
    # Drop any rows that have missing values in the 'Rating' column
    unrated_df = unrated_df.dropna(subset=['Rating'])
    
    X_unrated = unrated_df[['Rating']]
    predictions = model.predict(X_unrated)
    
    # Create a copy and add the predictions column
    unrated_df = unrated_df.copy()
    unrated_df['Predicted_User_Rating'] = predictions
    return unrated_df

def main():
    df = load_dataset()
    display_films(df)
    
    user_ratings = get_user_ratings(df)
    model = train_rating_model(df, user_ratings)
    
    unrated_df = predict_unrated_films(df, user_ratings, model)
    print("\nPredicted ratings for films you haven't rated:")
    for idx, row in unrated_df.iterrows():
        print(f"{row['Title']}: Predicted Rating: {row['Predicted_User_Rating']:.2f}")

if __name__ == "__main__":
    main()