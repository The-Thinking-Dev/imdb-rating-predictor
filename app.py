from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

# Load the dataset (ensure movies.csv is in your project folder)
df = pd.read_csv('movies.csv')

def train_rating_model(user_ratings):
    rated_indices = list(user_ratings.keys())
    rated_df = df.loc[rated_indices].copy()
    rated_df['User_Rating'] = rated_df.index.map(user_ratings)
    X_train = rated_df[['Rating']]
    y_train = rated_df['User_Rating']
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_unrated_films(user_ratings, model):
    unrated_df = df.drop(user_ratings.keys())
    # Drop rows with missing 'Rating' values
    unrated_df = unrated_df.dropna(subset=['Rating'])
    X_unrated = unrated_df[['Rating']]
    predictions = model.predict(X_unrated)
    unrated_df = unrated_df.copy()
    unrated_df['Predicted_User_Rating'] = predictions
    return unrated_df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_ratings = {}
        # For each film in the dataset, check if the form has a rating value.
        for idx in df.index:
            field_name = f"rating_{idx}"
            rating_value = request.form.get(field_name)
            if rating_value and rating_value != "":
                try:
                    rating = float(rating_value)
                    if 0 <= rating <= 5:
                        user_ratings[idx] = rating
                except ValueError:
                    continue
        if len(user_ratings) < 5:
            flash("Please rate at least 5 films.")
            return redirect(url_for('index'))
        model = train_rating_model(user_ratings)
        predictions_df = predict_unrated_films(user_ratings, model)
        predictions = predictions_df[['Title', 'Predicted_User_Rating']].to_dict(orient='records')
        return render_template('predictions.html', predictions=predictions)
    else:
        # Prepare a list of films for the form
        films_list = []
        for idx, row in df.iterrows():
            films_list.append({
                'index': idx,
                'title': row['Title'],
                'existing_rating': row['Rating']
            })
        return render_template('index.html', films=films_list)

if __name__ == '__main__':
    app.run(debug=True)