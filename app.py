import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, Response
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        df = pd.read_csv(
            "D:\Programming\MachineLearning\Music_Recom\static\SpotifyFeatures.csv")
        df = df.reset_index()
        df = df[:20000]

        selected_features = ['genre',  'artist_name', 'mode']

        for feature in selected_features:
            df[feature] = df[feature].fillna('')

        combined_features = df['genre']+' '+df['artist_name']+' '+df['mode'] 

        vectorizer = TfidfVectorizer()

        feature_vectors = vectorizer.fit_transform(combined_features)

        similarity = cosine_similarity(feature_vectors)
        # print(similarity.shape)

        Song_name = request.form['key']

        list_of_all_titles = df['track_name'].tolist()

        find_close_match = difflib.get_close_matches(
            Song_name, list_of_all_titles)

        close_match = find_close_match[0]

        index_of_the_song = df[df.track_name == close_match]['index'].values[0]

        # artist_of_the_song = df[df.track_name == close_match]['artist_name'].values[0]
        
        similarity_score = list(enumerate(similarity[index_of_the_song]))
        sorted_similar_song = sorted(
            similarity_score, key=lambda x: x[1], reverse=True)

        i = 1
        recom_song=[]

        for song in sorted_similar_song:
            index = song[0]
            title_from_index = df[df.index == index]['track_name']
            # artist_from_index = df[df.index==index]['artist_name']
            if (i < 16):
                i_value = list(title_from_index)
                i += 1
                recom_song += i_value

        return render_template('index.html',recom_song=recom_song, close_match=close_match,
         Song_name=Song_name)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
