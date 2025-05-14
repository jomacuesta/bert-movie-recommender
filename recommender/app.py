from flask import Flask, request, render_template
import pandas as pd
from models.model import SentenceTransformerRecommender
from recommender.utils import MovieRecommender
import os

app = Flask(__name__)

# Initialize environment
path_file = os.path.join(os.path.dirname(__file__), '../data/clean_data/movies_metadata_embeddings.csv')
df = pd.read_csv(path_file)
st_model = SentenceTransformerRecommender()
recommender = MovieRecommender(st_model, df, "st_embeddings")


# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        description = request.form.get('description', '').strip()
        top_n = int(request.form.get('top_n', 3))
        if description.strip():
            results = recommender.get_recommendations(description, top_n=top_n, order_by_vote=True).to_dict(orient='records')
    return render_template('index.html', results=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


