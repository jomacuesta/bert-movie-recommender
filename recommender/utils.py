import numpy as np
import ast


class MovieRecommender:
    def __init__(self, model, database, embedding_col="bert_embeddings"):
        self.model = model
        self.database = database.copy()
        self.database[embedding_col] = self.database[embedding_col].apply(parse_embedding)
        self.database_emb = np.stack(self.database[embedding_col].values)

    def get_recommendations(self, new_overview, top_n=5, order_by_vote: bool = False):

        new_overview_emb = self.model.encode(new_overview)
        similarities = calculate_similarity(self.database_emb, new_overview_emb)
        top_indices = similarities.argsort()[-top_n:][::-1]
        recommendations = self.database.iloc[top_indices][["original_title", "overview", "vote_average"]]

        if order_by_vote:
            recommendations = recommendations.sort_values(by="vote_average", ascending=False)

        return recommendations


def get_genres(genre_str):
    """
    Get the genres from the dataframe and return a list of unique genres.
    """
    genres = ast.literal_eval(genre_str)
    return [genre['name'] for genre in genres]


def get_model_embeddings(df, model, output_col, description_col='overview', **kwargs):
    """
    Get embeddings for the movie overviews.
    """
    texts = df[description_col].fillna('').tolist()
    embeddings = model.encode(texts, **kwargs)
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
    df[output_col] = embeddings
    return df


def calculate_similarity(embeddings, target_embedding):
    """
    Calculate the cosine similarity between the target embedding and all other embeddings.
    """
    similarities = np.dot(embeddings, target_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_embedding))
    return similarities


def calculate_similarity_matrix(embeddings):
    """
    Calculate the cosine similarity matrix between all embeddings.
    :param embeddings: NumPy array of shape (n, d), where n is the number of embeddings and d is the dimension.
    :return: Similarity matrix of shape (n, n).

    """
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    return similarity_matrix


def parse_embedding(embedding_str):
    embedding_list = ast.literal_eval(embedding_str)
    return np.array(embedding_list, dtype=np.float32)
