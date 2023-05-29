import os
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

if os.path.exists('example_model.csv'):
    data = pd.read_csv('example_model.csv')
else:
    print("Nessun modello da caricare. Esegui prima il training.")
    exit()

data['title_tokens'] = data['title'].apply(simple_preprocess)
data['content_tokens'] = data['content'].apply(simple_preprocess)


def get_vector(words):
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


data['title_vector'] = data['title_tokens'].apply(get_vector)
data['content_vector'] = data['content_tokens'].apply(get_vector)

feedback_scores = {}


def ask_question(question):
    question_vector = get_vector(simple_preprocess(question))
    title_similarities = cosine_similarity([question_vector], np.stack(data['title_vector'].values))
    content_similarities = cosine_similarity([question_vector], np.stack(data['content_vector'].values))
    question_feedback_scores = np.array([feedback_scores.get((question, title, content), 0) for title, content in zip(data['title'], data['content'])])
    title_scores = title_similarities[0] + question_feedback_scores
    content_scores = content_similarities[0] + question_feedback_scores
    top_title_indices = np.argsort(title_scores)[-5:][::-1]
    top_content_indices = np.argsort(content_scores)[-5:][::-1]
    return top_title_indices, top_content_indices


def main():
    while True:
        question = input("Type your question here (type 'exit' to exit): ")
        if question.lower() == 'exit':
            break

        title_index, content_index = ask_question(question)
        print(f"Answer: {data.loc[content_index, 'content']}\n")


if __name__ == "__main__":
    main()
