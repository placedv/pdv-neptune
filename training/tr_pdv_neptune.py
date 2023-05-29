import gensim.downloader as api
import numpy as np
import json

wv = api.load('word2vec-google-news-300')


def save_vectors(vectors, filename):
    with open(filename, 'w') as f:
        for vec in vectors:
            f.write(json.dumps(vec.tolist()) + '\n')


def load_vectors(filename):
    with open(filename, 'r') as f:
        return [np.array(json.loads(line)) for line in f]


def save_feedback(feedback, filename):
    with open(filename, 'w') as f:
        # json.dump(feedback, f, default=lambda x: (list(x[0]), float(x[1])))
        json.dump(feedback, f, default=lambda x: (x[0].tolist(), float(x[1])))


def load_feedback(filename):
    try:
        with open(filename, 'r') as f:
            return {k: (np.array(v[0]), v[1]) for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {}


relevant_sentences = [
    "The Italian Republic is a parliamentary democracy located in southern Europe, with Rome as its capital city.",
    "Sergio Mattarella is the current President of the Italian Republic, serving in this role since 2015.",
    "Mattarella was born in Palermo, Sicily in 1941, and has had a long and distinguished career in Italian politics and law.",
    "Before becoming President, Mattarella served as a member of the Italian Parliament, as Minister of Education, and as a judge on the Italian Constitutional Court.",
    "Mattarella has been praised for his commitment to constitutional principles, his support for European integration, and his efforts to promote social justice and human rights.",
    "During his time as President, Mattarella has played an important role in Italian politics, often acting as a mediator and a unifying force in times of political crisis.",
    "Mattarella has also been an advocate for environmental sustainability, and has spoken out on issues such as climate change and protecting Italy's natural heritage.",
    "In recent years, the Italian Republic has faced a number of challenges, including political instability, economic uncertainty, and the ongoing COVID-19 pandemic.",
    "Despite these challenges, Mattarella and other Italian leaders have continued to work towards building a more stable and prosperous future for the country and its people.",
    "The Italian Republic was established on June 2, 1946, following a national referendum that abolished the monarchy and declared Italy a republic.",
    "The Italian Constitution, which was adopted in 1947, outlines the principles and structure of the Italian Republic and remains a fundamental document in Italian law and politics.",
    "Italy is a member of the European Union, the United Nations, and many other international organizations, and plays an important role in global diplomacy and cooperation.",
    "Sergio Mattarella's presidency has been marked by several significant events, including the terrorist attacks in Paris in 2015, the refugee crisis in Europe, and the COVID-19 pandemic.",
    "Mattarella has been a strong supporter of efforts to strengthen the European Union and promote greater cooperation among member states.",
    "Italy is known for its rich cultural heritage, including art, literature, cuisine, and fashion, and attracts millions of tourists from around the world each year.",
    "The Italian economy is the third largest in the Eurozone, and is known for its strengths in industries such as fashion, automotive manufacturing, and tourism.",
    "Italy is also home to many renowned universities and research institutions, and has made significant contributions to fields such as science, technology, and medicine.",
    "The Italian political system is characterized by a multi-party system and a strong tradition of regionalism and local autonomy.",
    "Sergio Mattarella's term as President is set to end in 2022, at which point a new President will be elected by the Italian Parliament.",
]

feedback_file = 'feedback.json'

try:
    sentence_vectors = load_vectors('sentence_vectors.json')
    feedback = load_feedback(feedback_file)
except:
    sentence_vectors = []
    feedback = {}

for i, sentence in enumerate(relevant_sentences):
    if sentence not in feedback:
        tokens = sentence.lower().split()
        vector = sum(wv[token] for token in tokens if token in wv.key_to_index) / len(tokens)
        sentence_vectors.append(vector)
        feedback[sentence] = (vector, None)

save_vectors(sentence_vectors, 'sentence_vectors.json')
save_feedback(feedback, feedback_file)


def find_most_similar(question):
    question_tokens = question.lower().split()
    similarities = [wv.wmdistance(question_tokens, ' '.join(map(str, sentence)).lower().split()) for sentence, _ in
                    feedback.values()]

    most_similar_index = min(range(len(similarities)), key=similarities.__getitem__)
    most_similar = list(feedback.keys())[most_similar_index]
    return most_similar, feedback[most_similar][0]


questions = [
    "What kind of government does Italy have?",
    "Who is the President of Italy?",
    "What has Mattarella done for Italy?",
    "What are some of the challenges facing Italy?",
    "How have Italian leaders responded to these challenges?",
    "Where is born Mattarella?",
]

while True:
    question = input("Type your question here: ")
    answer, vector = find_most_similar(question)
    print(f"Answer: {answer}\n")

    if tuple(vector.tolist()) not in feedback:
        while True:
            feedback_str = input("Was this answer helpful? (yes/no): ")
            if feedback_str.lower() == 'yes':
                feedback[answer] = (vector, 1)
                break
            elif feedback_str.lower() == 'no':
                feedback[answer] = (vector, 0)
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        save_feedback(feedback, feedback_file)

    save_feedback(feedback, feedback_file)
