import tensorflow_hub as hub
from flask import Flask, request, jsonify
from scipy.spatial.distance import cosine

app = Flask(__name__)

# Загрузите Universal Sentence Encoder
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")


def get_most_similar_class(phrase, classes):
    phrase_embedding = model([phrase])
    max_similarity = 0
    most_similar_class = None
    most_similar_var = None

    for _class in classes:
        for var in _class['vars'].split('; '):
            var_embedding = model([var])
            similarity = 1 - cosine(phrase_embedding, var_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_class = _class['class']
                most_similar_var = var

    return most_similar_class, most_similar_var, max_similarity


@app.route('/fastClassify', methods=['POST'])
def fast_classify():
    data = request.get_json()
    phrase = data['phrase']
    classes = data['data']

    class_id, similar_var, score = get_most_similar_class(phrase, classes)

    # Проверка уверенности и возвращение default класса при необходимости
    if score < 0.35:
        for _class in classes:
            if _class['class'] == "default":
                class_id = _class['class']
                similar_var = "default"
                break

    response = {
        "class": class_id,
        "phrase": phrase,
        "score": score,
        "var": similar_var
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5016, ssl_context='adhoc')  # Включите ssl_context для использования HTTPS
