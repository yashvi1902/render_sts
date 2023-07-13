from flask import Flask, request, jsonify
from model_files.model_code import get_embeddings,get_similarity_score


app = Flask('app')

@app.route('/test', methods=['GET'])
def test():
    return 'Test!'

@app.route('/predict', methods=['POST'])
def predict():
    js_data = request.get_json()
    para_1= js_data["text1"]
    para_2 =js_data["text2"]
    embed_1 = get_embeddings(para_1)
    embed_2 = get_embeddings(para_2)
    embed_1= embed_1.reshape([1,768])
    embed_2 = embed_2.reshape([1,768])
    score = get_similarity_score(embed_1,embed_2)
    score = 0.5+0.5*float(score[0])
    result = {
        'similarity score' : round(score,4)
    }
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
