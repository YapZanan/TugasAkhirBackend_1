import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app, support_credentials=True)


def load_models():
    # build model and tokenizera
    model_name_dict = {
        'nllb-distilled-600M': 'GPT/Model',
        # 'nllb-1.3B': 'facebook/nllb-200-1.3B',
        #   'nllb-distilled-1.3B': 'model/nllb-distilled-1.3B',
        # 'nllb-3.3B': 'facebook/nllb-200-3.3B',
    }

    model_dict = {}

    for call_name, real_name in model_name_dict.items():
        print('\tLoading model: %s' % call_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(real_name)
        tokenizer = AutoTokenizer.from_pretrained(real_name)
        model_dict[call_name + '_model'] = model
        model_dict[call_name + '_tokenizer'] = tokenizer

    return model_dict


def generation(source, target, text):
    if len(model_dict) == 2:
        model_name = 'nllb-distilled-600M'

    start_time = time.time()
    source = flores_codes[source]
    target = flores_codes[target]

    model = model_dict[model_name + '_model']
    tokenizer = model_dict[model_name + '_tokenizer']

    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source, tgt_lang=target)
    output = translator(text, max_length=400)

    end_time = time.time()

    full_output = output
    output = output[0]['translation_text']
    result = {'inference_time': end_time - start_time,
              'source': source,
              'target': target,
              'result': output,
              'full_output': full_output}
    return result



@app.route('/generate', methods=['POST'])
@cross_origin()
def generate():
    data = request.get_json()
    source = data['source']
    target = data['target']
    text = data['text']

    result = translation(source, target, text)
    response = make_response(jsonify(result))
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response

@app.route('/language', methods=['GET'])
def getlanguages():
    return jsonify(list(flores_codes.keys()))

if __name__ == '__main__':
    global model_dict
    model_dict = load_models()
    app.run()