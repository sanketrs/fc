# from flask import Flask
#
# ##creating a flask app and naming it "app"
# app = Flask('app')
#
# @app.route('/test', methods=['GET'])
# def test():
#     return 'Pinging Model Application!!'
#
# if __name__ == '__main__':
#     app.run()
# # The run method starts our flask application service. The 3 parameters specify:
# # debug=True — restarts the application automatically when it encounters any change in the code
# # host=’0.0.0.0' — makes the web service public
# # port=9696 — the port that we use to access the application


import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_mpg


app = Flask('mpg_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    vehicle = request.get_json()
    print(vehicle)
    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = predict_mpg(vehicle, model)

    result = {
        'mpg_prediction': list(predictions)
    }
    return jsonify(result['mpg_prediction'])

@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model!!"


@app.route('/sanket', methods=['GET'])
def test():
    return 'Pinging Model Application UAE!!'


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)  #use this port and ip for local server check
