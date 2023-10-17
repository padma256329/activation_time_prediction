import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_time', methods=['POST'])
def predict_time():
    try:
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        output = model.predict(final_features)
        #output = round(output[0], 2)
        return render_template('index.html', prediction_text='Predicted activation time {}'.format(output[0]))
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
