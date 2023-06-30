from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

def Predictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1,18)
    rf_model = pickle.load(open('model.pkl', 'rb'))
    result = rf_model.predict(to_predict)
    return result[0]

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = Predictor(to_predict_list)     
        if int(result)==1:
            pred = 'Customer will Churn Off Probably.'
        else:
            pred = 'Customer will Not Churn Off Probably.'
        return render_template('result.html', prediction_text = pred)

if __name__ == '__main__':
    app.run(port = 3000)