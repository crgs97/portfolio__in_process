from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict',methods=['POST'])
def predict():
    PL = float(request.values['petal_length'])
    PW = float(request.values['petal_width'])
    SL = float(request.values['sepal_length'])
    SW = float(request.values['sepal_width'])

    values = np.array([[PL, PW, SL, SW]])
    output=model.predict(values)
    output=output.item()
    return render_template('resultpage.html',prediction_text="The predicted Iris Species is {}".format(output))

if __name__=='__main__':
    app.run(debug=True)