from flask import Flask, render_template, request
from keras.models import load_model
import pickle
import tensorflow as tf

app = Flask(__name__)

with open(r'count_vec.pkl','rb') as file:
    cv=pickle.load(file)
cla = load_model('mymodel.h5')
cla.compile(optimizer='adam',loss='binary_crossentropy')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tpredict')
@app.route('/', methods=['GET','POST'])
def page2():
    if request.method == 'POST':
        return render_template('index.html')
    if request.method == 'POST':
        topic = request.form['tweet']
        print("Hey " + topic)
        topic=cv.transform([topic])
        print("\n"+str(topic.shape)+"\n")
        with graph.as_default():
            y_pred = cla.predict(topic)
            print("pred is " + str(y_pred))
        if(y_pred > 0.5):
            topic = "Positive Tweet"
        else:
            topic = "Negative Tweet"
        return render_template('index.html',ypred=topic)

if __name__ == "__main__":
    app.run(debug=True)
    
    