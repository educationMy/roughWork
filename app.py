import imp
from operator import index
from pyexpat import model
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'}

from flask import Flask, session

app = Flask(__name__, static_url_path="", static_folder="")

app.secret_key = "super secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100*1024*1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return render_template('index.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('index.html', btn_css='btn btn-primary btn-block btn-large')
    return render_template('index.html', btn_css='btn btn-primary btn-block btn-large')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train',methods=['GET'])
def train():

    if os.path.isfile("model.h5") :
        os.remove("model.h5")

    if os.path.isfile("result.png") :
        os.remove("result.png")


    # ### Develop LSTM Models For Univariate Time Series Forecasting

    # In[ ]:


    # univariate lstm example
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten


    # In[ ]:


    # preparing independent and dependent features
    def prepare_data(timeseries_data, n_features):
        X, y =[],[]
        for i in range(len(timeseries_data)):
            # find the end of this pattern
            end_ix = i + n_features
            # check if we are beyond the sequence
            if end_ix > len(timeseries_data)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)


    # In[ ]:


    # define input sequence
    timeseries_data = [110, 125, 133, 146, 158, 172, 187, 196, 210]
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = prepare_data(timeseries_data, n_steps)


    # In[ ]:


    print(X),print(y)


    # In[ ]:


    X.shape


    # In[ ]:


    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))


    # ### Building LSTM Model

    # In[ ]:



    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=100000, verbose=1)

    # save the model
    # os.remove("model.h5")
    from keras.models import load_model
    model.save('model.h5')

    return render_template('index.html')


@app.route('/predict',methods=['GET'])
def predict():
    # ### Predicting For the next 10 data

    if os.path.isfile("model.h5") :
        
        from keras.models import load_model
        model = load_model('model.h5')

        # In[ ]:


        # demonstrate prediction for next 10 days

        # define input sequence
        timeseries_data = [110, 125, 133, 146, 158, 172, 187, 196, 210]
        # choose a number of time steps
        n_steps = 3
        n_features = 1

        x_input = np.array([187, 190, 210])
        # x_input = [187, 196, 210]
        temp_input=list(x_input)
        lst_output=[]
        i=0
        while(i<20):
            
            if(len(temp_input)>3):
                x_input=np.array(temp_input[1:])
                # x_input=temp_input[1:]
                print("{} day input {}".format(i,x_input))
                #print(x_input)
                x_input = x_input.reshape((1, n_steps, n_features))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.append(yhat[0][0])
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.append(yhat[0][0])
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i=i+1
            

        print(lst_output)


        # In[ ]:





        # In[ ]:


        timeseries_data


        # In[ ]:


        len(timeseries_data)


        # In[ ]:


        lst_output


        # In[ ]:


        # lst


        # ### Visualizaing The Output

        # In[ ]:


        import matplotlib.pyplot as plt


        # In[ ]:


        day_new=np.arange(1,10)
        day_pred=np.arange(10,30)


        # In[ ]:


        plt.plot(day_new,timeseries_data)
        plt.plot(day_pred,lst_output)
        plt.savefig('result.png', dpi=150);
        plt.close()

        return render_template('result.html', img = '/result.png')
    return "Please wait model is being trained"

if __name__ == "__main__":
    app.run(debug=True)
