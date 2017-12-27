from flask import Flask ,jsonify,request
from sklearn.externals import joblib
import numpy as np 

loaded_model = joblib.load('model.pkl')
X_test = np.array([[   6. ,    148.   ,   72.  ,    35.   ,    0.    ,  33.6     , 0.627  , 50.   ]])
# X_test = np.array([[   0. ,    0.   ,   0.  ,    0.   ,    0.    ,  0     , 0.627  , 0.   ]])
app = Flask(__name__)


@app.route('/')     
def list_store():
    results = loaded_model.predict(X_test)
    print (results)
    return (jsonify({'result':results[0]}))



app.run(port=5000)

