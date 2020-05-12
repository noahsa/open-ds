from flask import Flask

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, world"

@app.route('/model')
def model():
    train = pd.DataFrame(np.random.randint(5, size=(1000,2)), columns=['x','y'])

    reg = LinearRegression()
    reg.fit(train[['x']], train['y'])

    test = pd.DataFrame(np.random.randint(5, size=1000), columns=['x'])

    pred = reg.predict(test[['x']])
    return str(pred)

if __name__ ==  '__main__':
    app.run(host='0.0.0.0', port=8080)