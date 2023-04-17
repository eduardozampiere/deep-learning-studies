### Deep Learning basic tutorial
https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0

### Diferences between the optimizers
https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c
https://www.youtube.com/watch?v=rQ-BePyJK0E&list=PL-t7zzWJWPtygNTsgC_M8c9a-p5biCjho&index=8&t=908s

### Fashion MNIST database
https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download

### Fine Tunning
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

### How to save the best model
https://machinelearningmastery.com/save-load-keras-deep-learning-models/

```python
from keras.models import Sequential
from keras.models import model_from_json

best_model = Sequential() # the best treined model

# Save the best model
model_json = best_model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
    
best_model.save_weights('model.h5') # save the weights [IMPORTANT]

# Load the best model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json) # load the model
loaded_model.load_weights('model.h5') # load the weights [IMPORTANT]
```

### Recurrent Neural Networks
https://www.youtube.com/watch?v=qjrad0V0uJE

## LSTM (Long Short Term Memory)
https://colah.github.io/posts/2015-08-Understanding-LSTMs/


### About the project
* Multiple layers
  * Each .py file is a different model 
  * Models with multiplelayers using Sequential
* One layer
  * Each .py file is a different model
  * Models with one layer using Perceptron
* src
  * Contains the .csv files to train and test the models
* trainning_ml.py
  * A basic ML model w/ a single neuron, just to understand how a model works

### Run the project
if you are not in Mac M1, go to pyproject.toml and change the line
```toml
tensorflow-macos = "^2.12.0"
```
to
```toml
tensorflow = "^2.12.0"
```

To install:
```bash
conda create python=3.9 -n <env_name>
conda activate <env_name>
poetry init
poetry install
```
