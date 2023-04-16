### Deep Learning basic tutorial
https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0

### Diferences between the optimizers
https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c
https://www.youtube.com/watch?v=rQ-BePyJK0E&list=PL-t7zzWJWPtygNTsgC_M8c9a-p5biCjho&index=8&t=908s


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
```
tensorflow-macos = "^2.12.0"
```
to
```
tensorflow = "^2.12.0"
```
* conda create python=3.9 -n <env_name>
* conda activate <env_name>
* poetry init
* poetry install
