# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here

```

### Dataset Information
Include screenshot of the generated data

### OUTPUT

### Training Loss Vs Iteration Plot
Include your plot here
<img width="702" height="565" alt="542722564-fefb2612-e8df-4832-8537-2a261ac34280" src="https://github.com/user-attachments/assets/091fd089-65df-479e-8ded-bfc3aaad363c" />
### New Sample Data Prediction
Include your sample input and output here

<img width="874" height="142" alt="image" src="https://github.com/user-attachments/assets/9fb2c206-acb0-47cb-826c-afba2a762849" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
