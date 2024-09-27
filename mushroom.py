# April 2024

# Import Packages
import argparse
import numpy as np
import pandas as pd
import pdb
import os
import time

# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder # Convert Input Categorical to Numerical Data
from sklearn.preprocessing import LabelEncoder # Convert Categorical Outputs to Binary Outputs (1 and 0's)
from sklearn.pipeline import Pipeline # Transform all Features/Columns of DataFrame using OneHotEncoding

# Keras Packages
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import keras.utils

# PyTorch Packages
import torch
from torch import nn


# Define ROOT Directory where data is stored
ROOT = 'C:\\Users\\deasu\\OneDrive\\Documents\\CSC 3520\\homework_3\\mushroom_neural_network'

# Expand '~' to users home directory (C: for Windows)
ROOT = os.path.expanduser(ROOT)

# Current Directory (For saving new file to desired directory)
THIS = os.path.dirname(os.path.realpath(__file__))
pdb.set_trace()

# Command Line Arguments
parser = argparse.ArgumentParser(description="Use a Neural Network to classify mushrooms as either edible or poisonous")

# Force user to choose between Keras and PyTorch
model_group = parser.add_mutually_exclusive_group(required=True)

model_group.add_argument('-keras',
                         action='store_true',
                         help='Use a keras implementation of a neural network')
model_group.add_argument('-pytorch',
                         action='store_true',
                         help='Use a pytorch implementation of a neural network')

# Other Arguments
parser.add_argument('-xtrain',
                    default=os.path.join(ROOT, 'training_data.txt'),
                    help='Training data for the model')
parser.add_argument('-ytrain',
                    default=os.path.join(ROOT, 'training_labels.txt'),
                    help='Training labels (outputs) for the model')
parser.add_argument('-delimiter',
                    default='\t',
                    type=str,
                    help='Delimiter for training data')
parser.add_argument('-save',
                    action='store_true',
                    help='Save weights to file')
parser.add_argument('-epoch',
                    default=100,
                    type=int,
                    help='Number of epochs in the training process')
parser.add_argument('-batch',
                    default=50,
                    type=int,
                    help='Batch size for the training process')

# Main Function
def main(args):

    # Process Data
    training_data = pd.read_csv(args.xtrain, delimiter=args.delimiter, header=None).to_numpy()
    training_labels = pd.read_csv(args.ytrain, delimiter=args.delimiter, header=None).to_numpy()[:, 0] # Convert 2D to 1D
    
    # Shuffle Data
    indices = np.arange(len(training_labels))
    np.random.shuffle(indices)

    training_data = training_data[indices]
    training_labels = training_labels[indices]

    # Encode Output Training Labels with Binary Labels (0's and 1's)
    encoder = LabelEncoder()
    encoder.fit(training_labels)
    training_labels = encoder.transform(training_labels)
    
    # Seperate Column For Each Output Label/Class
    training_labels = keras.utils.to_categorical(training_labels==1, 2)
    
    # One Hot Encode each Input Feature (Column) of DataFrame
    spicy_encoder = OneHotEncoder() # Initialize OneHotEncoder
    pipeline = Pipeline(steps=[('encoder', spicy_encoder)]) # Used to encode all columns
    training_data = pipeline.fit_transform(training_data).toarray() # Save transformed dataframe
    
    # Essential Variables
    num_samples = len(training_data) # Before extraction
    num_features = len(training_data[0]) # After OneHotEncoding
    num_epochs = args.epoch
    batch_size = args.batch
    
    # Extract Testing Data (10% of Training Data)
    testing_data = training_data[-int(0.1 * num_samples) : ]
    testing_labels = training_labels[-int(0.1 * num_samples) : ]
    
    # Remove extracted testing data from training data
    training_data = training_data[ : -int(0.1 * num_samples)]
    training_labels = training_labels[ : -int(0.1 * num_samples)]
    
    # Update Number of Training Samples
    num_training_samples = len(training_data)
    
    # If user chose Keras model
    if args.keras:
        
        # Training Header
        print(f'\n================')
        print(f'TRAINING')
        print(f'================')
        
        # Initialize the Keras model
        model = Sequential()
        model.add(Input(shape=(num_features, ))) # Input layer, where number of neurons = number of features
        model.add(Dense(units=50, activation='relu', name='Hidden_Layer_1')) # Hidden layer 1 has 75 neurons
        # model.add(Dense(units=10, activation='relu', name='Hidden_Layer_2')) # Hidden layer 2 has 25 neurons
        model.add(Dense(units=2, activation='sigmoid', name='Output')) # Sigmoid used for binary classification as it outputs values between 0 and 1
        
        # Train the Keras network
        model.summary()
        input("Press <Enter> to train this network...")

        # Compile the Keras model
        model.compile(
            loss='binary_crossentropy', # Loss function for binary classification
            optimizer=Adam(learning_rate=0.001), # Initialize adaptive learning rate to 0.001
            metrics=['accuracy', 'precision', 'recall']) # Metrics to evaluate model on
        
        # Define an early stopping point for the model when desired accuracy is reached or the model stops improving. Early stopping can reduce overfitting
        callback = EarlyStopping(
            monitor='loss', # Improvement metric
            min_delta=1e-3, # Stop if accuracy is 100%
            patience=2, # Number of epochs to wait until stopping after no improvement is observed
            restore_best_weights=True, # If stopped due to lack of improvement, restore weights that had the lowest loss
            verbose=1) # If the model stops early, display confirmation
        
        # Train the keras model
        history = model.fit(training_data, training_labels, # Inputs & Outputs
            epochs=num_epochs, # Number of epochs
            batch_size=batch_size, # Batch size
            callbacks=[callback], # Early stopping point
            shuffle=True, # Shuffle the data since data was extracted from end of dataset (likely bias)
            validation_split=0.2, # Validation dataset (Keras automatically tunes hyperparameters)
            verbose=1) # Display information per epoch completed
        
        # Testing Header
        print(f'\n================')
        print(f'TESTING')
        print(f'================')
        
        # Evaluate the keras model
        training_metrics = model.evaluate(training_data, training_labels)
        testing_metrics = model.evaluate(testing_data, testing_labels)
        
        # Save Model (Including Weights & Configuration)
        if args.save:
            print('Saving PyTorch Neural Network Weights to File...')
            model.save('keras_nn.keras')
            print('Successfully Saved to \'keras_nn.keras\'')
        
        # Display Accuracy
        print(f'\n================')
        print(f'ACCURACY')
        print(f'================')
        
        print(f'Training Accuracy: {(training_metrics[1] * 100):.2f}%')
        print(f'Testing Accuracy: {(testing_metrics[1] * 100):.2f}%')
    
    # If user chose PyTorch model
    else:  # args.pytorch
        
        # Convert Training & Testing Data & Labels to Torch Tensors
        training_data = torch.tensor(training_data, dtype=torch.float32)
        training_labels = torch.tensor(training_labels, dtype=torch.float32)
        testing_data = torch.tensor(testing_data, dtype=torch.float32)
        testing_labels = torch.tensor(testing_labels, dtype=torch.float32)
        
        # Get cpu, gpu or mps device for training.
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        # Display Device
        print(f"Using {device} device")

        # Create neural network
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Network Architecture
                self.hidden1 = nn.Linear(num_features, 50) # Input Layer, where each feature is a Neuron
                self.act1 = nn.ReLU() # ReLU Activation Function
                # self.hidden2 = nn.Linear(100, 50) # Hidden Layer 1 of 100 Neurons
                # self.act2 = nn.ReLU()
                self.output = nn.Linear(50, 2) # Hidden Layer 2 of 50 Neurons (Output to 2 Output Nodes)
                self.act_output = nn.Sigmoid() # Binary Activiation Function
        
            def forward(self, x):
                x = self.act1(self.hidden1(x))
                # x = self.act2(self.hidden2(x))
                x = self.act_output(self.output(x))
                return x # Predictions
        
        # Initialize and Display Neural Network Architecture
        pytorch_model = NeuralNetwork().to(device)
        print(pytorch_model)
        
        # Train the pytorch network
        input("Press <Enter> to train this network...")
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=1e-3)
        
        # Training Header
        print(f'\n================')
        print(f'TRAINING')
        print(f'================')
        
        # Initialize Early Stopping Count & Break Point
        early_stop_value = 1e-3
        early_stop_count = 0
        early_stop_break = 2
        
        # Iterate over each epoch
        for epoch in range(num_epochs):
            
            # Initialize 'Loss' so it exists outside the following block
            loss = 0
            
            # Number of Batches sent per Epoch
            for sample_index in range(0, num_training_samples, batch_size):
                
                # Splice Training Set to get current batch
                batch_data = training_data[sample_index : sample_index + batch_size]

                # Forward pass the spliced data
                outputs = pytorch_model.forward(batch_data)
                
                # Splice Training Set to get current batch labels
                batch_labels = training_labels[sample_index : sample_index + batch_size]
                
                # Compute Loss             
                loss = loss_fn(outputs, batch_labels)  # Calculate the loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Summarize Progress at each Epoch        
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
            # If 100% Accuracy reached for 2 epochs
            if loss <= early_stop_value:
                early_stop_count += 1
                if early_stop_count == early_stop_break:
                    break
                
        # Save Model (Including Weights & Configuration)
        if args.save:
            print('Saving PyTorch Neural Network Weights to File...')
            torch.save(pytorch_model.state_dict(), 'pytorch_weights.pth')
            print('Successfully Saved to \'pytorch_weights.pth\'')

        # Compute Training and Testing Accuracies
        output = pytorch_model(testing_data)
        train_output = pytorch_model(training_data)
        
        accuracy = 0
        train_accuracy = 0
        
        # Testing Accuracy
        for sample_index in range(len(output)):
            if output[sample_index][0] > output[sample_index][1]:
                if testing_labels[sample_index][0] == 1:
                    accuracy += 1
            else:
                if testing_labels[sample_index][1] == 1:
                    accuracy += 1
        
        # Training Accuracy         
        for sample_index in range(len(train_output)):
            if train_output[sample_index][0] > train_output[sample_index][1]:
                if training_labels[sample_index][0] == 1:
                    train_accuracy += 1
            else:
                if training_labels[sample_index][1] == 1:
                    train_accuracy += 1
                    
        # Display Accuracy
        print(f'\n================')
        print(f'ACCURACY')
        print(f'================')
           
        print(f'Training Accuracy: {(train_accuracy / len(train_output)):.4f} ({(100 * train_accuracy / len(train_output)):.2f}%)')         
        print(f'Testing Accuracy: {(accuracy / len(output)):.4f} ({(100 * accuracy / len(output)):.2f}%)')
        
  
        
if __name__ == "__main__":
	main(parser.parse_args())