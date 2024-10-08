import os
import pandas as pd
import numpy as np
from cnn import CNN
from pyod.models.deep_svdd import DeepSVDD
import pickle

# Import the model trained for seizure detection
model = torch.load('seizure_detection_model.pth')

# Import your test EEG in numpy format with 1 seconds segments. 
# training data must be in the format of (samples, 1, time, channels), and labels (samples, )
test_data = np.load('EEG/train_EEG.npy')
test_label = np.load('EEG/train_label.npy')

# Load data
test_dataloader = (test_data, batch_size=128, shuffle=False)

# Get the features from the last convolutional layers from the trained model based on the index from the 4 metrics above. 
target_layer = model.maxpool2
cam_features = extract_cams(model, test_dataloader, target_layer)
cam_features = np.concatenate(cam_features, axis=0)
n_samples, n_channels, n_features_per_channel = cam_features.shape
cam_features= np.reshape(cam_features, (n_samples, n_channels * n_features_per_channel))

# Define the file names for each model
clf_names = ['DeepSVDD_false_positive.pkl','DeepSVDD_true_positive.pkl', 'DeepSVDD_false_negative.pkl', 'DeepSVDD_true_negative.pkl']

# Load each model and store them in a dictionary for easy access
models = {}

for clf_name in clf_names:
    with open(clf_name, 'rb') as file:
        model = pickle.load(file)
        models[clf_name.replace('.pkl', '')] = model 

# Access each model from the dictionary
false_positives_svdd = models['DeepSVDD_false_positive']
true_positives_svdd = models['DeepSVDD_true_positive']
false_negative_svdd = models['DeepSVDD_false_negative']
true_negative_svdd = models['DeepSVDD_true_negative']

# Make a prediction for seizure detection
model.eval()
with torch.no_grad():
    model.eval()
    for data in test_dataloader:
        # Extract the inputs from the batch
        inputs = data

        # Pass the inputs through the model to get predictions
        outputs = model(inputs)
        predicted = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(predicted, dim=1)
      
        # Convert the predictions to a list and append to all_predictions
        all_predictions_test += predicted.cpu().numpy().tolist()

# Make a prediction for uncertainty
false_positives_pred = false_positives_svdd.predict(cam_features)
true_positives_pred = true_positives_svdd.predict(cam_features)
false_negative_pred = false_negative_svdd.predict(cam_features)
true_negative_pred = true_negative_svdd.predict(cam_features)

# Detect Uncertainty for each predicted segments
uncertain_list = []
for FP, TP, FN, TN,i in zip(false_positives_pred, true_positives_pred, false_negative_pred, true_negative_pred, all_predictions_test):
    
    if i == 0: 
        if (FN == 0 or TN == 0) and (FP == 1 and TP == 1) :
            uncertain_list.append(0)
        else:
            uncertain_list.append(1)
    if i == 1:        
        if (FP == 0 or TP == 0) and (FN == 1 and TN == 1) :
            uncertain_list.append(0)
        else:
            uncertain_list.append(1) 

# estimate uncertainty % on patient level
print(len([i for i in range(len(uncertain_list)) if uncertain_list[i] == 1])/len(uncertain_list))
