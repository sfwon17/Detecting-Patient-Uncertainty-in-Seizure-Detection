import os
import pandas as pd
import numpy as np
from cnn import CNN
import torch.nn as nn
import torch.optim as optim
from utils.predicted_indices import get_indices
from utils.get_features import extract_cams
from utils.reshape_features import features_for_each_metrics
from pyod.models.deep_svdd import DeepSVDD
import pickle

# Import the model trained for seizure detection
model = torch.load('seizure_detection_model.pth')

# Import your training and validation EEG in numpy format with 1 seconds segments. 
# training data must be in the format of (samples, 1, time, channels), and labels (samples, )
train_data = np.load('EEG/train_EEG.npy')
train_label = np.load('EEG/train_label.npy')

# Load data
train_dataloader = (train_data, batch_size=6, shuffle=False)

# Make predictions on trained data
model.eval()
with torch.no_grad():
    for data in train_dataloader:
        # Pass the inputs through the model to get predictions
        outputs = model(data)

        # Apply softmax and get the predicted class
        predicted = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(predicted, dim=1)

        # Convert predictions to a list and extend all_predictions
        all_predictions.extend(predicted.cpu().numpy())

# Get the index of the predicted labels, against true labels.
false_positives = get_indices(all_predictions, train_label, pred_label=1, true_label=0)
true_positives = get_indices(all_predictions, train_label, pred_label=1, true_label=1)
true_negatives = get_indices(all_predictions, train_label, pred_label=0, true_label=0)
false_negatives = get_indices(all_predictions, train_label, pred_label=0, true_label=1)

# Get the features from the last convolutional layers from the trained model based on the index from the 4 metrics above. 
target_layer = model.maxpool2
cam_features = extract_cams(model, train_dataloader, target_layer)
cam_features = np.concatenate(cam_features, axis=0)

# group features based on the 4 of the metrics.
reshaped_features = features_for_each_metrics(cam_features, false_positives, true_positives, false_negatives, true_negatives)

false_positives_reshaped = reshaped_features['false_positives']
true_positives_reshaped = reshaped_features['true_positives']
false_negatives_reshaped = reshaped_features['false_negatives']
true_negatives_reshaped = reshaped_features['true_negatives']

# train 4 DeepSVDD
contamination = 0.05

# train DeepSVDD detector (Without-AE)
clf_name = 'DeepSVDD_false_positive'
use_ae = False
false_positives_svdd = DeepSVDD(use_ae=use_ae, epochs=10, contamination=contamination,random_state=1)
false_positives_svdd.fit(false_positives_reshaped)
# Save the model using pickle
with open(clf_name, 'wb') as file:
    pickle.dump(false_positives_svdd, file)

# train DeepSVDD detector (Without-AE)
clf_name = 'DeepSVDD_true_positive'
use_ae = False
true_positives_svdd = DeepSVDD(use_ae=use_ae, epochs=10, contamination=contamination,random_state=1)
true_positives_svdd.fit(true_positives_reshaped)
# Save the model using pickle
with open(clf_name, 'wb') as file:
    pickle.dump(true_positives_svdd, file)

# train DeepSVDD detector (Without-AE)
clf_name = 'DeepSVDD_false_negative'
use_ae = False
false_negative_svdd = DeepSVDD(use_ae=use_ae, epochs=10, contamination=contamination,random_state=1)
false_negative_svdd.fit(false_negatives_reshaped)
# Save the model using pickle
with open(clf_name, 'wb') as file:
    pickle.dump(false_negative_svdd, file)

# train DeepSVDD detector (Without-AE)
clf_name = 'DeepSVDD_true_negative'
use_ae = False
true_negative_svdd = DeepSVDD(use_ae=use_ae, epochs=10, contamination=contamination,random_state=1)
true_negative_svdd.fit(true_negatives_reshaped)
# Save the model using pickle
with open(clf_name, 'wb') as file:
    pickle.dump(true_negative_svdd, file)
