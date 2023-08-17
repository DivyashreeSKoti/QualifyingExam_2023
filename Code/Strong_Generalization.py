#!/usr/bin/env python
# coding: utf-8

import time

# Keep track of run time
start_time = time.time()

# ********* Start ************
# This section is to ensure to get the right directory when submitting a job
import sys
import os
current_dir = os.path.abspath('./')
sys.path.append(current_dir)

# Set the working directory to the directory containing the script
custom_path = current_dir

# Get the absolute path of the current script
script_dir = os.path.abspath(custom_path)

# ********* END ************

# ********* Start of main code *************
import DataLoader as DL
import SetTransformer_Extrapolating as ST

#to plot data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from itertools import islice
import numpy as np
import pandas as pd

print('arg: ',sys.argv)

folder_path = "../../../Data/3dbsf_txt/"

min_lines, max_lines = DL.get_min_max_lines(folder_path)
print(f"The minimum number of lines among all files: {min_lines}")
print(f"The maximum number of lines among all files: {max_lines}")

# Load data and corresponding targets
dataset, targets, labels = DL.load_dataset(folder_path)

# Get device to run on available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define global variable to keep track of uniquestring fr outputfile
file_str = ''

# Funtion to custom data split 80:20
def call_splitter(val_target):
    splitter = DL.DatasetSplitter(validation_ratio=0.2, shuffle=False)
    train_data,train_targets, val_data, val_targets, train_class_mapping, val_class_mapping = splitter.split_dataset_by_index(dataset, targets, val_target=val_target)
    return train_data,train_targets, val_data, val_targets, train_class_mapping, val_class_mapping

# Custom collate function to subssample the point cloud data
def custom_collate(subsample_size):
    def collate_fn(batch):
        subsamples = []
        for data, target in batch:
            num_samples = data.shape[0]
            current_subsample_size = min(subsample_size, num_samples)
            indices = random.sample(range(num_samples), subsample_size)
            subsample = data[indices]
            subsamples.append((subsample, target))

        data, targets = zip(*subsamples)
        data = torch.stack(data, dim=0)
        targets = torch.stack(targets, dim=0)

        return data, targets
    
    return collate_fn

# Defining the fine tuning model for Contrastive pretrained model
class FineTuneModel(nn.Module):
    def __init__(self, pretrained_model, additional_layers, final_layer):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.additional_layers = additional_layers
        self.final_layer = final_layer
    
    # get_embeddings is to get embeddings from standard set transformer
    def forward(self, inputs, get_embeddings=True, get_embeddings_additional_layer=False):
        _, outputs = self.pretrained_model(inputs, get_embeddings=get_embeddings)
        embeddings = self.additional_layers(outputs)
        outputs = self.final_layer(embeddings)
        if get_embeddings_additional_layer:
            return outputs, embeddings
        return outputs

# Define the batch size, training subsample size and validation subsample size
def call_dataloader(train_data,train_targets, val_data, val_targets):
    train_batch_size = 4
    val_batch_size = 1
    train_subsample_size = 8000
    val_subsample_size = 2000
    print('batch_size =', train_batch_size, 'subsample_size =', train_subsample_size)
    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_targets))
    val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_targets))

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=custom_collate(subsample_size=train_subsample_size))
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=custom_collate(subsample_size=val_subsample_size))

    train_total_DLbatches = len(train_dataloader)
    val_total_DLbatches = len(val_dataloader)
    print('train num batches',train_total_DLbatches)
    print('val num batches',val_total_DLbatches)

    return train_dataloader, val_dataloader, train_total_DLbatches, val_total_DLbatches

def create_model(
    # Define architecture for the model
    embed_dim = 64,
    num_heads = 16,
    num_induce = 128,
    stack=3,
    ff_activation="gelu",
    dropout=0.05,
    use_layernorm=False,
    pre_layernorm=False,
    is_final_block = False,
    num_classes = 75,
    load_model = None
):
    # Condition to check if pretrain model is provided
    global file_str
    if load_model:
        print('*************** Using pretrained ***************')
        pretrained_path = load_model+'.pth' # Path to the pretrained model file
        pretrained_model = torch.load(pretrained_path)

        num_classes = 75  # Number of output classes
        projecttion_dim = 128
        # Additional layers for pretrained model
        additional_layers = nn.Sequential(
            nn.Linear(pretrained_model.embed_dim, projecttion_dim), 
            nn.LeakyReLU(),
            nn.Dropout(p=0.1)
        )

        # Final layer 
        final_layer = nn.Sequential(
            nn.Linear((projecttion_dim), num_classes)
        )

        # Stack all layers
        pytorch_model = FineTuneModel(
            pretrained_model,
            additional_layers,
            final_layer
        )
        file_str = sys.argv[1].split('/')[-1] # To keep unique for the output files
    else:
        print('*************** New model ***************')

        pytorch_model = ST.PyTorchModel(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_induce=num_induce,
            stack=stack,
            ff_activation=ff_activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
            is_final_block = is_final_block,
            num_classes = num_classes
        )
        print('Details:', 'embed_dim =', embed_dim,
        'num_heads =', num_heads,
        'num_induce =', num_induce,
        'stack =', stack,
        'dropout =', dropout)
        # To keep unique for the output files
        file_str = str(embed_dim)+'_'+str(num_heads)+'_'+str(num_induce)+'_'+str(stack)+'_'+str(dropout)

    pytorch_model.to(device)
    return pytorch_model




def fit(val_target):
    train_data,train_targets, val_data, val_targets, train_class_mapping, val_class_mapping = call_splitter(val_target)
    train_dataloader, val_dataloader, train_total_DLbatches, val_total_DLbatches = call_dataloader(train_data,train_targets, val_data, val_targets)
    # Get model based on pretrained argument provided
    if len(sys.argv) > 1:
        pytorch_model = create_model(load_model = sys.argv[1])
    else:
        pytorch_model = create_model()
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(pytorch_model.parameters(), lr=1e-3)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Define the number of training epochs
    num_epochs = 250

    # Training loop
    for epoch in range(num_epochs):
        train_loss_total = 0.0
        total_train_correct = 0
        total_train_samples = 0
        # Convert the training data to PyTorch tensors
        for i,(batch_data, batch_targets) in enumerate(train_dataloader):
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)

            # Set the model to training mode
            pytorch_model.train()

            # Forward pass - Training
            train_outputs = pytorch_model(batch_data)

            train_loss = criterion(train_outputs, batch_targets.long())

            # Compute accuracy
            _, train_predicted = torch.max(train_outputs.data, 1)

            # Compute accuracy
            correct = (train_predicted == batch_targets).sum().item()
            
            # Accumulate the validation loss
            train_loss_total += train_loss.item()
        
        
            # Backward pass and optimization
            optimizer.zero_grad() #gradients are cleared before computing the gradients for the current batch
            train_loss.backward()
            optimizer.step()  #update the model parameters based on the computed gradients


            train_accuracy = correct / batch_data.size(0)
            total_train_correct += correct
            total_train_samples += batch_data.size(0)
            
            # Print the progress
            print(f"\rEpoch [{epoch+1}/{num_epochs}], Progress: {i+1}/{train_total_DLbatches}, Train Loss: {train_loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}", end="", flush=True)

        # Compute the average training loss and accuracy
        avg_train_loss = train_loss_total / train_total_DLbatches
        avg_train_accuracy = total_train_correct / total_train_samples
    
        # keep track of loss and accuracy for Monte carlo simulation
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
   
        # Print final results of epoch
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}", end="",flush=True) 
        print('')

    # ***** Evaluate on the left out class
    val_true_labels = []
    predicted_probs = []
    predicted_labels = []
    for batch_data, batch_targets in val_dataloader:
        val_true_labels.extend(batch_targets.long().tolist())
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        # Forward pass - Validation
        with torch.no_grad():
            val_outputs = pytorch_model(batch_data)
            val_probs = torch.softmax(val_outputs, dim=1)
            _, val_predicted = torch.max(val_outputs.data, 1)
            predicted_probs.extend(val_probs.tolist())
            predicted_labels.extend(val_predicted.tolist())
    return train_targets, predicted_probs, val_true_labels, train_class_mapping, val_class_mapping

# appended_probs to keep track of the 75*75 matrix
appended_probs = np.empty((0, 75))
appended_targets = np.empty(0)
for i in range(75):
    print('####### Left out ==> ', i, ' #######')
    train_targets, predicted_probs, val_true_labels, train_class_mapping, val_class_mapping = fit(i)

    val_reversed_mapping = {v: k for k, v in val_class_mapping.items()}
    val_original_targets = [val_reversed_mapping[true_label] for true_label in val_true_labels]

    train_reversed_mapping = {v: k for k, v in train_class_mapping.items()}
    # Revert the values back to keys using the reversed mapping
    train_original_targets = [train_reversed_mapping[train_target] for train_target in train_targets]

    predicted_probs = np.array(predicted_probs)

    appended_probs = np.concatenate((appended_probs, predicted_probs), axis=0)  # Concatenate along axis 0
    appended_targets = np.concatenate((appended_targets, val_original_targets), axis=0)  # Concatenate along axis 0
# Zip the predicted_probs matrix with the true labels
zipped = zip(appended_probs, appended_targets)

# Sort the zipped list based on the true labels
sorted_zipped = sorted(zipped, key=lambda x: x[1])

# Unzip the sorted list to separate the sorted predicted_probs and sorted true_labels
sorted_predicted_probs, sorted_true_labels = zip(*sorted_zipped)

# Convert the sorted_predicted_probs back to a numpy array
sorted_predicted_probs = np.array(sorted_predicted_probs)

print("Sorted_predicted_probs:\n",sorted_predicted_probs)
print('end')


# Plot the Confusion Matrix
plt.figure(figsize=(90, 90))
sns.set(font_scale=2.0) 
# , cbar_kws={'shrink': 0.8}
sns.heatmap(sorted_predicted_probs, cmap='Blues', annot=True, fmt='.2f')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
# Construct the absolute path
fp = os.path.join(script_dir)
# Resolve the absolute path
fp = os.path.abspath(fp)
temp_folder_path = fp + '/Confusion_Matrix/Strong_Generalization'
conf_matx_save = temp_folder_path+'/Confusion_'+str(os.getpid())+ '_' + file_str +'.png'
print('-----',conf_matx_save)
plt.savefig(conf_matx_save)

def write_to_csv(data):
    # Construct the absolute path
    fp = os.path.join(script_dir)
    # Resolve the absolute path
    fp = os.path.abspath(fp)
    temp_folder_path = fp + '/probability_output_files/Strong_Generalization/'+ file_str
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
        print(f"Folder created: {temp_folder_path}")
    else:
        print(f"Folder already exists: {temp_folder_path}")

    output_file = temp_folder_path +'/output_file_'+ str(os.getpid()) + '_' + file_str +'.csv'

    df = pd.DataFrame(data)
    
    df.to_csv(output_file, index=False)
        
# output_file to store the data
write_to_csv(sorted_predicted_probs)


# Stop the timer
end_time = time.time()

# Calculate the total time
total_time = end_time - start_time

# Print the total time
print("Total time:", total_time, "seconds")