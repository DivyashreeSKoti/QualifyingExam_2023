# Shell script to collect the loss and accuracy to plot Monte Carlo simulated learning plots
#!/bin/bash

cat Large\ Models/Pretrained_AdditionalLayers/NoDropout/* | grep '^Train accuracy' |  cut -f3- -d' ' > ./Large\ Models/Pretrained_AdditionalLayers/NoDropout/train_accuracy.txt

cat Large\ Models/Pretrained_AdditionalLayers/NoDropout/* | grep '^Validation accuracy' |  cut -f3- -d' ' > ./Large\ Models/Pretrained_AdditionalLayers/NoDropout/validation_accuracy.txt

cat Large\ Models/Pretrained_AdditionalLayers/NoDropout/* | grep '^Validation Loss' |  cut -f3- -d' ' > ./Large\ Models/Pretrained_AdditionalLayers/NoDropout/validation_loss.txt

cat Large\ Models/Pretrained_AdditionalLayers/NoDropout/* | grep '^Train Loss' |  cut -f3- -d' ' > ./Large\ Models/Pretrained_AdditionalLayers/NoDropout/train_loss.txt


# cat ./ContrastiveLearning_Model/ContrastiveLearnedModel_MonteCarloResults/* | grep '^Train Loss' |  cut -f3- -d' ' > ./ContrastiveLearning_Model/ContrastiveLearnedModel_MonteCarloResults/train_loss.txt

# cat ./ContrastiveLearning_Model/ContrastiveLearnedModel_MonteCarloResults/* | grep '^Validation Loss' |  cut -f3- -d' ' > ./ContrastiveLearning_Model/ContrastiveLearnedModel_MonteCarloResults/validation_loss.txt
