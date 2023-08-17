# Script to run the models
#!/bin/bash

# do sbatch --time=1800 --partition=research-dual-gpu --gpus=1 ContrastiveLearning_Pretrain.py; done

for ((x=0;x<10;x++)) do sbatch --time=1800 --partition=research-dual-gpu --gpus=1 Weak_Generalization.py ContrastiveLearning_Model/saved_models/masked_encoder_64_16_128_3_0.05; done

# for ((x=0;x<10;x++)) do sbatch --time=1800 --partition=research-dual-gpu --gpus=1 Strong_Generalization.py ContrastiveLearning_Model/saved_models/masked_encoder_64_16_128_3_0.05; done