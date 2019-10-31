# Encoder-Decoder-Translation
An encoder-decoder translation model with or without attention
# Dataset:
Multi30K English-German dataset has been used for training and testing purposes.
# Dependancies:
* Used Python Version:3.7.0
* Install necessary modules with `sudo pip3 install -r requiremnets.txt` command.
# Model Training and Testing:
To train and test the model --> `python3 train_and_test.py`
# Model Parameters:
For encoder-decoder without attention:
  * Encoder embedding dimension = 256
  * Deocder embedding dimension = 256
  * Hidden Dimension = 512
  * Number of Layers = 2
  * Encoder Dropout = 0.5
  * Decoder Dropout = 0.5

For encoder-decoder with attention:
  * Encoder Hidden dimension = 512
  * Decoder Hidden dimension = 512
# Author:
Subhrajit Dey(@subro608)
  
