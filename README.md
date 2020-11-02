# Albert-Adverserial-Attack

###### Disclaimer: This repository has been heavily influenced by the original code-base of the paper (https://github.com/robinjia/adversarial-squad) and contains a simplified re-implementation for personal learning purposes.

### Setting-up:
Conda is recommended for setting up this repository.
-	conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
-	pip install transformers
-	pip install pytorch-transformers
-	conda install nltk

### Running the code:
The main file to be run is ```main.py```. It does not require any command-line arguments however there is a variable on top of the file that needs to be set:
-	squad_dataset_filepath:	Set it to the filepath of squad dataset. Please note that AddAny is a very computation intensive technique hence it will take forever to run it on the complete squad dataset. So you can either the number of extracted questions at line 57, in order to obtain a suitable amount (example of it is provided in lines 58-60).

Additionally, at the top of ```Adverserial.py``` there are five arguments that can be specified:
-	NUM_EPOCHS: Number of epochs to be run in each Mega Epoch.
-	NUM_SAMPLE: Number of alternative words to be tried in each iteration.
-	NUM_ADDITIONS: Sequence length of the sentence to be added.
-	NUM_SEARCHES_PER_MEGA_EPOCH: 
Defines both the number of mega-epochs to be run (length of list), and the number of parallel searches in each mega-epoch. For example, an input of [1, 2, 4] would mean that 3 mega-epochs will be run. In first mega-epoch only one sequence will be initialized and searched, in second mega-epoch one more sequence will be initialized (making the total two), in the third mega-epoch 2 more sequences will be initialized.
-	PRINT_RESULTS_PER_ITERATION: 
Flag to turn on/off printing of results in each iteration.

```TODO: Accept the above mentioned arguments through command-line instead of hard-coded values.```
