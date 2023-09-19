#  MFAE
MFAE: Multilevel Feature Aggregation Enhanced Drug-Target Affnity Prediction for Drug Repurposing against Colorectal Cancer
This research introduces the first comprehensive drug-target affnity dataset for CRC-specific targets, P2X4 and mTOR. Utilizing a novel deep learning MFAE model, specific inhibitors are systematically sourced from ChEMBL. The meticulously fine-tuned model then screens FDA-approved drugs, effectively pinpointing potential lead compounds for CRC treatment.

## File list
- Data: the processed dataset of CRC dataset.
- Model: the trained model weights of MFAE.
- Result: the evaluate result of model.
- CRC_data.csv: the original CRC dataset.
- readme.md: the readme file.
- split_dataset.py: split dataset into train, val, test with 5 folds on Data folder. And the esamble dataset is in Data/bagging_data folder.
- training.py: train the model with the dataset in Data folder.
- test.py: test the model with the dataset in Data folder.
- utils.py: some useful functions.
- model.py: the code about model.
- dataset.py: the code about dataset.


## Requirements
```
numpy==1.24.3
pandas==1.5.3
scikit_learn==1.3.0
scipy==1.10.1
torch==1.12.1
torch_geometric==2.3.1
torchvision==0.13.1
tqdm==4.65.0
```


## Run Code
Step 1: Split the dataset into train, val, test with 5 folds on Data folder. And the esamble dataset is in Data/bagging_data folder.
```
python split_dataset.py
```


Step 2: Train the model with the dataset in Data folder.
```
python train.py
```


Step 3: Test the trained model.
```
python test.py
```


