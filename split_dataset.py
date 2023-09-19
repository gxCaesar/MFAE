import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


dataset_name = 'Data'
df = pd.read_csv(f'./CRC_data.csv')

k_folds = 5
random_state = 0
kf = KFold(n_splits=k_folds, random_state=random_state,shuffle=True)
fold_indices = list(kf.split(df))

for fold, (train_indices, test_indices) in enumerate(fold_indices):
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    train_df, val_df = train_test_split(train_df, test_size=1/8, random_state=random_state)

    train_df.to_csv(f'./{dataset_name}/fold_{fold+1}_train.csv', index=False)
    val_df.to_csv(f'./{dataset_name}/fold_{fold+1}_val.csv', index=False)
    test_df.to_csv(f'./{dataset_name}/fold_{fold+1}_test.csv', index=False)
    print(f'fold {fold+1} done.')
    print(f'train size: {len(train_df)}')
    print(f'val size: {len(val_df)}')
    print(f'test size: {len(test_df)}')
    print('')

for random_state in range(1,4):
    for fold in range(1,6):
        df = pd.read_csv("./Data/fold_{fold}_train.csv".format(fold=fold))
        new_df = df.sample(n = len(df),replace = True,random_state = random_state)
        new_df.to_csv("./Data/bagging_data/fold_{fold}_train_sample_{random}.csv".format(fold=fold,random = random_state),index = False)