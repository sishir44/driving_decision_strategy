import pandas as pd
import numpy as np

'''
def clean_label(label):
    return label.lstrip(',').rstrip(',').replace(',,', ',')

data = pd.read_csv('DrivingDataset/dataset.csv')
#data.drop('index', axis=1, inplace=True)
#data.to_csv('mydata.csv', index=False)

df_labeled = data.dropna(subset=['trajectory_id','start_time','end_time','rpm_average','rpm_medium','rpm_max','rpm_std','speed_average','speed_medium',
                                        'speed_max','speed_std','labels'])
df_labeled.loc[:,'labels'] = df_labeled['labels'].apply(lambda x: clean_label(x))

all_labels = df_labeled['labels'].unique()
print("Example of trajectory labels:")
for label in all_labels[0:5]:
    print(label)

#We can filter out single modal trajectories by taking the labels which do not contain a comma:
single_modality_labels = [elem for elem in all_labels if ',' not in elem]

df_single_modality = df_labeled[df_labeled['labels'].isin(single_modality_labels)]
df_single_modality.to_csv('mydata.csv', index=False)

mask = np.random.rand(len(df_single_modality)) < 0.7
df_train = df_single_modality[mask]
df_test = df_single_modality[~mask]

print(len(df_train))
'''
data = pd.read_csv('DrivingDataset/dataset.csv')
labels = data.labels.unique()
print(labels)
