import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os

class data_preprocess():
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
    
    def data_sampling(self, n):
        # this function tackles the problem of data imbalance
        pass
    
    def detect_outliers(self, data, threshold=3):
        mean = np.mean(data['delta_phi'])
        std_dev = np.std(data['delta_phi'])
        outliers = []
        for i in data['delta_phi']:
            z_score = (i - mean) / std_dev
            if np.abs(z_score) > threshold:
                outliers.append(i)
        data_without_outliers = data[~data['delta_phi'].isin(outliers)]
        return data_without_outliers
    
    def combing_data(self):
        path = 'data'
        file_names = os.listdir(path)
        # filter out the files that are not starting with 'real_ufrac'
        substances = [file_name.split('_')[-1].split('.')[0] for file_name in file_names if file_name.startswith('real_ufrac')]
        data_total = pd.DataFrame()
        for substance in substances:
            data = pd.read_csv(f'{path}/real_ufrac_{substance}.csv')
            data['substance'] = substance
            data_total = pd.concat([data_total, data[['density', 'temperature', 'delta_phi','substance']]], axis=0)
        if not os.path.exists(f'{path}/processed_data'):
            os.mkdir(f'{path}/processed_data')
        data_total.to_csv(f'{path}/processed_data/combined_data_{len(substances)}.csv', index=False)
        # create a txt file to store the description of the data
        # combined_data_{len(substances)}.csv: combined data of all (list of substances)
        with open(f'{path}/processed_data/data_description.txt', 'w') as f:
            f.write(f'combined_data_{len(substances)}.csv: combined data of substances: {substances}')
        print('Data combined and saved')
        self.one_hot_encoding(data_total, save=True)
        return data_total
    
    def one_hot_encoding(self, data, save=False):
        data['substance'] = data['substance'].astype(str)
        data = pd.get_dummies(data, columns=['substance'], dtype=int)
        if save:
            data.to_csv('data/processed_data/one_hot_encoded_data.csv', index=False)
        return data
    
    def add_features(self, data):
        # all features except the target variable
        supplementary_features = pd.read_csv('data/processed_data/supplementary_features_3.csv')
        combined_data_features = data.merge(supplementary_features, on='substance', how='left')
        # drop the columns that are not int or float
        combined_data_features = combined_data_features.drop(columns=['substance', 'SMILES'])
        # put delta_phi at the end
        combined_data_features = combined_data_features[[col for col in combined_data_features.columns if col != 'delta_phi'] + ['delta_phi']]
        combined_data_features.to_csv('data/processed_data/combined_data_add_features_3.csv', index=False)
        return combined_data_features
    
    def normalize_data(self, data, exclude_columns):
        df = data.copy()
        columns_to_standardize = [col for col in df.columns if col not in exclude_columns]
        scaler = MinMaxScaler()
        df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
        return df
    
    def formula_constraint(data):
        data.loc[data['density'] < 0.001, 'delta_phi'] = 0
        temperatures = data['temperature'].unique()
        # Step 2: Get unique substance parameters, filter out rows with missing values
        new_data = data[['temperature','Boiling_Point', 'Molecular_Weight', 'Melting_Point']].drop_duplicates()
        new_data['density'] = 10000*max(data['density'])
        new_data['delta_phi'] = 0
        data = pd.concat([data, new_data], ignore_index=True)
        return data
        
'''if __name__ == '__main__':
    dp = data_preprocessing()
    #dp.combing_data()
    data = pd.read_csv('data/processed_data/combined_data_add_features_3.csv')
    n_before = data.shape[0]
    #print(np.mean(data['delta_phi']))
    data_filtered = dp.detect_outliers(data)
    n_after = data_filtered.shape[0]
    print(f'Number of outliers removed: {n_before - n_after}')
    data_filtered_normalized = dp.normalize_data(data_filtered, exclude_columns=['Boiling_Point', 'Molecular_Weight', 'Melting_Point'])
    data_filtered_normalized.to_csv('data/processed_data/combined_data_add_features_3_filtered_normalized.csv', index=False)'''
    