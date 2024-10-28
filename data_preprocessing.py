import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

base_dir =  os.path.dirname(__file__)
class data_preprocess():
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
    
    def data_sampling(self, n):
        # this function tackles the problem of data imbalance
        pass
    
    def rescale_to_custom_range(self, data, min_val=1, max_val=100):
        # Get the minimum and maximum of the original data
        data_min = np.min(data)
        data_max = np.max(data)
        # Apply the rescaling formula
        rescaled_data = (data - data_min)*(max_val - min_val) / (data_max - data_min)  + min_val
        return rescaled_data
    
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
        path = base_dir+'/data'
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
            data.to_csv(base_dir+'/data/processed_data/one_hot_encoded_data.csv', index=False)
        return data
    
    def feature_engineering(self):
        supplementary_features = pd.DataFrame(columns=['substance', 'SMILES', 'TPSA', 'Molecular_Weight',
                                                       'LogP','BertzCT'])
        substances_dict = {'methane':'C',
                           'argon':'[Ar]',
                           'water':'O',}
        for substance, smiles in substances_dict.items():
            mol = Chem.MolFromSmiles(smiles)
            TPSA = Descriptors.TPSA(mol)
            Molecular_Weight = Descriptors.MolWt(mol)
            LogP = Descriptors.MolLogP(mol)
            BertzCT = Descriptors.BertzCT(mol)
            NumAtoms = mol.GetNumAtoms()
            BalabanJ = Descriptors.BalabanJ(mol)
            Ipc = Descriptors.Ipc(mol)
            HallKierAlpha = Descriptors.HallKierAlpha(mol)
            NumHAcceptors = Descriptors.NumHAcceptors(mol)
            NumAmideBonds = Descriptors.NumAmideBonds(mol)
            supplementary_features = pd.concat([supplementary_features, pd.DataFrame([[substance, smiles, TPSA, Molecular_Weight, LogP, BertzCT,
                                                                                       NumAtoms, BalabanJ, Ipc, HallKierAlpha, NumHAcceptors, NumAmideBonds]],
                                                                                     columns=['substance', 'SMILES', 'TPSA', 'MW', 'LogP', 'BertzCT',
                                                                                              'NumAtoms', 'BalabanJ', 'Ipc', 'HallKierAlpha', 'NumHAcceptors', 'NumAmideBonds',
                                                                                              ])], axis=0)
            print(f'{substance} done')
            # drop the columns that are all 0
        supplementary_features = supplementary_features.loc[:, (supplementary_features != 0).any(axis=0)]
        supplementary_features = supplementary_features.select_dtypes(include=['int64', 'float64'])
        supplementary_features = supplementary_features.loc[:, (supplementary_features != supplementary_features.iloc[0]).any()]
        number_of_features = supplementary_features.shape[1]
        supplementary_features.to_csv(base_dir+f'/data/processed_data/supplementary_features_{int(number_of_features)}.csv', index=False)
        return supplementary_features
    
    def add_features(self, data, supplementary_features):
        combined_data_features = data.merge(supplementary_features, on='substance', how='left')
        # drop the columns that are not int or float
        combined_data_features = combined_data_features.drop(columns=['substance', 'SMILES'])
        number_of_features = combined_data_features.shape[1]-1
        # put delta_phi at the end
        combined_data_features = combined_data_features[[col for col in combined_data_features.columns if col != 'delta_phi'] + ['delta_phi']]
        combined_data_features.to_csv(base_dir+f'/data/processed_data/all_add_features_{str(number_of_features)}.csv', index=False)
        return combined_data_features
    
    def normalize_data(self, data, exclude_columns):
        df = data.copy()
        columns_to_standardize = [col for col in df.columns if col not in exclude_columns]
        scaler = MinMaxScaler()
        #df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
        for col in columns_to_standardize:
            df[col] = self.rescale_to_custom_range(df[col])
        #df.to_csv(base_dir+'/data/processed_data/all_add_features_filtered_normalized.csv', index=False)
        return df
    
        
if __name__ == '__main__':
    dp = data_preprocess()
    supplement_features= dp.feature_engineering()
    data = dp.combing_data()
    data = dp.add_features(data, supplement_features)
    n_before = data.shape[0]
    data_filtered = dp.detect_outliers(data)
    n_after = data_filtered.shape[0]
    print(f'Number of outliers removed: {n_before - n_after}')
    #data_filtered.to_csv(base_dir+'/data/processed_data/combined_data_add_features_3_filtered.csv', index=False)
    data_filtered_normalized = dp.normalize_data(data_filtered, exclude_columns=['delta_phi'])
    data_filtered_normalized.to_csv(base_dir+'/data/processed_data/all_add_features_filtered_normalized.csv', index=False)
    