import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
# OPTION OF NORMALIZATION: rescale_to_custom_range, normalize_data (reduced temperature)
# OPTION OF FEATURE ENGINEERING: add_features, one_hot_encoding, rdkit

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
    
    def normalize_data(self, data):
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
    
    def combing_data(self, mode = 'all'):
        path = base_dir+'/data'
        file_names = os.listdir(path)
        # filter out the files that are not starting with 'real_ufrac'
        substances = [file_name.split('_')[-1].split('.')[0] for file_name in file_names if file_name.startswith('real_ufrac')]
        if mode == 'all':
            data_total = pd.DataFrame()
            for substance in substances:
                data = pd.read_csv(f'{path}/real_ufrac_{substance}.csv')
                data['substance'] = substance
                data_total = pd.concat([data_total, data[['density', 'temperature', 'delta_phi','substance']]], axis=0)
            if not os.path.exists(f'{path}/processed_data'):
                os.mkdir(f'{path}/processed_data')
            data_total.to_csv(f'{path}/processed_data/all_{len(substances)}.csv', index=False)
            # create a txt file to store the description of the data
            # combined_data_{len(substances)}.csv: combined data of all (list of substances)
            with open(f'{path}/processed_data/data_description.txt', 'w') as f:
                f.write(f'all_{len(substances)}.csv: combined data of substances: {substances}')
            print('Data combined and saved')
            
        if mode == 'separate':
            # create a combined data excluding water
            data_total = pd.DataFrame()
            data_water = pd.DataFrame()
            for substance in substances:
                if substance != 'water':
                    data = pd.read_csv(f'{path}/real_ufrac_{substance}.csv')
                    data['substance'] = substance
                    data_total = pd.concat([data_total, data[['density', 'temperature', 'delta_phi','substance']]], axis=0)
                else:
                    data = pd.read_csv(f'{path}/real_ufrac_{substance}.csv')
                    data['substance'] = substance
                    data_water = pd.concat([data_water, data[['density', 'temperature', 'delta_phi','substance']]], axis=0)
            if not os.path.exists(f'{path}/processed_data'):
                os.mkdir(f'{path}/processed_data')
            data_total.to_csv(f'{path}/processed_data/exclude_water.csv', index=False)
            data_water.to_csv(f'{path}/processed_data/water.csv', index=False)
            # write the description of the data in the existing txt file, append the new data
            with open(f'{path}/processed_data/data_description.txt', 'a') as f:
                f.write(f'\n exclude_water.csv: combined data of substances: {substances}')
                f.write(f'\n water.csv: data of water')
            print('Data combined and saved')
        return data_total
    
    def one_hot_encoding(self, data, save=False):
        data['substance'] = data['substance'].astype(str)
        data = pd.get_dummies(data, columns=['substance'], dtype=int)
        if save:
            data.to_csv(base_dir+'/data/processed_data/one_hot_encoded_data.csv', index=False)
        return data
    
    def feature_engineering(self):
        supplementary_features = pd.DataFrame()
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
            supplementary_features = pd.concat([supplementary_features, pd.DataFrame([[substance, TPSA, Molecular_Weight, LogP, BertzCT,
                                                                                       NumAtoms, BalabanJ, Ipc, HallKierAlpha, NumHAcceptors]],
                                                                                     columns=['substance', 'TPSA', 'MW', 'LogP', 'BertzCT',
                                                                                              'NumAtoms', 'BalabanJ', 'Ipc', 'HallKierAlpha', 'NumHAcceptors', 
                                                                                              ])], axis=0)
            print(f'{substance} done')
            # drop the columns that are all 0
        supplementary_features = supplementary_features.loc[:, (supplementary_features != 0).any(axis=0)]
        supplementary_features = supplementary_features.loc[:, (supplementary_features != supplementary_features.iloc[0]).any()]
        number_of_features = supplementary_features.shape[1]
        supplementary_features.to_csv(base_dir+f'/data/processed_data/supplementary_features_{int(number_of_features)}.csv', index=False)
        return supplementary_features
    
    def feature_engineering_manual(self):
        data = {
            "substance": ["argon", "methane", "water"],
            "nu": [13.24120923, 13.67421233, 20],
            "Sigma / Angstrom": [3.4013733, 3.73761057, 3.00143591],
            "Epsilon/kB / Kelvin": [122.95766323, 158.69985269, 398.302260],
            "Kappa_AB": [0, 0, 0.0588848599],
            "epsilon_AB": [0, 0, 2699.63068]}
        df = pd.DataFrame(data)
        csv_path = base_dir+"/data/processed_data/supplementary_features_manual.csv"
        df.to_csv(csv_path, index=False)
        return df
    
    def add_features(self, data, supplementary_features):
        combined_data_features = data.merge(supplementary_features, on='substance', how='left')
        combined_data_features = combined_data_features.drop(columns=['substance'])
        number_of_features = combined_data_features.shape[1]-1
        combined_data_features = combined_data_features[[col for col in combined_data_features.columns if col != 'delta_phi'] + ['delta_phi']]
        #combined_data_features.to_csv(base_dir+f'/data/processed_data/all_add_features_{str(number_of_features)}.csv', index=False)
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
    
    def feature_selection(self,data):
        X = data.drop(columns=['delta_phi'])
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        data_filtered = data.drop(columns=to_drop)
        print(f'Number of features removed due to high correlation: {len(to_drop)}')
        return data_filtered
        
    
        
if __name__ == '__main__':
    dp = data_preprocess()
    supplement_features= dp.feature_engineering_manual()
    #supplement_features = pd.read_csv(base_dir+'/data/processed_data/supplementary_features_6.csv')
    #data = dp.combing_data(mode = 'separate')
    data = pd.read_csv(base_dir+'/data/processed_data/all_3.csv')
    data = dp.add_features(data, supplement_features)
    data.to_csv(base_dir+'/data/processed_data/all_add_features_manual.csv', index=False)
    '''
    n_before = data.shape[0]
    data_filtered = dp.detect_outliers(data)
    n_after = data_filtered.shape[0]
    print(f'Number of outliers removed: {n_before - n_after}')

    data_filtered = dp.feature_selection(data_filtered)
    
    exclude_columns = data_filtered.columns.difference(['density', 'temperature'])
    data_filtered_normalized = dp.normalize_data(data_filtered, exclude_columns=exclude_columns)
    
    data_filtered_normalized.to_csv(base_dir+'/data/processed_data/exclude_water_add_features_normalized.csv', index=False)
    '''
    
    