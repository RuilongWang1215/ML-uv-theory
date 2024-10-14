import pandas as pd
import numpy as np
import os

class data_preprocessing():
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
    
    def data_sampling(self, n):
        # this function tackles the problem of data imbalance
        pass
    
    def detect_outliers(data, threshold=3):
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

if __name__ == '__main__':
    dp = data_preprocessing()
    dp.combing_data()
    