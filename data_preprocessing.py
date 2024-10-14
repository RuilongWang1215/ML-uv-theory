import pandas as pd
import numpy as np
import os

class data_preprocessing():
    def __init__(self, data):
        self.data = data # pd.DataFrame
        self.X = None
        self.y = None
    
    def data_sampling(self, n):
        # this function tackles the problem of data imbalance
        pass
    
    def detect_outliers(data, threshold=3):
        mean = np.mean(self.data['delta_phi'])
        std_dev = np.std(self.data['delta_phi'])
        outliers = []
        for i in data['delta_phi']:
            z_score = (i - mean) / std_dev
            if np.abs(z_score) > threshold:
                outliers.append(i)
        data_without_outliers = self.data[~self.data['delta_phi'].isin(outliers)]
        return data_without_outliers
    
    def combing_data(self):
        # this function combines the data from different sources
        path = 'data'
        file_names = os.listdir(path)
        # filter out the files that are not starting with 'real_ufrac'
        substances = [file_name.split('_')[-1].split('.')[0] for file_name in file_names if file_name.startswith('real_ufrac')]
        data_total = pd.DataFrame()
        for substance in substances:
            data = pd.read_csv(f'{path}/real_ufrac_{substance}.csv')
            data_total = pd.concat([data_total, data[['density', 'temperature', 'delta_phi']]], axis=0)
        if not os.path.exists(f'{path}/process_data'):
            os.mkdir(f'{path}/process_data')
        data_total.to_csv(f'{path}/process_data/combined_data.csv', index=False)
        return data_total
        