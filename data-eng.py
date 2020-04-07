import pandas
import numpy as np
import matplotlib.pyplot as plt


train_data = pandas.read_csv('data/train.csv')


# for feature in train_data.keys():
#     if train_data[feature].dtype in [np.float, np.int]:
#         train_data.boxplot(column=feature)
#         plt.show()


outliers = ['LotFrontage',
            'LotArea',
            'MasVnrArea',
            'BsmtFinSF1',
            'BsmtFinSF2',
            'TotalBsmtSF',
            '1stFlrSF',
            'LowQualFinSF',
            'GrLivArea',
            'OpenPorchSF']

for feature in outliers:
    col = train_data[feature]
    col = col[col.between(col.quantile(.15), col.quantile(.85))]
    train_data[feature] = col

for feature in outliers:
    if train_data[feature].dtype in [np.float, np.int]:
        train_data.boxplot(column=feature)
        plt.show()