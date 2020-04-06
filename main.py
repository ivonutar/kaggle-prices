import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class DataTransformator:

    def __init__(self, dataset):
        self.dataset = dataset

    def data_trans(self):
        # Too many NaNs
        self.dataset['Alley'].fillna('NotAvailable', inplace=True)  # ['Grvl' 'Pave']
        self.dataset['PoolQC'].fillna('NotAvailable', inplace=True)  # ['Ex' 'Fa' 'Gd']
        self.dataset['Fence'].fillna('NotAvailable', inplace=True)  # ['MnPrv' 'GdWo' 'GdPrv' 'MnWw']
        self.dataset['MiscFeature'].fillna('NotAvailable', inplace=True)  # ['Shed' 'Gar2' 'Othr' 'TenC']

        # Numbers
        for feature in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']:
            self.dataset[feature].fillna(self.dataset[feature].median(), inplace=True)

        # Strings
        for feature in ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical',
                        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
            self.dataset[feature].fillna(self.dataset[feature].mode()[0], inplace=True)

        # Change to categories
        # Fill rest of NaNs
        for feature in self.dataset:
            if self.dataset[feature].dtype == np.object:
                self.dataset[feature].fillna(self.dataset[feature].mode()[0], inplace=True)
                self.dataset[feature] = self.dataset[feature].astype('category')
                self.dataset[feature] = self.dataset[feature].cat.codes
            else:
                self.dataset[feature].fillna(self.dataset[feature].median(), inplace=True)


# Load training data
train_data = pandas.read_csv('data/train.csv')
target = 'SalePrice'

d_train = DataTransformator(train_data)
d_train.data_trans()

# Split training set
X_train, X_test, y_train, y_test = train_test_split(d_train.dataset, d_train.dataset[target], random_state=np.random)


# Try model
model = RandomForestRegressor()
X_train.drop(target, axis=1, inplace=True)
X_test.drop(target, axis=1, inplace=True)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

test_data = pandas.read_csv('data/test.csv')
d_test = DataTransformator(test_data)
d_test.data_trans()

preds = model.predict(d_test.dataset)

submission = pandas.DataFrame()
submission['Id'] = d_test.dataset['Id']
submission['SalePrice'] = preds
submission.to_csv('predictions.csv', index=False)
