import pandas
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
# pandas.set_option('display.max_colwidth', -1)


class DataTransformator:
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

    low_corr = [
        'MoSold',
        '3SsnPorch',
        'BsmtFinSF2',
        'BsmtHalfBath',
        'LowQualFinSF',
        'YrSold'
    ]

    def __init__(self, dataset):
        self.dataset = dataset

    def data_trans(self):
        # Too many NaNs
        # Low cardinality, use OneHot
        for feature in ['Alley', 'PoolQC', 'Fence', 'MiscFeature']:
            self.dataset[feature].fillna('NotAvailable', inplace=True)

        # Numbers
        for feature in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']:
            self.dataset[feature].fillna(self.dataset[feature].median(), inplace=True)

        # Strings
        for feature in ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                        'Electrical',
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
                if feature == 'GarageYtBlt':
                    self.dataset[feature].fillna(self.dataset[feature].min(), inplace=True)
                else:
                    self.dataset[feature].fillna(self.dataset[feature].median(), inplace=True)

        self.dataset.drop('LotShape', inplace=True, axis=1)

    def data_eng(self):
        pass
        # YearRemodAdd and YearBuilt
        # self.dataset['Remodeled'] = self.dataset['YearRemodAdd'] - self.dataset['YearBuilt']
        # self.dataset['Remodeled'] = self.dataset['Remodeled'].apply(lambda x: 1 if x > 0 else 0)
        # self.dataset.drop('YearRemodAdd', inplace=True, axis=1)
        # self.dataset.drop('YearBuilt', inplace=True, axis=1)

        # for feature in self.outliers:
        #     col = self.dataset[feature]
        #     col = col[col.between(col.quantile(.15), col.quantile(.85))]
        #     self.dataset[feature] = col
        #     self.dataset[feature].fillna(self.dataset[feature].median(), inplace=True)


# Load training data
train_data = pandas.read_csv('data/train.csv')
target = 'SalePrice'

d_train = DataTransformator(train_data)
d_train.data_trans()
d_train.data_eng()

# Split training set
X_train, X_test, y_train, y_test = train_test_split(d_train.dataset, d_train.dataset[target], random_state=np.random)

# Drop target feature
X_train.drop(target, axis=1, inplace=True)
X_test.drop(target, axis=1, inplace=True)

# model = XGBRegressor()
# parameters = {'nthread': [4],
#               'learning_rate': [.03, 0.05, .07],
#               'max_depth': [1, 3, 5, 7, 9],
#               'min_child_weight': [2, 4, 6],
#               'silent': [1],
#               'subsample': [0.5, 0.7],
#               'colsample_bytree': [0.5, 0.7],
#               'n_estimators': [100]}
# xgb_grid = GridSearchCV(model,
#                         parameters,
#                         cv=5,
#                         n_jobs=4,
#                         verbose=True)
#
# xgb_grid.fit(X_train,
#              y_train)
#
# print(xgb_grid.best_score_)
# print(xgb_grid.best_params_)

#
best_params = {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 6, 'n_estimators': 100, 'nthread': 4, 'silent': 1, 'subsample': 0.5}


model = XGBRegressor(**best_params)
# Fit and score
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Score: {}".format(score))

# Test data predictions
test_data = pandas.read_csv('data/test.csv')
d_test = DataTransformator(test_data)
d_test.data_trans()
d_test.data_eng()

preds = model.predict(d_test.dataset)

# Store to CSV file
submission = pandas.DataFrame()
submission['Id'] = d_test.dataset['Id']
submission['SalePrice'] = preds
submission.to_csv('predictions.csv', index=False)
