import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV


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
                self.dataset[feature].fillna(self.dataset[feature].mean(), inplace=True)

        self.dataset.drop('LotShape', inplace=True, axis=1)

    def data_eng(self):
        # YearRemodAdd and YearBuilt
        self.dataset['Remodeled'] = self.dataset['YearRemodAdd'] - self.dataset['YearBuilt']
        self.dataset['Remodeled'] = self.dataset['Remodeled'].apply(lambda x: 1 if x > 0 else 0)
        self.dataset.drop('YearRemodAdd', inplace=True, axis=1)
        self.dataset.drop('YearBuilt', inplace=True, axis=1)


# Load training data
train_data = pandas.read_csv('data/train.csv')
target = 'SalePrice'

d_train = DataTransformator(train_data)
d_train.data_trans()
d_train.data_eng()

# Split training set
X_train, X_test, y_train, y_test = train_test_split(d_train.dataset, d_train.dataset[target], random_state=np.random)

# model = GradientBoostingRegressor(learning_rate=0.1,
#                                   max_depth=6,
#                                   max_features=0.3,
#                                   min_samples_leaf=3,
#                                   n_estimators=100)
X_train.drop(target, axis=1, inplace=True)
X_test.drop(target, axis=1, inplace=True)

# param_grid = {'n_estimators': [10, 30, 60, 100, 150],
#               'learning_rate': [0.1, 0.05, 0.02, 0.01],
#               'max_depth': [2, 4, 6, 8, 10],
#               'min_samples_leaf': [3, 5, 9, 17],
#               'max_features': [1.0, 0.3, 0.7]
#               }

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# print("Best params: {}".format(grid_search.best_params_))
# Best params: {'learning_rate': 0.1, 'max_depth': 4, 'max_features': 0.3, 'min_samples_leaf': 9, 'n_estimators': 100}

best_params = {'learning_rate': 0.1, 'max_depth': 4, 'max_features': 0.3, 'min_samples_leaf': 9, 'n_estimators': 100}
model = GradientBoostingRegressor(**best_params)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Score: {}".format(score))

test_data = pandas.read_csv('data/test.csv')
d_test = DataTransformator(test_data)
d_test.data_trans()
d_test.data_eng()

preds = model.predict(d_test.dataset)

submission = pandas.DataFrame()
submission['Id'] = d_test.dataset['Id']
submission['SalePrice'] = preds
submission.to_csv('predictions.csv', index=False)
