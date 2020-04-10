import pandas
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer

pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', -1)

# Load training data
train_data = pandas.read_csv('data/train.csv')
target = 'SalePrice'

#
# # Change to categories
# # Fill rest of NaNs
# for feature in train_data:
#     if train_data[feature].dtype == np.object:
#         train_data[feature].fillna(train_data[feature].mode()[0], inplace=True)
#         train_data[feature] = train_data[feature].astype('category')
#         train_data[feature] = train_data[feature].cat.codes
#     else:
#         if feature == 'GarageYtBlt':
#             train_data[feature].fillna(train_data[feature].min(), inplace=True)
#         else:
#             train_data[feature].fillna(train_data[feature].median(), inplace=True)

# train_data.drop('LotShape', inplace=True, axis=1)

# Split training set
X_train, X_test, y_train, y_test = train_test_split(train_data, train_data[target], random_state=np.random)

# Drop target feature
X_train.drop(target, axis=1, inplace=True)
X_test.drop(target, axis=1, inplace=True)


best_params = {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 6,
               'n_estimators': 100, 'nthread': 4, 'silent': 1, 'subsample': 0.5}
model = XGBRegressor(**best_params)

notavailable_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='NotAvailable')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

median_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

freq_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

other_obj = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

n = (X_train.dtypes != 'object')
num_cols = list(n[n].index)

preprocessor = ColumnTransformer(
    transformers=[
        ('na', notavailable_transformer, ['Alley', 'PoolQC', 'Fence', 'MiscFeature']),
        ('med', median_transformer, ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']),
        ('frq', freq_transformer, ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                   'BsmtFinType2', 'Electrical',
                                   'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']),
        ('oth', other_obj, object_cols),
        ('med2', median_transformer, num_cols),
    ], remainder='drop')

pipe = Pipeline(steps=[
    ('prep', preprocessor),
    ('model', model)
])

# Fit and score
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print("Score: {}".format(score))
preds_test = pipe.predict(X_test)
print("MAE: {}".format(mean_absolute_error(y_test, preds_test)))

# Test data predictions
test_data = pandas.read_csv('data/test.csv')

preds = pipe.predict(test_data)

# Store to CSV file
submission = pandas.DataFrame()
submission['Id'] = test_data['Id']
submission['SalePrice'] = preds
submission.to_csv('predictions.csv', index=False)
