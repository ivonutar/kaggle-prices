import pandas

# Load training data
train_data = pandas.read_csv('data/train.csv')

# Columns with NaNs
# print(train_data.columns[train_data.isna().any()])
# ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual',
# 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
# 'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
# 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
# 'MiscFeature']

# Too many NaNs
train_data['Alley'].fillna('NotAvailable', inplace=True)  # ['Grvl' 'Pave']
train_data['PoolQC'].fillna('NotAvailable', inplace=True)  # ['Ex' 'Fa' 'Gd']
train_data['Fence'].fillna('NotAvailable', inplace=True)  # ['MnPrv' 'GdWo' 'GdPrv' 'MnWw']
train_data['MiscFeature'].fillna('NotAvailable', inplace=True)  # ['Shed' 'Gar2' 'Othr' 'TenC']


# Numbers
for feature in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']:
    train_data[feature].fillna(train_data[feature].median())

# Strings
for feature in ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical',
                'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train_data[feature].fillna(train_data[feature].mode()[0])


