import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
from io import StringIO
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, recall_score, \
    mean_squared_error, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
df_taxi_zones = pd.read_csv('taxi+_zone_lookup.csv')

# Initial discussion
print(df.columns)
print(df.dtypes)
print(df.nunique())
warnings.filterwarnings("ignore")
print(len(df.index))
print(df.isna().sum())
print(df.head().to_string())
print(type(df))

print(df_taxi_zones.head())
print(df.describe().round(2).to_string())

df['fare_amount'] = df['fare_amount'].abs()
df['extra'] = df['extra'].abs()
df['mta_tax'] = df['mta_tax'].abs()
df['tip_amount'] = df['tip_amount'].abs()
df['tolls_amount'] = df['tolls_amount'].abs()
df['improvement_surcharge'] = df['improvement_surcharge'].abs()
df['total_amount'] = df['total_amount'].abs()
df['congestion_surcharge'] = df['congestion_surcharge'].abs()
df['airport_fee'] = df['airport_fee'].abs()

passenger_mode = df['passenger_count'].mode()[0]
df['passenger_count'].fillna(passenger_mode, inplace=True)
ratecodeID_mode = df['RatecodeID'].mode()[0]
df['RatecodeID'].fillna(ratecodeID_mode, inplace=True)
store_and_fwd_flag_mode = df['store_and_fwd_flag'].mode()[0]
df['store_and_fwd_flag'].fillna(store_and_fwd_flag_mode, inplace=True)
df['congestion_surcharge'].fillna(0.0, inplace=True)
df['airport_fee'].fillna(0.00, inplace=True)

print(df.isna().sum())

store_and_fwd_flag_mapping = {'N': 0, 'Y': 1}

df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace(store_and_fwd_flag_mapping)
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
df['pickup_day'] = df['tpep_pickup_datetime'].dt.dayofweek
df['dropoff_day'] = df['tpep_dropoff_datetime'].dt.dayofweek
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

df['tip_given'] = (df['tip_amount'] > 0).astype(int)  # Categorical Target

print(df.head())

df = df.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)

sns.boxplot(data=df, x='total_amount')
plt.title('A boxplot depicting the total amount before outlier removal')
plt.tight_layout()
plt.show()


# Outlier Removal
def QQ_calc(temp):
    for column_name in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[column_name]) and (column_name != 'passenger_count'
                                                                 and column_name != 'payment_type'
                                                                 and column_name != 'tip_given'):
            Q1 = temp[column_name].quantile(0.25)
            Q3 = temp[column_name].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            temp = temp[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return temp


df = QQ_calc(df)

sns.boxplot(data=df, x='total_amount')
plt.title('A boxplot depicting the total amount after outlier removal')
plt.tight_layout()
plt.show()

# Downsampling to 50k observations
df_sampled = df.sample(n=50000, random_state=5805)
df_sampled.reset_index(drop=True, inplace=True)
print('Length of downsampled data = ', len(df_sampled))

# Dataframe for Apriori
df_categories = df_sampled[['PULocationID', 'DOLocationID', 'payment_type',
                            'tip_given', 'pickup_day', 'dropoff_day']]

reverse_weekday_mapping = {6: 'Sunday', 0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                           3: 'Thursday', 4: 'Friday', 5: 'Saturday'}
payment_type_mapping = {1: 'Credit card', 2: 'Cash', 3: 'No charge', 4: 'Dispute', 5: 'Unknown', 6: 'Voided trip',
                        0: 'Voided trip'}
yes_no_mapping = {1: 'Yes', 0: 'No'}

df_categories['pickup_day'] = df_categories['pickup_day'].replace(reverse_weekday_mapping)
df_categories['dropoff_day'] = df_categories['dropoff_day'].replace(reverse_weekday_mapping)
df_categories['payment_type'] = df_categories['payment_type'].replace(payment_type_mapping)
df_categories['tip_given'] = df_categories['tip_given'].replace(yes_no_mapping)
df_categories = pd.merge(df_categories, df_taxi_zones, left_on='PULocationID', right_on='LocationID', how='left')
df_categories = df_categories.drop(columns=['PULocationID', 'LocationID'])

df_categories = pd.merge(df_categories, df_taxi_zones, left_on='DOLocationID', right_on='LocationID', how='left')
df_categories = df_categories.drop(columns=['DOLocationID', 'LocationID', 'Zone_x',
                                            'service_zone_x', 'Zone_y', 'service_zone_y'])
df_categories = df_categories.rename(columns={'Borough_x': 'pickup_borough', 'Borough_y': 'dropoff_borough'})

print(df_categories.head().to_string())
print(df_categories['pickup_borough'].unique())
print(df_categories['dropoff_borough'].unique())
print(df_categories['payment_type'].unique())
# ================================================
# Phase I - Preprocessing
# ================================================
# Preprocessing for Numerical Target - total_amount
# ================================================

# PCA
X = df_sampled.drop(['total_amount'], axis=1)
y = df_sampled['total_amount']

scalar = StandardScaler()
X_std = scalar.fit_transform(X)
X_std = pd.DataFrame(X_std, columns=X.columns)

pca = PCA(n_components=21, svd_solver='full')
pca.fit(X_std)

exp = pd.DataFrame(pca.explained_variance_ratio_.round(3), columns=['explained variance ratio'])

for idx, elem in enumerate(np.cumsum(pca.explained_variance_ratio_)):
    if elem >= 0.95:
        print('Number of features needed to explain 95% of the dependent variance is ', idx + 1)
        break

plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1),
         np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1))
plt.grid()
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Analysis')
plt.tight_layout()
plt.show()

plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1),
         np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1))
plt.axhline(y=0.95, color='red', linestyle='--', label='y = 0.95 (Horizontal Line)')
plt.axvline(x=np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][0] + 1, color='green', linestyle='--',
            label='x = Component #')
plt.grid()
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Analysis - Tracing number of components above 95% threshold')
plt.tight_layout()
plt.show()

# Condition Number
cond_no = np.linalg.cond(X_std)

print(f'Condition number of original feature space = {cond_no: .3f}')

# Correlation Matrix
corr_matrix = X_std.corr()
plt.figure(figsize=(40, 20))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.title('Correlation Coefficient between features')
plt.tight_layout()
plt.show()

# PCA on reduced no of features
pca_red = PCA(n_components=9, svd_solver='full')
pca_red.fit(X_std)
X_PCA_reduced = pca_red.transform(X_std)

exp = pd.DataFrame(pca_red.explained_variance_ratio_.round(3), columns=['explained variance ratio'])

plt.plot(np.arange(1, len(np.cumsum(pca_red.explained_variance_ratio_)) + 1, 1),
         np.cumsum(pca_red.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca_red.explained_variance_ratio_)) + 1, 1))
plt.grid()
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Reduced PCA Analysis')
plt.tight_layout()
plt.show()

plt.plot(np.arange(1, len(np.cumsum(pca_red.explained_variance_ratio_)) + 1, 1),
         np.cumsum(pca_red.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca_red.explained_variance_ratio_)) + 1, 1))
plt.axhline(y=0.95, color='red', linestyle='--', label='y = 0.95 (Horizontal Line)')
plt.axvline(x=np.where(np.cumsum(pca_red.explained_variance_ratio_) >= 0.95)[0][0] + 1, color='green', linestyle='--',
            label='x = Component #')
plt.grid()
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Reduced PCA Analysis - Tracing number of components above 95% threshold')
plt.tight_layout()
plt.show()

# Condition Number
cond_no_PCA = np.linalg.cond(X_PCA_reduced)

print(f'Condition number of reduced feature space = {cond_no_PCA: .3f}')

# SVD Analysis
svd = TruncatedSVD(n_components=21, random_state=5805)
X_svd = svd.fit_transform(X_std)
svd_val = svd.singular_values_

svd_pca = TruncatedSVD(n_components=9, random_state=5805)
X_svd_PCA = svd_pca.fit_transform(X_PCA_reduced)
svd_val_PCA = svd_pca.singular_values_

print(f'Original Condition Number = {cond_no: .3f}')
print(f'svd values of original matrix = ', svd_val.round(3))
print(f'Reduced Condition Number = {cond_no_PCA: .3f}')
print(f'svd values of reduced matrix = ', svd_val_PCA.round(3))

# Random Forest Analysis
X_train, _, Y_train, Y_test = train_test_split(X, y, shuffle=True, random_state=5805, test_size=0.2)
regressor = RandomForestRegressor(random_state=5805)
regressor.fit(X_train, Y_train)

features = X_train.columns
importances = regressor.feature_importances_
indices = np.argsort(importances)
threshold = 0.050
unimp_indices = [features[i] for i in indices if importances[i] < threshold]
imp_indices = [features[i] for i in indices if importances[i] > threshold]
print('Decided Threshold = ', threshold)
print('Candidates features for dropping = ', unimp_indices)
print('Final selected features = ', imp_indices)

features = X_train.columns
importances = regressor.feature_importances_
indices = np.argsort(importances)

plt.figure()
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Feature Importances via Random Forest Regressor')
plt.tight_layout()
plt.show()


# VIF analysis
def feature_calculator(_df):
    print(_df)
    max_value_col = _df.loc[_df['VIF'].idxmax()]
    if max_value_col['VIF'] < 10:
        return None
    else:
        removed_feature = max_value_col['feature']
        print("removed feature: ", removed_feature)
        return removed_feature


def feature_remover(_X):
    status = True
    while status:
        vif_Data = pd.DataFrame()
        vif_Data['feature'] = _X.columns
        vif_Data['VIF'] = [variance_inflation_factor(_X.values, i) for i in range(len(_X.columns))]
        na_vif = vif_Data[vif_Data['VIF'].isna()]
        dropping = na_vif['feature'].values.tolist()
        for col in dropping:
            _X = _X.drop([col], axis=1)
        vif_Data = vif_Data.dropna(axis=0)
        removed_col = feature_calculator(vif_Data)
        if removed_col is None:
            status = False
            break
        else:
            _X = _X.drop(removed_col, axis=1)
    return _X


reg_df = feature_remover(X)
print('Final Selected Columns by VIF:')
print(reg_df.columns.tolist())

print("=" * 50)
print("Final selected feature is PCA and Condition Number Analysis")
print("=" * 50)

# Covariance Matrix
cov_matrix = X.cov()
plt.figure(figsize=(40, 20))
sns.heatmap(cov_matrix, annot=True, cmap='YlGnBu')
plt.title('Covariance Matrix between features before removing features')
plt.tight_layout()
plt.show()

X_num = X.drop(['RatecodeID', 'store_and_fwd_flag', 'mta_tax', 'tolls_amount', 'improvement_surcharge',
                'congestion_surcharge', 'airport_fee', 'tip_given', 'dropoff_day', 'trip_duration',
                'tip_amount', 'fare_amount'], axis=1)
y_num = y.copy()

# Covariance Matrix after removal of features
cov_matrix_num = X_num.cov()
plt.figure(figsize=(40, 20))
sns.heatmap(cov_matrix_num, annot=True, cmap='YlGnBu')
plt.title('Covariance Matrix between features before removing features')
plt.tight_layout()
plt.show()

# ================================================
# Preprocessing for Categorical Target - tip_given
# ================================================
print("=" * 50)
print("Checking if the target variable is balanced in the original dataframe")
print(df['tip_given'].value_counts())
print("=" * 50)

print('Balancing the dataset by undersampling the majority class')
minority_class = df[df['tip_given'] == 0]
majority_class = df[df['tip_given'] == 1]
majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=5805)
balanced_df = pd.concat([minority_class, majority_downsampled])

print("Class count of tip_given:", balanced_df['tip_given'].value_counts())
balanced_df = balanced_df.drop(['tip_amount'], axis=1)
# PCA
print('=' * 50)
print('Dimensionality reduction on Categorical Target')
print('=' * 50)
df_sampled = balanced_df.sample(n=50000, random_state=5805)
df_sampled.reset_index(drop=True, inplace=True)
print('Length of downsampled data = ', len(df_sampled))

X = df_sampled.drop(['tip_given'], axis=1)
y = df_sampled['tip_given']

scalar_c = StandardScaler()
X_std = scalar_c.fit_transform(X)
X_std = pd.DataFrame(X_std, columns=X.columns)

pca_c = PCA(n_components=20, svd_solver='full')
pca_c.fit(X_std)

exp = pd.DataFrame(pca_c.explained_variance_ratio_.round(3), columns=['explained variance ratio'])

for idx, elem in enumerate(np.cumsum(pca_c.explained_variance_ratio_)):
    if elem >= 0.95:
        print('Number of features needed to explain 95% of the dependent variance is ', idx + 1)
        break

plt.plot(np.arange(1, len(np.cumsum(pca_c.explained_variance_ratio_)) + 1, 1),
         np.cumsum(pca_c.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca_c.explained_variance_ratio_)) + 1, 1))
plt.grid()
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Analysis')
plt.tight_layout()
plt.show()

plt.plot(np.arange(1, len(np.cumsum(pca_c.explained_variance_ratio_)) + 1, 1),
         np.cumsum(pca_c.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca_c.explained_variance_ratio_)) + 1, 1))
plt.axhline(y=0.95, color='red', linestyle='--', label='y = 0.95 (Horizontal Line)')
plt.axvline(x=np.where(np.cumsum(pca_c.explained_variance_ratio_) >= 0.95)[0][0] + 1, color='green', linestyle='--',
            label='x = Component #')
plt.grid()
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Analysis - Tracing number of components above 95% threshold')
plt.tight_layout()
plt.show()

# Condition Number
cond_no = np.linalg.cond(X_std)

print(f'Condition number of original feature space = {cond_no: .3f}')

# Correlation Matrix
corr_matrix = X_std.corr()
plt.figure(figsize=(40, 20))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.title('Correlation Coefficient between features')
plt.tight_layout()
plt.show()

# PCA on reduced no of features
pca_red = PCA(n_components=8, svd_solver='full')
pca_red.fit(X_std)
X_PCA_reduced = pca_red.transform(X_std)

exp = pd.DataFrame(pca_red.explained_variance_ratio_.round(3), columns=['explained variance ratio'])

plt.plot(np.arange(1, len(np.cumsum(pca_red.explained_variance_ratio_)) + 1, 1),
         np.cumsum(pca_red.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca_red.explained_variance_ratio_)) + 1, 1))
plt.grid()
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Reduced PCA Analysis')
plt.tight_layout()
plt.show()

plt.plot(np.arange(1, len(np.cumsum(pca_red.explained_variance_ratio_)) + 1, 1),
         np.cumsum(pca_red.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca_red.explained_variance_ratio_)) + 1, 1))
plt.axhline(y=0.95, color='red', linestyle='--', label='y = 0.95 (Horizontal Line)')
plt.axvline(x=np.where(np.cumsum(pca_red.explained_variance_ratio_) >= 0.95)[0][0] + 1, color='green', linestyle='--',
            label='x = Component #')
plt.grid()
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Reduced PCA Analysis - Tracing number of components above 95% threshold')
plt.tight_layout()
plt.show()

# Condition Number
cond_no_PCA = np.linalg.cond(X_PCA_reduced)

print(f'Condition number of reduced feature space = {cond_no_PCA: .3f}')

# SVD Analysis
svd = TruncatedSVD(n_components=20, random_state=5805)
X_svd = svd.fit_transform(X_std)
svd_val = svd.singular_values_

svd_pca = TruncatedSVD(n_components=8, random_state=5805)
X_svd_PCA = svd_pca.fit_transform(X_PCA_reduced)
svd_val_PCA = svd_pca.singular_values_

print(f'Original Condition Number = {cond_no: .3f}')
print(f'svd values of original matrix = ', svd_val.round(3))
print(f'Reduced Condition Number = {cond_no_PCA: .3f}')
print(f'svd values of reduced matrix = ', svd_val_PCA.round(3))

# Random Forest Analysis
X_train, _, Y_train, Y_test = train_test_split(X, y, shuffle=True, random_state=5805, test_size=0.2)
regressor = RandomForestRegressor(random_state=5805)
regressor.fit(X_train, Y_train)

features = X_train.columns
importances = regressor.feature_importances_
indices = np.argsort(importances)
threshold = 0.050
unimp_indices = [features[i] for i in indices if importances[i] < threshold]
imp_indices = [features[i] for i in indices if importances[i] > threshold]
print('Decided Threshold = ', threshold)
print('Candidates features for dropping = ', unimp_indices)
print('Final selected features = ', imp_indices)

features = X_train.columns
importances = regressor.feature_importances_
indices = np.argsort(importances)

plt.figure()
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Feature Importances via Random Forest Regressor')
plt.tight_layout()
plt.show()

# VIF Analysis
cat_df = feature_remover(X)
print('Final Selected Columns by VIF:')
print(cat_df.columns.tolist())

print("=" * 50)
print("Final selected feature is PCA and Condition Number Analysis")
print("=" * 50)

# Covariance Matrix
cov_matrix = X.cov()
plt.figure(figsize=(40, 20))
sns.heatmap(cov_matrix, annot=True, cmap='YlGnBu')
plt.title('Covariance Matrix between features before removing features')
plt.tight_layout()
plt.show()

X_cat = X.drop(['RatecodeID', 'store_and_fwd_flag', 'mta_tax', 'tolls_amount', 'improvement_surcharge',
                'congestion_surcharge', 'airport_fee', 'dropoff_day', 'trip_distance', 'extra', 'fare_amount',
                'total_amount'], axis=1)
y_cat = y.copy()

# Covariance Matrix after removal of features
cov_matrix_cat = X_cat.cov()
plt.figure(figsize=(40, 20))
sns.heatmap(cov_matrix_cat, annot=True, cmap='YlGnBu')
plt.title('Covariance Matrix between features before removing features')
plt.tight_layout()
plt.show()

# ================================================
# Phase II - Regression Analysis
# ================================================
sc = StandardScaler()
sc_y = StandardScaler()

org_mean = y_num.mean()
org_std = y_num.std()

X_std_num = sc.fit_transform(X_num)
X_std_num = pd.DataFrame(X_std_num, columns=X_num.columns)
y_num_df = pd.DataFrame(y_num, columns=['total_amount'])
y_std_num = sc_y.fit_transform(y_num_df)
y_std_num = pd.DataFrame(y_std_num, columns=['total_amount'])

X_train, X_test, y_train, y_test = train_test_split(X_std_num, y_std_num, shuffle=True, random_state=5805,
                                                    test_size=0.2)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()


def highest_p_value(mod):
    summary_text = mod.summary().tables[1].as_text()
    summary_df = pd.read_csv(StringIO(summary_text), delimiter='\s+', skiprows=[0], header=0)
    highest, variable_with_highest_p_value = 0, 0
    filtered_summary_df = summary_df[summary_df['P>|t|'] > 0.01]
    if not filtered_summary_df.empty:
        highest = filtered_summary_df['P>|t|'].max()
        variable_with_highest_p_value = filtered_summary_df.loc[filtered_summary_df['P>|t|'].idxmax(), 'coef']
    AIC_val = mod.aic
    BIC_val = mod.bic
    adj_R_sq_val = mod.rsquared_adj
    return AIC_val, BIC_val, adj_R_sq_val, highest, variable_with_highest_p_value


def feature_dropper(X_train_set, column):
    X_train_set.drop([column], axis=1, inplace=True)
    return X_train_set


res = PrettyTable()
res.field_names = ['AIC', 'BIC', 'Adjusted-R-Squared', 'Candidate Feature for Elimination',
                   'p_value of Candidate Feature', 'Dropped']
AIC_cur, BIC_cur, adj_R_sq_cur, p_value_drop_candidate, drop_candidate = highest_p_value(model)
res.add_row([AIC_cur.round(3), BIC_cur.round(3), adj_R_sq_cur.round(3), drop_candidate, p_value_drop_candidate, 'None'])
dropped_list = []

X_train = feature_dropper(X_train, drop_candidate)
model = sm.OLS(y_train, X_train).fit()
dropped_list_curr = dropped_list.copy()
dropped_list_curr += [drop_candidate]
dropped_list += [drop_candidate]
dropped_string = ""
for item in dropped_list_curr:
    dropped_string += f''''{item}' '''
AIC_curr, BIC_curr, adj_R_sq_curr, p_value_drop_candidates, drop_candidate_new = highest_p_value(model)
drop_candidate = drop_candidate_new
print(model.summary())
if drop_candidate == 0:
    res.add_row([AIC_curr.round(3), BIC_curr.round(3), adj_R_sq_curr.round(3), 'N/A', 'Adj R-Squared threshold '
                                                                                      'exceeded for remaining '
                                                                                      'features', dropped_string])
res.add_row(
    [AIC_curr.round(3), BIC_curr.round(3), adj_R_sq_curr.round(3), drop_candidate_new, p_value_drop_candidates,
     dropped_string])

X_train = feature_dropper(X_train, drop_candidate)
model = sm.OLS(y_train, X_train).fit()
dropped_list_curr = dropped_list.copy()
dropped_list_curr += [drop_candidate]
dropped_list += [drop_candidate]
dropped_string = ""
for item in dropped_list_curr:
    dropped_string += f''''{item}' '''
AIC_curr, BIC_curr, adj_R_sq_curr, p_value_drop_candidates, drop_candidate_new = highest_p_value(model)
drop_candidate = drop_candidate_new
print(model.summary())
if drop_candidate == 0:
    res.add_row([AIC_curr.round(3), BIC_curr.round(3), adj_R_sq_curr.round(3), 'N/A', 'Adj R-Squared threshold '
                                                                                      'exceeded for remaining '
                                                                                      'features', dropped_string])
res.add_row(
    [AIC_curr.round(3), BIC_curr.round(3), adj_R_sq_curr.round(3), drop_candidate_new, p_value_drop_candidates,
     dropped_string])

X_train = feature_dropper(X_train, drop_candidate)
model = sm.OLS(y_train, X_train).fit()
dropped_list_curr = dropped_list.copy()
dropped_list_curr += [drop_candidate]
dropped_list += [drop_candidate]
dropped_string = ""
for item in dropped_list_curr:
    dropped_string += f''''{item}' '''
AIC_curr, BIC_curr, adj_R_sq_curr, p_value_drop_candidates, drop_candidate_new = highest_p_value(model)
drop_candidate = drop_candidate_new
print(model.summary())
if drop_candidate == 0:
    res.add_row([AIC_curr.round(3), BIC_curr.round(3), adj_R_sq_curr.round(3), 'N/A', 'Adj R-Squared threshold '
                                                                                      'exceeded for remaining '
                                                                                      'features', dropped_string])
res.add_row(
    [AIC_curr.round(3), BIC_curr.round(3), adj_R_sq_curr.round(3), drop_candidate_new, p_value_drop_candidates,
     dropped_string])

X_train = feature_dropper(X_train, drop_candidate)
model = sm.OLS(y_train, X_train).fit()
dropped_list_curr = dropped_list.copy()
dropped_list_curr += [drop_candidate]
dropped_list += [drop_candidate]
dropped_string = ""
for item in dropped_list_curr:
    dropped_string += f''''{item}' '''
AIC_curr, BIC_curr, adj_R_sq_curr, p_value_drop_candidates, drop_candidate_new = highest_p_value(model)
drop_candidate = drop_candidate_new
print(model.summary())
if drop_candidate == 0:
    res.add_row([AIC_curr.round(3), BIC_curr.round(3), adj_R_sq_curr.round(3), 'N/A', 'Adj R-Squared threshold '
                                                                                      'exceeded for remaining '
                                                                                      'features', dropped_string])

print(res.get_string(title='Summary of Model after dropping several Features'))

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()

intercept = model.params['const']
coefficients = model.params.drop('const')
equation = f'total_amount = {intercept:.3f}'
for feature, coef in coefficients.items():
    if coef < 0:
        equation += f' {coef:.3f} * {feature}'
    else:
        equation += f' + {coef:.3f} * {feature}'
print("Backwards Elimination Regression Equation:")
print(equation)

model = sm.OLS(y_train, X_train).fit()

for col in dropped_list:
    if col != 'const':
        X_test = feature_dropper(X_test, col)
X_test_dropped_copy = X_test.copy()
y_pred = model.predict(X_test)

mse_model = mean_squared_error(y_test, y_pred)
print(f'The MSE of the standardized prediction is = {mse_model: .3f}')


def de_standardized(data: pd.DataFrame, mean_org, std_org):
    return (data * std_org) + mean_org


comparison = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred})

print(comparison.head().round(3).to_string())
idx = np.arange(len(y_test))

plt.plot(idx, y_test, label='Actual Values')
plt.plot(idx, y_pred, label='Predicted Values', alpha=0.8)
plt.xlabel('# of Samples')
plt.ylabel('USD($)')
plt.title('Actual vs Predicted total_amount')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

y_test = de_standardized(y_test, org_mean, org_std).squeeze()
y_pred = de_standardized(y_pred, org_mean, org_std)

plt.scatter(y_test, y_pred)
plt.xlabel('Actual total_amount')
plt.ylabel('Predicted total_amount')
plt.title('Test Set vs Predicted total_amount')
plt.tight_layout()
plt.show()


compare = PrettyTable()
compare.field_names = ['Model', 'AIC', 'BIC', 'R-Squared', 'Adjusted-R-Squared', 'MSE']

compare.add_row(['Backward Stepwise Regression', model.aic.round(3), model.bic.round(3), model.rsquared.round(3),
                 model.rsquared_adj.round(3), mse_model.round(3)])
print(compare.get_string(title='Statistics of Stepwise Regression Model'))

# Confidence Interval
X_test = X_test_dropped_copy.copy()
predictions = model.get_prediction(X_test)
prediction_intervals = predictions.conf_int(alpha=0.05)  # Confidence level 95%
pred_summ = predictions.summary_frame(alpha=0.05)

summ_lower_bounds = de_standardized(pred_summ['obs_ci_lower'], org_mean, org_std)
summ_upper_bounds = de_standardized(pred_summ['obs_ci_upper'], org_mean, org_std)
lower_bounds = de_standardized(prediction_intervals[:, 0], org_mean, org_std)
upper_bounds = de_standardized(prediction_intervals[:, 1], org_mean, org_std)

indices = np.arange(len(y_test))

plt.plot(indices, y_pred, label='Predicted total_amount', color='b', linestyle='-')
plt.fill_between(indices, lower_bounds, upper_bounds, alpha=0.7, label='Prediction Interval')
plt.xlabel('# of Samples')
plt.ylabel('USD($)')
plt.title('Predicted Values with Confidence Interval for Stepwise Regression')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.plot(indices, y_pred, label='Predicted total_amount', color='b', linestyle='-')
plt.fill_between(indices, summ_lower_bounds, summ_upper_bounds, alpha=0.7, label='Prediction Interval')
plt.xlabel('# of Samples')
plt.ylabel('USD($)')
plt.title('Predicted Values with Prediction Interval for Stepwise Regression', fontsize=10)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ================================================
# Phase III - Classification Analysis
# ================================================
# We don't have to standardize our dataset for Decision Tree
X_train, X_test, y_train, y_test = train_test_split(X_cat, y_cat, shuffle=True, random_state=5805, test_size=0.2)

# ================================================
# Decision Tree Analysis - Pre Pruning
# ================================================

clf = DecisionTreeClassifier(random_state=5805)
clf.fit(X_train, y_train)

tuned_parameters = {
    'max_depth': [3, 5, 8],
    'min_samples_split': [10, 12, 13],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [2, 3, 4],
    'splitter': ['best', 'random'],
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid_search = GridSearchCV(clf, tuned_parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_
print("Best Parameters (pre pruning): ", grid_search.best_params_)
print("Best Accuracy (using grid search/ pre pruning): ", grid_search.best_score_.round(2))

best_clf.fit(X_train, y_train)

plt.figure(figsize=(40, 20))
plot_tree(best_clf, filled=True, feature_names=X.columns, fontsize=8, rounded=True)
plt.show()

y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set (with the best model/ pre pruning): {accuracy:.2f}")

cnf_matrix = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cnf_matrix[0][0], cnf_matrix[0][1], cnf_matrix[1][0], cnf_matrix[1][1]
recall_pre = TP / (TP + FN)
specificity_pre = TN / (TN + FP)
f1_pre = f1_score(y_test, y_pred)
y_proba_pre = best_clf.predict_proba(X_test)[::, -1]
pre_fpr, pre_tpr, _ = roc_curve(y_test, y_proba_pre)
auc_pre = roc_auc_score(y_test, y_proba_pre)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - Pre Prunning Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(pre_fpr, pre_tpr, label=f'Pre pruning auc = {auc_pre:.2f}')
plt.plot(pre_fpr, pre_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - Pre pruning')
plt.tight_layout()
plt.show()

p = PrettyTable()
p.field_names = ['Model', 'True Negative', 'False Positive', 'False Negative', 'True Positive', 'Accuracy', 'Recall',
                 'Specificity', 'F1 Score', 'AUC']


p.add_row(['Pre pruning Decision Tree', cnf_matrix[0, 0], cnf_matrix[0, 1], cnf_matrix[1, 0], cnf_matrix[1, 1],
           f'{accuracy: .2f}',
           f'{recall_pre: .2f}', f'{specificity_pre: .2f}', f'{f1_pre: .2f}', f'{auc_pre: .2f}'])


# Cross-Validation
def cross_validator(classifier, X_, y_, clf_name):
    n_splits = 5 if clf_name not in ('SVM (linear)', 'SVM (poly)', 'SVM (rbf)',
                                     'Random Forest (stacking)', 'Random Forest (Bagging)',
                                     'Random Forest (boosting)') else 3

    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)

    accuracy_scores = []

    for train_index, test_index in stratified_kfold.split(X_, y_):
        X_train_cv, X_test_cv = X_.iloc[train_index], X_.iloc[test_index]
        y_train_cv, y_test_cv = y_.iloc[train_index], y_.iloc[test_index]

        classifier.fit(X_train_cv, y_train_cv)

        y_pred_cv = classifier.predict(X_test_cv)

        accuracy_cv = accuracy_score(y_test_cv, y_pred_cv)
        accuracy_scores.append(accuracy_cv)

    c = PrettyTable()
    c.field_names = ['Fold', 'Accuracy']
    for i, acc in enumerate(accuracy_scores, 1):
        c.add_row([i, acc.round(2)])

    mean_accuracy = np.mean(accuracy_scores)
    print(c.get_string(title=f'k = {n_splits} fold accuracy'))
    print(f'{clf_name} classifier, Accuracy Mean = {mean_accuracy: .2f}')


cross_validator(best_clf, X_cat, y_cat, 'Pre-pruned Decision Tree')

# ================================================
# Decision Tree Analysis - Post Pruning
# ================================================
path = clf.cost_complexity_pruning_path(X_train, y_train)
alphas = path['ccp_alphas']

accuracy_train, accuracy_test = [], []
for i in alphas:
    clf_x = DecisionTreeClassifier(random_state=5805, ccp_alpha=i)
    clf_x.fit(X_train, y_train)
    y_train_pred = clf_x.predict(X_train)
    y_test_pred = clf_x.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))

fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(alphas, accuracy_train, label="train",
        drawstyle="steps-post")
ax.plot(alphas, accuracy_test, label="test",
        drawstyle="steps-post")
ax.legend()
plt.grid()
plt.tight_layout()
plt.show()

max_ccp = alphas[accuracy_test.index(max(accuracy_test))]
print('Max alpha = ', max_ccp.round(5))

clf2 = DecisionTreeClassifier(random_state=5805, ccp_alpha=max_ccp)
clf2.fit(X_train, y_train)
y_train_pred = clf2.predict(X_train)
y_test_pred = clf2.predict(X_test)
print(f'Train accuracy (post pruning) {accuracy_score(y_train, y_train_pred): .2f}')
print(f'Test accuracy (post pruning) {accuracy_score(y_test, y_test_pred): .2f}')
accuracy2 = accuracy_score(y_test, y_test_pred)
plt.figure(figsize=(16, 8))
tree.plot_tree(clf2, rounded=True, filled=True)
plt.show()

cnf_matrix2 = confusion_matrix(y_test, y_test_pred)
TN, FP, FN, TP = cnf_matrix2[0][0], cnf_matrix2[0][1], cnf_matrix2[1][0], cnf_matrix2[1][1]
recall_post = TP / (TP + FN)
specificity_post = TN / (TN + FP)
f1_post = f1_score(y_test, y_pred)
y_proba_post = clf2.predict_proba(X_test)[::, -1]
post_fpr, post_tpr, _ = roc_curve(y_test, y_proba_post)
auc_post = roc_auc_score(y_test, y_proba_post)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix2, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - Post Prunned Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(pre_fpr, pre_fpr, 'r--')
plt.plot(post_fpr, post_tpr, label=f'Post pruning auc = {auc_post:.2f}')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - Post pruning')
plt.tight_layout()
plt.show()

p.add_row(
    ['Post pruning Decision Tree', cnf_matrix2[0, 0], cnf_matrix2[0, 1], cnf_matrix2[1, 0], cnf_matrix2[1, 1],
     f'{accuracy2: .2f}',
     f'{recall_post: .2f}', f'{specificity_post: .2f}', f'{f1_post: .2f}', f'{auc_post: .2f}'])

cross_validator(clf2, X_cat, y_cat, 'Post-pruned Decision Tree')

# ================================================
# Logistic Regression
# ================================================
model_logistic = LogisticRegression()

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

grid_search = GridSearchCV(model_logistic, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters (Logistic Regression): ", grid_search.best_params_)

best_model_logistic = grid_search.best_estimator_
y_pred_logistic = best_model_logistic.predict(X_test)
cnf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
y_proba_logistic = best_model_logistic.predict_proba(X_test)[:, 1]
logistic_fpr, logistic_tpr, _ = roc_curve(y_test, y_proba_logistic)
auc_logistic = roc_auc_score(y_test, y_proba_logistic)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

TN, FP, FN, TP = cnf_matrix_logistic[0][0], cnf_matrix_logistic[0][1], cnf_matrix_logistic[1][0], \
    cnf_matrix_logistic[1][1]
specificity_logistic = TN / (TN + FP)
recall_logistic = TP / (TP + FN)
f1_logistic = f1_score(y_test, y_pred_logistic)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_logistic, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(logistic_fpr, logistic_tpr, label=f'Logistic auc = {auc_logistic:.2f}')
plt.plot(logistic_fpr, logistic_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - Logistic Regression')
plt.tight_layout()
plt.show()

p.add_row(['Logistic Regression', cnf_matrix_logistic[0, 0], cnf_matrix_logistic[0, 1], cnf_matrix_logistic[1, 0],
           cnf_matrix_logistic[1, 1], f'{accuracy_logistic: .2f}',
           f'{recall_logistic: .2f}', f'{specificity_logistic: .2f}', f'{f1_logistic: .2f}', f'{auc_logistic: .2f}'])

cross_validator(model_logistic, X_cat, y_cat, 'Logistic Regression')

# =======================================
# KNN Classifier
# =======================================
model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)

sc_cat = StandardScaler()

X_std_cat = sc_cat.fit_transform(X_cat)
X_std_cat = pd.DataFrame(X_std_cat, columns=X_cat.columns)

X_train, X_test, y_train, y_test = train_test_split(X_std_cat, y_cat, shuffle=True, random_state=5805, test_size=0.2)


model.fit(X_train, y_train)

k_range = list(range(1, 21))
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',
                    return_train_score=True, verbose=1)
grid.fit(X_train, y_train)

results = grid.cv_results_
k_values = results['param_n_neighbors'].data
error_rates = 1 - results['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(k_values, error_rates, marker='o', linestyle='dashed', markersize=8)
plt.title('KNN Grid Search with Cross-Validation: k vs. Error Rate')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()

best_k = grid.best_params_['n_neighbors']
best_score = grid.best_score_
print(f"Best k: {best_k}")

# Checking Accuracy on Test Data
nn = KNeighborsClassifier(n_neighbors=best_k)
nn.fit(X_train, y_train)
y_pred_g = nn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_g)
cnf_matrix_knn = confusion_matrix(y_test, y_pred_g)

TN, FP, FN, TP = cnf_matrix_knn[0][0], cnf_matrix_knn[0][1], cnf_matrix_knn[1][0], cnf_matrix_knn[1][1]
recall_knn = TP / (TP + FN)
specificity_knn = TN / (TN + FP)
f1_knn = f1_score(y_test, y_pred_g)
y_proba_knn = nn.predict_proba(X_test)[::, -1]
knn_fpr, knn_tpr, _ = roc_curve(y_test, y_proba_knn)
auc_knn = roc_auc_score(y_test, y_proba_knn)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_knn, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(knn_fpr, knn_tpr, label=f'KNN auc = {auc_knn:.2f}')
plt.plot(knn_fpr, knn_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - KNN')
plt.tight_layout()
plt.show()

p.add_row(['KNN', cnf_matrix_knn[0, 0], cnf_matrix_knn[0, 1], cnf_matrix_knn[1, 0],
           cnf_matrix_knn[1, 1], f'{accuracy_knn: .2f}',
           f'{recall_knn: .2f}', f'{specificity_knn: .2f}', f'{f1_knn: .2f}', f'{auc_knn: .2f}'])

cross_validator(nn, X_cat, y_cat, 'KNN')


# =======================================
# SVM Classifier - linear
# =======================================
model_svm_linear = SVC(kernel='linear', probability=True)

param_grid = {'C': [0.01, 1, 10]}

grid_search = GridSearchCV(model_svm_linear, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters (SVM Linear): ", grid_search.best_params_)

best_model_svm_linear = grid_search.best_estimator_
y_pred_svm_linear = best_model_svm_linear.predict(X_test)
cnf_matrix_svm_linear = confusion_matrix(y_test, y_pred_svm_linear)
y_proba_svm_linear = best_model_svm_linear.predict_proba(X_test)[:, 1]
svm_linear_fpr, svm_linear_tpr, _ = roc_curve(y_test, y_proba_svm_linear)
auc_svm_linear = roc_auc_score(y_test, y_proba_svm_linear)
accuracy_svm_linear = accuracy_score(y_test, y_pred_svm_linear)

TN, FP, FN, TP = cnf_matrix_svm_linear[0][0], cnf_matrix_svm_linear[0][1], cnf_matrix_svm_linear[1][0], \
    cnf_matrix_svm_linear[1][1]
specificity_svm_linear = TN / (TN + FP)
recall_svm_linear = TP / (TP + FN)
f1_svm_linear = f1_score(y_test, y_pred_svm_linear)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_svm_linear, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - SVM (linear)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(svm_linear_fpr, svm_linear_tpr, label=f'SVM (linear) auc = {auc_svm_linear:.2f}')
plt.plot(svm_linear_fpr, svm_linear_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - SVM Classifier (linear)')
plt.tight_layout()
plt.show()

p.add_row(['SVM Classifier (linear)', cnf_matrix_svm_linear[0, 0], cnf_matrix_svm_linear[0, 1],
           cnf_matrix_svm_linear[1, 0], cnf_matrix_svm_linear[1, 1], f'{accuracy_svm_linear: .2f}',
           f'{recall_svm_linear: .2f}', f'{specificity_svm_linear: .2f}',
           f'{f1_svm_linear: .2f}', f'{auc_svm_linear: .2f}'])

cross_validator(model_svm_linear, X_std_cat, y_cat, 'SVM (linear)')


# =======================================
# SVM Classifier - poly
# =======================================
X_train, X_test, y_train, y_test = train_test_split(X_std_cat, y_cat, shuffle=True, random_state=5805, test_size=0.2)

model_svm_poly = SVC(kernel='poly', probability=True)

param_grid = {'C': [1, 2, 4],
              'degree': [2, 3, 4]}

grid_search = GridSearchCV(model_svm_poly, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters (SVM Polynomial): ", grid_search.best_params_)

best_model_svm_poly = grid_search.best_estimator_
y_pred_svm_poly = best_model_svm_poly.predict(X_test)
cnf_matrix_svm_poly = confusion_matrix(y_test, y_pred_svm_poly)
y_proba_svm_poly = best_model_svm_poly.predict_proba(X_test)[:, 1]
svm_poly_fpr, svm_poly_tpr, _ = roc_curve(y_test, y_proba_svm_poly)
auc_svm_poly = roc_auc_score(y_test, y_proba_svm_poly)
accuracy_svm_poly = accuracy_score(y_test, y_pred_svm_poly)

TN, FP, FN, TP = cnf_matrix_svm_poly[0][0], cnf_matrix_svm_poly[0][1], cnf_matrix_svm_poly[1][0], \
    cnf_matrix_svm_poly[1][1]
specificity_svm_poly = TN / (TN + FP)
recall_svm_poly = TP / (TP + FN)
f1_svm_poly = f1_score(y_test, y_pred_svm_poly)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_svm_poly, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - SVM (poly)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(svm_poly_fpr, svm_poly_tpr, label=f'SVM (poly) auc = {auc_svm_poly:.2f}')
plt.plot(svm_poly_fpr, svm_poly_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - SVM Classifier (poly)')
plt.tight_layout()
plt.show()

p.add_row(['SVM Classifier (poly)', cnf_matrix_svm_poly[0, 0], cnf_matrix_svm_poly[0, 1],
           cnf_matrix_svm_poly[1, 0], cnf_matrix_svm_poly[1, 1], f'{accuracy_svm_poly: .2f}',
           f'{recall_svm_poly: .2f}', f'{specificity_svm_poly: .2f}',
           f'{f1_svm_poly: .2f}', f'{auc_svm_poly: .2f}'])

cross_validator(model_svm_poly, X_std_cat, y_cat, 'SVM (poly)')


# =======================================
# SVM Classifier - radial base kernel
# =======================================

X_train, X_test, y_train, y_test = train_test_split(X_std_cat, y_cat, shuffle=True, random_state=5805, test_size=0.2)

model_svm_rbf = SVC(kernel='rbf', probability=True)

param_grid = {'C': [0.1, 1, 10],
              'gamma': ['scale', 'auto']}

grid_search = GridSearchCV(model_svm_rbf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters (SVM rbf): ", grid_search.best_params_)

best_model_svm_rbf = grid_search.best_estimator_
y_pred_svm_rbf = best_model_svm_rbf.predict(X_test)
cnf_matrix_svm_rbf = confusion_matrix(y_test, y_pred_svm_rbf)
y_proba_svm_rbf = best_model_svm_rbf.predict_proba(X_test)[:, 1]
svm_rbf_fpr, svm_rbf_tpr, _ = roc_curve(y_test, y_proba_svm_rbf)
auc_svm_rbf = roc_auc_score(y_test, y_proba_svm_rbf)
accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)


TN, FP, FN, TP = cnf_matrix_svm_rbf[0][0], cnf_matrix_svm_rbf[0][1], cnf_matrix_svm_rbf[1][0], \
    cnf_matrix_svm_rbf[1][1]
specificity_svm_rbf = TN / (TN + FP)
recall_svm_rbf = TP / (TP + FN)
f1_svm_rbf = f1_score(y_test, y_pred_svm_rbf)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_svm_rbf, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - SVM (rbf)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(svm_rbf_fpr, svm_rbf_tpr, label=f'SVM (rbf) auc = {auc_svm_rbf:.2f}')
plt.plot(svm_rbf_fpr, svm_rbf_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - SVM Classifier (rbf)')
plt.tight_layout()
plt.show()

p.add_row(['SVM Classifier (rbf)', cnf_matrix_svm_rbf[0, 0], cnf_matrix_svm_rbf[0, 1],
           cnf_matrix_svm_rbf[1, 0], cnf_matrix_svm_rbf[1, 1], f'{accuracy_svm_rbf: .2f}',
           f'{recall_svm_rbf: .2f}', f'{specificity_svm_rbf: .2f}',
           f'{f1_svm_rbf: .2f}', f'{auc_svm_rbf: .2f}'])

cross_validator(model_svm_rbf, X_std_cat, y_cat, 'SVM (rbf)')

# =======================================
# Naive Bayes Classifier
# =======================================

X_train, X_test, y_train, y_test = train_test_split(X_cat, y_cat, shuffle=True, random_state=5805, test_size=0.2)

model_naive = GaussianNB()
model_naive.fit(X_train, y_train)

y_pred_naive = model_naive.predict(X_test)
cnf_matrix_naive = confusion_matrix(y_test, y_pred_naive)
y_proba_naive = model_naive.predict_proba(X_test)[::, -1]
naive_fpr, naive_tpr, _ = roc_curve(y_test, y_proba_naive)
auc_naive = roc_auc_score(y_test, y_proba_naive)
accuracy_naive = accuracy_score(y_test, y_pred_naive)

TN, FP, FN, TP = cnf_matrix_naive[0][0], cnf_matrix_naive[0][1], cnf_matrix_naive[1][0], \
    cnf_matrix_naive[1][1]
specificity_naive = TN / (TN + FP)
recall_naive = TP / (TP + FN)
f1_naive = f1_score(y_test, y_pred_naive)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_naive, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(naive_fpr, naive_tpr, label=f'Naive Bayes auc = {auc_naive:.2f}')
plt.plot(naive_fpr, naive_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - Naive Bayes Classifier')
plt.tight_layout()
plt.show()

p.add_row(['Naive Bayes Classifier', cnf_matrix_naive[0, 0], cnf_matrix_naive[0, 1],
           cnf_matrix_naive[1, 0], cnf_matrix_naive[1, 1], f'{accuracy_naive: .2f}',
           f'{recall_naive: .2f}', f'{specificity_naive: .2f}',
           f'{f1_naive: .2f}', f'{auc_naive: .2f}'])

cross_validator(model_naive, X_cat, y_cat, 'Naive Bayes')

# =======================================
# Random Forest Classifier - Bagging
# =======================================
base_classifier = RandomForestClassifier(random_state=5805)

model_bagging = BaggingClassifier(base_classifier, random_state=5805)

param_grid = {
    'base_estimator__n_estimators': [10, 15, 20],
    'n_estimators': [10, 15, 20],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.5, 0.7, 1.0]
}

grid_search = GridSearchCV(model_bagging, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters (Random Forest Bagging): ", grid_search.best_params_)

best_model_bagging = grid_search.best_estimator_
y_pred_bagging = best_model_bagging.predict(X_test)
cnf_matrix_bagging = confusion_matrix(y_test, y_pred_bagging)
y_proba_bagging = best_model_bagging.predict_proba(X_test)[:, 1]
bagging_fpr, bagging_tpr, _ = roc_curve(y_test, y_proba_bagging)
auc_bagging = roc_auc_score(y_test, y_proba_bagging)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)


TN, FP, FN, TP = cnf_matrix_bagging[0][0], cnf_matrix_bagging[0][1], cnf_matrix_bagging[1][0], \
    cnf_matrix_bagging[1][1]
specificity_bagging = TN / (TN + FP)
recall_bagging = TP / (TP + FN)
f1_bagging = f1_score(y_test, y_pred_bagging)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_bagging, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - Random Forest (Bagging)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(bagging_fpr, bagging_tpr, label=f'Random Forest (Bagging) auc = {auc_bagging:.2f}')
plt.plot(bagging_fpr, bagging_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - Random Forest Classifier (Bagging)')
plt.tight_layout()
plt.show()

p.add_row(['Bagging Random Forest Classifier', cnf_matrix_bagging[0, 0], cnf_matrix_bagging[0, 1],
           cnf_matrix_bagging[1, 0], cnf_matrix_bagging[1, 1], f'{accuracy_bagging: .2f}',
           f'{recall_bagging: .2f}', f'{specificity_bagging: .2f}',
           f'{f1_bagging: .2f}', f'{auc_bagging: .2f}'])

cross_validator(model_bagging, X_cat, y_cat, 'Random Forest (Bagging)')

# =======================================
# Random Forest Classifier - Stacking
# =======================================
base_classifiers = [
    ('log', LogisticRegression()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
]

model_stacking = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=RandomForestClassifier(random_state=5805)
)

param_grid = {
    'final_estimator__n_estimators': [10, 15, 20],
    'final_estimator__max_depth': [5, 10],
    'final_estimator__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model_stacking, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters (Random Forest Stacking): ", grid_search.best_params_)

best_model_stacking = grid_search.best_estimator_
y_pred_stacking = best_model_stacking.predict(X_test)
cnf_matrix_stacking = confusion_matrix(y_test, y_pred_stacking)
y_proba_stacking = best_model_stacking.predict_proba(X_test)[:, 1]
stacking_fpr, stacking_tpr, _ = roc_curve(y_test, y_proba_stacking)
auc_stacking = roc_auc_score(y_test, y_proba_stacking)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)


TN, FP, FN, TP = cnf_matrix_stacking[0][0], cnf_matrix_stacking[0][1], cnf_matrix_stacking[1][0], \
    cnf_matrix_stacking[1][1]
specificity_stacking = TN / (TN + FP)
recall_stacking = TP / (TP + FN)
f1_stacking = f1_score(y_test, y_pred_stacking)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_stacking, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - Random Forest (Stacking)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(stacking_fpr, stacking_tpr, label=f'Random Forest (stacking) auc = {auc_stacking:.2f}')
plt.plot(stacking_fpr, stacking_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - Random Forest Classifier (stacking)')
plt.tight_layout()
plt.show()

p.add_row(['Stacking Random Forest Classifier', cnf_matrix_stacking[0, 0], cnf_matrix_stacking[0, 1],
           cnf_matrix_stacking[1, 0], cnf_matrix_stacking[1, 1], f'{accuracy_stacking: .2f}',
           f'{recall_stacking: .2f}', f'{specificity_stacking: .2f}',
           f'{f1_stacking: .2f}', f'{auc_stacking: .2f}'])

cross_validator(model_stacking, X_cat, y_cat, 'Random Forest (stacking)')

# =======================================
# Random Forest Classifier - Boosting
# =======================================
base_classifier = RandomForestClassifier(random_state=5805)

model_boosting = AdaBoostClassifier(base_estimator=base_classifier, random_state=5805)

param_grid = {
    'base_estimator__n_estimators': [10, 15, 20],
    'n_estimators': [10, 15, 20],
    'learning_rate': [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(model_boosting, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters (Random Forest Boosting): ", grid_search.best_params_)

best_model_boosting = grid_search.best_estimator_
y_pred_boosting = best_model_boosting.predict(X_test)
cnf_matrix_boosting = confusion_matrix(y_test, y_pred_boosting)
y_proba_boosting = best_model_boosting.predict_proba(X_test)[:, 1]
boosting_fpr, boosting_tpr, _ = roc_curve(y_test, y_proba_boosting)
auc_boosting = roc_auc_score(y_test, y_proba_boosting)
accuracy_boosting = accuracy_score(y_test, y_pred_boosting)

TN, FP, FN, TP = cnf_matrix_boosting[0][0], cnf_matrix_boosting[0][1], cnf_matrix_boosting[1][0], \
    cnf_matrix_boosting[1][1]
specificity_boosting = TN / (TN + FP)
recall_boosting = TP / (TP + FN)
f1_boosting = f1_score(y_test, y_pred_boosting)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_boosting, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - Random Forest (Boosting)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(boosting_fpr, boosting_tpr, label=f'Random Forest (boosting) auc = {auc_boosting:.2f}')
plt.plot(boosting_fpr, boosting_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - Random Forest Classifier (boosting)')
plt.tight_layout()
plt.show()

p.add_row(['Boosting Random Forest Classifier', cnf_matrix_boosting[0, 0], cnf_matrix_boosting[0, 1],
           cnf_matrix_boosting[1, 0], cnf_matrix_boosting[1, 1], f'{accuracy_boosting: .2f}',
           f'{recall_boosting: .2f}', f'{specificity_boosting: .2f}',
           f'{f1_boosting: .2f}', f'{auc_boosting: .2f}'])

cross_validator(model_boosting, X_cat, y_cat, 'Random Forest (boosting)')

# =======================================
# Neural Network
# =======================================
model_neural = MLPClassifier(random_state=5805)

param_grid = {
    'hidden_layer_sizes': [(64, 32), (32, 16), (16, 8)],
    'max_iter': [50, 100, 200],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(model_neural, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters (Neural Network): ", grid_search.best_params_)

best_model_neural = grid_search.best_estimator_
y_pred_neural = best_model_neural.predict(X_test)
cnf_matrix_neural = confusion_matrix(y_test, y_pred_neural)
y_proba_neural = best_model_neural.predict_proba(X_test)[:, 1]
neural_fpr, neural_tpr, _ = roc_curve(y_test, y_proba_neural)
auc_neural = roc_auc_score(y_test, y_proba_neural)
accuracy_neural = accuracy_score(y_test, y_pred_neural)

TN, FP, FN, TP = cnf_matrix_neural[0][0], cnf_matrix_neural[0][1], cnf_matrix_neural[1][0], \
    cnf_matrix_neural[1][1]
specificity_neural = TN / (TN + FP)
recall_neural = TP / (TP + FN)
f1_neural = f1_score(y_test, y_pred_neural)

plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_neural, annot=True, fmt='d', cmap='inferno')
plt.title('Confusion Matrix - Multi Layered Perceptron')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure()
plt.plot(neural_fpr, neural_tpr, label=f'Neural Network auc = {auc_neural:.2f}')
plt.plot(neural_fpr, neural_fpr, 'r--')
plt.legend(loc=4)
plt.grid()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - Neural Network Classifier')
plt.tight_layout()
plt.show()

p.add_row(['Neural Network Classifier', cnf_matrix_neural[0, 0], cnf_matrix_neural[0, 1],
           cnf_matrix_neural[1, 0], cnf_matrix_neural[1, 1], f'{accuracy_neural: .2f}',
           f'{recall_neural: .2f}', f'{specificity_neural: .2f}',
           f'{f1_neural: .2f}', f'{auc_neural: .2f}'])

cross_validator(model_neural, X_cat, y_cat, 'Neural Network')

print(p.get_string(title='Model Comparison Table'))


# =====================================
# Phase IV - Clustering and Association
# =====================================
# Association Rule Mining - Apriori
# =====================================
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df_categories = df_categories.fillna('')
transactions = df_categories.values.tolist()

a = TransactionEncoder()
a_data = a.fit_transform(transactions)
df_apriori = pd.DataFrame(a_data, columns=a.columns_)
print(df_apriori)

df_apriori = apriori(df_apriori, min_support=0.2, use_colnames=True, verbose=1)
print(df_apriori)
df_ar = association_rules(df_apriori, metric='confidence', min_threshold=0.6)
df_ar = df_ar.sort_values(['confidence', 'lift'], ascending=[False, False])
print(df_ar.to_string())

# =====================================
# Clustering - Kmeans
# =====================================

# ========================================================
# Elbow Method Within cluster Sum of Squared Errors (WSS)
# ========================================================
from sklearn.cluster import KMeans

def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # Calculate squared Euclidean distance for each point from its cluster center
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += sum((points[i] - curr_center) ** 2)

        sse.append(curr_sse)
    return sse


k = 12
sse = calculate_WSS(X_cat.values, k)
plt.figure()
plt.plot(np.arange(1, k + 1, 1), sse)
plt.xticks(np.arange(1, k + 1, 1))
plt.grid()
plt.xlabel('k')
plt.ylabel('WSS')
plt.title('k selection in k-mean Elbow Algorithm')
plt.show()

# ========================================================
# Silhouette Method for selection of K
# ========================================================
from sklearn.metrics import silhouette_score

sil = []
kmax = 12

for k in range(2, kmax + 1):
    kmeans = KMeans(n_clusters=k).fit(X_cat.values)
    labels = kmeans.labels_
    sil.append(silhouette_score(X_cat.values, labels, metric='euclidean'))

plt.figure()
plt.plot(np.arange(2, k + 1, 1), sil, 'bx-')
plt.xticks(np.arange(2, k + 1, 1))
plt.grid()
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method')
plt.show()
