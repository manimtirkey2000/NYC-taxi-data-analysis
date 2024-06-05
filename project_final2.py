import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
from prettytable import PrettyTable
from scipy.stats import shapiro, kstest, normaltest

title_font = {'family': 'serif', 'color': 'blue'}
label_font = {'family': 'serif', 'color': 'darkred'}

df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
df_taxi_zones = pd.read_csv('taxi+_zone_lookup.csv')

# Initial discussion
print(df.columns)
print(df.dtypes)

table2 = PrettyTable()

# Add column names
table2.field_names = ['Column', 'Unique Values']

# Add data to the table
for column in df.columns:
    table2.add_row([column, df[column].nunique()])

# Print the table
print(table2.get_string(title='Number of Unique values in Original Dataset'))

warnings.filterwarnings("ignore")
print(len(df.index))
print(df.isna().sum())
print(df.head().to_string())
print(type(df))

print(df_taxi_zones.head())

yx = PrettyTable()

yx.field_names = df_taxi_zones.columns

for i in range(len(df_taxi_zones.index)):
    yx.add_row(df_taxi_zones.iloc[i, :])


print(yx.get_string(title='Taxi Zone Dictionary'))


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

df_temp = df.copy()

category_mapping_fwd_flag = {'N': 0, 'Y': 1}
reverse_day_mapping = {1.0: 'Sunday', 2.0: 'Monday', 3.0: 'Tuesday', 4.0: 'Wednesday', 5.0: 'Thursday', 6.0: 'Friday',
                       7.0: 'Saturday'}
weekday_mapping = {'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7}

category_mapping = {1: 'Credit card', 2: 'Cash', 3: 'No charge', 4: 'Dispute', 5: 'Unknown', 6: 'Voided trip',
                    0: 'Voided trip'}
reversed_mapping = {
    'Credit card': 1.0,
    'Cash': 2.0,
    'No charge': 3.0,
    'Dispute': 4.0,
    'Unknown': 5.0,
    'Voided trip': 6.0
}
df['payment_type'] = df['payment_type'].replace(category_mapping)
category_mapping_ratecode = {1.0: 'Standard rate', 2.0: 'JFK', 3.0: 'Newark', 4.0: 'Nassau/Westchester',
                             5.0: 'Negotiated fare', 6.0: 'Group ride', 99.0: 'Unknown'}
reverse_category_mapping_ratecode = {'Standard rate': 1.0, 'JFK': 2.0, 'Newark': 3.0, 'Nassau/Westchester': 4.0,
                                     'Negotiated fare': 5.0, 'Group ride': 6.0, 'Unknown': 99.0}
df['RatecodeID'] = df['RatecodeID'].replace(category_mapping_ratecode)

df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
df['pickup_day'] = df['tpep_pickup_datetime'].dt.day_name()
df['dropoff_day'] = df['tpep_dropoff_datetime'].dt.day_name()
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['dropoff_hour'] = df['tpep_dropoff_datetime'].dt.hour

df_temp['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
df_temp['pickup_day'] = df['tpep_pickup_datetime'].dt.day
df_temp['dropoff_day'] = df['tpep_dropoff_datetime'].dt.day
df_temp['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df_temp['dropoff_hour'] = df['tpep_dropoff_datetime'].dt.hour
df_temp['store_and_fwd_flag'] = df_temp['store_and_fwd_flag'].replace(category_mapping_fwd_flag)

grouper = df.groupby([pd.Grouper(key='pickup_day'), pd.Grouper(key='pickup_hour')])
temp_df = grouper['total_amount'].mean().round(2)
temp_df = temp_df.to_frame().reset_index()
temp_df.set_index('pickup_hour')
temp_df = temp_df.pivot(index='pickup_hour', columns='pickup_day', values='total_amount')

table = PrettyTable()

# Add column names
table.field_names = ['Hour'] + list(temp_df.columns)

# Add data to the table
for index, row in temp_df.iterrows():
    table.add_row([index] + list(row))

# Print the table
print(table)

print(df.head().round(2).to_string())

df = df.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)
df_temp = df_temp.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)
df_outlier = df.copy()

# ================
# Outlier Removal
# ================

plt.boxplot(df['total_amount'], vert=False, patch_artist=True)
plt.title('Boxplot - total_amount with outliers', fontdict=title_font)
plt.xlabel('total_amount', fontdict=label_font)
plt.show()


def QQ_calc(temp):
    for column_name in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[column_name]) and (column_name != 'passenger_count'
                                                                 and column_name != 'payment_type'
                                                                 and column_name != 'congestion_surcharge'):
            Q1 = temp[column_name].quantile(0.25)
            Q3 = temp[column_name].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            temp = temp[(temp[column_name] >= lower_bound) & (temp[column_name] <= upper_bound)]

    return temp


df = QQ_calc(df)

plt.boxplot(df['total_amount'], vert=False, patch_artist=True)
plt.title('Boxplot - total_amount without outliers', fontdict=title_font)
plt.xlabel('total_amount', fontdict=label_font)
plt.show()

df_temp = QQ_calc(df_temp)

# Drop off Grouped sum, tip, airport fee
pickup_tip_df = df.groupby('PULocationID')[['fare_amount', 'trip_distance', 'tip_amount']].sum()

grouper2 = df.groupby([pd.Grouper(key='PULocationID'), pd.Grouper(key='payment_type')])
temp_df2 = grouper2['trip_distance'].mean()
temp_df2 = temp_df2.to_frame().reset_index()
temp_df2.set_index('PULocationID')
temp_df2 = temp_df2.pivot(index='payment_type', columns='PULocationID', values='trip_distance')

# =============
# PCA
# =============
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df_sampled = df.sample(n=50000, random_state=5764)
df_sampled.reset_index(drop=True, inplace=True)

df_sampled3 = df.sample(n=5000, random_state=5805)
df_sampled3.reset_index(drop=True, inplace=True)

df_sampled3['store_and_fwd_flag'] = df_sampled3['store_and_fwd_flag'].replace(category_mapping_fwd_flag)
df_sampled3['payment_type'] = df_sampled3['payment_type'].replace(reversed_mapping)
df_sampled3['pickup_day'] = df_sampled3['pickup_day'].replace(weekday_mapping)
df_sampled3['dropoff_day'] = df_sampled3['dropoff_day'].replace(weekday_mapping)
df_sampled3['RatecodeID'] = df_sampled3['RatecodeID'].replace(reverse_category_mapping_ratecode)

df_sampled3 = df_sampled3.drop(['mta_tax', 'tolls_amount', 'improvement_surcharge', 'airport_fee'], axis=1)


def normal_test(x, selected_test):
    if selected_test == 'Shapiro Test':
        stats, p = shapiro(x)
        return stats, p

    elif selected_test == 'K_S test':
        np.random.seed(5764)

        mean = np.mean(x)
        std = np.std(x)
        dist = np.random.normal(mean, std, len(x))
        stats, p = kstest(x, dist)

        return stats, p

    elif selected_test == 'Da_k_squared':
        stats, p = normaltest(x)

        return stats, p


stats_table = PrettyTable()

stats_table.field_names = ['Feature', 'Da_k_squared Test', 'K_S test', 'Shapiro Test', 'Verdict']

for col in df_sampled3.columns:
    stats1, p1 = normal_test(df_sampled3[col], 'Da_k_squared')
    stats2, p2 = normal_test(df_sampled3[col], 'K_S test')
    stats3, p3 = normal_test(df_sampled3[col], 'Shapiro Test')

    verdict = 'Normal'
    if p1 < 0.01 and p2 < 0.01 and p3 < 0.01:
        verdict = 'Not Normal'

    stats_table.add_row([col, f'Stats = {stats1: .2f}, p = {p1 :.2f}', f'Stats = {stats2: .2f}, p = {p2 :.2f}',
                         f'Stats = {stats3: .2f}, p = {p3: .2f}', verdict])

print(stats_table.get_string(title='Normality Test'))

df_sampled2 = df_temp.sample(n=10000, random_state=5764)
df_sampled2.reset_index(drop=True, inplace=True)

X_PCA = df_sampled2.drop(['total_amount'], axis=1)
y_PCA = df_sampled['total_amount']

scalar = StandardScaler()
X_std = scalar.fit_transform(X_PCA)
X_std = pd.DataFrame(X_std, columns=X_PCA.columns)

pca = PCA(n_components=20, svd_solver='full')
pca.fit(X_std)


exp = np.cumsum(pca.explained_variance_ratio_).round(2)

pca_p = PrettyTable()

pca_p.field_names = ['Component', 'Cumulative Explained Variance Ratio']

for i in range(len(exp)):
    pca_p.add_row([i + 1, exp[i]])

print(pca_p.get_string(title='PCA Analysis'))


pca2 = PCA(n_components=9, svd_solver='full')
pca2.fit(X_std)
X_PCA2 = pca2.transform(X_std)

_, d_raw, _ = np.linalg.svd(X_std)
_, d_pca, _ = np.linalg.svd(X_PCA2)

pq = PrettyTable()
pq.field_names = [f'Raw Condition Number = {np.linalg.cond(X_std): .2f}',
                  f'Transformed Condition Number = {np.linalg.cond(X_PCA2): .2f}']

for i, (raw_val, pca_val) in enumerate(zip(d_raw, d_pca)):
    pq.add_row([f'{raw_val :.2f}', f'{pca_val :.2f}'])

for i in range(9, 20):
    pq.add_row([f'{d_raw[i]: .2f}', '-'])

print(pq.get_string(title='Singular Values'))


# ===========
# Statistics
# ===========
print(df.describe().round(2).to_string())

sns.kdeplot(data=df_sampled3[['trip_distance', 'total_amount', 'tip_amount',
                              'fare_amount', 'trip_duration']])
plt.title("Multivariate Kernel Density Estimate - Between Numerical Features", fontdict=title_font)
plt.tight_layout()
plt.show()

# ===========
# EDA
# ===========

# Distplot - fare_amount
sns.set_style("whitegrid")
sns.distplot(df['fare_amount'], kde=True, color='blue')
plt.xlabel("fare_amount", fontdict=label_font)
plt.ylabel("Density", fontdict=label_font)
plt.title("Distribution Plot - Fare Amount", fontdict=title_font)
plt.tight_layout()
plt.show()

# Countplot - passenger_count
sns.countplot(data=df, x='passenger_count')
plt.xlabel('Count', fontdict=label_font)
plt.ylabel('passenger_count', fontdict=label_font)
plt.title('Countplot - passenger_count', fontdict=title_font)
plt.show()

# KDE plot - tip_amount
custom_palette = ["#FF5733", "#33FF57"]
sns.kdeplot(data=df, x="tip_amount", fill=True, alpha=0.6, palette=custom_palette, linewidth=3)
plt.xlabel('tip_amount', fontdict=label_font)
plt.ylabel('Density', fontdict=label_font)
plt.grid()
plt.title('Filled KDE Plot for tip_amount', fontdict=title_font)
plt.tight_layout()
plt.show()

# Bar Plot - Stacked
cross_tab = pd.crosstab(df['payment_type'], df['VendorID'])
ax = cross_tab.plot(kind='bar', stacked=True, figsize=(8, 6))
ax.set_xlabel('Payment_type', fontdict=label_font)
ax.set_ylabel('Count', fontdict=label_font)
ax.set_title('Stacked Barplot of Payment Type by VendorID', fontdict=title_font)
ax.legend(title='VendorID', loc='upper right', bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.show()

# Bar Plot - Grouped
plt.figure(figsize=(10, 6))
sns.barplot(x="RatecodeID", y="total_amount", hue="VendorID", data=df, ci=None)
plt.xlabel("RatecodeID", fontdict=label_font)
plt.ylabel("total_amount", fontdict=label_font)
plt.title("Grouped Barplot of total_amount by RatecodeID and VendorID", fontdict=title_font)
plt.legend(title='VendorID', loc='upper right', bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.show()

# Count plot - days of week
figure, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
sns.countplot(x='pickup_day', data=df, ax=ax[0])
ax[0].set_title('Number of Pickups done on each day of the week', fontdict=title_font)
sns.countplot(x='dropoff_day', data=df, ax=ax[1])
ax[1].set_title('Number of dropoffs done on each day of the week', fontdict=title_font)
plt.tight_layout()
plt.show()

# Pie plot - Pickups and Dropoffs locations
pickup_counts = df['PULocationID'].value_counts()
dropoff_counts = df['DOLocationID'].value_counts()

# Top 10 pickup locations
top_10_pickup_locations_id = pickup_counts.head(10)
filtered_pickup_df = df_taxi_zones[df_taxi_zones['LocationID'].isin(top_10_pickup_locations_id.index)]
pickup_zone_names_list = filtered_pickup_df['Zone'].tolist()

# Top 10 dropoff locations
top_10_dropoff_locations_id = dropoff_counts.head(10)
filtered_dropoff_df = df_taxi_zones[df_taxi_zones['LocationID'].isin(top_10_dropoff_locations_id.index)]
dropoff_zone_names_list = filtered_dropoff_df['Zone'].tolist()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Pie chart for top 10 pickup locations
axes[0].pie(top_10_pickup_locations_id.values, labels=pickup_zone_names_list, autopct='%1.1f%%', startangle=140)
axes[0].set_title("Top 10 Most Frequent Pickup Locations", fontdict=title_font)

# Pie chart for top 10 dropoff locations
axes[1].pie(top_10_dropoff_locations_id.values, labels=dropoff_zone_names_list, autopct='%1.1f%%', startangle=140)
axes[1].set_title("Top 10 Most Frequent Dropoff Locations", fontdict=title_font)
plt.tight_layout()
plt.show()

# Boxenplot - Payment type vs Total Amount
sns.boxenplot(data=df, x='payment_type', y='total_amount')
plt.xlabel('payment_type', fontdict=label_font)
plt.ylabel('total_amount', fontdict=label_font)
plt.title('A boxenplot depicting the total amount vs payment type', fontdict=title_font)
plt.tight_layout()
plt.show()

# Reg plot - Duration of trip
sns.set(style="whitegrid")
sns.regplot(data=df, y="trip_duration", x="total_amount", ci=None, line_kws={'color': 'red'})
plt.xlabel('total_amount', fontdict=label_font)
plt.ylabel('trip_duration', fontdict=label_font)
plt.title('Regression Plot - Target Trip Duration', fontdict=title_font)
plt.tight_layout()
plt.show()

# Violin plot - Duration per day
sns.violinplot(data=df, y="total_amount", x="pickup_day")
plt.xlabel('pickup_day', fontdict=label_font)
plt.ylabel('total_amount', fontdict=label_font)
plt.title("Violin Plot of Total Amount per Day of Week", fontdict=title_font)
plt.show()

# Rug Plot
plt.figure(figsize=(10, 4))
sns.kdeplot(df['pickup_hour'], fill=True, color="skyblue", label="Kernel Density Estimate")
sns.rugplot(x=df['pickup_hour'])
plt.xlabel('pickup_hour', fontdict=label_font)
plt.title('Rug Plot for pickup_hour', fontdict=title_font)
plt.tight_layout()
plt.show()

# Hexbin 2
plt.figure(figsize=(20, 16))
sns.set_style("whitegrid")
sns.jointplot(data=df, y='tip_amount', x='total_amount', kind='hex', color='green')
plt.xlabel("total_amount", fontdict=label_font)
plt.ylabel("tip_amount", fontdict=label_font)
plt.title("Hexbin Plot - total_amount vs tip_amount", fontdict=title_font)
plt.tight_layout()
plt.show()

# Line Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='trip_distance', y='trip_duration', data=df)
plt.xlabel('trip_distance', fontdict=label_font)
plt.ylabel('trip_duration', fontdict=label_font)
plt.title('Line Plot of trip_distance vs trip_duration', fontdict=title_font)
plt.tight_layout()
plt.show()

# Heatmap with cbar
df_pivot = df.pivot_table(index="passenger_count", columns="pickup_day", values="fare_amount")
plt.figure(figsize=(10, 8))
sns.heatmap(df_pivot, annot=True, cmap="YlGnBu", cbar=True)
plt.xlabel('pickup_day', fontdict=label_font)
plt.ylabel('passenger_count', fontdict=label_font)
plt.title('fare_amount Heatmap', fontdict=title_font)
plt.show()

# Strip Plot
plt.figure(figsize=(8, 6))
sns.stripplot(x="pickup_day", y="tip_amount", data=df_outlier, jitter=True, palette="Set1")
plt.title("Strip Plot of Tip Amount by Day", fontdict=title_font)
plt.xlabel("Day of the Week (Pickup)", fontdict=label_font)
plt.ylabel("Tip Amount", fontdict=label_font)
plt.show()

# Cluster Map
df_pivot = df.pivot_table(index="payment_type", columns="pickup_day", values="total_amount")
plt.figure(figsize=(40, 20))
sns.clustermap(df_pivot, cmap="coolwarm", standard_scale=1)
plt.title("Cluster Map of Total Amount by Payment Type and Pickup Day", fontdict=title_font)
plt.tight_layout()
plt.show()

# Swarm Plot
temp_df2 = temp_df2.stack().reset_index(name='trip_distance').rename(columns={'level_0': 'payment_type',
                                                                              'level_1': 'PULocationID'})

plt.figure(figsize=(10, 8))
sns.swarmplot(x='payment_type', y='trip_distance', data=temp_df2, size=8, palette='Set2')
plt.title('Swarm Plot - trip_distance vs payment type', fontdict=title_font)
plt.xlabel('Payment Type', fontdict=label_font)
plt.ylabel('Trip Distance', fontdict=label_font)
plt.tight_layout()
plt.show()

# Histogram with KDE - Pickup Location
plt.figure()
sns.histplot(df['PULocationID'], kde=True)
plt.xlabel('PULocationID', fontdict=label_font)
plt.ylabel('Frequency', fontdict=label_font)
plt.title('Histogram with KDE - Pickup Location ID', fontdict=title_font)
plt.tight_layout()
plt.show()

# Histogram with KDE - Dropoff Location
plt.figure()
sns.histplot(df['DOLocationID'], kde=True)
plt.xlabel('DOLocationID', fontdict=label_font)
plt.ylabel('Frequency', fontdict=label_font)
plt.title('Histogram with KDE - Dropoff Location ID', fontdict=title_font)
plt.tight_layout()
plt.show()

# Histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df['pickup_hour'], bins=30, kde=True, ax=axes[0])
axes[0].set_xlabel('pickup_hour', fontdict=label_font)
axes[0].set_ylabel('Frequency', fontdict=label_font)
axes[0].set_title('Histogram with KDE - Pickup Hour', fontdict=title_font)

sns.histplot(df['dropoff_hour'], bins=30, kde=True, ax=axes[1], color='orange')
axes[1].set_xlabel('dropoff_hour', fontdict=label_font)
axes[1].set_ylabel('Frequency', fontdict=label_font)
axes[1].set_title('Histogram with KDE - Dropoff Hour', fontdict=title_font)

plt.tight_layout()
plt.show()

# Histogram with KDE - Fare amount
plt.figure()
sns.histplot(df['fare_amount'], kde=True)
plt.xlabel('fare_amount', fontdict=label_font)
plt.ylabel('Frequency', fontdict=label_font)
plt.title('Histogram with KDE - fare_amount', fontdict=title_font)
plt.tight_layout()
plt.show()

# Histogram with KDE - trip_duration
plt.figure()
sns.histplot(df['trip_duration'], kde=True)
plt.xlabel('trip_duration', fontdict=label_font)
plt.ylabel('Frequency', fontdict=label_font)
plt.title('Histogram with KDE - trip_duration', fontdict=title_font)
plt.tight_layout()
plt.show()

# Joint Plot with KDE
sns.jointplot(x="congestion_surcharge", y="trip_duration", data=df, kind="scatter", marginal_kws=dict(bins=60))
plt.suptitle("Joint Plot - congestion_surcharge vs trip_duration", fontdict=title_font)
plt.xlabel('congestion_surcharge', fontdict=label_font)
plt.ylabel('trip_duration', fontdict=label_font)
plt.tight_layout()
plt.show()

# QQ Plot
sm.qqplot(df['total_amount'], line='45', fit=True)
plt.xlabel('Theoretical Quantiles', fontdict=label_font)
plt.ylabel('Sample Quantiles', fontdict=label_font)
plt.title("QQ Plot - total_amount", fontdict=title_font)
plt.show()

# Pairplot
sns.pairplot(df_sampled[['trip_distance', 'trip_duration', 'total_amount', 'RatecodeID']], hue='RatecodeID')
plt.title('Pairplot - Trip Distance vs Duration vs Total Amount')
plt.show()

# Multivariate KDE
sns.kdeplot(data=df_sampled[['pickup_hour', 'trip_duration']], x='pickup_hour', y='trip_duration', fill=True,
            cmap="Blues", levels=5)
plt.xlabel('pickup_hour', fontdict=label_font)
plt.ylabel('trip_duration', fontdict=label_font)
plt.title('Bivariate Kernel Density Plot', fontdict=title_font)
plt.show()


# Contour Plot
def plot_contour(ax, x, y, cmap, label):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    sc = ax.scatter(x, y, c=z, cmap=cmap, s=10, edgecolor='b', marker='o', label=label)

    x_range = np.linspace(x.min(), x.max(), 265)
    y_range = np.linspace(y.min(), y.max(), 265)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = gaussian_kde(xy)(np.vstack([x_grid.ravel(), y_grid.ravel()]))

    cset = ax.contour(x_grid, y_grid, z_grid.reshape(x_grid.shape), levels=10, alpha=0.7, cmap=cmap)

    return sc, cset


# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# XY Plane
sc_xy, cset_xy = plot_contour(axs[0], pickup_tip_df['trip_distance'], pickup_tip_df['fare_amount'], 'Blues',
                              'Total trip_distance vs Total fare_amount Plane')
axs[0].set_xlabel('Total Trip Distance')
axs[0].set_ylabel('Total Fare Amount')
axs[0].set_title('Trip Distance vs Fare Amount', fontdict=title_font)

# Add colorbar for XY Plane
norm_xy = Normalize()
cb_xy = fig.colorbar(sc_xy, ax=axs[0], label='Density', norm=norm_xy)

# XZ Plane
sc_xz, cset_xz = plot_contour(axs[1], pickup_tip_df['trip_distance'], pickup_tip_df['tip_amount'], 'Greens',
                              'Trip Distance vs Tip Amount Plane')
axs[1].set_xlabel('Total Trip Distance', fontdict=label_font)
axs[1].set_ylabel('Total Tip Amount', fontdict=label_font)
axs[1].set_title('Trip Distance vs Tip Amount', fontdict=title_font)

norm_xz = Normalize()
cb_xz = fig.colorbar(sc_xz, ax=axs[1], label='Density', norm=norm_xz)

sc_yz, cset_yz = plot_contour(axs[2], pickup_tip_df['fare_amount'], pickup_tip_df['tip_amount'], 'Reds',
                              'Fare Amount vs Tip Amount Plane')
axs[2].set_xlabel('Total Fare Amount', fontdict=label_font)
axs[2].set_ylabel('Total Tip Amount', fontdict=label_font)
axs[2].set_title('Fare Amount vs Tip Amount', fontdict=title_font)

norm_yz = Normalize()
cb_yz = fig.colorbar(sc_yz, ax=axs[2], label='Density', norm=norm_yz)

plt.tight_layout()
plt.show()

# 3D plot
fig = go.Figure()

# XYZ Plane
fig.add_trace(go.Scatter3d(
    x=pickup_tip_df['trip_distance'],
    y=pickup_tip_df['fare_amount'],
    z=pickup_tip_df['tip_amount'],
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
    ),
    name='XYZ Plane'
))

# XZ Plane
fig.add_trace(go.Scatter3d(
    x=pickup_tip_df['trip_distance'],
    y=pickup_tip_df['fare_amount'] * 0,
    z=pickup_tip_df['tip_amount'],
    mode='markers',
    marker=dict(
        size=5,
        color='yellow',
    ),
    name='XZ Plane'
))

# XY Plane
fig.add_trace(go.Scatter3d(
    x=pickup_tip_df['trip_distance'],
    y=pickup_tip_df['fare_amount'],
    z=pickup_tip_df['tip_amount'] * 0,
    mode='markers',
    marker=dict(
        size=5,
        color='green',
    ),
    name='XY Plane'
))

# YZ Plane
fig.add_trace(go.Scatter3d(
    x=pickup_tip_df['trip_distance'] * 0,
    y=pickup_tip_df['fare_amount'],
    z=pickup_tip_df['tip_amount'],
    mode='markers',
    marker=dict(
        size=5,
        color='red',
    ),
    name='YZ Plane'
))

fig.update_layout(scene=dict(
    xaxis=dict(title='Total Trip Distance'),
    yaxis=dict(title='Total Fare Amount'),
    zaxis=dict(title='Total Tip Amount'),
    aspectmode='cube',
))

# Show the plot
fig.show()
