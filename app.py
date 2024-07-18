import pandas as pd
from dash import Dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import plotly.express as px
import random
import warnings
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from scipy.stats import shapiro, boxcox, kstest, normaltest
from dash import html as dhtml

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
df_taxi_zones = pd.read_csv('taxi+_zone_lookup.csv')
warnings.filterwarnings("ignore")
df['fare_amount'] = df['fare_amount'].abs()
df['extra'] = df['extra'].abs()
df['mta_tax'] = df['mta_tax'].abs()
df['tip_amount'] = df['tip_amount'].abs()
df['tolls_amount'] = df['tolls_amount'].abs()
df['improvement_surcharge'] = df['improvement_surcharge'].abs()
df['total_amount'] = df['total_amount'].abs()
df['congestion_surcharge'] = df['congestion_surcharge'].abs()
df['airport_fee'] = df['airport_fee'].abs()

df_unclean = df.copy()
missing_percentage = (df_unclean.isna().mean() * 100).round(2)
initial_fig = go.Figure()

initial_fig.add_trace(go.Bar(
    x=missing_percentage.index,
    y=missing_percentage.values,
    text=missing_percentage.values,
    textposition='auto',
    marker_color='blue',
))

# Set plot properties
initial_fig.update_layout(
    title='Percentage of Missing Values in Each Column',
    xaxis_title='Columns',
    yaxis_title='Percentage Missing',
    bargap=0.2,  # Gap between bars
)

# initial_fig = px.imshow(df_unclean.isna(), labels=dict(color="Missing Values"),
#                         color_continuous_scale=["white", "red"])

passenger_mode = df['passenger_count'].mode()[0]
df['passenger_count'].fillna(passenger_mode, inplace=True)
ratecodeID_mode = df['RatecodeID'].mode()[0]
df['RatecodeID'].fillna(ratecodeID_mode, inplace=True)
store_and_fwd_flag_mode = df['store_and_fwd_flag'].mode()[0]
extra_mean = df['extra'].mean()
mta_tax_mean = df['mta_tax'].mean()
tip_amount_mean = df['tip_amount'].mean()
tolls_amount_mean = df['tolls_amount'].mean()
improvement_surcharge_mean = df['improvement_surcharge'].mean()
congestion_surcharge_mean = df['congestion_surcharge'].mean()
df['store_and_fwd_flag'].fillna(store_and_fwd_flag_mode, inplace=True)
df['congestion_surcharge'].fillna(0.0, inplace=True)
df['airport_fee'].fillna(0.00, inplace=True)

category_mapping_fwd_flag = {'N': 0, 'Y': 1}
weekday_mapping = {'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7}
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace(category_mapping_fwd_flag)
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
df['pickup_day'] = df['tpep_pickup_datetime'].dt.day_name().replace(weekday_mapping)
df['dropoff_day'] = df['tpep_dropoff_datetime'].dt.day_name().replace(weekday_mapping)
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df = df.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)

# Cleaned Data stored and described
df_clean = df.head(50).round(2)
traces = go.Table(
    header=dict(values=df_clean.columns),
    cells=dict(values=[df_clean[col] for col in df_clean.columns])
)

layout = go.Layout(title='First Few Columns of the cleaned Dataset')
cleaned_fig = go.Figure(data=[traces], layout=layout)

summary_stats = df.describe().transpose()

table_trace = go.Table(
    header=dict(values=['Column', 'Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max']),
    cells=dict(values=[summary_stats.index] + [summary_stats[col].round(2) for col in summary_stats.columns])
)

table_layout = go.Layout(title='Summary Statistics of Cleaned DataFrame')

initial_table = go.Figure()
table_fig = go.Figure(data=[table_trace], layout=table_layout)

cleaned_csv_string = df_clean.to_csv(index=False, encoding='utf-8')
cleaned_csv_string = "data:text/csv;charset=utf-8," + cleaned_csv_string
download_link = cleaned_csv_string
download_filename = "cleaned_data.csv"

# Grouped Weekly Data
grouper = df.groupby([pd.Grouper(key='pickup_day'), pd.Grouper(key='pickup_hour')])
temp_df = grouper['total_amount'].mean()
temp_df = temp_df.to_frame().reset_index()
temp_df.set_index('pickup_hour')
temp_df = temp_df.pivot(index='pickup_hour', columns='pickup_day', values='total_amount')

# Grouped Borough Data
pickup_df = df.groupby('PULocationID').size().reset_index(name='PUcount')
dropoff_df = df.groupby('DOLocationID').size().reset_index(name='DOcount')
location_info = df_taxi_zones

pickup_df = pd.merge(pickup_df, location_info, left_on='PULocationID', right_on='LocationID', how='left')
pickup_df.drop('LocationID', axis=1, inplace=True)
dropoff_df = pd.merge(dropoff_df, location_info, left_on='DOLocationID', right_on='LocationID', how='left')
dropoff_df.drop('LocationID', axis=1, inplace=True)

pickupzone_std_df = df.groupby('PULocationID')['total_amount'].std().reset_index()
std_mean = pickupzone_std_df.mean()
pickupzone_std_df.fillna(std_mean, inplace=True)

pickupzone_std_df = pd.merge(pickupzone_std_df, location_info, left_on='PULocationID', right_on='LocationID',
                             how='left')
pickupzone_std_df.drop('LocationID', axis=1, inplace=True)
pickupzone_std_df.drop('Borough', axis=1, inplace=True)
pickupzone_std_df.drop('Zone', axis=1, inplace=True)
pickupzone_std_df.drop('service_zone', axis=1, inplace=True)

dropoffzone_std_df = df.groupby('DOLocationID')['total_amount'].std().reset_index()
std_mean = dropoffzone_std_df.mean()
dropoffzone_std_df.fillna(std_mean, inplace=True)

dropoffzone_std_df = pd.merge(dropoffzone_std_df, location_info, left_on='DOLocationID', right_on='LocationID',
                              how='left')
dropoffzone_std_df.drop('LocationID', axis=1, inplace=True)
dropoffzone_std_df.drop('Borough', axis=1, inplace=True)

paymentType_sum_df = df.groupby('payment_type')['total_amount'].sum()
paymentType_sum_df = paymentType_sum_df.reset_index(name='total_amount')

# Drop off Grouped sum, tip, airport fee
dropoff_tip_df = df.groupby('PULocationID')[['total_amount', 'airport_fee', 'tip_amount']].sum()


# Outlier Removal
def QQ_calc(temp):
    for column_name in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[column_name]) and (column_name != 'passenger_count'
                                                                 and column_name != 'payment_type'):
            Q1 = temp[column_name].quantile(0.25)
            Q3 = temp[column_name].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            temp = temp[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return temp


df = QQ_calc(df)


def QQ_calc_2(temp):
    for column_name in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[column_name]):
            Q1 = temp[column_name].quantile(0.25)
            Q3 = temp[column_name].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            temp = temp[(temp[column_name] >= lower_bound) & (temp[column_name] <= upper_bound)]

    return temp


category_mapping = {1: 'Credit card', 2: 'Cash', 3: 'No charge', 4: 'Dispute', 5: 'Unknown', 6: 'Voided trip',
                    0: 'Voided trip'}
reversed_mapping = {
    'credit card': 1.0,
    'cash': 2.0,
    'no charge': 3.0,
    'dispute': 4.0,
    'unknown': 5.0,
    'voided trip': 6.0
}
paymentType_sum_df['payment_type'] = paymentType_sum_df['payment_type'].replace(category_mapping)

df_sampled = df.sample(n=50000, random_state=5805)
df_sampled.reset_index(drop=True, inplace=True)

df_sampled3 = df.sample(n=5000, random_state=5805)
df_sampled3.reset_index(drop=True, inplace=True)

df_sampled2 = df.sample(n=100000, random_state=5905)
df_sampled2.reset_index(drop=True, inplace=True)
df_outlier = df_sampled2.copy()
df_wo_outlier = QQ_calc_2(df_sampled2)

# =============
# Linear model for total_amount
# =============

X = df_sampled.drop(['total_amount'], axis=1)
y = df_sampled['total_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=5705, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# print(comparison.head().round(2).to_string())

# plt.scatter(y_test, y_pred)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs Predicted Values')
# plt.show()

mse_model = mean_squared_error(y_test, y_pred)
# print(f'The MSE of the linear regression prediction is = {mse_model: .3f}')

coefficients = model.coef_
intercept = model.intercept_
feature_names = X.columns

# equation = f"The linear regression equation: y = {intercept:.2f} "
# for i, (coef, feature_name) in enumerate(zip(coefficients, feature_names)):
#     equation += f"+ {coef:.2f} * {feature_name} "
# print(equation)


# =============
# PCA
# =============
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

filter_cols = []
for col in df_sampled.columns:
    if len(df_sampled[col].unique()) < 2:
        filter_cols.append(col)

df_filtered = df_sampled.drop(filter_cols, axis=1)
X_PCA = df_sampled.drop(['total_amount'], axis=1)
# y_PCA = df_sampled['total_amount']

scalar = StandardScaler()
X_std = scalar.fit_transform(X_PCA)
X_std = pd.DataFrame(X_std, columns=X_PCA.columns)

pca = PCA(n_components=20, svd_solver='full')
pca.fit(X_std)

exp = np.cumsum(pca.explained_variance_ratio_).round(2)

# ==========
# App layout
# ==========
my_app = Dash('my app', external_stylesheets=external_stylesheets)
my_app.layout = html.Div(
    [html.H3("CS5764 Final Term Project - Information Visualization", style={"textAlign": "center"}),
     dcc.Tabs(id='ftp-tabs',
              children=[
                  dcc.Tab(label='Dataset Description', value='q9'),
                  dcc.Tab(label='Dataframe Cleaner', value='q7'),
                  dcc.Tab(label='PCA Analysis', value='q4'),
                  dcc.Tab(label='Outlier Detector', value='q5'),
                  dcc.Tab(label='Normality Test', value='q6'),
                  dcc.Tab(label='Pearson Heatmap', value='q8'),
                  dcc.Tab(label='Pickup Hour Graph', value='q1'),
                  dcc.Tab(label='NYC Borough Analysis', value='q2'),
                  dcc.Tab(label='Taxi Price Estimator ', value='q3'),
                  dcc.Tab(label='About Me', value='q10')
              ],
              style={}),
     html.Div(id='layout')
     ]
)
# ================
# Data Dictionary
# ================
bullet_points = ["VendorID - A code indicating the TPEP provider that provided the record. "
                 "1= Creative Mobile Technologies, LLC; 2= VeriFone Inc.",
                 "tpep_pickup_datetime - The date and time when the meter was engaged.",
                 "tpep_dropoff_datetime - The date and time when the meter was disengaged.",
                 "Passenger_count - The number of passengers in the vehicle.",
                 "Trip_distance - The elapsed trip distance in miles reported by the taximeter.",
                 "PULocationID - TLC Taxi Zone in which the taximeter was engaged",
                 "DOLocationID - TLC Taxi Zone in which the taximeter was disengaged",
                 "RateCodeID -  The final rate code in effect at the end of the trip. "
                 "1= Standard rate 2=JF 3=Newark 4=Nassau "
                 "or Westchester 5=Negotiated fare 6=Group ride",
                 "Store_and_fwd_flag - This flag indicates whether the trip record was held "
                 "in vehicle memory before sending to the vendor, aka “store and forward,” "
                 "because the vehicle did not have a connection to the server.",
                 "Payment_type - A numeric code signifying how the passenger paid for the trip. "
                 "1= Credit card 2= Cash 3= No charge "
                 "4= Dispute 5= Unknown 6= Voided trip",
                 "Fare_amount - The time-and-distance fare calculated by the meter.",
                 "Extra - Miscellaneous extras and surcharges. Currently, "
                 "this only includes the $0.50 and $1 rush hour and overnight charges.",
                 "MTA_tax - $0.50 MTA tax that is automatically triggered based on the metered rate in use.",
                 "Improvement_surcharge - $0.30 improvement surcharge assessed trips at the flag drop. "
                 "The improvement surcharge began being levied in 2015.",
                 "Tip_amount - Tip amount – This field is automatically populated for credit card tips. "
                 "Cash tips are not included.",
                 "Tolls_amount - Total amount of all tolls paid in trip.",
                 "Total_amount - The total amount charged to passengers. Does not include cash tips.",
                 "Congestion_Surcharge - Total amount collected in trip for NYS congestion surcharge.",
                 "Airport_fee - $1.25 for pick up only at LaGuardia and John F. Kennedy Airports"]

question9_layout = html.Div([
    html.H1("NYC Taxi Dataset Dictionary", style={'text-align': 'center', 'font-weight': 'bold'}),
    html.Figure(
        children=[
            html.Img(
                src=my_app.get_asset_url('taxis.png'),
            ),
        ],
        style={'text-align': 'center', 'margin': '20px'},
    ),
    html.P([
        '''
        Delve into the extensive dataset of New York City's yellow taxis, a valuable resource for data scientists. 
        This dataset 
        (''',
        html.A("click here to download",
               href="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"),
        ''') serves as a comprehensive record of urban mobility, capturing intricate details of each taxi journey. 
        Ranging from temporal data points like pickup and drop-off times to spatial dimensions involving specific 
        locations, this dataset provides a detailed perspective on urban transit patterns. Moreover, it includes
         quantitative metrics such as trip distance, fare composition, payment methods, and passenger counts. 
         Supplied by tech companies under the TPEP/LPEP initiatives, this data represents a goldmine for analyzing 
         urban transportation trends, understanding fare  dynamics, and exploring passenger behavior. 
         Beyond just taxis and their routes, it serves as a window into the lifeblood
        of the city, offering limitless possibilities for data-driven insights and informing urban planning strategies.
        '''
    ]),
    html.P(['''
    This dataset is a collection of around 3 million trips made in a span of 1 month in the time 
    period of January 2023 
    (''',
            html.A("visit the source",
                   href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page", target="_blank"),
            '''). In this dataset our potential target is total amount paid by the rider and for PCA analysis
    we have considered the remaining features (described in the given dictionary) as independent features and ran a 
    thorough EDA to look for various patterns and trends in the yellow taxi framework of NYC.
    ''']),

    html.H2('Feature Description:'),
    html.Ul([html.Li(point) for point in bullet_points]),
])

# ================
# Pearson Heatmap
# ================
question8_layout = html.Div(children=[
    html.H1("Pearson Correlation Coefficient", style={'text-align': 'center'}),
    dcc.Dropdown(id='col-checklist',
                 options=[{'label': i, 'value': i} for i in df_sampled.columns],
                 value=df_sampled.columns[:],
                 style={
                     'fontSize': 10,
                     'margin-top': '20px'
                 }, multi=True),
    dcc.Loading(
        id="loading-graph",
        type='circle',
        children=[
            dcc.Graph(id='cor-matrix-graph'),
        ],
    ),
])


@my_app.callback(
    [Output('cor-matrix-graph', 'figure')],
    [Input('col-checklist', 'value')]
)
def update_graph(selected_col):
    corr_df = df_sampled[selected_col].corr()
    heatmap_trace = go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='YlGnBu',
    )

    heatmap_layout = go.Layout(
        title='Correlation Heatmap (Sparse data shows NaN values)',
    )
    fig = go.Figure(data=[heatmap_trace], layout=heatmap_layout)

    return [fig]


# ================
# Dataframe Cleaner
# ================
question7_layout = html.Div([
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=[
            dcc.Graph(id='bar-plot', figure=initial_fig),
            html.Button('Clean DataFrame', id='clean-button', n_clicks=0, style={"textAlign": "center"}),
            html.Div([
                dhtml.A(dhtml.Button('Download Cleaned DataFrame', id='download-button'), id='download-link',
                        style={'display': 'none'})
            ]),
            dcc.Graph(id='describe')
        ]
    ),
])


@my_app.callback(
    [Output('bar-plot', 'figure'),
     Output('download-link', 'style'),
     Output('download-link', 'href'),
     Output('download-link', 'download'),
     Output('describe', 'figure'),
     Output('describe', 'style'),
     Output('clean-button', 'style')],
    [Input('clean-button', 'n_clicks')]
)
def update_cleaned_data(n_clicks):
    if n_clicks > 0:
        return (cleaned_fig, {'display': 'block'}, download_link, download_filename,
                table_fig, {'display': 'block'}, {'display': 'none'})

    return initial_fig, {'display': 'none'}, "", "", "", {'display': 'none'}, {"textAlign": "center"}


# ================
# Outlier Detector
# ================
question5_layout = html.Div([
    html.H1("Outlier Detector"),
    html.Div([
        dcc.Dropdown(
            id='in-column-dropdown',
            options=[{'label': col, 'value': col} for col in df_outlier.columns],
            value='trip_distance',
            multi=False
        ),
        html.Br(),
        html.Button('Generate Boxplot', id='generate-boxplot-btn', n_clicks=0),
    ]),
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=[
            dcc.Graph(
                id='boxplot-outlier',
            ),
            html.Br(),
            html.Br(),
            dcc.Graph(
                id='boxplot-outlier-wo'
            ),
        ]
    ),
])


# Callback to display box plot before removing outliers
@my_app.callback(
    [Output('boxplot-outlier', 'figure'),
     Output('boxplot-outlier-wo', 'figure')],
    [Input('generate-boxplot-btn', 'n_clicks')],
    [State('in-column-dropdown', 'value')]
)
def display_boxplot_before(n_clicks, selected_feature):
    fig = {
        'data': [
            go.Box(y=df_outlier[selected_feature], name=selected_feature),
        ],
        'layout': go.Layout(title=f'Boxplot - {selected_feature} with outliers'),
    }

    fig2 = {
        'data': [
            go.Box(y=df_wo_outlier[selected_feature], name=selected_feature),
        ],
        'layout': go.Layout(title=f'Boxplot - {selected_feature} w/o outliers'),
    }
    return fig, fig2


# ================
# Normality Test
# ================
question6_layout = html.Div([
    html.Label('Select a feature:'),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in df_filtered.columns],
        value=df_filtered.columns[0],
    ),
    html.Label('Select a test:'),
    dcc.Dropdown(
        id='column-normal-test-dropdown',
        options=[
            {'label': 'Da_k_squared', 'value': 'Da_k_squared'},
            {'label': 'K_S test', 'value': 'K_S test'},
            {'label': 'Shapiro Test', 'value': 'Shapiro Test'}, ],
        value='Shapiro Test',
    ),
    html.Div(id='shapiro-output', style={'text-align': 'center', 'font-size': '20px'}),
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=[
            dcc.Graph(id='original-histogram', figure={}),
            dcc.Graph(id='transformed-histogram', figure={}),
        ]
    )
])


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


@my_app.callback(
    [Output('shapiro-output', 'children'),
     Output('original-histogram', 'figure'),
     Output('transformed-histogram', 'figure')],
    [Input('column-dropdown', 'value'),
     Input('column-normal-test-dropdown', 'value'), ]
)
def update_output(selected_column, selected_test):
    column_data = df_sampled3[selected_column]

    stats, p = normal_test(column_data, selected_test)
    norm_stat = 'Normal' if p > 0.01 else 'Not Normal'

    shapiro_output = f'{selected_test}: statistics = {stats:.2f}, p-value = {p:.2f}\n'
    shapiro_output += f'{selected_column} feature looks {norm_stat}'

    original_histogram = px.histogram(df_sampled3, x=selected_column, title=f'Original Histogram of {selected_column}')

    transformed_histogram = {}
    if norm_stat == 'Not Normal':
        shifted_data = column_data - column_data.min() + 1e-6
        transformed_data, _ = boxcox(shifted_data)
        df_transformed = pd.DataFrame({selected_column: transformed_data})

        transformed_histogram = px.histogram(df_transformed, x=selected_column,
                                             title=f'Box-Cox Transformed Histogram of {selected_column}')
        transformed_histogram.update_layout(xaxis_title=f'Normalized {selected_column}', )

    return shapiro_output, original_histogram, transformed_histogram


# ================
# PCA Plot
# ================
question4_layout = html.Div([
    html.H3('PCA Analysis'),
    html.P("Select Number of Components:"),
    dcc.Slider(
        id='no-component-slider',
        min=1,
        max=20,
        step=1,
        value=8,
    ),
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=[
            dcc.Graph(id='PCA-graph')
        ]
    ),
    dcc.Tooltip(
        id="graph-tooltip-0",
        background_color="lightgreen",
        border_color="blue"),
])


@my_app.callback(
    Output(component_id='PCA-graph', component_property='figure'),
    [Input(component_id='no-component-slider', component_property='value')],
)
def update_q4(no_component):
    time.sleep(1)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(1, no_component + 1),
                             y=exp,
                             mode='lines',
                             name='Cumulative Explained Variance'))

    threshold_component = np.where(exp >= 0.95)[0][0] + 1
    if threshold_component <= no_component:
        fig.add_shape(
            go.layout.Shape(
                type='line',
                x0=1,
                x1=len(exp),
                y0=0.95,
                y1=0.95,
                line=dict(color='red', width=2, dash='dash'),
                name='0.95 Threshold'
            )
        )

        fig.add_shape(
            go.layout.Shape(
                type='line',
                x0=threshold_component,
                x1=threshold_component,
                y0=0,
                y1=1,
                line=dict(color='green', width=2, dash='dash'),
                name=f'Component #{threshold_component}'
            )
        )

    title_PCA = 'PCA Analysis' if threshold_component > no_component \
        else 'PCA Analysis - tracing component with >95% cumulative explained variance'
    fig.update_layout(
        title=f'{title_PCA}',
        xaxis=dict(title='Number of components'),
        yaxis=dict(title='Cumulative explained variance'),
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        showlegend=True,
    )

    return fig


@my_app.callback(
    Output("graph-tooltip-0", "show"),
    Output("graph-tooltip-0", "bbox"),
    Output("graph-tooltip-0", "children"),
    Input("PCA-graph", "hoverData"),
)
def update_tooltip_content(hoverData):
    if hoverData is None:
        return no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]

    children = [
        html.P(f"Component: {pt['x']}, Cumulative Explained Variance Ratio: {pt['y']: .2f}")
    ]

    return True, bbox, children


# ================
# Line Plot
# ===============
question1_layout = html.Div([
    html.H3('Weekly Pickup Hour Analysis'),
    html.H5('Total Amount Vs Pickup Hour'),
    html.P("Pick Day of Week:"),
    dcc.Checklist(id='check1',
                  options=[
                      {"label": "Sunday", 'value': 1},
                      {'label': "Monday", 'value': 2},
                      {'label': "Tuesday", 'value': 3},
                      {'label': "Wednesday", 'value': 4},
                      {'label': "Thursday", 'value': 5},
                      {'label': "Friday", 'value': 6},
                      {'label': "Saturday", 'value': 7}
                  ], value=[2], inline=True),
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=[
            dcc.Graph(id='pickuphour-graph'),
            html.Button("Download Raw Dataframe", id="btn_csv"),
            dcc.Store(id='selected-columns-store', storage_type='memory'),
            dcc.Download(id="download-dataframe-csv"),
        ]
    ),
    dcc.Tooltip(
        id="graph-tooltip-1",
        background_color="lightgreen",
        border_color="blue"),
])


@my_app.callback(
    Output(component_id='pickuphour-graph', component_property='figure'),
    [Input(component_id='check1', component_property='value')],
)
def update_q1(selected_days):
    time.sleep(1)
    data = []
    day_dict = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
    for day in selected_days:
        trace = go.Scatter(
            x=temp_df.index,
            y=temp_df[day],
            mode='lines',
            name=day_dict[day]
        )
        data.append(trace)

    layout = go.Layout(
        title='Line plot - Total Amount vs Pickup Hour',
        xaxis=dict(title='Pickup Hour'),
        yaxis=dict(title='Total Amount'),
        showlegend=True,
        legend=dict(x=0, y=1)
    )

    return {'data': data, 'layout': layout}


@my_app.callback(
    Output("graph-tooltip-1", "show"),
    Output("graph-tooltip-1", "bbox"),
    Output("graph-tooltip-1", "children"),
    Input("pickuphour-graph", "hoverData"),
)
def update_tooltip_content(hoverData):
    if hoverData is None:
        return no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]

    children = [
        html.P(f"Hour: {pt['x']}, Avg Total Amount: {pt['y']: .2f}")
    ]

    return True, bbox, children


@my_app.callback(
    Output('selected-columns-store', 'data'),
    [Input('check1', 'value')]
)
def store_selected_columns(selected_columns):
    return selected_columns


@my_app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("btn_csv", "n_clicks")],
    [State("selected-columns-store", "data")],
    prevent_initial_call=True,
)
def download_q1(n_clicks, selected_days):
    day_dict = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
    if n_clicks and selected_days:
        new_df = temp_df[selected_days]
        new_df.rename(columns=day_dict, inplace=True)
        return dcc.send_data_frame(new_df.to_csv, "weekday_Data.csv")


# # ================
# # Area Plot
# # ===============
question2_layout = html.Div(
    [html.H3('NYC Taxi Borough Information'),
     html.H4('Select a Borough:'),
     dcc.RadioItems(
         id='borough-dropdown',
         options=[
             {'label': borough, 'value': borough} for borough in pickup_df['Borough'].unique()
         ],
         value=pickup_df['Borough'].unique()[1],
         labelStyle={'display': 'inline-block', 'margin-right': '10px'}
     ),
     dcc.Loading(
         id="loading-graph",
         type="circle",
         children=[
             html.Div(
                 id='image-container',
                 children=[
                     html.Img(id='borough-image', src='', style={'width': '20%', 'margin': 'auto', 'display': 'block'}),
                 ]
             ),
             html.Div(id='borough-info', style={'text-align': 'center'}),
             html.Br(),
             html.Br(),
             dcc.Graph(id='top5PU-graph'),

             html.Br(),
             dcc.Graph(id='top5DO-graph'),
         ]
     ),
     dcc.Tooltip(
         id="graph-tooltip-3",
         background_color="pink",
         border_color="blue"),
     dcc.Tooltip(
         id="graph-tooltip-4",
         background_color="pink",
         border_color="blue"),
     ]),


@my_app.callback(
    Output('borough-image', 'src'),
    [Input('borough-dropdown', 'value')]
)
def update_borough_image(selected_borough):
    image_path = my_app.get_asset_url(f'borough_images/{selected_borough.lower()}.png')
    return image_path


@my_app.callback(
    [Output('borough-info', 'children'),
     Output('top5PU-graph', 'figure'),
     Output('top5DO-graph', 'figure')],
    [Input('borough-dropdown', 'value')]
)
def update_borough_info(selected_borough):
    time.sleep(1)
    borough_PU_zones = pickup_df[pickup_df['Borough'] == selected_borough]
    top_10_data_PU = borough_PU_zones.sort_values(by='PUcount', ascending=False).head(10)
    if len(top_10_data_PU.index) > 2:
        fig1 = px.area(top_10_data_PU, x='Zone', y='PUcount', title=f'Top 10 Pickup Locations for {selected_borough}',
                       labels={'Zone': 'Zone', 'PUcount': 'PUcount'},
                       template='plotly_dark')
        fig1.update_layout(xaxis=dict(tickangle=45))

    else:
        fig1 = px.histogram(top_10_data_PU, x='Zone', y='PUcount',
                            title=f'Top 10 Pickup Locations for {selected_borough}',
                            labels={'Zone': 'Zone', 'PUcount': 'PUcount'},
                            template='plotly_dark')
        fig1.update_layout(xaxis=dict(tickangle=45))

    borough_DO_zones = dropoff_df[dropoff_df['Borough'] == selected_borough]
    top_10_data_DO = borough_DO_zones.sort_values(by='DOcount', ascending=False).head(10)
    if len(top_10_data_DO.index) > 2:
        fig2 = px.area(top_10_data_DO, x='Zone', y='DOcount', title=f'Top 10 Dropoff Locations for {selected_borough}',
                       labels={'Zone': 'Zone', 'DOcount': 'DOcount'},
                       template='plotly_dark')
        fig2.update_layout(xaxis=dict(tickangle=45))

    else:
        fig2 = px.histogram(top_10_data_DO, x='Zone', y='DOcount',
                            title=f'Top 10 Dropoff Locations for {selected_borough}',
                            labels={'Zone': 'Zone', 'DOcount': 'DOcount'},
                            template='plotly_dark')
        fig2.update_layout(xaxis=dict(tickangle=45))

    return f"{selected_borough} Borough", fig1, fig2


@my_app.callback(
    Output("graph-tooltip-3", "show"),
    Output("graph-tooltip-3", "bbox"),
    Output("graph-tooltip-3", "children"),
    Input("top5PU-graph", "hoverData"),
)
def update_tooltip_content(hoverData):
    if hoverData is None:
        return no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]

    children = [
        html.P(f"Zone: {pt['x']}, Count: {pt['y']: .2f}")
    ]

    return True, bbox, children


@my_app.callback(
    Output("graph-tooltip-4", "show"),
    Output("graph-tooltip-4", "bbox"),
    Output("graph-tooltip-4", "children"),
    Input("top5DO-graph", "hoverData"),
)
def update_tooltip_content(hoverData):
    if hoverData is None:
        return no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]

    children = [
        html.P(f"Zone: {pt['x']}, Total Pickups: {pt['y']: .2f}")
    ]

    return True, bbox, children


vendor_options = [{'label': '1 - Creative Mobile Technologies, LLC', 'value': 1},
                  {'label': f'2 - VeriFone Inc.', 'value': 2}]
rate_code_options = [{'label': str(i), 'value': i} for i in range(1, 6)]
PU_location_options = [{'label': f'{i}', 'value': j} for i, j in zip(pickup_df['Zone'], pickup_df['PULocationID'])]
DO_location_options = [{'label': f'{i}', 'value': j} for i, j in zip(dropoff_df['Zone'], dropoff_df['DOLocationID'])]
day_options = [{'label': f'Sunday', 'value': 1},
               {'label': f'Monday', 'value': 2},
               {'label': f'Tuesday', 'value': 3},
               {'label': f'Wednesday', 'value': 4},
               {'label': f'Thursday', 'value': 5},
               {'label': f'Friday', 'value': 6},
               {'label': f'Saturday', 'value': 7}]
payment_options = [{'label': i, 'value': reversed_mapping[i.lower()]} for i in reversed_mapping.keys()]
accepted_payment_types = ['credit card', 'cash', 'no charge', 'dispute', 'unknown', 'voided trip']

# ================
# About Me
# ===============
question10_layout = html.Div([
    html.Header("About Me", style={'text-align': 'center', 'font-size': '40px',
                                   'weight': 'bold', 'font': 'Arial, sans-serif;'}),
    html.H2('Manim Tirkey', style={'text-align': 'center', 'margin': '20px', 'weight': 'bold'}),
    html.Figure(
        children=[
            html.Img(
                src=my_app.get_asset_url('me.png'),
                style={'width': '300px', 'height': '300px'},
            )
        ],
        style={'text-align': 'center', 'margin': '20px'},
    ),
    html.P('Email: mtirkey@vt.edu', style={'text-align': 'center', 'margin': '20px'}),
    html.Div([
        html.A("LinkedIn Profile", href="https://www.linkedin.com/in/manim-tirkey-934a36204/", target="_blank"),
    ], style={'text-align': 'center'})

])

# ================
# Price Estimator
# ===============
question3_layout = html.Div([
    html.Header("Taxi Price Estimator", style={'text-align': 'center', 'font-size': '40px'}),
    html.Figure(
        children=[
            html.Img(
                src=my_app.get_asset_url('for_hire_logo.png'),
                style={'width': '300px', 'height': '300px'},
            ),
        ],
        style={'text-align': 'center', 'margin': '20px'},
    ),
    html.Label('VendorID'),
    dcc.Dropdown(
        id='vendor-dropdown',
        options=vendor_options,
        value=1,
        multi=False,
        style={'width': '50%'},
        placeholder="Select VendorID"
    ),

    html.Label('Passenger Count'),
    dcc.Input(
        id='passenger-input',
        type='number',
        value=1,
        placeholder='Enter Passenger Count',
        style={'width': '50%'}
    ),
    html.Div(id='error-message-passenger'),

    html.Label('Trip Distance (select a range) (in Miles)'),
    dcc.RangeSlider(
        id='distance-slider',
        min=0,
        max=10,
        step=0.1,
        marks={i: str(i) for i in range(11)},
        value=[0, 10],
    ),
    html.Label('Rate Code ID'),
    dcc.Dropdown(
        id='rate-code-dropdown',
        options=[
            {'label': '1 - Standard rate', 'value': 1.0},
            {'label': '2 - JFK', 'value': 2.0},
            {'label': '3 - Newark', 'value': 3.0},
            {'label': '4 - Nassau/Westchester', 'value': 4.0},
            {'label': '5 - Negotiated fare', 'value': 5.0},
            {'label': '6 - Group ride', 'value': 6.0},
            {'label': '7 - Unknown', 'value': 99.0}
        ],
        value=1,
        multi=False,
        style={'width': '50%'},
        placeholder="Select RateCodeID"
    ),

    html.Label('Pickup Location (Additional Charges incurred for Airport Pickups)'),
    dcc.Dropdown(
        id='pickup-location-dropdown',
        options=PU_location_options,
        multi=False,
        style={'width': '50%'},
        placeholder="Select Pickup Location",
        value=3
    ),

    html.Label('Dropoff Location'),
    dcc.Dropdown(
        id='dropoff-location-dropdown',
        options=DO_location_options,
        multi=False,
        style={'width': '50%'},
        placeholder="Select Dropoff Location",
        value=2
    ),

    html.Div([
        html.Label('Payment Type', style={'display': 'inline-block'}),
        html.Img(
            src=my_app.get_asset_url('question_logo.png'),
            style={'width': '20px', 'height': '20px', 'margin-left': '5px', 'margin-top': '5px', 'cursor': 'pointer'},
            id='logo-image',
            title='Accepted payment types are ' + ', '.join(
                payment_option['label'] for payment_option in payment_options)
        ),
    ]),
    dcc.Textarea(
        id='payment-type-textarea',
        placeholder='Enter Payment Type',
        style={'width': '50%', 'position': 'relative', 'display': 'inline-block'},
        value='credit card'
    ),
    html.Div(id='error-message'),
    html.Label('Base Fare ($)'),
    dcc.Slider(
        id='base-fare-slider',
        min=0,
        max=50,
        step=1,
        marks={i: str(i) for i in range(51)},
        value=10,
    ),

    html.Label('Expected Trip Duration (select a range) (in Minutes)'),
    dcc.RangeSlider(
        id='trip-duration-slider',
        min=0,
        max=120,
        step=5,
        marks={i: str(i) for i in range(0, 121, 5)},
        value=[30, 60],
    ),

    html.Label('Pickup Day'),
    dcc.Dropdown(
        id='pickup-day-dropdown',
        options=day_options,
        multi=False,
        style={'width': '50%'},
        placeholder="Select Pickup Day",
        value=2
    ),

    html.Label('Dropoff Day'),
    dcc.Dropdown(
        id='dropoff-day-dropdown',
        options=day_options,
        multi=False,
        style={'width': '50%'},
        placeholder="Select Dropoff Day",
        value=2
    ),

    html.Label('Pickup Hour (24 Hr Format)'),
    dcc.Slider(
        id='pickup-hour-slider',
        min=0,
        max=23,
        step=1,
        marks={i: str(i) for i in range(24)},
        value=12,
    ),
    dcc.Loading(
        id="loading-estimate",
        type="circle",
        children=[
            html.Div(id='selected-values', style={'margin-top': '20px', 'text-align': 'center', 'font-size': '40px'}),
            html.Div(id='estimated-value', style={'margin-top': '20px', 'text-align': 'center', 'font-size': '40px'}),
        ]
    ),
    html.Br(),
])


@my_app.callback(
    Output('error-message', 'children'),
    [Input('payment-type-textarea', 'value')]
)
def update_error_message(payment_type):
    if payment_type.lower() not in accepted_payment_types:
        return html.Div('Error: Invalid payment type!', style={'color': 'red'})
    else:
        return html.Div()


@my_app.callback(
    Output('error-message-passenger', 'children'),
    [Input('passenger-input', 'value')]
)
def update_error_message(passenger):
    if passenger <= 0:
        return html.Div('Error: Invalid Input!', style={'color': 'red'})
    else:
        return html.Div()


@my_app.callback(
    [Output('selected-values', 'children'),
     Output('estimated-value', 'children')],
    Input('vendor-dropdown', 'value'),
    Input('passenger-input', 'value'),
    Input('distance-slider', 'value'),
    Input('rate-code-dropdown', 'value'),
    Input('pickup-location-dropdown', 'value'),
    Input('dropoff-location-dropdown', 'value'),
    Input('payment-type-textarea', 'value'),
    Input('base-fare-slider', 'value'),
    Input('trip-duration-slider', 'value'),
    Input('pickup-day-dropdown', 'value'),
    Input('dropoff-day-dropdown', 'value'),
    Input('pickup-hour-slider', 'value'),
)
def display_selected_values(
        vendor, passenger_count, trip_distance, rate_code,
        pickup_location, dropoff_location, payment_type,
        base_fare, trip_duration, pickup_day, dropoff_day, pickup_hour
):
    average_distance = sum(trip_distance) / len(trip_distance) if trip_distance else 0
    average_duration = sum(trip_duration) / len(trip_duration) if trip_duration else 0

    selected_payment_type = reversed_mapping[payment_type.lower()]

    airport_fee = 0.0
    if pickup_location in (1, 132, 139):
        airport_fee = 1.25

    model_input = pd.DataFrame({
        'VendorID': [vendor],
        'passenger_count': [passenger_count],
        'trip_distance': [average_distance],
        'RatecodeID': [rate_code],
        'store_and_fwd_flag': [0],
        'PULocationID': [pickup_location],
        'DOLocationID': [dropoff_location],
        'payment_type': [selected_payment_type],
        'fare_amount': [base_fare],
        'extra': [extra_mean],
        'mta_tax': [mta_tax_mean],
        'tip_amount': [tip_amount_mean],
        'tolls_amount': [tolls_amount_mean],
        'improvement_surcharge': [improvement_surcharge_mean],
        'congestion_surcharge': [congestion_surcharge_mean],
        'airport_fee': [airport_fee],
        'trip_duration': [average_duration],
        'pickup_day': [pickup_day],
        'dropoff_day': [dropoff_day],
        'pickup_hour': [pickup_hour],
    })
    prediction = model.predict(model_input)
    time.sleep(1)
    std = pickupzone_std_df[pickupzone_std_df['PULocationID'] == pickup_location]['total_amount']
    probable_fare = f'Most Probable fare = {prediction[0]: .2f}$'
    np.random.seed(5075)
    lower_limit = prediction[0] - std.item() if prediction[0] - std.item() >= base_fare \
        else random.uniform(5.0, 10.0)
    upper_limit = prediction[0] + std.item()
    output_str = f'This trip will cost the passenger between {lower_limit: .2f}$ - {upper_limit:.2f}$'
    return [output_str, probable_fare]


# ================
# Parent call back
# ===============
@my_app.callback(
    Output(component_id='layout', component_property='children'),
    Input(component_id='ftp-tabs', component_property='value')
)
def update_layout(ques):
    if ques == 'q1':
        return question1_layout
    elif ques == 'q2':
        return question2_layout
    elif ques == 'q3':
        return question3_layout
    elif ques == 'q4':
        return question4_layout
    elif ques == 'q5':
        return question5_layout
    elif ques == 'q6':
        return question6_layout
    elif ques == 'q7':
        return question7_layout
    elif ques == 'q8':
        return question8_layout
    elif ques == 'q9':
        return question9_layout
    elif ques == 'q10':
        return question10_layout
    else:
        return question9_layout


my_app.run_server(
    port=8050,
    host='0.0.0.0'
)
