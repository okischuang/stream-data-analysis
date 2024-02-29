import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
dataset_path = './final_dataset.csv'  # Update this path to your dataset
cdn_logs = pd.read_csv(dataset_path)

# EDA for Content Strategy
def content_strategy_eda(cdn_logs):
    # Identifying top contents based on total watch time
    top_contents_watch_time = cdn_logs.groupby('asset_id')['watch_time'].sum().sort_values(ascending=False).head(10)
    
    # Distribution of viewer experience scores
    viewer_experience_distribution = cdn_logs['viewer_experience_score'].dropna()

    # Visualization
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_contents_watch_time.values, y=top_contents_watch_time.index, palette="viridis")
    plt.title('Top 10 Contents by Total Watch Time')
    plt.xlabel('Total Watch Time')
    plt.ylabel('Content (Asset ID)')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(viewer_experience_distribution, bins=20, kde=True, color="skyblue")
    plt.title('Distribution of Viewer Experience Scores')
    plt.xlabel('Viewer Experience Score')
    plt.ylabel('Frequency')
    plt.show()

# Model for Predicting Viewer Interest
def model_viewer_interest(cdn_logs):
    # Feature Engineering and Model Preparation
    features = ['viewer_device_category', 'country_name', 'browser', 'watch_time']
    data_for_model = cdn_logs[features]
    data_for_model['high_interest'] = data_for_model['watch_time'] > data_for_model['watch_time'].median()

    X = data_for_model.drop('high_interest', axis=1)
    y = data_for_model['high_interest']

    categorical_features = ['viewer_device_category', 'country_name', 'browser']
    numerical_features = ['watch_time']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))])

    # Splitting the dataset and Training the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

# Ad Strategy Analysis
def ad_strategy_analysis(cdn_logs):
    # Analyzing the impact of ads on viewer engagement
    ad_impact_watch_time = cdn_logs.groupby('view_has_ad')['watch_time'].mean()
    ad_impact_experience_score = cdn_logs.groupby('view_has_ad')['viewer_experience_score'].mean()
    ad_drop_off_rate = cdn_logs.groupby('view_has_ad')['exit_before_video_start'].mean()
    print(ad_drop_off_rate)


    # Visualization
    plt.figure(figsize=(10, 5))
    ad_impact_watch_time.plot(kind='bar', title="Impact of Ads on Watch Time")
    plt.ylabel('Average Watch Time')
    plt.xticks([0, 1], ['Without Ads', 'With Ads'], rotation=0)
    plt.show()

    plt.figure(figsize=(10, 5))
    ad_impact_experience_score.plot(kind='bar', title="Impact of Ads on Viewer Experience Score")
    plt.ylabel('Average Viewer Experience Score')
    plt.xticks([0, 1], ['Without Ads', 'With Ads'], rotation=0)
    plt.show()

    plt.figure(figsize=(10, 5))
    ad_drop_off_rate.plot(kind='bar', title="Impact of Ads on Drop-off Rate")
    plt.ylabel('Avarage Drop-off Rate')
    plt.xticks([0, 1], ['Without Ads', 'With Ads'], rotation=0)
    plt.show()

# Execute the functions
content_strategy_eda(cdn_logs)
model_viewer_interest(cdn_logs)
ad_strategy_analysis(cdn_logs)
