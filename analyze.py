import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load the dataset
dataset_path = './final_dataset.csv'  # Update this path to your dataset
stream_logs = pd.read_csv(dataset_path)

# EDA for content popularity
def viewer_experience_by_content(stream_logs):
    # Check if the fields exist in the dataset
    fields_to_check = ['video_content_type', 'video_quality_score', 'video_duration']
    for field in fields_to_check:
        if field not in stream_logs.columns:
            print(f"Field '{field}' not found in the dataset.")
        else:
            print(f"Field '{field}' is present in the dataset.")

    print(stream_logs.columns)
    # If the fields are present, perform the EDA
    if 'video_content_type' in stream_logs.columns:
        content_type_popularity = stream_logs.groupby('video_content_type')[['watch_time', 'viewer_experience_score']].mean()

    if 'video_quality_score' in stream_logs.columns:
        # Define bins for quality score ranges
        max_quality_score = stream_logs['video_quality_score'].max()
        quality_bins = list(range(0, int(max_quality_score) + 20, 20))
        quality_labels = [f'{i}-{i+19}' for i in quality_bins[:-1]]
        stream_logs['quality_score_range'] = pd.cut(stream_logs['video_quality_score'], bins=quality_bins, labels=quality_labels, include_lowest=True)

        # Group by 'quality_score_range' and calculate the average watch time and viewer experience score
        quality_score_ranges_popularity = stream_logs.groupby('quality_score_range')[['watch_time', 'viewer_experience_score']].mean().reset_index()


    if 'video_duration' in stream_logs.columns:
        # Create duration categories if 'video_duration' is in seconds (or convert accordingly)
        duration_bins = [0, 300, 600, 1200, 1800, 3600, float('inf')]
        duration_labels = ['<5min', '5-10min', '10-20min', '20-30min', '30-60min', '>60min']
        stream_logs['video_duration_category'] = pd.cut(stream_logs['video_duration'], bins=duration_bins, labels=duration_labels)
        duration_popularity = stream_logs.groupby('video_duration_category')[['watch_time', 'viewer_experience_score']].mean()

    # Output the results
    print(content_type_popularity)
    print(quality_score_ranges_popularity)
    print(duration_popularity)

    # Visualize 'video_content_type' with respect to average watch time
    plt.figure(figsize=(10, 6))
    sns.barplot(data=content_type_popularity, x='video_content_type', y='watch_time', palette='Set2')
    plt.title('Content Popularity by Video Content Type - Watch Time')
    plt.xlabel('Video Content Type')
    plt.ylabel('Average Watch Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visualize 'video_content_type' with respect to viewer experience score
    plt.figure(figsize=(10, 6))
    sns.barplot(data=content_type_popularity, x='video_content_type', y='viewer_experience_score', palette='Set3')
    plt.title('Content Popularity by Video Content Type - Viewer Experience Score')
    plt.xlabel('Video Content Type')
    plt.ylabel('Average Viewer Experience Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visualization for 'quality_score_range' with respect to average watch time
    plt.figure(figsize=(14, 6))
    sns.barplot(data=quality_score_ranges_popularity, x='quality_score_range', y='watch_time', palette='Set2')
    plt.title('Content Popularity by Video Quality Score Range - Watch Time')
    plt.xlabel('Video Quality Score Range')
    plt.ylabel('Average Watch Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visualization for 'quality_score_range' with respect to viewer experience score
    plt.figure(figsize=(14, 6))
    sns.barplot(data=quality_score_ranges_popularity, x='quality_score_range', y='viewer_experience_score', palette='Set3')
    plt.title('Content Popularity by Video Quality Score Range - Viewer Experience Score')
    plt.xlabel('Video Quality Score Range')
    plt.ylabel('Average Viewer Experience Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visualize 'video_duration' with respect to average watch time
    plt.figure(figsize=(10, 6))
    sns.barplot(data=duration_popularity, x='video_duration_category', y='watch_time', palette='Set2')
    plt.title('Content Popularity by Video Duration - Watch Time')
    plt.xlabel('Video Duration')
    plt.ylabel('Average Watch Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visualize 'video_duration' with respect to viewer experience score
    plt.figure(figsize=(10, 6))
    sns.barplot(data=duration_popularity, x='video_duration_category', y='viewer_experience_score', palette='Set3')
    plt.title('Content Popularity by Video Duration - Viewer Experience Score')
    plt.xlabel('Video Duration')
    plt.ylabel('Average Viewer Experience Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def viewer_experience_by_demographics(stream_logs):
    # Check if the fields exist in the dataset
    fields_to_check = ['viewer_device_category', 'browser', 'country']
    for field in fields_to_check:
        if field not in stream_logs.columns:
            print(f"Field '{field}' not found in the dataset.")
        else:
            print(f"Field '{field}' is present in the dataset.")

    print(stream_logs.columns)
    # If the fields are present, perform the EDA
    if 'viewer_device_category' in stream_logs.columns:
        engagement_by_device_category = stream_logs.groupby('viewer_device_category')[['watch_time', 'viewer_experience_score']].mean()

    if 'browser' in stream_logs.columns:
        engagement_by_browser = stream_logs.groupby('browser')[['watch_time', 'viewer_experience_score']].mean()

    if 'country' in stream_logs.columns:
        engagement_by_country = stream_logs.groupby('country')[['watch_time', 'viewer_experience_score']].mean()

    # Output the results
    print(engagement_by_device_category)
    print(engagement_by_browser)
    print(engagement_by_country)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=engagement_by_device_category, x='viewer_device_category', y='watch_time', palette='Set2')
    plt.title('Engagement by Device Category - Watch Time')
    plt.xlabel('Video Device Category')
    plt.ylabel('Average Watch Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=engagement_by_device_category, x='viewer_device_category', y='viewer_experience_score', palette='Set3')
    plt.title('Engagement by Device Category - Viewer Experience Score')
    plt.xlabel('Video Device Category')
    plt.ylabel('Average Viewer Experience Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    sns.barplot(data=engagement_by_browser, x='browser', y='watch_time', palette='Set2')
    plt.title('Engagement by Browser - Watch Time')
    plt.xlabel('Browser')
    plt.ylabel('Average Watch Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    sns.barplot(data=engagement_by_browser, x='browser', y='viewer_experience_score', palette='Set3')
    plt.title('Engagement by Browser - Viewer Experience Score')
    plt.xlabel('Browser')
    plt.ylabel('Average Viewer Experience Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=engagement_by_country, x='country', y='watch_time', palette='Set2')
    plt.title('Engagement by Country - Watch Time')
    plt.xlabel('Country')
    plt.ylabel('Average Watch Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visualize 'video_duration' with respect to viewer experience score
    plt.figure(figsize=(10, 6))
    # Visualize 'video_duration' with respect to average watch time
    sns.barplot(data=engagement_by_country, x='country', y='viewer_experience_score', palette='Set3')
    plt.title('Engagement by Country - Viewer Experience Score')
    plt.xlabel('country')
    plt.ylabel('Average Viewer Experience Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# EDA for Content Strategy
def content_strategy_eda(stream_logs):
    

    # Identifying top contents based on total watch time
    top_contents_watch_time = stream_logs.groupby('asset_id')['watch_time'].sum().sort_values(ascending=False).head(10)
    
    # Distribution of viewer experience scores
    viewer_experience_distribution = stream_logs['viewer_experience_score'].dropna()

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
def model_viewer_interest(stream_logs):
    # Simplified Feature Engineering
    # Selecting a subset of features for the mode   l
    features = ['viewer_device_category', 'country_name', 'browser', 'watch_time']
    data_for_model = stream_logs[features]
    # Defining the target variable based on watch_time (high interest = top 50% of watch_time)
    data_for_model['high_interest'] = data_for_model['watch_time'] > data_for_model['watch_time'].median()

    # Preparing features and target variable
    X = data_for_model.drop('high_interest', axis=1)
    y = data_for_model['high_interest']

    # Encoding categorical variables and normalizing numerical features
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
def ad_strategy_analysis(stream_logs):
    # Analyzing the impact of ads on viewer engagement
    ad_impact_watch_time = stream_logs.groupby('view_has_ad')['watch_time'].mean()
    ad_impact_experience_score = stream_logs.groupby('view_has_ad')['viewer_experience_score'].mean()
    ad_drop_off_rate = stream_logs.groupby('view_has_ad')['exit_before_video_start'].mean()
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

    # Segmented Analysis by Ad Quantity
    segmented_by_quantity = stream_logs.groupby('ad_quantity')[['watch_time', 'viewer_experience_score']].mean().reset_index()

    # Segmented Analysis by Ad Length
    # Creating bins for ad length to facilitate segmented analysis
    bins = [0, 30, 60, 90, 120]  # Defining bins for ad length ranges
    labels = ['1-30', '31-60', '61-90', '91-120']  # Label for each bin
    stream_logs['ad_length_segment'] = pd.cut(stream_logs['ad_length'], bins=bins, labels=labels, right=False)

    segmented_by_length = stream_logs.groupby('ad_length_segment')[['watch_time', 'viewer_experience_score']].mean().reset_index()

    # Segmented Analysis by Ad Quantity - Watch Time
    sns.barplot(x='ad_quantity', y='watch_time', data=segmented_by_quantity, palette='coolwarm')
    plt.title('Average Watch Time by Ad Quantity')
    plt.xlabel('Ad Quantity')
    plt.ylabel('Average Watch Time')
    plt.show()

    # Segmented Analysis by Ad Quantity - Viewer Experience Score
    sns.barplot(x='ad_quantity', y='viewer_experience_score', data=segmented_by_quantity, palette='coolwarm')
    plt.title('Average Viewer Experience Score by Ad Quantity')
    plt.xlabel('Ad Quantity')
    plt.ylabel('Average Viewer Experience Score')
    plt.show()

    # Segmented Analysis by Ad Length - Watch Time
    sns.barplot(x='ad_length_segment', y='watch_time', data=segmented_by_length, palette='viridis')
    plt.title('Average Watch Time by Ad Length Segment')
    plt.xlabel('Ad Length Segment (seconds)')
    plt.ylabel('Average Watch Time')
    plt.show()

    # Segmented Analysis by Ad Length - Viewer Experience Score
    sns.barplot(x='ad_length_segment', y='viewer_experience_score', data=segmented_by_length, palette='viridis')
    plt.title('Average Viewer Experience Score by Ad Length Segment')
    plt.xlabel('Ad Length Segment (seconds)')
    plt.ylabel('Average Viewer Experience Score')
    plt.show()

def video_perf_engagement_correlation_analysis(stream_logs):
    # Calculate 'view_seek_latency' as 'view_seek_duration' divided by 'view_seek_count'
    # To avoid division by zero, we'll replace zeros in 'view_seek_count' with NaN and then use fillna to handle them.
    stream_logs['view_seek_count'] = stream_logs['view_seek_count'].replace(0, np.nan)
    stream_logs['view_seek_latency'] = stream_logs['view_seek_duration'] / stream_logs['view_seek_count']
    
    # Fill NaN values resulted from the division with 0 for cases where 'view_seek_count' was originally 0
    stream_logs['view_seek_latency'] = stream_logs['view_seek_latency'].fillna(0)

    # Now we'll evaluate the impact of these performance metrics on viewer engagement
    # We'll use 'watch_time' and 'viewer_experience_score' as engagement metrics
    performance_metrics = ['startup_time_score', 'weighted_average_bitrate', 'view_seek_latency']
    engagement_metrics = ['watch_time', 'viewer_experience_score']

    # Calculate correlations between performance metrics and engagement metrics
    performance_impact_corr = stream_logs[performance_metrics + engagement_metrics].corr().loc[performance_metrics, engagement_metrics]

    # Visualize the correlations
    sns.heatmap(performance_impact_corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation of Video Performance Metrics with Viewer Engagement')
    plt.show()

# Execute the functions
# viewer_experience_by_content(stream_logs)
# viewer_experience_by_demographics(stream_logs)
# content_strategy_eda(stream_logs)
# model_viewer_interest(stream_logs)
# ad_strategy_analysis(stream_logs)
# simulate_model_result()
video_perf_engagement_correlation_analysis(stream_logs)

def simulate_model_result():
    # Sample data for viewer segments and their optimal ad length predicted by the model
    viewer_segments = ['Sports Enthusiasts', 'Movie Lovers', 'News Followers', 'Gamers', 'Music Fans']
    optimal_ad_length = [90, 120, 60, 45, 30]  # Simulated optimal ad lengths in seconds

    # Creating the chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=optimal_ad_length, y=viewer_segments, palette='tab10')
    plt.title('Predicted Optimal Ad Length for Viewer Segments')
    plt.xlabel('Optimal Ad Length (seconds)')
    plt.ylabel('Viewer Segment')
    plt.xlim(0, 150)  # Extend x-axis to show a range up to 150 seconds for clarity
    plt.grid(axis='x')

    # Show the plot
    plt.show()
