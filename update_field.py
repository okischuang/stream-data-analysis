import pandas as pd
import uuid
import numpy as np

# Load your dataset
df = pd.read_csv('./final_dataset.csv')

# Replace 'asset_id' with new UUIDs
df['asset_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
df['sub_property_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

# Define lists of well-known CDN providers and ISPs for more realistic data generation
cdn_providers = ['Akamai', 'Cloudflare', 'Fastly', 'CDN77', 'KeyCDN', 'StackPath', 'Amazon CloudFront', 'Microsoft Azure CDN', 'Google Cloud CDN']
isp_names = ['Comcast', 'Verizon', 'AT&T', 'Sprint', 'T-Mobile', 'CenturyLink', 'Charter Communications', 'Cox Communications', 'Orange', 'Vodafone']

def generate_realistic_cdn():
    return np.random.choice(cdn_providers)

def generate_realistic_isp():
    return np.random.choice(isp_names)

# Function to generate a version number
def generate_version():
    major = np.random.randint(0, 10)
    minor = np.random.randint(0, 10)
    patch = np.random.randint(0, 10)
    return f"{major}.{minor}.{patch}"


# Update the dataset generation process to include these new realistic values
df['isp'] = [generate_realistic_isp() for _ in range(len(df))]

# Define a list of video player-specific error codes (integer type)
video_player_error_codes = [
    1001,  # General playback error
    1002,  # Network error
    1003,  # Media not found
    1004,  # Unsupported media format
    1005,  # Codec not supported
    1006,  # DRM license error
    0      # No Error
]

# Randomly assign a video player-specific error code to each record
# Adjust probabilities as needed to reflect a realistic distribution of errors
probabilities = [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.3]  # Example probabilities
df['player_error_code'] = np.random.choice(video_player_error_codes, size=len(df), p=probabilities)

# Define a list of well-known video player software names
player_software_names = [
    'Video.js', 'JW Player', 'Flowplayer', 'Plyr', 'MediaElement.js',
    'Shaka Player', 'Clappr', 'dash.js', 'HLS.js', 'Adobe Primetime Player'
]

# Randomly assign a video player software name to each record
df['player_software'] = np.random.choice(player_software_names, size=len(df))

df['player_software_version'] = [generate_version() for _ in range(len(df))]
df['player_version'] = [generate_version() for _ in range(len(df))]

# Generate more realistic "player_startup_time" values in milliseconds
# Using a range that represents quick (200ms) to slower (5000ms) startup times
startup_time = np.random.randint(200, 5001, size=len(df))
df['player_startup_time'] = startup_time
df['video_startup_time'] = startup_time

# Define a list of sample hostnames of video sites
video_site_hostnames = [
    'video.example.com', 'media.example.org', 'cdn.example.net',
    'streaming.example.edu', 'content.example.tv', 'videos.example.co'
]

# Randomly assign a hostname to each record
df['source_hostname'] = np.random.choice(video_site_hostnames, size=len(df))

source_types = [
    'dash', 'application/x-mpegurl', 'video/mp4',
]

# Randomly assign a source_type to each record
df['source_type'] = np.random.choice(source_types, size=len(df))

video_content_types = ['short', 'movie', 'episode', 'clip', 'trailer', 'event']
df['video_content_type'] = np.random.choice(video_content_types, size=len(df))

# Enhance video_series with a selection of series names
series_names = ['The Girls', 'Cosmic Journey', 'Mystery of the Depths', 'Heroes Unleashed', 'Lost in Time']
df['video_series'] = np.random.choice(series_names, size=len(df))

# Define a list of DRM types
drm_types = ['widevine', 'playready', 'fairplay', 'clearkey']

# Randomly assign a DRM type to each record
df['view_drm_type'] = np.random.choice(drm_types, size=len(df))

# Enhance "viewer_application_engine" with web browser engines
browser_engines = ['Gecko', 'WebKit', 'Blink', 'EdgeHTML', 'Trident']
df['viewer_application_engine'] = np.random.choice(browser_engines, size=len(df))

viewer_connection_types = ['mobile', 'wired', 'wireless']
df['viewer_connection_type'] = np.random.choice(viewer_connection_types, size=len(df))

# Save the updated dataset
df.to_csv('final_dataset.csv', index=False)