import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

def prepare_data(player_xy):
    player_xy = player_xy[player_xy[:, 0].argsort()]
    player_xy = player_xy[~np.any(player_xy[:, 1:] == 0, axis=1)]
    _, unique_indices = np.unique(player_xy[:, 0], return_index=True)
    return player_xy[unique_indices]

def remove_outliers(player_xy):
    scaler = StandardScaler()
    X = scaler.fit_transform(player_xy[:, 1:])
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    clusters = dbscan.fit_predict(X)
    return player_xy[clusters != -1]

def smooth_and_interpolate(player_xy):
    f = interpolate.interp1d(player_xy[:, 0], player_xy[:, 1:], axis=0, kind='linear', fill_value='extrapolate')
    frames = np.arange(player_xy[0, 0], player_xy[-1, 0] + 1)
    interpolated = f(frames)
    
    window_length = min(len(frames) // 5, 101)  # Increased window size
    window_length = window_length if window_length % 2 == 1 else window_length + 1
    window_length = max(window_length, 5)
    polyorder = min(3, window_length - 2)
    
    smoothed = savgol_filter(interpolated, window_length, polyorder, axis=0)
    return np.column_stack((frames, smoothed))

def connect_segments(segments, max_gap=50):
    connected = []
    current_segment = segments[0]
    for segment in segments[1:]:
        if segment[0, 0] - current_segment[-1, 0] <= max_gap:
            current_segment = np.vstack((current_segment, segment))
        else:
            connected.append(current_segment)
            current_segment = segment
    connected.append(current_segment)
    return connected

def process_player_trajectory(player_xy):
    player_xy = prepare_data(player_xy)
    player_xy = remove_outliers(player_xy)
    segments = connect_segments([player_xy])
    smoothed_segments = [smooth_and_interpolate(seg) for seg in segments if len(seg) > 5]
    
    # Detect acceleration peaks for each segment
    peaks = [detect_acceleration_peaks(seg) for seg in smoothed_segments]
    
    return smoothed_segments, peaks

def detect_acceleration_peaks(segment, min_distance=10, prominence=0.5):
    # Calculate velocity
    velocity = np.diff(segment[:, 1:], axis=0)
    # Calculate acceleration
    acceleration = np.diff(velocity, axis=0)
    # Calculate magnitude of acceleration
    acc_magnitude = np.linalg.norm(acceleration, axis=1)
    
    # Find peaks in acceleration magnitude
    peaks, _ = find_peaks(acc_magnitude, distance=min_distance, prominence=prominence)
    
    # Add 2 to peaks index to account for two diff operations
    return segment[peaks + 2]

def visualize_trajectory(original, processed, peaks):
    plt.figure(figsize=(12, 6))
    plt.plot(original[:, 1], original[:, 2], 'b.', alpha=0.3, label='Original')
    for segment in processed:
        plt.plot(segment[:, 1], segment[:, 2], 'r-', linewidth=2)
    
    # Plot acceleration peaks
    for peak_set in peaks:
        plt.plot(peak_set[:, 1], peak_set[:, 2], 'go', markersize=8, label='Acceleration Peaks')
    
    plt.legend()
    plt.title('Player Trajectory: Original vs Processed with Acceleration Peaks')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

# Usage Example
#
# Populate your player path data here... should be a numpy array with 3 columns: frame, x, y
player_xy = np.array([
    [1, 100, 100],
    [2, 105, 105],
    [3, 110, 110],
    [4, 115, 115],
    [5, 120, 120],
    [6, 125, 125],
    [7, 130, 130],
])

# Print the shape of the array to confirm it's in the right structure
print(f"Shape of player_xy: {player_xy.shape}")


processed_path, acceleration_peaks = process_player_trajectory(player_xy)
visualize_trajectory(player_xy, processed_trajectory, acceleration_peaks)
