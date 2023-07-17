import os
import glob

class Interpolator: 
    def __init__(self, data_dir, prefix):
        self.data_dir = data_dir
        self.prefix = prefix

# Function to perform sliding window linear interpolation
def sliding_window_interpolation(features, flags, window_size):
    interpolated_features = np.copy(features)

    # Iterate through each position in the flags array
    for i in range(features.shape[0]):
        if flags[i] == FLAG_NONE:
            start_idx = max(0, i - window_size // 2)
            end_idx = min(i + window_size // 2 + 1, features.shape[0])

            # Check if there are any OK features in the window
            if FLAG_OK in flags[start_idx:end_idx]:
                window_features = features[start_idx:end_idx]
                window_flags = flags[start_idx:end_idx]

                # Find indices of OK features
                ok_indices = np.where(window_flags == FLAG_OK)[0]

                if len(ok_indices) > 0:
                    # Interpolate missing feature based on non-missing features
                    non_missing_features = window_features[ok_indices]
                    interpolated_feature = np.mean(non_missing_features, axis=0)

                    interpolated_features[i] = interpolated_feature
                else:
                    # Fill missing feature with nearest non-missing feature
                    distances = np.abs(start_idx + ok_indices - i)
                    nearest_index = start_idx + ok_indices[np.argmin(distances)]

                    interpolated_features[i] = features[nearest_index]

    return interpolated_features

# Example usage
# Assuming you have the 'flags' and 'features' arrays

# Define the flag values
FLAG_OK = 0
FLAG_NONE = 1

# Specify the window size
window_size = 5

# Perform sliding window interpolation
interpolated_features = sliding_window_interpolation(features, flags, window_size)

# Print the interpolated features
print("Interpolated Features:")
print(interpolated_features)

            