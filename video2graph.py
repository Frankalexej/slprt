# Preprocessing
# Use Mediapipe to turn video to detected hand landmarks (only hands at the moment)
# 1. Detect
# 2. Write into graph file (?)
# 3. Add it on original video, and save the video to another dir in src
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main(): 
    print("Hello world! ")

if __name__ == "__main__":
    main()