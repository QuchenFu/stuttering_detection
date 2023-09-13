import os
import librosa

# Function to check the length of audio files
def check_audio_length(folder_path):
    # Iterate through all files and subfolders in the given folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an audio file (you can modify this condition)
            if file.endswith(('.wav', '.mp3', '.ogg')):  # Add more audio formats if needed
                audio_path = os.path.join(root, file)
                try:
                    # Load the audio file with librosa
                    y, sr = librosa.load(audio_path, sr=16000)
                    # Check if the length of the audio is 48000 samples
                    if len(y) != 48000:
                        print(f"File {audio_path} does not have a length of 48000 samples.")
                except Exception as e:
                    print(f"Error processing file {audio_path}: {str(e)}")

# Specify the folder path to start the search
folder_path = 'C:\\Users\\quchenfu\\Downloads\\SEP_28K_CLIP'
check_audio_length(folder_path)
