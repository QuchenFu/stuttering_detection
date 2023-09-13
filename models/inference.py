import librosa
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from scipy.io import wavfile
# Load the ONNX model
onnx_model_path = "C:\\Users\\quchenfu\\Downloads\\stuttering_detection\\model.onnx"  # Replace with the path to your ONNX model file
session = onnxruntime.InferenceSession(onnx_model_path)
label=['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
# Load the audio file
audio_file = "C:\\Users\\quchenfu\\Downloads\\Placeholder for-20230829_101338-Meeting Recording.wav"  # Replace with the path to your audio file
# audio_file = "C:\\Users\\quchenfu\\Downloads\\IStutterSoWhat_0.wav" 
audio, sr = librosa.load(audio_file, sr=None)
# Parameters for chunking
chunk_length = 3  # 3 seconds per chunk
overlap_length = 2  # 2 seconds overlap
sample_rate = sr

# Calculate the number of chunks
num_samples = len(audio)
num_frames = int(np.ceil(num_samples / (sample_rate * chunk_length)))

# Initialize a list to store the last number from each clip
last_numbers = []

# Process each chunk
for frame_idx in range(num_frames):
    start = frame_idx * sample_rate * (chunk_length - overlap_length)
    end = start + sample_rate * chunk_length
    
    # Extract a chunk from the audio
    chunk = audio[start:end]
    
    # Perform inference on the chunk
    # You need to adapt this part based on your model's input requirements
    # Replace 'features' with the appropriate input name from your model
    input_name = session.get_inputs()[0].name
    input_data = np.expand_dims(chunk, axis=0).astype(np.float32)
    input_data = librosa.feature.melspectrogram(y=input_data, sr=16000)
    outputs = session.run(None, {input_name: input_data})
    outputs = np.exp(outputs) / np.sum(np.exp(outputs))
    # Assuming the last number is the one you want to save
    last_number = outputs[0][0][-2]
    if last_number>0.6:
        wav_file_name = f"output_clip_{frame_idx}.wav"
        wavfile.write(wav_file_name, sample_rate, (chunk * 32767.0).astype(np.int16))

    # Append the last number to the list
    last_numbers.append(last_number)

# Plot the numbers
plt.plot(last_numbers)
plt.xlabel("Clip Index")
plt.ylabel("Last Number")
plt.title("Last Number from Each Clip")
plt.show()
plt.savefig("last_number.png")
