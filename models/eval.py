import onnxruntime
import numpy as np
import librosa

# Load the ONNX model
onnx_model_path = "C:\\Users\\quchenfu\\Downloads\\stuttering_detection\\model.onnx"  # Replace with the path to your ONNX model file
session = onnxruntime.InferenceSession(onnx_model_path)
label=['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
# Create sample input data (replace with your actual data)
array_shape = (1, 48000)
dummy_input = np.random.rand(*array_shape)
input_data = librosa.feature.melspectrogram(y=dummy_input, sr=16000)

# Define the input name based on the ONNX model (you can find this in the ONNX model)
input_name = session.get_inputs()[0].name

# Prepare the input data as a dictionary
input_dict = {input_name: input_data.astype(np.float32)}

# Perform inference
outputs = session.run(None, input_dict)

# Process the model outputs (replace with your post-processing logic)
output_data = outputs[0][0]
softmax_scores = np.exp(output_data) / np.sum(np.exp(output_data))
print(softmax_scores)


