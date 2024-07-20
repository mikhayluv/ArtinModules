import numpy as np
import soundfile as sf
from openvino.inference_engine import IENetwork, IEPlugin

# Path to the model files
model_xml = 'wav2vec2.xml'
model_bin = 'wav2vec2.bin'

# Load network using IENetwork
net = IENetwork(model=model_xml, weights=model_bin)

# Initialize the plugin
plugin = IEPlugin(device="CPU")

# Load the IENetwork into the plugin
exec_net = plugin.load(network=net)

# Input and output information
input_blob = next(iter(net.inputs))
output_blob = next(iter(net.outputs))

def predict_emotions_audio(file_path):
    # Load the audio file
    data, samplerate = sf.read(file_path)

    # Preprocessing for the audio to fit the model's input requirements
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # Convert to mono by averaging channels
    data = np.expand_dims(data, axis=0)

    # Perform inference
    result = exec_net.infer({input_blob: data})

    # Processing output
    output = result[output_blob]
    output = output.flatten()

    # Emotion analysis
    LABELS = ['neutral', 'happiness', 'sadness', 'enthusiasm', 'fear', 'anger', 'disgust']
    emotions_list = {}
    max_pred = -1
    max_label = ""
    for i, label in enumerate(LABELS):
        score = output[i]
        emotions_list[label] = '{:.10f}'.format(score)
        if score > max_pred:
            max_pred = score
            max_label = label

    return max_label, emotions_list


# Example usage:
file_path = 'TEST_REC.wav'
emotion, scores = predict_emotions_audio(file_path)
print(f"Predicted Emotion: {emotion}")
print(f"Scores: {scores}")
