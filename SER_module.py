import torch
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.models import HuggingFaceModel

model = HuggingFaceModel.Voice.Wav2Vec2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LABELS = ['neutral', 'happiness', 'sadness', 'enthusiasm', 'fear', 'anger', 'disgust']

def predict_emotions_audio(file_path):
    vr = VoiceRecognizer(model, device)
    result = vr.recognize(file_path)
    emotions_list = {}
    max_pred = -1
    max_label = ""
    for i in LABELS:
        emotions_list[i] = result.get(i)
        emotions_list[i] = '{:.10f}'.format(emotions_list[i])
        if float(emotions_list[i]) > float(max_pred):
            max_pred = emotions_list[i]
            max_label = i

    return max_label, emotions_list

