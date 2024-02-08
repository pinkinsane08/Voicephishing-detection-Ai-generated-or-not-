import streamlit as st
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from glob import glob
import io
import librosa
import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.io.wavfile import read


#function to load the audio file
def load_audio(audiopath, sampling_rate=22000):
    if isinstance(audiopath, str):
        if audiopath.endswith('.mp3'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        else:
            assert False, f"Unsupported audio format provided: {audiopath[-4:]}"
    elif isinstance(audiopath, io.BytesIO):
        audio, lsr = torchaudio.load(audiopath)
        audio = audio[0]  # Remove any channel data

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with audio data. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio.unsqueeze(0)

# Function for classifier
def classify_audio_clip(clip):
    """
    Returns whether or not the classifier thinks the given clip came from AI generation.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: The probability of the audio clip being Al-generated.
    """
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4, resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(0)  # Corrected variable usage
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]

st.set_page_config(layout="wide")

def main():

    st.title("Vishing Detection")
    #file uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3"])

    if uploaded_file is not None:
        if st.button("Analyze Audio"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.info("Your results are below")
                #load abd classifythe audio file
                audio_clip = load_audio(uploaded_file)
                result = classify_audio_clip(audio_clip)
                st.info(f"Result Probability: {result}")
                st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI Generated. ")

            with col2:
                st.info("Your uploaded audio is below")
                st.audio(uploaded_file)
                fig = px.line()
                fig.add_scatter(x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
                fig.update_layout(
                    title="Waveform Plot",
                    xaxis_title="Time",
                    yaxis_title="Amplitude"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.info("Disclaimer")
                st.warning("These classification or detection mechanisms are not always accurate. They should be considered as a strong signal and not the ultimate decision makers.")

if __name__ == "__main__":
    main() 