import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
from pydub import AudioSegment
from sklearn.metrics import accuracy_score
import random

# Placeholder models and song database
GENRES = ['Pop', 'Rock', 'Jazz', 'Hip-Hop', 'Classical']
MOODS = ['Happy', 'Sad', 'Energetic', 'Calm']
SONG_DB = {
    'Pop': ['Song A', 'Song B', 'Song C'],
    'Rock': ['Song D', 'Song E', 'Song F'],
    'Jazz': ['Song G', 'Song H', 'Song I'],
    'Hip-Hop': ['Song J', 'Song K', 'Song L'],
    'Classical': ['Song M', 'Song N', 'Song O']
}

# Page Setup
st.set_page_config(page_title="Trasetone", layout="centered")
st.title("🎶 Trasetone: Mapping Genre and Mood for Smart Music Recommendation")

# File Upload
audio_file = st.file_uploader("Upload your audio file (MP3/WAV)", type=["mp3", "wav"])

if audio_file:
    # Load audio
    st.audio(audio_file, format='audio/wav')
    
    # Convert to WAV if MP3
    if audio_file.type == "audio/mp3":
        audio = AudioSegment.from_mp3(audio_file)
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio_bytes = io.BytesIO()
        audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)
        y, sr = librosa.load(audio_bytes, sr=22050)
    else:
        y, sr = librosa.load(audio_file, sr=22050)
    
    # Preprocessing
    y = librosa.util.normalize(y)

    st.subheader("🎧 Audio Feature Visualizations")

    # Waveform
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # Spectrogram
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig, ax = plt.subplots()
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title("Spectrogram")
    st.pyplot(fig)

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig, ax = plt.subplots()
    librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    ax.set_title("MFCC")
    st.pyplot(fig)

    st.subheader("🎼 Genre and Mood Classification")

    # Dummy prediction
    genre = random.choice(GENRES)
    mood = random.choice(MOODS)
    st.markdown(f"- **Genre:** {genre}")
    st.markdown(f"- **Mood:** {mood}")

    st.subheader("📻 Smart Song Recommendation")
    recommendations = random.sample(SONG_DB[genre], 3)
    for song in recommendations:
        st.write(f"- {song}")

    st.subheader("📊 Model Performance (Placeholder)")
    st.markdown("- **Confidence Score:** 87.4%")
    st.markdown("- **Processing Time:** 1.2s")

    with st.expander("🔍 View Raw Audio Features"):
        zcr = librosa.feature.zero_crossing_rate(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        st.write(f"Zero Crossing Rate (mean): {np.mean(zcr):.4f}")
        st.write(f"Estimated Tempo: {tempo:.2f} BPM")

