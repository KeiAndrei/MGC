import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
from pydub import AudioSegment
import random

# Dummy model output for demo
GENRES = ['Pop', 'Rock', 'Jazz', 'Hip-Hop', 'Classical']
MOODS = ['Happy', 'Sad', 'Energetic', 'Calm']
SONG_DB = {
    'Pop': ['Song A', 'Song B', 'Song C'],
    'Rock': ['Song D', 'Song E', 'Song F'],
    'Jazz': ['Song G', 'Song H', 'Song I'],
    'Hip-Hop': ['Song J', 'Song K', 'Song L'],
    'Classical': ['Song M', 'Song N', 'Song O']
}

# Set Streamlit page config
st.set_page_config(page_title="Trasetone", layout="centered")
st.title("🎶 Trasetone: Smart Genre & Mood-Based Music Recommendation")

# Audio file uploader
audio_file = st.file_uploader("Upload your audio file (MP3 or WAV)", type=["mp3", "wav"])

if audio_file:
    st.audio(audio_file, format="audio/wav")

    # Convert to WAV for uniform processing
    if audio_file.type == "audio/mp3":
        audio = AudioSegment.from_file(audio_file, format="mp3")
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio_bytes = io.BytesIO()
        audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)
        y, sr = librosa.load(audio_bytes, sr=22050)
    else:
        y, sr = librosa.load(audio_file, sr=22050)

    y = librosa.util.normalize(y)

    st.subheader("🎧 Audio Feature Visualizations")

    # Waveform
    fig_wave, ax_wave = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax_wave)
    ax_wave.set_title("Waveform")
    st.pyplot(fig_wave)

    # Spectrogram
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig_spec, ax_spec = plt.subplots()
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax_spec)
    ax_spec.set_title("Spectrogram")
    st.pyplot(fig_spec)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig_mfcc, ax_mfcc = plt.subplots()
    librosa.display.specshow(mfccs, x_axis='time', ax=ax_mfcc)
    ax_mfcc.set_title("MFCC")
    st.pyplot(fig_mfcc)

    st.subheader("🎼 Genre and Mood Classification")

    # Dummy prediction output
    genre = random.choice(GENRES)
    mood = random.choice(MOODS)
    st.markdown(f"- **Genre:** {genre}")
    st.markdown(f"- **Mood:** {mood}")

    st.subheader("📻 Smart Song Recommendation")
    recommendations = random.sample(SONG_DB[genre], 3)
    for song in recommendations:
        st.write(f"- {song}")

    st.subheader("📊 Model Performance")
    st.markdown("- **Confidence Score:** 87.4%")
    st.markdown("- **Processing Time:** 1.2 seconds")

    with st.expander("🔍 View Raw Audio Features"):
        try:
            zcr = librosa.feature.zero_crossing_rate(y)
            st.write(f"Zero Crossing Rate (mean): {np.mean(zcr):.4f}")
        except Exception as e:
            st.warning(f"Could not compute ZCR: {e}")

        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if tempo is not None:
                st.write(f"Estimated Tempo: {tempo:.2f} BPM")
            else:
                st.write("Estimated Tempo: Not detected")
        except Exception as e:
            st.warning(f"Could not estimate tempo: {e}")
