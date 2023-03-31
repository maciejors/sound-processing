import numpy as np
import streamlit as st
import plotly.express as px

from app import store
from app.components.file_uploader import file_uploader


st.markdown('# Sound analysis')
file_uploader()

if store.get_wavfile() is not None:
    wavfile = store.get_wavfile()

    # audio player
    st.audio(wavfile.samples, sample_rate=wavfile.sample_rate_per_channel)

    # basic info
    st.markdown('## Basic audio properties')
    st.markdown(f'- Audio length: {wavfile.audio_length_sec:.2f}s')
    st.markdown(f'- Number of channels: {wavfile.n_channels}')
    st.markdown(f'- Total number of samples: {wavfile.n_samples}')
    st.markdown(f'- Number of samples per channel: {wavfile.n_samples_per_channel}')
    st.markdown(f'- Number of samples per second: {wavfile.sample_rate}')
    st.markdown(f'- Number of samples in a channel per second: {wavfile.sample_rate_per_channel}')

    # plotting signals
    st.markdown('## Signals per channel')
    timestamps_samples = np.linspace(0, wavfile.audio_length_sec, num=wavfile.n_samples_per_channel)
    plot = px.line(
        x=timestamps_samples,
        y=wavfile.samples,
        title=f'Signal plot',
        labels={'x': 'Time (in seconds)', 'y': 'Amplitude'},
    )
    st.plotly_chart(plot, use_container_width=True)
