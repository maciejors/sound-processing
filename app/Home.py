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
    st.audio(wavfile.samples, sample_rate=wavfile.sample_rate)

    # basic info
    st.markdown('## Basic audio properties')
    st.markdown(f'- Audio length: {wavfile.audio_length_sec:.2f}s')
    st.markdown(f'- Number of channels: {wavfile.n_channels}')
    st.markdown(f'- Total number of samples: {wavfile.n_samples}')
    st.markdown(f'- Number of samples per second: {wavfile.sample_rate}')

    # plotting signals
    st.markdown('## Signals per channel')
    # TODO: this part should later be moved into separate file
    timestamps_samples = np.linspace(0, wavfile.audio_length_sec, num=wavfile.n_samples)
    for channel_id, channel_samples in enumerate(wavfile.samples):
        plot = px.line(
            x=timestamps_samples,
            y=channel_samples,
            title=f'Channel {channel_id + 1}',
            labels={'x': 'Time (in seconds)', 'y': 'Amplitude'},
        )
        st.plotly_chart(plot, use_container_width=True)
