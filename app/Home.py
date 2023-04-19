import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json

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

    # csv
    st.markdown('## Generate JSON File')
    if st.button('Generate JSON'):
        features = wavfile.get_features()
        json_str = json.dumps(features)
        st.download_button(
            label='Download JSON',
            data=json_str,
            file_name='data.json',
            mime='text/json'
        )

    # plotting signals
    st.markdown('## Averaged signal')
    timestamps_samples = np.linspace(0, wavfile.audio_length_sec, num=wavfile.n_samples)
    plot = px.line(
        x=timestamps_samples,
        y=wavfile.samples,
        title=f'Signal plot',
        labels={'x': 'Time (in seconds)', 'y': 'Amplitude'},
    )
    plot.add_shape(
        type='line',
        x0=timestamps_samples[0], y0=np.min(wavfile.samples),
        x1=timestamps_samples[0], y1=np.max(wavfile.samples),
        line=dict(color='red', width=2)
    )
    plot.add_shape(
        type='line',
        x0=timestamps_samples[-1], y0=np.min(wavfile.samples),
        x1=timestamps_samples[-1], y1=np.max(wavfile.samples),
        line=dict(color='red', width=2)
    )
    st.plotly_chart(plot, use_container_width=True)
