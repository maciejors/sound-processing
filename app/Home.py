import numpy as np
import streamlit as st
import plotly.express as px

from app import store
from app.components.file_uploader import file_uploader
from app.core.exporting import Bundler

st.markdown('# Sound analysis')
file_uploader()

if store.get_signal() is not None:
    signal = store.get_signal()

    # audio boundaries
    st.markdown('## Audio boundaries')
    start_id, end_id = st.slider(
        'Set the range of audio to analyse',
        value=signal.boundaries,
        min_value=0, max_value=signal.n_samples_all,
        format="",
    )
    start_time_s = start_id / signal.sample_rate
    end_time_s = end_id / signal.sample_rate
    st.markdown(f'Selected interval: {start_time_s:.2f}s - {end_time_s:.2f}s')
    if st.button('Apply boundaries'):
        signal.set_boundaries(start_id, end_id)

    # audio player
    st.audio(signal.samples, sample_rate=signal.sample_rate)

    # basic info
    st.markdown('## Basic audio properties')
    st.markdown(f'- Audio length: {signal.audio_length_sec:.2f}s')
    st.markdown(f'- Number of channels: {signal.n_channels}')
    st.markdown(f'- Total number of samples: {signal.n_samples}')
    st.markdown(f'- Number of samples per second: {signal.sample_rate}')

    # csv
    st.markdown('## Export features to JSON')
    if st.button('Generate JSON'):
        json_str = Bundler(signal).export_json()
        st.download_button(
            label='Download JSON',
            data=json_str,
            file_name='data.json',
            mime='text/json'
        )

    # plotting signals
    st.markdown('## Averaged signal')
    timestamps_samples = np.linspace(0, signal.all_audio_length_sec, num=signal.n_samples_all)
    plot = px.line(
        x=timestamps_samples,
        y=signal.samples_all,
        title=f'Full signal plot',
        labels={'x': 'Time (in seconds)', 'y': 'Amplitude'},
    )
    plot.add_shape(
        type='line',
        x0=timestamps_samples[signal.boundaries[0]], y0=np.min(signal.samples),
        x1=timestamps_samples[signal.boundaries[0]], y1=np.max(signal.samples),
        line=dict(color='red', width=2)
    )
    plot.add_shape(
        type='line',
        x0=timestamps_samples[signal.boundaries[1] - 1], y0=np.min(signal.samples),
        x1=timestamps_samples[signal.boundaries[1] - 1], y1=np.max(signal.samples),
        line=dict(color='red', width=2)
    )
    st.plotly_chart(plot, use_container_width=True)
