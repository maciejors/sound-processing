import streamlit as st
import numpy as np
import plotly.express as px

from app import store
from app.core.windowing import Windowing

st.markdown('# Windowing')

if store.get_signal() is not None:
    signal = store.get_signal()
    windowing = Windowing(signal)

    selected_func_name = st.selectbox(
        'Select a windowing function',
        windowing.function_map.keys(),
    )
    windowing.set_windowing_func(selected_func_name)

    # Signal plot
    st.markdown('---')
    st.markdown('## Signal plots')
    timestamps_samples = np.linspace(0, signal.audio_length_sec, num=signal.n_samples)

    original_signal_col, windowed_signal_col = st.columns([1, 1])
    with original_signal_col:
        plot = px.line(
            x=timestamps_samples,
            y=signal.samples,
            title='Original signal',
            labels={'x': 'Time (in seconds)', 'y': 'Amplitude'},
        )
        st.plotly_chart(plot, use_container_width=True)
    with windowed_signal_col:
        plot = px.line(
            x=timestamps_samples,
            y=windowing.windowed_samples(),
            title='Windowed signal',
            labels={'x': 'Time (in seconds)', 'y': 'Amplitude'},
        )
        st.plotly_chart(plot, use_container_width=True)

    # Frequency plot
    st.markdown('---')
    st.markdown('## Frequency domain plots')

    original_signal_col, windowed_signal_col = st.columns([1, 1])
    with original_signal_col:
        plot = px.line(
            x=signal.fft_freqs_full,
            y=signal.fft_magn_spectr_full,
            title='Original frequency domain',
            labels={'x': 'Time (in seconds)', 'y': 'Magnitude'},
        )
        st.plotly_chart(plot, use_container_width=True)
    with windowed_signal_col:
        plot = px.line(
            x=windowing.windowed_fft_freqs(),
            y=windowing.windowed_fft_magn_spectr(),
            title='Frequency domain after windowing',
            labels={'x': 'Time (in seconds)', 'y': 'Magnitude'},
        )
        st.plotly_chart(plot, use_container_width=True)

    # Spectrogram
    st.markdown('---')
    st.markdown('## Spectrogram')
    timestamps_frames = np.linspace(0, signal.audio_length_sec, num=signal.n_frames)

    plot = px.imshow(
        x=timestamps_frames,
        img=windowing.spectrogram(),
        color_continuous_scale='RdBu_r',
        aspect='auto',
        labels={'x': 'Time [s]', 'y': 'Frequency'},
    )
    st.plotly_chart(plot, use_container_width=True)
