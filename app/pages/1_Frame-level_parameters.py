import streamlit as st
import numpy as np
import plotly.express as px

from app import store

st.markdown('# Frame-level parameters')

if store.get_wavfile() is not None:
    wav = store.get_wavfile()
    timestamps_frames = np.linspace(0, wav.audio_length_sec, num=wav.n_frames)

    # Volume
    st.markdown('---')
    plot = px.line(
        x=timestamps_frames,
        y=wav.volume,
        title=f'Volume',
        labels={'x': 'Time (in seconds)', 'y': 'Volume'},
    )
    st.plotly_chart(plot, use_container_width=True)

    # STE
    st.markdown('---')
    plot = px.line(
        x=timestamps_frames,
        y=wav.short_time_energy,
        title=f'Short time energy',
        labels={'x': 'Time (in seconds)', 'y': 'STE'},
    )
    st.plotly_chart(plot, use_container_width=True)

    # ZCR
    st.markdown('---')
    plot = px.line(
        x=timestamps_frames,
        y=wav.zero_crossing_rate,
        title=f'Zero crossing rate',
        labels={'x': 'Time (in seconds)', 'y': 'ZCR'},
    )
    st.plotly_chart(plot, use_container_width=True)

    # Silent Ratio
    st.markdown('---')
    zcr_col, vol_col = st.columns([1, 1])
    with zcr_col:
        zcr_threshold = st.number_input(label='ZCR threshold')
    with vol_col:
        volume_threshold = st.number_input(label='Volume threshold')
    st.markdown(f'#### Silent ratio: {wav.get_silence_rate(zcr_threshold, volume_threshold):.2f}')

    # F0
    st.markdown('---')
    plot = px.line(
        x=timestamps_frames,
        y=wav.fundamental_frequency,
        title=f'Fundamental frequency',
        labels={'x': 'Time (in seconds)', 'y': 'F0'},
    )
    st.plotly_chart(plot, use_container_width=True)

