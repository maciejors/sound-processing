import streamlit as st
import numpy as np
import plotly.express as px

from app import store

st.markdown('# Clip-level parameters')

if store.get_signal() is not None:
    signal = store.get_signal()
    timestamps_frames = np.linspace(0, signal.audio_length_sec, num=signal.n_frames)

    st.markdown('---')
    st.markdown('## Audio type')
    st.markdown(f'{signal.get_audio_type()}')

    st.markdown('---')
    st.markdown('## Volume-based')
    # VSTD
    st.markdown(f'VSTD: {signal.vstd:.2f}')
    # VDR
    st.markdown(f'Volume dynamic range: {signal.volume_dynamic_range:.2f}')
    # Volume undulation
    st.markdown(f'Volume undulation: {signal.volume_undulation:.2f}')

    st.markdown('---')
    st.markdown('## Energy-based')
    # LSTER
    st.markdown(f'Low short time energy ratio: {signal.low_short_time_energy_ratio:.2f}')
    # VDR
    st.markdown(f'Energy entropy: {signal.energy_entropy():.2f}')

    st.markdown('---')
    st.markdown('## ZCR-based')
    # VSTD
    st.markdown(f'Standard deviation of ZCR: {signal.zstd:.2f}')
    # VDR
    st.markdown(f'High zero crossing rate ratio: {signal.hzcrr:.2f}')

