import streamlit as st
import numpy as np
import plotly.express as px

from app import store

st.markdown('# Clip-level parameters')

if store.get_wavfile() is not None:
    wav = store.get_wavfile()
    timestamps_frames = np.linspace(0, wav.audio_length_sec, num=wav.n_frames)

    st.markdown('---')
    st.markdown('## Volume-based')
    # VSTD
    st.markdown(f'VSTD: {wav.vstd:.2f}')
    # VDR
    st.markdown(f'Volume dynamic range: {wav.volume_dynamic_range:.2f}')
    # Volume undulation
    st.markdown(f'Volume undulation: {wav.volume_undulation:.2f}')

    st.markdown('---')
    st.markdown('## Energy-based')
    # LSTER
    st.markdown(f'Low short time energy ratio: {wav.low_short_time_energy_ratio:.2f}')
    # VDR
    st.markdown(f'Energy entropy: {wav.energy_entropy():.2f}')

    st.markdown('---')
    st.markdown('## ZCR-based')
    # VSTD
    st.markdown(f'Standard deviation of ZCR: {wav.zstd:.2f}')
    # VDR
    st.markdown(f'High zero crossing rate ratio: {wav.hzcrr:.2f}')

