import streamlit as st
import numpy as np
import plotly.express as px

from app import store
from app.core.cepstrum import Cepstrum

st.markdown('# Cepstrum')

if store.get_signal() is not None:
    signal = store.get_signal()
    cepstrum = Cepstrum(signal)

    timestamps_frames = np.linspace(0, signal.audio_length_sec, num=signal.n_frames)

    st.markdown('---')
    plot = px.line(
        x=timestamps_frames,
        y=cepstrum.f0(),
        title=f'Fundamental frequency (using cepstrum)',
        labels={'x': 'Time (in seconds)', 'y': 'F0'},
    )
    st.plotly_chart(plot, use_container_width=True)
