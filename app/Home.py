import streamlit as st

from app import store
from app.components.file_uploader import file_uploader


st.markdown('# Sound analysis')
file_uploader()

if store.get_wavfile() is not None:
    wavfile = store.get_wavfile()

    # audio player
    st.audio(wavfile.samples, sample_rate=wavfile.sample_rate)
