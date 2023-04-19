import streamlit as st

from app import store


def file_uploader():
    wavfile_raw = st.file_uploader(
        'Import a .wav file to get started:',
        type=['wav'],
    )
    if wavfile_raw is not None:
        store.update_wavfile(wavfile_raw)
