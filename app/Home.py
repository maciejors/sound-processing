import streamlit as st


st.markdown('# Sound analysis')

wav_file = st.file_uploader(
    'Import a .wav file to get started:',
    type=['wav'],
)
if wav_file is not None:
    # TODO: load the file and display charts
    pass
