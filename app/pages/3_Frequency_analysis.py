import streamlit as st
import numpy as np
import plotly.express as px

from app import store
from app.core.freq_analysis import FrequencyAnalyser

st.markdown('# Frequency analysis')

if store.get_signal() is not None:
    signal = store.get_signal()
    timestamps_frames = np.linspace(0, signal.audio_length_sec, num=signal.n_frames)

    fa = FrequencyAnalyser(signal)

    # Frequency domain
    st.markdown('---')
    st.markdown('## Frequency domain of a clip')
    plot = px.line(
        x=signal.fft_freq_vals_full,
        y=signal.fft_magn_spectr_full,
        title=f'Frequency domain',
        labels={'x': 'Frequency [Hz]', 'y': 'Magnitude'},
    )
    st.plotly_chart(plot, use_container_width=True)

    # other stuff
    st.markdown('---')
    st.markdown('## Frequency-based parameters (frame-level)')

    # Volume
    plot = px.line(
        x=timestamps_frames,
        y=fa.volume(),
        title=f'Volume',
        labels={'x': 'Time (in seconds)', 'y': 'Volume'},
    )
    st.plotly_chart(plot, use_container_width=True)

    # Frequency centroids
    st.markdown('---')
    plot = px.line(
        x=timestamps_frames,
        y=fa.frequency_cetroids(),
        title=f'Frequency centroids',
        labels={'x': 'Time (in seconds)', 'y': 'FC'},
    )
    st.plotly_chart(plot, use_container_width=True)

    # Effective bandwidth
    st.markdown('---')
    plot = px.line(
        x=timestamps_frames,
        y=fa.effective_bandwidth(),
        title=f'Effective bandwidth',
        labels={'x': 'Time (in seconds)', 'y': 'EB'},
    )
    st.plotly_chart(plot, use_container_width=True)

    # ersb
    st.markdown('---')
    ersb1_col, ersb2_col, ersb3_col = st.columns([1, 1, 1])
    with ersb1_col:
        ersb1 = fa.ersb1()
        plot = px.line(
            x=timestamps_frames,
            y=ersb1,
            title=f'Energy Ratio Subband (0-630 Hz)',
            labels={'x': 'Time (in seconds)', 'y': 'ERSB1'},
        )
        st.plotly_chart(plot, use_container_width=True)
    with ersb2_col:
        ersb2 = fa.ersb2()
        plot = px.line(
            x=timestamps_frames,
            y=ersb2,
            title=f'Energy Ratio Subband (630-1720 Hz)',
            labels={'x': 'Time (in seconds)', 'y': 'ERSB2'},
        )
        st.plotly_chart(plot, use_container_width=True)
    with ersb3_col:
        ersb3 = fa.ersb3()
        plot = px.line(
            x=timestamps_frames,
            y=ersb3,
            title=f'Energy Ratio Subband (1720-4400 Hz)',
            labels={'x': 'Time (in seconds)', 'y': 'ERSB3'},
        )
        st.plotly_chart(plot, use_container_width=True)

    # Spectral flatness
    st.markdown('---')
    plot = px.line(
        x=timestamps_frames,
        y=fa.spectral_flatness(),
        title=f'Spectral flatness measure',
        labels={'x': 'Time (in seconds)', 'y': 'SFM'},
    )
    st.plotly_chart(plot, use_container_width=True)

    # Spectral crest factor
    st.markdown('---')
    plot = px.line(
        x=timestamps_frames,
        y=fa.spectral_crest_factor(),
        title=f'Spectral crest factor',
        labels={'x': 'Time (in seconds)', 'y': 'SCF'},
    )
    st.plotly_chart(plot, use_container_width=True)
