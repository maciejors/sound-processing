import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    classes = wav.get_frame_types(zcr_threshold, volume_threshold)
    classes = np.repeat(classes, wav.n_samples_per_frame)
    class_to_color = {0: 'red', 1: 'yellow', 2: 'green'}
    timestamps_samples = np.linspace(0, wav.all_audio_length_sec, num=wav.n_samples_all)
    # create a list of trace objects, one for each class
    traces = []
    for cls, color in class_to_color.items():
        # filter the x and y values for this class
        x_cls = timestamps_samples[classes == cls]
        y_cls = wav.samples_all[classes == cls]
        
        # create a scatter trace with the filtered x and y values
        trace = go.Scatter(
            x=x_cls,
            y=y_cls,
            mode='markers',
            marker=dict(color=color),
            name= "Silence" if cls == 0 else "Unvoiced" if cls == 1 else "Voiced"
        )
        
        # add the trace to the list
        traces.append(trace)

    # create a layout object and a figure object
    layout = go.Layout(title='Scatter Plot with Colored Classes')
    fig = go.Figure(data=traces, layout=layout)

    # show the figure
    fig.show()

    # F0
    st.markdown('---')
    plot = px.line(
        x=timestamps_frames,
        y=wav.fundamental_frequency,
        title=f'Fundamental frequency',
        labels={'x': 'Time (in seconds)', 'y': 'F0'},
    )
    st.plotly_chart(plot, use_container_width=True)

