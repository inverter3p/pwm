import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from scipy.fft import fft, fftfreq, rfft, rfftfreq, irfft, dst
from scipy.signal import blackman

def thd(abs_data):
    sq_sum = np.sum(np.square(abs_data))
    sq_harmonics = sq_sum - (max(abs_data))**2
    return 100 * np.sqrt(sq_harmonics) / max(abs_data)

# Sidebar section
st.sidebar.info('** 1. Set modulation parameters**')
x = st.sidebar.slider('Modulation Index', 0.0, 3.5, 1.0)
fcarrier = st.sidebar.slider('Frequency Modulation Index', 10, 50, 20)

# Main content
st.title("Sinusoidal PWM (SPWM) Inverter")

pwm = st.radio("Select PWM scheme", ('1P Bipolar SPWM', '1P Unipolar SPWM', '3P SPWM', '3P HI-SPWM'))

# Display relevant images based on selection
image_mapping = {
    '3P SPWM': 'inverter.png',
    '1P Unipolar SPWM': 'inv1p.png',
    '1P Bipolar SPWM': 'inv1p.png',
    '3P HI-SPWM': 'inv3pwith3rd.png'
}
image = Image.open(image_mapping[pwm])
st.image(image)

# Modulation parameters
st.info('** Modulation parameters**')
col1, col3 = st.columns([1, 1])
with
