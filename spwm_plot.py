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
with col1:
    st.write(f"$$M_a = {x}$$")
    st.write(f"$$F_c = {fcarrier}$$")
    st.sidebar.info('** 2. Set DC link voltage (VDC)**')
    Vdc = st.sidebar.slider('Vdc', 100, 600, 100)
    st.write(f"$$V_{{DC}} = {Vdc} \: V$$")

# Plotting modulation signals
plt.style.use('ggplot')
sample_rate = 4000
duration = 1
time = np.arange(0, duration, 1 / sample_rate)
pii = np.pi
ma = x

fc = fcarrier * 1  # Carrier frequency
tri = signal.sawtooth(2 * pii * fc * time, 0.5)
tri_FB = tri * 1

# Generate sine waves for different phases
sine_A = ma * np.sin(2 * pii * time)
sine_B = ma * np.sin(2 * pii * time + 2 * pii / 3)
sine_C = ma * np.sin(2 * pii * time - 2 * pii / 3)

# Generate plot based on selected PWM scheme
plotpwm = plt.figure(figsize=[7, 3])
plt.plot(time, tri_FB, 'gray')
plt.plot(time, sine_A, 'red')
plt.axis([0, 1, -1.5, 1.5])
plt.xticks([])
plt.yticks([-0.5, 0, -1, 0.5, 1])
plt.text(0.25, 1.05, 'a', color='r')

if pwm == '1P Unipolar SPWM':
    plt.plot(time, -sine_A, 'blue')
    plt.text(0.92, 1.05, 'b', color='b')
if pwm == '3P SPWM':
    plt.plot(time, sine_B, 'blue')
    plt.plot(time, sine_C, 'black')
    plt.text(0.92, 1.05, 'b', color='b')
    plt.text(0.57, 1.05, 'c', color='k')

# Display the plot using the latest Streamlit function
st.pyplot(plotpwm)

# Phase-leg voltage calculation and plotting
pwm_A = (sine_A >= tri_FB) * 1
VA = pwm_A * Vdc
Vsin_A = (sine_A + 1) * 0.5 * Vdc
phase_A = plt.figure(figsize=[7, 3])
plt.plot(time, VA, 'gray')
plt.plot(time, Vsin_A, 'red')
plt.text(0.3, max(Vsin_A) + 2, f'{max(Vsin_A):.2f} V')
plt.axis([0, 1, -0.25 * Vdc, 1.25 * Vdc])
plt.xticks([])
plt.yticks([0, 0.5 * Vdc, Vdc])

st.pyplot(phase_A)
