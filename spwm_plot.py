import streamlit as st
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from scipy.fft import fft, fftfreq, rfft,rfftfreq, irfft, dst
from scipy.signal import blackman

def thd(abs_data):
    sq_sum=0.0
    for r in range( len(abs_data)):
        sq_sum = sq_sum + (abs_data[r])**2

    sq_harmonics = sq_sum -(max(abs_data))**2
    thd = 100*sq_harmonics**0.5 / max(abs_data)
    return thd

st.sidebar.info('** 1. Set modulation parameters**')
x = st.sidebar.slider('Modulation Index',0.0,3.5,1.0)
fcarrier = st.sidebar.slider('Frequency Modulation Index',10,50,15)

pii = np.pi
sample_rate = 4000
duration = 1
time=np.arange(0,duration,1/sample_rate)
f1 = 1
fc= fcarrier*f1
tri=signal.sawtooth(2 * np.pi * fc * time,0.5)
# tri_FB = (tri+1)/2
tri_FB = tri*1


st.write(""" # Sinusoidal PWM (SPWM) Inverter""")

pwm = st.radio("Select PWM scheme",('1P Bipolar SPWM','1P Unipolar SPWM','3P SPWM','3P HI-SPWM'))
if pwm == '3P SPWM':
    image = Image.open('inverter.png')
    st.image(image)
elif pwm == '1P Unipolar SPWM':
    image = Image.open('inv1p.png')
    st.image(image)
elif pwm == '1P Bipolar SPWM':
    image = Image.open('inv1p.png')
    st.image(image)
else:
    image = Image.open('inv3pwith3rd.png')
    st.image(image)

st.info('** Modulation parameters**')
col1,col3 = st.columns([1,1])
with col1:
    st.write("$$M_a =$$  ",str(x))
    st.write("$$F_c =$$  ",str(fcarrier))
    st.sidebar.info('** 2. Set DC link voltage (VDC)**')
    Vdc = st.sidebar.slider('Vdc',100,600,100)
    sVdc = "$$V_{DC} = \:$$" + str(Vdc) +'$$\: V$$'
    st.write(sVdc)

with col3:
    if pwm == '3P HI-SPWM':
        V1 = st.sidebar.slider('Amplitude of Fundamental Harmonic',0.0,1.25,1.15)
        st.write("$$h_1 =$$  ",str(V1))
        V3 = st.sidebar.slider('Amplitude of 3rd Harmonic',0.0,0.25,0.1916)
        st.write("$$h_3 =$$  ",str(V3)) 
        st.write('Default: ','$h_{3} = \dfrac{h_{1}}{6}$')

ma = x
if pwm == '3P HI-SPWM':
    first = V1
    third = V3
else:
    first = 1
    third = 0
sine_A = (first*ma*np.sin(2*pii*time))  
sine_B = (first*ma*np.sin(2*pii*time+2*pii/3))
sine_C = (first*ma*np.sin(2*pii*time-2*pii/3))
sine_3A = (third*ma*np.sin(2*pii*3*time))



plt.style.use('ggplot')

plotpwm = plt.figure(figsize=[7,3])
plt.plot(time,tri_FB,'gray')
plt.plot(time,sine_A,'red')
plt.axis([0,1,-1.5,1.5])
plt.xticks(ticks=[])
plt.yticks([-0.5,0,-1,0.5,1])
plt.text(0.25,1.05,'a',color='r')
if pwm == '1P Unipolar SPWM':
    plt.plot(time,-sine_A,'blue')
    plt.text(0.92,1.05,'b',color='b')
if pwm == '3P SPWM':
    plt.plot(time,sine_B,'blue')
    plt.plot(time,sine_C,'black')
    plt.text(0.92,1.05,'b',color='b')
    plt.text(0.57,1.05,'c',color='k')
if pwm == '3P HI-SPWM':
    plt.plot(time,sine_B,'blue')
    plt.plot(time,sine_C,'black')
    plt.plot(time,sine_3A,'green')
    plt.text(0.25,0,'3rd',color='green')
    plt.text(0.92,1.05,'b',color='b')
    plt.text(0.57,1.05,'c',color='k')



if pwm == '3P HI-SPWM':
    plotpwm3rd = plt.figure(figsize=[7,3])
    plt.plot(time,tri_FB,'gray')
    plt.plot(time,sine_A+sine_3A,'red')
    plt.plot(time,sine_B+sine_3A,'blue')
    plt.plot(time,sine_C+sine_3A,'black')
    plt.axis([0,1,-1.5,1.5])
    plt.xticks(ticks=[])
    plt.yticks([0,round(min(sine_A+sine_3A),2),round(max(sine_A+sine_3A),2)])
    plt.text(0.22,1.1,'a+3rd',color='r')
    plt.text(0.89,1.1,'b+3rd',color='b')
    plt.text(0.54,1.1,'c+3rd',color='k')
    # plt.title('Modulation Signal')

col1,col2 = st.columns([6,2])

if pwm != '3P HI-SPWM':
    with col1:
        st.write(plotpwm)
        
    with col2:
        st.write('**Modulation Signals**')
        st.latex('a = M_{a}\sin(\omega t)')
        if pwm == '1P Unipolar SPWM':
            st.latex('b = -M_{a}\sin(\omega t)')
        if pwm  == '3P SPWM':
            st.latex('b = M_{a}\sin(\omega t + \dfrac{2\pi}{3})')
            st.latex('c = M_{a}\sin(\omega t - \dfrac{2\pi}{3})')

elif pwm == '3P HI-SPWM':    
    with col1:
        st.write(plotpwm)
        st.write(plotpwm3rd)
    with col2:
        st.write('**Modulation Signals**')
        st.latex('a = M_{a}h_{1} \sin(\omega t)')
        st.latex('b = M_{a}h_{1}\sin(\omega t + \dfrac{2\pi}{3})')
        st.latex('c = M_{a}h_{1}\sin(\omega t - \dfrac{2\pi}{3})')
        st.latex('3rd = M_{a}h_{3} \sin(3\omega t)')





# st.info('**PWM Gate Signals**')
col4,col5 = st.columns([6,2])

pwm_A = (sine_A+sine_3A>=tri_FB)*1

if pwm == '1P Bipolar SPWM':
    pwm_B = 1-(sine_A >=tri_FB)*1 
    plotgate_A = plt.figure(figsize=[7,2])
    plt.plot(time,pwm_A,'red')
    plt.title('Leg A PWM gate signal')
    plt.axis([0,1,-0.1,1.1])
    plt.yticks([0,0.5,1.0])
    col4.write(plotgate_A)
    
    plotgate_B = plt.figure(figsize=[7,2])
    plt.plot(time,pwm_B,'blue')
    plt.title('Leg B PWM gate signal')
    plt.axis([0,1,-0.1,1.1])
    plt.yticks([0,0.5,1.0])
    col4.write(plotgate_B)
    
elif pwm == '1P Unipolar SPWM':
    pwm_B = (-sine_A >=tri_FB)*1 
    plotgate_A = plt.figure(figsize=[7,2])
    plt.plot(time,pwm_A,'red')
    plt.title('Leg A PWM gate signal')
    plt.axis([0,1,-0.1,1.1])
    plt.yticks([0,0.5,1.0])
    col4.write(plotgate_A)
    
    plotgate_B = plt.figure(figsize=[7,2])
    plt.plot(time,pwm_B,'blue')
    plt.title('Leg B PWM gate signal')
    plt.axis([0,1,-0.1,1.1])
    plt.yticks([0,0.5,1.0])
    col4.write(plotgate_B)


else:
    pwm_B = (sine_B+sine_3A>=tri_FB)*1
    pwm_C = (sine_C+sine_3A>=tri_FB)*1
    plotgate_A = plt.figure(figsize=[7,2])
    plt.plot(time,pwm_A,'red')
    plt.title('Leg A PWM gate signal')
    plt.axis([0,1,-0.1,1.1])
    plt.yticks([0,0.5,1.0])
    col4.write(plotgate_A)
    
    plotgate_B = plt.figure(figsize=[7,2])
    plt.plot(time,pwm_B,'blue')
    plt.title('Leg B PWM gate signal')
    plt.axis([0,1,-0.1,1.1])
    plt.yticks([0,0.5,1.0])
    col4.write(plotgate_B)

    plotgate_C = plt.figure(figsize=[7,2])
    plt.plot(time,pwm_C,'black')
    plt.title('Leg C PWM gate signal')
    plt.axis([0,1,-0.1,1.1])
    plt.yticks([0,0.5,1.0])
    col4.write(plotgate_C)

    


if ma >1 :
    st.error('Over modulation is yet to support  !!!')
else:
    st.info('**Phase-leg Voltage**')
    col4,col5 = st.columns([6,2])

    VA = pwm_A*Vdc
    Vsin_A = (sine_A+sine_3A+1)*0.5*Vdc
    phase_A = plt.figure(figsize=[7,3])
    plt.plot(time,VA,'gray')
    plt.plot(time,Vsin_A,'red')
    plt.text(0.3,max(Vsin_A)+2,'{0:.2f}'.format(max(Vsin_A))+" V")
    # plt.plot(time,(sine_3A+1)*0.5)
    plt.axis([0,1,-0.25*Vdc,1.25*Vdc])
    plt.xticks([])
    # plt.yticks([0,0.5*Vdc,1*Vdc],['0','Vdc/2','Vdc'])
    plt.yticks([0,0.5*Vdc,Vdc])
    col4.write(phase_A)
    if pwm != '3P HI-SPWM':
        col5.latex(''' V_{A0} = \dfrac{V_{DC}}{2}+ \dfrac{V_{DC}}{2}[{M_{a}\sin(\omega t)}]''')
    elif pwm == '3P Hi-SPWM':
        col5.latex(''' V_{A0} = \dfrac{V_{DC}}{2}+ \dfrac{V_{DC}}{2}[{M_{a}h_{1}\sin(\omega t)+h_{3}\sin(3\omega t)}]''')

    # ao = "V_{A0,pk}="+'{0:.2f}'.format(max(Vsin_A))+"\: V"
    # col5.latex(ao)

    col4,col5 = st.columns([6,2])
    # pwm_B = (sine_B+sine_3A>=tri_FB)*1
    VB = pwm_B*Vdc
    if pwm == '1P Unipolar SPWM' or pwm == '1P Bipolar SPWM':
        Vsin_B = (-sine_A+sine_3A+1)*0.5*Vdc
    else:
        Vsin_B = (sine_B+sine_3A+1)*0.5*Vdc

    phase_B= plt.figure(figsize=[7,3])
    plt.plot(time,VB,'gray')
    plt.plot(time,Vsin_B,'blue')
    plt.text(0.75,max(Vsin_B)+2,'{0:.2f}'.format(max(Vsin_B))+" V")
    # plt.plot(time,(sine_3A+1)*0.5)
    plt.axis([0,1,-0.25*Vdc,1.25*Vdc])
    plt.xticks([])
    plt.yticks([0,0.5*Vdc,1*Vdc])
    col4.write(phase_B)
    if pwm == '1P Unipolar SPWM' or pwm == '1P Bipolar SPWM':
        col5.latex(''' V_{B0} = \dfrac{V_{DC}}{2}- \dfrac{V_{DC}}{2}[{M_{a}\sin(\omega t)}]''')

    elif pwm == '3P SPWM':
        col5.latex(''' V_{B0} = \dfrac{V_{DC}}{2}+ \dfrac{V_{DC}}{2}[{M_{a}\sin(\omega t+\dfrac{2\pi}{3})}]''')
    elif pwm == '3P HI-SPWM':
        col5.latex(''' V_{B0} = \dfrac{V_{DC}}{2}+ \dfrac{V_{DC}}{2}[{M_{a}h_{1}\sin(\omega t+\dfrac{2\pi}{3})+h_{3}\sin(3\omega t)}]''')



    if pwm == '3P SPWM' or pwm == '3P HI-SPWM':
        col4,col5 = st.columns([6,2])
        # pwm_C = (sine_C+sine_3A>=tri_FB)*1
        VC = pwm_C*Vdc
        Vsin_C = (sine_C+sine_3A+1)*0.5*Vdc
        phase_C = plt.figure(figsize=[7,3])
        plt.plot(time,VC,'gray')
        plt.plot(time,Vsin_C,'black')
        plt.text(0.5,max(Vsin_C)+2,'{0:.2f}'.format(max(Vsin_C))+" V")
        # plt.plot(time,(sine_3A+1)*0.5)
        plt.axis([0,1,-0.25*Vdc,1.25*Vdc])
        plt.xticks([])
        plt.yticks([0,0.5*Vdc,1*Vdc])
        col4.write(phase_C)
        if pwm =='3P SPWM':
            col5.latex(''' V_{C0} = \dfrac{V_{DC}}{2}+ \dfrac{V_{DC}}{2}[{M_{a}\sin(\omega t - \dfrac{2\pi}{3})}]''')
        else:
            col5.latex(''' V_{C0} = \dfrac{V_{DC}}{2}+ \dfrac{V_{DC}}{2}[{M_{a}h_{1}\sin(\omega t - \dfrac{2\pi}{3})+h_{3}\sin(3\omega t)}]''')
        # co = "V_{C0,pk}="+'{0:.2f}'.format(max(Vsin_C))+"\: V"
        # col5.latex(co)

    st.write('**Line-Line Voltage**')

    VAB = VA-VB
    if pwm == '3P SPWM' or pwm == '3P HI-SPWM':
        VBC = VB - VC
        VCA = VC -VA


    col5,col6 = st.columns([6,2])
    with col5:
        plotVab = plt.figure(figsize=[7,3])
        plt.plot(time,VAB,'gray')
        plt.plot(time,Vsin_A-Vsin_B,'red')
        plt.axis([0,1,-1.25*Vdc,1.25*Vdc])
        plt.xticks([])
        plt.yticks([0,Vdc,-Vdc])
        st.write(plotVab)
    with col6:
        st.latex(''' V_{AB}= V_{A0}-V_{B0}''')
        if pwm =='1P Bipolar SPWM' or pwm == '1P Unipolar SPWM':
            st.latex(''' V_{AB,1}= M_a V_{dc}\sin(\omega t)''')
        elif pwm == '3P SPWM':
            st.latex(''' V_{AB,1}= M_a V_{dc}\dfrac{\sqrt{3}}{2}\sin(\omega t-\dfrac{\pi}{6})''')
        else:
            st.latex(''' V_{AB,1}= M_a V_{dc}h_{1}\dfrac{\sqrt{3}}{2}\sin(\omega t-\dfrac{\pi}{6})''')
        ab = "V_{AB,1}="+'{0:.2f}'.format(max(Vsin_A-Vsin_B))+"\: V"
        ab_rms = "V_{AB,1,rms}="+'{0:.2f}'.format(max(Vsin_A-Vsin_B)/np.sqrt(2))+"\: V"
        st.latex(ab)
        st.latex(ab_rms)

    if pwm == '3P SPWM' or pwm == '3P HI-SPWM':
        VBC = VB - VC
        VCA = VC -VA
        col5,col6 = st.columns([6,2])
        with col5:
            plotVbc = plt.figure(figsize=[7,3])
            plt.plot(time,VBC,'gray')
            plt.plot(time,Vsin_B-Vsin_C,'blue')
            plt.axis([0,1,-1.25*Vdc,1.25*Vdc])
            plt.xticks([])
            plt.yticks([0,Vdc,-Vdc])
            st.write(plotVbc)
        with col6:
            st.latex(''' V_{BC} = V_{B0}-V_{C0}''')
            if pwm =='SPWM':
                st.latex(''' V_{BC,1}= M_a V_{dc}\dfrac{\sqrt{3}}{2}\sin(\omega t-\dfrac{\pi}{6} + \dfrac{2\pi}{3} )''')
            else:
                st.latex(''' V_{BC,1}= M_a V_{dc}h_{1}\dfrac{\sqrt{3}}{2}\sin(\omega t-\dfrac{\pi}{6} + \dfrac{2\pi}{3} )''')
            
            bc = "V_{BC,1}="+'{0:.2f}'.format(max(Vsin_B-Vsin_C))+"\: V"
            bc_rms = "V_{AB,1,rms}="+'{0:.2f}'.format(max(Vsin_B-Vsin_C)/np.sqrt(2))+"\: V"
            st.latex(bc)
            st.latex(bc_rms)

        col5,col6 = st.columns([6,2])
        with col5:
            plotVca = plt.figure(figsize=[7,3])
            plt.plot(time,VCA,'gray')
            plt.plot(time,Vsin_C-Vsin_A,'black')
            plt.axis([0,1,-1.25*Vdc,1.25*Vdc])
            plt.xticks([])
            plt.yticks([0,Vdc,-Vdc])
            st.write(plotVca)
        with col6:
            st.latex(''' V_{CA} = V_{C0}-V_{A0}''')
            if pwm == '3P SPWM':
                st.latex(''' V_{CA,1}= M_a V_{dc}\dfrac{\sqrt{3}}{2}\sin(\omega t-\dfrac{\pi}{6}-\dfrac{2\pi}{3})''')
            else: 
                st.latex(''' V_{CA,1}= M_a V_{dc}h_{1}\dfrac{\sqrt{3}}{2}\sin(\omega t-\dfrac{\pi}{6}-\dfrac{2\pi}{3})''')

            ca = "V_{CA,1}="+'{0:.2f}'.format(max(Vsin_C-Vsin_A))+"\: V"
            ca_rms = "V_{AB,1,rms}="+'{0:.2f}'.format(max(Vsin_C-Vsin_A)/np.sqrt(2))+"\: V"
            st.latex(ca)
            st.latex(ca_rms)

    

    # Number of samples in normalized_tone
    N = sample_rate*duration

    # yf_abs = np.abs(rfft(VABFB)/(N/2))
    yf_abs = np.abs(rfft(VAB)/(N/2))
    xf = rfftfreq(N,1/sample_rate)

    # w = blackman(N)
    # ywf = rfft(VAB_UniFB*w)/(N/2)


    st.sidebar.info('** FFT Spectrum **')
    fft_cal = st.sidebar.button('FFT Calculation')

    if fft_cal:
        st.info('** Fundamental Voltage from FFT Spectrum **')
        col1,col2 = st.columns([6,2])
        plotfft = plt.figure(figsize=[7,3])
        plt.bar(xf, yf_abs,width=1)
        plt.axis([0,200,-0.2,1.25*Vdc])
        plt.xticks(np.arange(0, 200, step=10))
        plt.xlabel('n-th Harmonic')
        plt.title('FFT Plot')
        col1.write(plotfft)

        yf_abs100 = yf_abs[:100]
        v1 = 'V_{L-L,1} = ' + str(format((yf_abs[1]),'.2f'))+' \:V'
        vrms = 'V_{L-L,1,rms}= \dfrac{V_{L-L,pk}}{\sqrt{2}} = ' + str(format((yf_abs[1])/np.sqrt(2),'.2f'))+'\:V'
        thdv = ('THD_V = '+ str(format(thd(yf_abs100),'.2f'))+'\: \%')
        if pwm == '3P SPWM':
            v1 = 'V_{L-L,pk} = M_{a}V_{DC}\dfrac{\sqrt{3}}{2} =\:' + str(format(max(yf_abs),'.2f'))+' \:V'
            # col2.latex('''V_{L-L,pk} = M_{a}V_{DC}\dfrac{\sqrt{3}}{2}''')
        elif pwm == '3P SPWM':
            v1 = 'V_{L-L,pk} = M_{a}h_{1}V_{DC}\dfrac{\sqrt{3}}{2} =\:' + str(format(max(yf_abs),'.2f'))+' \:V'
        col2.latex(v1)
        # col2.latex('V_{L-L,rms}= \dfrac{V_{L-L,pk}}{\sqrt{2}}')
        col2.latex(vrms)
        col2.latex(thdv)
        col2.write('(Calculate upto 100th harmonic)')
        st.sidebar.write('## Done')
