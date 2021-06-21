from PIL import Image
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumtrapz

st.set_page_config(
    page_title="PWM of Buck Converter",
    page_icon="🎮",
    initial_sidebar_state="expanded",
)

st.write(""" # Buck Converter """)
image = Image.open('buck.png')
bckon = Image.open('buck_on.png')
bckoff = Image.open('buck_off.png')
st.image(image,use_column_width = True)

st.sidebar.info('** 1. Select Input Voltage **')
vin = st.sidebar.slider('Vin (V)',10,100,50,step=10)
vout = st.sidebar.slider('Vout (V)',vin*0.1,vin*0.9,vin*0.5,step=2.0)
iout = st.sidebar.slider('Iout (A)',0.5,10.0,5.0,step=0.5)

st.sidebar.info('** 2. Modulation Parameters**')
duty = vout/vin
st.sidebar.write('Modulation Index (M)', str(round(duty,2)))
fc = st.sidebar.slider('Frequency (Hz)',5000,20000,10000,step=1000)

st.sidebar.info('** 3. Select Inductor Value **')
lbk = st.sidebar.slider('Inductance (mH)',0.05,2.0,step=0.05)
votext = str(vout)

st.info('**Parameters for Vout = '+str(vout)+'V **')
col1,col2,col3,col4 = st.beta_columns(4)
vint = 'V_{in} = ' + str(vin) +'\:\\text{V}'
col1.latex(vint)
dutyt = '\\text{M} = '+ str(round(duty,2))
col2.latex(dutyt)
fct = 'f_{sw} = ' + str(fc) + '\:\\text{kHz}'
col3.latex(fct)
lt = 'L_{buck}='+ str(lbk)+'\:\\text{mH}'
col4.latex(lt)

st.info('** PWM Pattern **')
pii = np.pi
sample_rate = 4000
duration = 1
time=np.arange(0,duration,1/sample_rate)
f1 = 1
fsw = fc*f1/1000
tri=signal.sawtooth(2 * np.pi * fsw * time,0.99)
tri_FB = (tri+1)/2
dd = time/time*duty

plt.style.use('ggplot')

plotpwm = plt.figure(figsize=[7,2])
plt.plot(time,tri_FB,'gray')
plt.plot(time,dd,'darkblue')
plt.text(1.02,duty,'M',color='darkblue')




st.pyplot(plotpwm)



pwm = (dd>=tri_FB)*1
plotduty = plt.figure(figsize=[7,2])
plt.plot(time,pwm,'darkblue')
plt.xlabel('Time (ms)')
plt.yticks([0,0.5,1.00])

st.pyplot(plotduty)

col1,col2 = st.beta_columns(2)
Ton = duty/fsw
ton = 'T_{ON} = \:'+str(round(Ton,3))+'\:\\text{ms}'
col1.latex(ton)
Toff = (1-duty)/fsw
toff = 'T_{OFF} = \:'+str(round(Toff,3))+'\: \\text{ms}'
col2.latex(toff)
textduty = '\\text{Duty Cycle} \:D = \dfrac{T_{ON}}{T_{ON}+T_{OFF}} = \:' + str(round(duty,2))
st.latex(textduty)

col1,col2 = st.beta_columns(2)
with col1:
    st.info('** When S is ON **')
    st.image(bckon)
    st.latex('V_{L} = V_{in}-V_{out}')
    st.latex('\Delta I_{L} = \dfrac{V_{L}\\times T_{ON}}{L_{buck}}')

with col2:
    st.info('** When S is OFF **')
    st.image(bckoff)
    st.latex('V_{L} = -V_{out}')
    st.latex('\Delta I_{L} = \dfrac{V_{L}\\times T_{OFF}}{L_{buck}}')

VL = pwm*(vin-vout)+(pwm-1)*vout
vlplot = plt.figure(figsize=[7,2])
plt.plot(time,VL,'darkblue',label='$V_{L}$')
plt.text(0.02,max(VL)+2,'$V_{in}-V_{out}$')
plt.text(0.02,min(VL)-5,'$-V_{out}$')
plt.title('$V_{L}$')
# plt.legend(frameon=False, loc='lower center', ncol=3)
plt.axis([0,1,-vout-20,vin])
# plt.xlabel('Time (ms)')
plt.yticks([0,-vout,(vin-vout)])
plt.fill_between(time,VL,alpha=0.3,color='blue')

delta_il = cumtrapz(VL/lbk,time,initial=0)
ilplot = plt.figure(figsize=[7,2])
iL = delta_il+iout-np.average(delta_il)
plt.plot(time,iL,'darkblue',label='$I_{L}$')
plt.plot(time,time/time*iout,'r')
plt.text(0.9,iout+0.2,'$I_{out}$',color='red')
plt.legend(frameon=False, loc='lower center', ncol=3)
plt.xlabel('Time (ms)')
plt.axis([0,1,0,max(iL)+5])
# plt.yticks([0,-vout,(vin-vout)])



mode =True
if min(iL)<0:
    st.sidebar.error('Discontinous Mode!!!')
    mode = False
#     plt.axis([0,1,0,max(iL)+5])
else:
    st.sidebar.info('Continuous Mode')
    mode = True
#     plt.axis([0,1,0,max])

st.write(vlplot)
st.write(ilplot)


if mode:
    st.info('** In steady state, use averaging approach **')
    col1,col2 = st.beta_columns(2)
    with col1:
        st.write('** Volt-second balance of inductance voltage **')
        st.latex('(V_{in}-V_{out})T_{ON} + (-V_{out}T_{OFF}) = 0')
        st.latex('\dfrac{V_{out}}{V_{in}} = \dfrac{T_{ON}}{T_{ON}+T_{OFF}} = D')

    with col2:
        st.write('** Average value of **','** $I_{L}$ **')
        iltext = 'I_{L,avg} = I_{out}=\:'+str(iout) +'\:A'
        st.latex(iltext)
        iripple = max(iL) - min(iL)
        irippletext = '\Delta I_{L} = \:' + str(round(iripple,2)) +'A'
        st.latex(irippletext) 
        ripplefactor = iripple/iout
        ripfactext = '\\text{Ripple factor} = \dfrac{\Delta I_{L}}{I_{L,avg}}=\:'+str(round(ripplefactor,2))
        st.latex(ripfactext)
else:
    st.error('Revise parameters for continuouse conduction mode')
