import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
# from google.colab import files
from matplotlib import gridspec

A = [1, 0]  # A-axis
B = [-0.5, 0.866] #B-axis
C = [-0.5, -0.866] # C-axis

def checkii(xpos,ypos):
  if i==0:
    a = plt.text(xpos, ypos, '$\omega t = 0$')
  elif i==1:
    a = plt.text(xpos, ypos, '$\omega t = \pi /6$')
  elif i==6 :
    a = plt.text(xpos, ypos, '$\omega t = \pi$')
  elif i==12 :
    a = plt.text(xpos, ypos, '$\omega t = 2\pi$')
  else:
    a = plt.text(xpos, ypos, '$\omega t =$ '+str(i)+'$\pi /6$')
  return a

def plotvec(org,vx,vy,pos,nol):
  ax1 = fig.add_subplot(pos)
  org_x =  org[0]
  org_y =  org[1]
  if np.size(vx)<4:
    
    if np.size(vx)>2 and nol !='Y' :
      Q = plt.quiver(org_x, org_y, vx, vy, angles='xy', scale_units='xy', color=['r','b','black'], scale=1)
      plt.text(vx[0],vy[0],'A')
      plt.text(vx[1],vy[1],'B')
      plt.text(vx[2],vy[2],'C')
    elif np.size(vx)>2 and nol =='Y' :
      Q = plt.quiver(org_x, org_y, vx, vy, angles='xy', scale_units='xy', color=['r','b','g'], scale=1)
      plt.text(vx[0],vy[0],'$\\alpha $')
      plt.text(vx[2],vy[2],'$\\beta$')  
    else:
      Q = plt.quiver(org_x, org_y, vx, vy, angles='xy', scale_units='xy', color=['r','b'], scale=1)
      plt.text(vx[0],vy[0],'$\\alpha $')
      plt.text(vx[1],vy[1],'$\\beta$')
  else :
    Q = plt.quiver(org_x, org_y, vx, vy, angles='xy', scale_units='xy', color=['r','b','black','g'], scale=1)
    plt.text(vx[0],vy[0],'A')
    plt.text(org_x[2],org_y[2],'B')
    plt.text(vx[3],vy[3],'C')

  plt.xlim(-1.5*amp, 1.5*amp)
  plt.ylim(-1.5*amp, 1.5*amp)
  ax1.set_aspect('equal', adjustable='box')  
  
  checkii(-1.43,1.3)
  return ax



def ABC2ab(abc,scaling):
  alpha = scaling*(abc[0]*1-abc[1]*0.5-abc[2]*0.5)
  beta = scaling*(abc[0]*0+abc[1]*0.866-abc[2]*0.866)
  return [alpha,beta]

##########################๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒๒#######################
#############                                  ####################################
###################################################################################
st.set_page_config(
    page_title="Space Vector Transformation",
    page_icon="😵",
    # layout="wide",
    initial_sidebar_state="expanded",
)

i = st.sidebar.slider('Step',0,12,step=1) 
st.sidebar.info('** Set amplitude**')
 

ampA = st.sidebar.slider('Phase A Amplitude',1,10,1,step=1)
ampB = st.sidebar.slider('Phase B Amplitude',1,10,1,step=1)
ampC = st.sidebar.slider('Phase C Amplitude',1,10,1,step=1)

amp = max(ampA,ampB,ampC)
step = 13
pii = np.pi
wt = np.linspace(0,2*pii,step)
Vec_A = ampA*np.sin(wt)
Vec_B = ampB*np.sin(wt + 2*pii/3)
Vec_C = ampC*np.sin(wt -2*pii/3)

a = 'A(t) = '+ str(ampA)+'\sin(\omega t)'
b = 'B(t) = '+ str(ampB)+'\sin(\omega t + 2\pi/3)'
c = 'C(t) = '+ str(ampC)+'\sin(\omega t - 2\pi/3)'
st.sidebar.latex(a)
st.sidebar.latex(b)
st.sidebar.latex(c)
#########################################
#   Plot sine wave ABC
#########################################


wt2 = np.linspace(0,2*pii,101)
sinA = ampA*np.sin(wt2)
sinB = ampB*np.sin(wt2 + 2*pii/3)
sinC = ampC*np.sin(wt2 -2*pii/3)

plt.style.use('ggplot')

abc_plot = plt.figure(figsize=[10,4])
ax = abc_plot.add_subplot(111)
plt.plot(wt2,sinA,'r',label='A')
plt.plot(wt2,sinB,'b',label='B')
plt.plot(wt2,sinC,'black',label='C')
plt.xlabel('$\omega t$')
plt.fill_between(wt2,sinA,alpha=0.1,color='red')
plt.fill_between(wt2,sinB,alpha=0.1,color='blue')
plt.fill_between(wt2,sinC,alpha=0.1,color='black')
plt.xlim(0, 2*pii)
plt.ylim(-amp*1.5, amp*1.5)
plt.title('Three-Phase Waveforms')
plt.legend(frameon=False, loc='lower center', ncol=3)\


grid_x_ticks_major = np.arange(0, 2*pii+pii/3, pii/3 )
ax.set_xticks(grid_x_ticks_major)
ax.set_xticklabels(['0', '$\pi /3$', '$2\pi /3$', '$\pi$', '$4\pi /3$','$5\pi /3$','$2\pi$'])
# ax.grid(which='major', linestyle='--')

arrow_loc = i*2*pii/(step-1)

plt.arrow(arrow_loc,0,0,np.sin(arrow_loc)*ampA,color='red',width=0.04,length_includes_head=True)
plt.plot(arrow_loc,np.sin(arrow_loc)*ampA,'r+',markersize= 15)
plt.arrow(arrow_loc,0,0,np.sin(arrow_loc+2*pii/3)*ampB,color='blue',width=0.04,length_includes_head=True)
plt.plot(arrow_loc,np.sin(arrow_loc+2*pii/3)*ampB,'b+',markersize= 15)
plt.arrow(arrow_loc,0,0,np.sin(arrow_loc-2*pii/3)*ampC,color='black',width=0.04,length_includes_head=True)
plt.plot(arrow_loc,np.sin(arrow_loc-2*pii/3)*ampC,'k+',markersize= 15)


plt.axvline(arrow_loc, -1.5, 1.5,color='grey')
checkii(i*2*pii/(step-1)+0.1,1.3)


#########################################
#   Plot Vector ABC
#########################################



#########  Plot ABC Vectors  #############
abc = [Vec_A[i],Vec_B[i],Vec_C[i]]
Vec_xabc = [A[0]*abc[0],B[0]*abc[1],C[0]*abc[2]]
Vec_yabc = [A[1]*abc[0],B[1]*abc[1],C[1]*abc[2]]
Org = [[0]*np.size(Vec_xabc),[0]*np.size(Vec_yabc)]
fig = plt.figure(figsize=[10,5])
# ax1 = fig.add_subplot(121)
plotvec(Org,Vec_xabc,Vec_yabc,121,'N')
plt.title('Space Vector ABC')

#########  Plot Sum ABC Vectors  #############
Vec_sumxabc = Vec_xabc+[Vec_xabc[0]+Vec_xabc[1]+Vec_xabc[2]]
Vec_sumyabc = Vec_yabc+[Vec_yabc[0]+Vec_yabc[1]+Vec_yabc[2]]
Org_x = [0, Vec_xabc[0], Vec_xabc[0]+Vec_xabc[1],0]
Org_y = [0, Vec_yabc[0], Vec_yabc[0]+Vec_yabc[1],0]
Org = [Org_x,Org_y]
plotvec(Org,Vec_sumxabc,Vec_sumyabc,122,'N')
plt.title('Sum of Space Vector ABC')

#############################################3


st.info('## ABC Reference Frame')
st.write(abc_plot)
col1,col2,col3 = st.beta_columns(3)
vec_a = '\\vec{A} = '+ str(round(np.sin(arrow_loc)*ampA,3))+'\\angle{0^\circ}'
vec_b = '\\vec{B} = '+ str(round(np.sin(arrow_loc+2*pii/3)*ampB,3))+'\\angle{120^\circ}'
vec_c = '\\vec{C} = '+ str(round(np.sin(arrow_loc-2*pii/3)*ampC,3))+'\\angle{240^\circ}'
col1.latex(vec_a)
col2.latex(vec_b)
col3.latex(vec_c)

with st.beta_expander('See ABC  Space Vector'):
    st.write(fig)
# fname="Vector_abc_step"+str(i)+".png"
# plt.savefig(fname)
# files.download(fname) 

#################################################
#########  Plot alpha beta Vectors  #############
#################################################
st.info('## Alpha Beta Reference Frame')
st.latex('\\begin{bmatrix} F_{\\alpha} \\\ F_{\\beta} \\\ F_{0} \end{bmatrix} = \
  C\\begin{bmatrix} 1 & -\\frac{1}{2} & -\\frac{1}{2} \\\ 0 & \\frac{\sqrt{3}}{2} & -\\frac{\sqrt{3}}{2} \\\ \
   \\frac{1}{2} & \\frac{1}{2} & \\frac{1}{2}    \end{bmatrix} \\begin{bmatrix} F_{A} \\\ F_{B} \\\ F_{C} \end{bmatrix}  ')

scaling=1.0
sf = st.radio('Scaling Factor (C)', ('1','2/3','sqrt{2/3}'))
if sf =='1':
  scaling = 1.0
elif sf == '2/3':
  scaling = 2/3
else:
  scaling = np.sqrt(2/3)

sin_ab= ABC2ab([sinA,sinB,sinC],scaling)
ab = ABC2ab(abc,scaling)  ### ABC to alpha beta Tranformation 
ab_plot = plt.figure(figsize=[10,4])
ax = ab_plot.add_subplot(111)


plt.plot(wt2,sin_ab[0],'r',label='$\\alpha$')
plt.plot(wt2,sin_ab[1],'b',label='$\\beta$')
plt.fill_between(wt2,sin_ab[0],alpha=0.1,color='red')
plt.fill_between(wt2,sin_ab[1],alpha=0.1,color='blue')
plt.xlim(0, 2*pii)
plt.ylim(-amp*1.8, amp*1.8)
plt.title('Equivalent $\\alpha \\beta$-Reference Frame Waveforms')
plt.xlabel('$\omega t$')
plt.legend(frameon=False, loc='lower center', ncol=3)
grid_x_ticks_major = np.arange(0, 2*pii+pii/3, pii/3 )
ax.set_xticks(grid_x_ticks_major)
ax.set_xticklabels(['0', '$\pi /3$', '$2\pi /3$', '$\pi$', '$4\pi /3$','$5\pi /3$','$2\pi$'])
# ax.grid(which='major', linestyle='--')
plt.plot(arrow_loc,ab[0],'r+',markersize= 15)
plt.plot(arrow_loc,ab[1],'b+',markersize= 15)


plt.axvline(i*2*pii/(step-1), -15, 15,color='grey')
checkii(i*2*pii/(step-1)+0.1,1.3)

######################################################
fig = plt.figure(figsize=[10,5])
ab = ABC2ab(abc,scaling)  ### ABC to alpha beta Tranformation 
Vec_xab = [ab[0],0]
Vec_yab = [0,ab[1]]
plotvec([[0,0],[0,0]],Vec_xab,Vec_yab,121,'N')
plt.title('Space Vector $\\alpha \\beta$')

#########  Plot sum alpha beta Vectors  #############
Vec_sumxab = Vec_xab+[Vec_xab[0]+Vec_xab[1]]
Vec_sumyab = Vec_yab+[Vec_yab[0]+Vec_yab[1]]
Org_x = [0, Vec_xab[0], 0]
Org_y = [0, Vec_yab[0], 0]
Org = [Org_x,Org_y]
plotvec(Org,Vec_sumxab,Vec_sumyab,122,'Y')
plt.title('Sum of Space Vector $\\alpha \\beta$')
# fname="Vector_ab_step"+str(i)+".png"
# plt.savefig(fname)
# files.download(fname) 



st.write(ab_plot)

vec_alpha = '\\vec{\\alpha} = '+ str(round(ab[0],3))+'\\angle{0^\circ}'
vec_beta = '\\vec{\\beta} = '+ str(round(ab[1],3))+'\\angle{90^\circ}'

amp_alpha = max(sin_ab[0])
amp_beta = max(sin_ab[1])
alpha = '\\alpha(t) = '+str(round(amp_alpha,3))+'\sin(\omega t)'
beta = '\\beta(t)= '+ str(round(amp_beta,3))+'\sin(\omega t + \pi/2)'


st.sidebar.info('** Clarke Transformation **')
st.sidebar.write('Scaling = ',sf)
st.sidebar.latex(alpha)
st.sidebar.latex(beta)
col1,col2,col3 = st.beta_columns([10,1,10])
col1.latex(alpha)
col3.latex(beta)
col1.latex(vec_alpha)
col3.latex(vec_beta)

with st.beta_expander('See alpha beta  Vector'):
    st.write(fig)
    

# print(theta*180/pii)
