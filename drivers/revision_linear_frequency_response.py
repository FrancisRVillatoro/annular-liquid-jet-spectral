from __future__ import annotations
import os,sys,csv
import numpy as np
from scipy.optimize import root
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','src'))
from annular_spectral import Parameters,UnsteadySpectral,solve_steady
N=32
base=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.)
w,prob=solve_steady(N,base)
op=UnsteadySpectral(N,base,dealias=True)
y0=op.state_from_steady(w,prob)
op.volume_reference=op.volume(op.unpack(y0,0.)[1],float(y0[-1]))
# Closed-loop equilibrium of the actual unsteady RHS.
sol=root(lambda y:op.rhs(0.,y),y0,method='hybr',tol=1e-12,options={'maxfev':50000})
ye=sol.x
print('eq_res',np.max(np.abs(op.rhs(0.,ye))),'success',sol.success,flush=True)
J=op.jacobian(ye,eps=3e-8)
# Input vector: derivative wrt imposed nozzle velocity u0. At a quarter phase
# sin=1 and du0/dt=0; with ramp_cycles=0 the amplitude is exactly delta u0.
Stref=.1;tq=1/(4*Stref);eps=1e-6
def f_with_amp(a):
 p=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.,amplitude=a,St=Stref,ramp_cycles=0.)
 q=UnsteadySpectral(N,p,dealias=True);q.volume_reference=op.volume_reference
 return q.rhs(tq,ye)
B=(f_with_amp(eps)-f_with_amp(-eps))/(2*eps)
# Output derivatives.
n=ye.size
CL=np.zeros(n);CL[-1]=1.
def outC(y):
 m,S,u,v,L=op.unpack(y,0.);return op.pressure_coefficient(S,L)
f0=outC(ye);CC=np.empty(n)
for k in range(n):
 h=2e-7*max(1.,abs(ye[k]));yp=ye.copy();ym=ye.copy();yp[k]+=h;ym[k]-=h;CC[k]=(outC(yp)-outC(ym))/(2*h)
L0=ye[-1]
# Static transfer from same linear system.
x0=np.linalg.solve(-J,B);gL0=abs(CL@x0)/L0;gC0=abs(CC@x0)
print('static',gL0,gC0,flush=True)
Sts=np.linspace(.005,.405,401);rows=[]
for St in Sts:
 om=2*np.pi*St;x=np.linalg.solve(1j*om*np.eye(n)-J,B);HL=CL@x;HC=CC@x
 rows.append({'St':St,'gain_L':abs(HL)/L0,'gain_L_norm':abs(HL)/L0/gL0,'gain_Cpn':abs(HC),'gain_Cpn_norm':abs(HC)/gC0})
arr=np.array([r['gain_Cpn_norm'] for r in rows]);i=int(np.argmax(arr));print('peak C',rows[i],flush=True)
arrL=np.array([r['gain_L_norm'] for r in rows]);print('L monotone',bool(np.all(np.diff(arrL)>0)),'last',rows[-1],flush=True)
path=os.path.join(HERE,'..','data','revision_linear_frequency_response.csv')
with open(path,'w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print('WROTE',path)
