from __future__ import annotations
import os,sys,csv
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','src'))
from annular_spectral import Parameters,UnsteadySpectral,solve_steady

def fourier(tt,ff,St,K=4):
 cols=[np.ones_like(tt)]
 for k in range(1,K+1): cols += [np.cos(2*np.pi*k*St*tt),np.sin(2*np.pi*k*St*tt)]
 A=np.column_stack(cols);c,*_=np.linalg.lstsq(A,ff,rcond=None)
 amps=[]
 for k in range(K): amps.append(float(np.hypot(c[1+2*k],c[2+2*k])))
 return float(c[0]),amps

def run(N=32,amp=.02,St=.1,rtol=1e-11,atol=1e-13):
 base=Parameters(Fr=10,We=50,theta0_deg=0);w,prob=solve_steady(N,base)
 p=Parameters(Fr=10,We=50,theta0_deg=0,amplitude=amp,St=St,ramp_cycles=2.)
 T=1/St;NC=max(8,int(np.ceil(70/T))+5);tf=NC*T
 op=UnsteadySpectral(N,p,dealias=True);y0=op.state_from_steady(w,prob);op.volume_reference=op.volume(op.unpack(y0,0)[1],float(y0[-1]))
 sol=op.integrate(y0,tf,method='DOP853',rtol=rtol,atol=atol,max_step=.05)
 # 120 pts/cycle from dense output over last 3 cycles
 tt=np.linspace((NC-3)*T,NC*T,361)
 LL=[];CC=[]
 for t in tt:
  m,S,u,v,L=op.unpack(sol.sol(float(t)),float(t));LL.append(L);CC.append(op.pressure_coefficient(S,L))
 LL=np.array(LL);CC=np.array(CC);Lm,aL=fourier(tt,LL,St);Cm,aC=fourier(tt,CC,St)
 phase=np.linspace(0,T,121,endpoint=False);curves=[]
 for cyc in (NC-2,NC-1):
  vals=[]
  for ph in phase:
   t=cyc*T+ph;m,S,u,v,L=op.unpack(sol.sol(t),t);vals.append((L,op.pressure_coefficient(S,L)))
  curves.append(np.array(vals))
 dL=float(np.max(np.abs(curves[1][:,0]-curves[0][:,0])));dC=float(np.max(np.abs(curves[1][:,1]-curves[0][:,1])))
 return {'N':N,'amp':amp,'St':St,'rtol':rtol,'atol':atol,'NC':NC,'discarded_cycles':NC-3,'analysed_cycles':3,
  'gain_L':aL[0]/(amp*Lm),'gain_Cpn':aC[0]/amp,
  'L_A2_over_A1':aL[1]/aL[0],'L_A3_over_A1':aL[2]/aL[0],'L_A4_over_A1':aL[3]/aL[0],
  'C_A2_over_A1':aC[1]/aC[0],'C_A3_over_A1':aC[2]/aC[0],'C_A4_over_A1':aC[3]/aC[0],
  'cycle_rel_L':dL/max(np.max(np.abs(curves[1][:,0])),1e-15),'cycle_rel_Cpn':dC/max(np.max(np.abs(curves[1][:,1])),1e-15),'n_clamped':op.n_clamped}

cases=[(24,.02,.1,1e-11,1e-13),(32,.02,.1,1e-11,1e-13),(48,.02,.1,1e-11,1e-13),
       (32,.02,.1,1e-9,1e-11),(32,.01,.1,1e-11,1e-13),(32,.04,.1,1e-11,1e-13)]
rows=[]
for c in cases:
 print('RUN',c,flush=True);r=run(*c);rows.append(r);print(r,flush=True)
path=os.path.join(HERE,'..','data','revision_forced_reproducibility.csv')
with open(path,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('WROTE',path)
