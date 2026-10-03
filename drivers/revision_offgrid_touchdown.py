from __future__ import annotations
import os,sys,csv
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','src'))
from annular_spectral import Parameters,UnsteadySpectral,solve_steady
from offgrid import fine_operators,offgrid_residual_detailed,boundary_residuals,modal_tail

TREF=10.75383

def run(N):
 p=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.,body_amplitude=.5,St_g=.5,ramp_cycles=0.)
 base=Parameters(Fr=10.,We=50.,theta0_deg=0.)
 w,prob=solve_steady(N,base); op=UnsteadySpectral(N,p,dealias=True)
 y0=op.state_from_steady(w,prob);op.volume_reference=op.volume(op.unpack(y0,0)[1],float(y0[-1]))
 ts=[0.,2.,5.,8.,10.,10.5,10.7,TREF-.02]
 sol=op.integrate(y0,max(ts),method='DOP853',rtol=1e-11,atol=1e-13,max_step=.01,t_eval=ts)
 ef,P=fine_operators(op,eta_min=.05,eta_max=1.-1e-6,M=2001)
 rows=[]
 for t,y in zip(sol.t,sol.y.T):
  d=offgrid_residual_detailed(op,float(t),y,ef,P); b=boundary_residuals(op,float(t),y); tail=modal_tail(op,y,float(t))
  row={'N':N,'t':float(t),'abs_max':d['abs_max'],'rel_max':d['rel_max'],'tail_max':max(tail.values()),
       'bc_max':max(b.values())}
  for q in ('m','S','u','v'):
   row[f'abs_{q}']=d[q]['abs'];row[f'rel_{q}']=d[q]['rel'];row[f'tail_{q}']=tail[q]
  rows.append(row)
 return rows

rows=[]
for N in (32,48,64,96):
 print('RUN',N,flush=True);rr=run(N);rows+=rr;print(rr[-1],flush=True)
path=os.path.join(HERE,'..','data','revision_offgrid_touchdown.csv')
with open(path,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('WROTE',path)
