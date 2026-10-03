"""Global-admissibility event for every forced case retained in the audit.

The integration is stopped by the zero of the minimum of the complete
Chebyshev interpolant, so no post-contact continuation or positivity clamp
is used to define these event times.
"""
from __future__ import annotations
import os,sys,csv
import numpy as np
from numpy.polynomial.chebyshev import chebder,chebroots,chebval
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','src'))
from annular_spectral import Parameters,UnsteadySpectral,solve_steady

def coeffs(f):
    g=np.asarray(f)[::-1]; n=len(g)-1
    ext=np.concatenate([g,g[-2:0:-1]])
    c=np.real(np.fft.fft(ext))[:n+1]/n
    c[0]*=.5; c[n]*=.5
    return c

def pmin(f):
    c=coeffs(f); rr=chebroots(chebder(c))
    rr=rr[np.abs(rr.imag)<1e-8].real
    rr=rr[(rr>-1+1e-10)&(rr<1-1e-10)]
    cand=np.r_[-1.,rr]
    vals=chebval(cand,c); j=int(np.argmin(vals))
    return float(vals[j]),float((cand[j]+1)/2)

def run(name,p,N,tf):
    base=Parameters(Fr=p.Fr,We=p.We,theta0_deg=p.theta0_deg,
                    Cpmax=p.Cpmax,pressure_ratio0=p.pressure_ratio0)
    w,prob=solve_steady(N,base)
    op=UnsteadySpectral(N,p,dealias=True)
    y0=op.state_from_steady(w,prob)
    op.volume_reference=op.volume(op.unpack(y0,0.)[1],float(y0[-1]))
    def ev(t,y):
        return pmin(op.unpack(y,float(t))[1])[0]
    ev.terminal=True; ev.direction=-1
    sol=op.integrate(y0,tf,method='DOP853',rtol=1e-10,atol=1e-12,
                     max_step=.02,events=ev)
    if sol.t_events[0].size:
        t=float(sol.t_events[0][0]); y=sol.sol(t)
        m,S,u,v,L=op.unpack(y,t); mn,eta=pmin(S)
        row={'case':name,'N':N,'t_contact':t,'eta_contact':eta,'S_tip':float(S[-1]),
             'L':float(L),'z_contact':float(L*eta),'minS':mn,
             'n_clamped':int(op.n_clamped),'success':bool(sol.success)}
    else:
        row={'case':name,'N':N,'t_contact':np.nan,'eta_contact':np.nan,
             'S_tip':np.nan,'L':float(sol.y[-1,-1]),'z_contact':np.nan,'minS':np.nan,
             'n_clamped':int(op.n_clamped),'success':bool(sol.success),'t_end':float(sol.t[-1])}
    print(row,flush=True); return row

cases=[
 ('body_A0.5_Stg0.5', Parameters(Fr=10,We=50,theta0_deg=0,Cpmax=1,pressure_ratio0=1,
     body_amplitude=.5,St_g=.5,ramp_cycles=0),20.),
 ('nozzle_a0.25_St0.1_abrupt', Parameters(Fr=10,We=50,theta0_deg=0,amplitude=.25,St=.1,ramp_cycles=0),40.),
 ('nozzle_a0.25_St0.1_ramp2', Parameters(Fr=10,We=50,theta0_deg=0,amplitude=.25,St=.1,ramp_cycles=2),45.),
 ('theta15_a0.1_St0.1_abrupt', Parameters(Fr=10,We=50,theta0_deg=15,amplitude=.1,St=.1,ramp_cycles=0),30.),
 ('theta15_a0.1_St0.1_ramp2', Parameters(Fr=10,We=50,theta0_deg=15,amplitude=.1,St=.1,ramp_cycles=2),40.),
 ('nozzle_a0.02_St0.5_ramp2', Parameters(Fr=10,We=50,theta0_deg=0,amplitude=.02,St=.5,ramp_cycles=2),20.),
]
rows=[]
for name,p,tf in cases:
    for N in (48,64):
        rows.append(run(name,p,N,tf))
keys=sorted({k for r in rows for k in r})
path=os.path.join(HERE,'..','data','revision_all_event_cases.csv')
with open(path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
print('WROTE',path)
