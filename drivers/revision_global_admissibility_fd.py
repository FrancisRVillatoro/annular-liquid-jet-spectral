"""Independent finite-difference check of first interior closure."""
from __future__ import annotations
import os,sys,csv
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','src'))
from annular_spectral import Parameters, solve_steady
from upwind_fd import UpwindFD


def setup(M,Nseed=96):
    p=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.,
                 amplitude=0.,ramp_cycles=0.,body_amplitude=0.5,St_g=0.5)
    base=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.)
    w,prob=solve_steady(Nseed,base)
    op=UpwindFD(M,p,impose_tip=True)
    y0=op.from_spectral(w,prob)
    return op,y0


def run(M,eps=1e-8,rtol=1e-10,atol=1e-12,max_step=0.01,tf=20.):
    op,y0=setup(M)
    om=1.0-op.eta[:-1]
    def ev(t,y):
        R=op.unpack(y)[1]
        S=R[:-1]/om
        return float(np.min(S)-eps)
    ev.terminal=True; ev.direction=-1
    try:
        sol=op.integrate(y0,tf,rtol=rtol,atol=atol,max_step=max_step,events=ev,dense_output=True)
    except FloatingPointError as e:
        return dict(M=M,t_global=np.nan,error=str(e))
    if not sol.t_events[0].size:
        return dict(M=M,t_global=np.nan,status=sol.status,last=sol.t[-1])
    t=float(sol.t_events[0][0]); y=sol.sol(t)
    m,R,u,v,L=op.unpack(y)
    S=R[:-1]/om; j=int(np.argmin(S))
    return dict(M=M,t_global=t,eta_global=float(op.eta[j]),minS=float(S[j]),
                S_near_tip=float(S[-1]),L=float(L),R_tip=float(R[-1]),
                m_min=float(m.min()))

def main():
    rows=[]
    for M in (80,160,320,640,1280):
        print('RUN',M,flush=True)
        r=run(M); rows.append(r); print(r,flush=True)
    os.makedirs(os.path.join(HERE,'..','data'),exist_ok=True)
    keys=sorted({k for r in rows for k in r})
    path=os.path.join(HERE,'..','data','revision_global_admissibility_fd.csv')
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    print('WROTE',path)
if __name__=='__main__': main()
