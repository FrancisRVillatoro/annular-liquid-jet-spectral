"""Precise first interior contact from dense-output + exact polynomial minimum."""
from __future__ import annotations
import os,sys,csv
import numpy as np
from scipy.optimize import brentq
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
    rr=rr[(rr>-1+1e-11)&(rr<1-1e-11)]
    cand=np.r_[-1.0,rr]  # include nozzle, exclude tracked endpoint
    vals=chebval(cand,c); j=int(np.argmin(vals))
    return float(vals[j]),float((cand[j]+1)/2)

def setup(N):
    p=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.,
        amplitude=0.,ramp_cycles=0.,body_amplitude=.5,St_g=.5)
    base=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.)
    w,prob=solve_steady(N,base); op=UnsteadySpectral(N,p,dealias=True)
    y0=op.state_from_steady(w,prob); op.volume_reference=op.volume(op.unpack(y0,0)[1],float(y0[-1]))
    return op,y0

def locate(N,eps=0.,rtol=1e-10,atol=1e-12,method='DOP853',max_step=.01):
    op,y0=setup(N)
    tf=10.785 if N<=24 else 10.765
    sol=op.integrate(y0,tf,method=method,rtol=rtol,atol=atol,max_step=max_step)
    def h(t):
        y=sol.sol(float(t)); S=op.unpack(y,float(t))[1]; return pmin(S)[0]-eps
    # search last 0.1 time units for first sign change
    tt=np.linspace(max(0,tf-.15),tf,601); hv=np.array([h(t) for t in tt])
    ix=np.where((hv[:-1]>0)&(hv[1:]<=0))[0]
    if not len(ix):
        return {'N':N,'t_global':np.nan,'hmin':float(hv.min())}
    i=int(ix[0]); tg=brentq(h,float(tt[i]),float(tt[i+1]),xtol=5e-13,rtol=5e-15)
    y=sol.sol(tg); m,S,u,v,L=op.unpack(y,tg); mn,eta=pmin(S)
    DS=op.D@S
    # derivative S_eta at contact, evaluated via Cheb derivative at x=2eta-1; d/deta=2 d/dx
    c=coeffs(S); dc=chebder(c); Seta=2*chebval(2*eta-1,dc)
    Reta=-S+(1-op.eta)*DS
    return {'N':N,'t_global':tg,'minS':mn,'eta_global':eta,'S_eta_contact':float(Seta),
            'S_tip':float(S[-1]),'Reta_tip':float(Reta[-1]),'L':float(L),
            'm_min':float(m.min()),'u_min':float(u.min()),'n_clamped':op.n_clamped}

def main():
    rows=[]
    for N in (24,32,48,64,96,128,160):
        print('RUN',N,flush=True); r=locate(N); rows.append(r); print(r,flush=True)
    os.makedirs(os.path.join(HERE,'..','data'),exist_ok=True)
    path=os.path.join(HERE,'..','data','revision_global_admissibility_precise.csv')
    keys=sorted({k for r in rows for k in r})
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    print('WROTE',path)
if __name__=='__main__': main()
