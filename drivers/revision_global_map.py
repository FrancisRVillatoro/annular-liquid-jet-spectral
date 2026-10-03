from __future__ import annotations
import os,sys
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.chebyshev import chebder,chebroots,chebval
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','src'))
from annular_spectral import Parameters,UnsteadySpectral,solve_steady

def coeffs(f):
    g=np.asarray(f)[::-1]; n=len(g)-1
    ext=np.concatenate([g,g[-2:0:-1]])
    c=np.real(np.fft.fft(ext))[:n+1]/n; c[0]*=.5; c[n]*=.5
    return c

def pmin(f):
    c=coeffs(f); rr=chebroots(chebder(c)); rr=rr[np.abs(rr.imag)<1e-8].real
    rr=rr[(rr>-1+1e-10)&(rr<1-1e-10)]
    cand=np.r_[-1.,rr]; vals=chebval(cand,c); j=int(np.argmin(vals))
    return float(vals[j]),float((cand[j]+1)/2)

def run(A,N,tf=40.,Stg=.5):
    p=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.,body_amplitude=A,St_g=Stg,ramp_cycles=0.)
    base=Parameters(Fr=10.,We=50.,theta0_deg=0.)
    w,prob=solve_steady(N,base); op=UnsteadySpectral(N,p,dealias=True)
    y0=op.state_from_steady(w,prob); op.volume_reference=op.volume(op.unpack(y0,0)[1],float(y0[-1]))
    sol=op.integrate(y0,tf,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.02)
    # scan dense output; stop at first sign change
    tt=np.arange(0.,float(sol.t[-1])+1e-12,.01)
    prev=pmin(op.unpack(sol.sol(0.),0.)[1])[0]
    best=(prev,0.,0.)
    event=None
    for t in tt[1:]:
        v,eta=pmin(op.unpack(sol.sol(float(t)),float(t))[1])
        if v<best[0]: best=(v,float(t),eta)
        if prev>0 and v<=0:
            lo=float(t-.01); hi=float(t)
            fun=lambda x:pmin(op.unpack(sol.sol(float(x)),float(x))[1])[0]
            tc=brentq(fun,lo,hi,xtol=5e-12,rtol=5e-15)
            mn,ec=pmin(op.unpack(sol.sol(tc),tc)[1]); S=op.unpack(sol.sol(tc),tc)[1]
            event=(tc,ec,float(S[-1])); break
        prev=v
    print('A',A,'Stg',Stg,'N',N,'status',sol.status,'tend',sol.t[-1],'event',event,'best',best,'clamp',op.n_clamped,flush=True)

for A in (.40,.45,.50):
    for N in (32,48,64): run(A,N)
