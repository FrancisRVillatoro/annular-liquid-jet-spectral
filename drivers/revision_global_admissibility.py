"""Revision audit: first loss of global admissibility for the membrane case.

The absorbed representation R=(1-eta)S enforces the endpoint condition
R(1)=0, but a single connected annular jet additionally requires S>0 on
0<=eta<1.  This driver locates the first zero of the interpolating
polynomial S in the whole interval and compares it with the later endpoint
transversality event S(1)=0.
"""
from __future__ import annotations
import os, sys, csv
import numpy as np
from numpy.polynomial.chebyshev import chebder, chebroots, chebval

HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,"..","src"))
from annular_spectral import Parameters, UnsteadySpectral, solve_steady


def bary_matrix(nodes, xnew):
    n=len(nodes)-1
    w=np.ones(n+1); w[0]=w[-1]=0.5; w*=(-1.0)**np.arange(n+1)
    X=xnew[:,None]-nodes[None,:]
    P=np.empty((len(xnew),n+1))
    close=np.isclose(X,0.0,atol=2e-15,rtol=0)
    for i in range(len(xnew)):
        jj=np.flatnonzero(close[i])
        if jj.size:
            P[i]=0.0; P[i,jj[0]]=1.0
        else:
            q=w/X[i]; P[i]=q/q.sum()
    return P


def cheb_coeffs_from_cgl_values(f):
    # f is ordered eta=0..1.  Reverse to x=2eta-1=1..-1.
    g=np.asarray(f)[::-1]
    n=len(g)-1
    ext=np.concatenate([g,g[-2:0:-1]])
    c=np.real(np.fft.fft(ext))[:n+1]/n
    c[0]*=0.5; c[n]*=0.5
    return c


def exact_poly_min(f, interior_only=False):
    c=cheb_coeffs_from_cgl_values(f)
    rr=chebroots(chebder(c))
    rr=rr[np.abs(rr.imag)<5e-10].real
    rr=rr[(rr>-1-1e-12)&(rr<1+1e-12)]
    cand=np.r_[-1.0,rr,1.0]
    if interior_only:
        cand=cand[np.abs(cand-1.0)>1e-10]  # exclude eta=1 endpoint
    vals=chebval(cand,c)
    i=int(np.argmin(vals))
    return float(vals[i]), float((cand[i]+1.0)/2.0)


def setup(N, dealias=True):
    p=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.,
                 amplitude=0.,ramp_cycles=0.,body_amplitude=0.5,St_g=0.5)
    base=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.)
    w,prob=solve_steady(N,base)
    op=UnsteadySpectral(N,p,dealias=dealias)
    y0=op.state_from_steady(w,prob)
    op.volume_reference=op.volume(op.unpack(y0,0.0)[1],float(y0[-1]))
    return op,y0


def run_case(N, eps=1e-8, rtol=1e-10, atol=1e-12, method="DOP853", max_step=0.02,
             scan_M=4001, tf=20.0):
    op,y0=setup(N)
    eta_scan=np.linspace(0.0,1.0,scan_M)
    P=bary_matrix(op.eta,eta_scan)

    def ev_global(t,y):
        S=op.unpack(y,t)[1]
        return float(np.min(P@S)-eps)
    ev_global.terminal=True; ev_global.direction=-1

    sol=op.integrate(y0,tf,method=method,rtol=rtol,atol=atol,max_step=max_step,
                     events=ev_global)
    if not sol.t_events[0].size:
        return dict(N=N,eps=eps,method=method,rtol=rtol,t_global=np.nan,
                    eta_global=np.nan,S_tip=np.nan,L=np.nan,n_clamped=op.n_clamped)
    tg=float(sol.t_events[0][0])
    yg=sol.sol(tg)
    m,S,u,v,L=op.unpack(yg,tg)
    mn,loc=exact_poly_min(S,interior_only=True)
    return dict(N=N,eps=eps,method=method,rtol=rtol,t_global=tg,
                minS_exact=mn,eta_global=loc,S_tip=float(S[-1]),L=float(L),
                m_min=float(m.min()),u_min=float(u.min()),n_clamped=op.n_clamped)


def main():
    out=[]
    for N in (24,32,48,64,96):
        print('RUN',N,flush=True)
        row=run_case(N)
        out.append(row); print(row,flush=True)
    # tolerance scan at N=48
    for method,rt,at in (("DOP853",1e-8,1e-10),("DOP853",1e-10,1e-12),
                         ("DOP853",1e-12,1e-14),("Radau",1e-9,1e-11)):
        print('TOL',method,rt,flush=True)
        row=run_case(48,method=method,rtol=rt,atol=at)
        row['kind']='tolerance'; out.append(row); print(row,flush=True)
    path=os.path.join(HERE,"..","data","revision_global_admissibility.csv")
    keys=sorted({k for r in out for k in r})
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(out)
    print('WROTE',path)

if __name__=='__main__': main()
