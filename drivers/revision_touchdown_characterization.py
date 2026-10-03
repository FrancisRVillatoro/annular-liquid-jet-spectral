from __future__ import annotations
import os,sys,csv
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.chebyshev import chebder,chebroots,chebval
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','src'))
from annular_spectral import Parameters,UnsteadySpectral,solve_steady

def coeffs(f):
 g=np.asarray(f)[::-1]; n=len(g)-1; ext=np.concatenate([g,g[-2:0:-1]])
 c=np.real(np.fft.fft(ext))[:n+1]/n; c[0]*=.5; c[n]*=.5; return c
def pmin(f):
 c=coeffs(f); rr=chebroots(chebder(c)); rr=rr[np.abs(rr.imag)<1e-8].real; rr=rr[(rr>-1+1e-11)&(rr<1-1e-11)]
 cand=np.r_[-1.,rr]; vals=chebval(cand,c); j=int(np.argmin(vals)); return float(vals[j]),float((cand[j]+1)/2),c
def locate(N):
 p=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.,body_amplitude=.5,St_g=.5,ramp_cycles=0.)
 base=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.)
 w,prob=solve_steady(N,base); op=UnsteadySpectral(N,p,dealias=True); y0=op.state_from_steady(w,prob); op.volume_reference=op.volume(op.unpack(y0,0)[1],float(y0[-1]))
 sol=op.integrate(y0,10.765,method='DOP853',rtol=1e-11,atol=1e-13,max_step=.005)
 def h(t): return pmin(op.unpack(sol.sol(t),t)[1])[0]
 tt=np.linspace(10.73,10.765,501); hv=np.array([h(t) for t in tt]); ix=np.where((hv[:-1]>0)&(hv[1:]<=0))[0][0]
 tg=brentq(h,tt[ix],tt[ix+1],xtol=2e-13,rtol=5e-15)
 y=sol.sol(tg); m,S,u,v,L=op.unpack(y,tg); mn,eta,c=pmin(S); x=2*eta-1
 Seta=2*chebval(x,chebder(c)); Setaaa=4*chebval(x,chebder(c,2))
 Retaeta=(1-eta)*Setaaa - 2*Seta
 # semidiscrete time derivative polynomial dS/dt at contact
 dy=op.rhs(tg,y); dS_nodes=np.r_[0.0,dy[N:2*N]]
 ct=coeffs(dS_nodes); St=chebval(x,ct)
 z=L*eta; Rzz=Retaeta/(L*L); quad=.5*Rzz
 # Ldot and tip terms still finite at first contact
 Ldot=float(dy[-1]); tip_term=float(L*v[-1]/S[-1])
 return dict(N=N,t_touch=tg,eta_touch=eta,z_touch=z,L=L,S_tip=S[-1],
   S_eta=Seta,S_etaeta=Setaaa,R_etaeta=Retaeta,R_zz=Rzz,quad_coeff_z=quad,
   S_t=St,Ldot=Ldot,tip_Lv_over_S=tip_term,min_m=m.min(),min_u=u.min())

def main():
 rows=[]
 for N in (48,64,96,128,160):
  print('RUN',N,flush=True); r=locate(N); rows.append(r); print(r,flush=True)
 os.makedirs(os.path.join(HERE,'..','data'),exist_ok=True); path=os.path.join(HERE,'..','data','revision_touchdown_characterization.csv')
 keys=list(rows[0].keys())
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
 print('WROTE',path)
if __name__=='__main__': main()
