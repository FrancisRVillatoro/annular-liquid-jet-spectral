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
 c=np.real(np.fft.fft(ext))[:n+1]/n; c[0]*=.5;c[n]*=.5;return c

def pmin(f):
 c=coeffs(f); rr=chebroots(chebder(c)); rr=rr[np.abs(rr.imag)<1e-8].real; rr=rr[(rr>-1+1e-9)&(rr<1-1e-9)]
 cand=np.r_[-1.,rr]; vals=chebval(cand,c);j=int(np.argmin(vals));return float(vals[j]),float((cand[j]+1)/2)

def run(St,N=32,amp=.02):
 base=Parameters(Fr=10,We=50,theta0_deg=0); w,prob=solve_steady(N,base)
 p=Parameters(Fr=10,We=50,theta0_deg=0,amplitude=amp,St=St,ramp_cycles=2.)
 T=1/St; NC=max(8,int(np.ceil(70/T))+5); tf=NC*T
 op=UnsteadySpectral(N,p,dealias=True); y0=op.state_from_steady(w,prob);op.volume_reference=op.volume(op.unpack(y0,0)[1],float(y0[-1]))
 sol=op.integrate(y0,tf,method='DOP853',rtol=1e-11,atol=1e-13,max_step=.05)
 # accepted time states are adequate to flag positivity; exact pmin of polynomial at accepted states
 best=(1e99,0,0,0); eventbr=None
 prev=None;pt=None
 for t,y in zip(sol.t,sol.y.T):
  S=op.unpack(y,float(t))[1]; v,e=pmin(S)
  if v<best[0]: best=(v,float(t),e,float(S[-1]))
  if prev is not None and prev>0 and v<=0 and eventbr is None: eventbr=(pt,float(t))
  prev=v;pt=float(t)
 ev=None
 if eventbr and sol.sol is not None:
  fun=lambda x:pmin(op.unpack(sol.sol(x),x)[1])[0]
  tc=brentq(fun,*eventbr); S=op.unpack(sol.sol(tc),tc)[1]; mn,e=pmin(S); ev=(tc,e,float(S[-1]))
 # cycle differences L,S min at phases using dense output if run succeeded
 cyc=[]
 if sol.sol is not None and sol.t[-1]>=tf-1e-8:
  phases=np.linspace(0,T,121,endpoint=False)
  curves=[]
  for k in (NC-3,NC-2,NC-1):
   vals=[]
   for ph in phases:
    t=k*T+ph; m,S,u,v,L=op.unpack(sol.sol(t),t); vals.append((L,op.pressure_coefficient(S,L),pmin(S)[0]))
   curves.append(np.array(vals))
  for j,name in enumerate(('L','Cpn','minS')):
   d12=np.max(np.abs(curves[2][:,j]-curves[1][:,j])); scale=max(np.max(np.abs(curves[2][:,j])),1e-14)
   cyc.append((name,d12,d12/scale))
 row={'St':St,'N':N,'NC':NC,'t_final':tf,'success':bool(sol.success),
      't_end':float(sol.t[-1]),'n_clamped':int(op.n_clamped),
      'minS_best':best[0],'t_best':best[1],'eta_best':best[2],'S_tip_best':best[3]}
 if ev is not None:
  row.update({'t_contact':ev[0],'eta_contact':ev[1],'S_tip_contact':ev[2]})
 else:
  row.update({'t_contact':np.nan,'eta_contact':np.nan,'S_tip_contact':np.nan})
 for name,d,rel in cyc:
  row['cycle_diff_'+name]=float(d); row['cycle_rel_'+name]=float(rel)
 print('St',St,'N',N,'NC',NC,'tf',tf,'success',sol.success,'tend',sol.t[-1],'event',ev,'best',best,'cycle',cyc,'clamp',op.n_clamped,flush=True)
 return row

def main():
 rows=[run(St) for St in (.1,.2,.3,.4,.405,.410)]
 keys=sorted({k for r in rows for k in r})
 path=os.path.join(HERE,'..','data','revision_frequency_admissibility.csv')
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
 print('WROTE',path)

if __name__=='__main__':
 main()
