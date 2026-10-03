from __future__ import annotations
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebder,chebroots,chebval
from scipy.optimize import brentq
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.join(HERE,'..')
sys.path.insert(0,os.path.join(ROOT,'src'))
from annular_spectral import Parameters,UnsteadySpectral,solve_steady


def coeffs(f):
 g=np.asarray(f)[::-1]; n=len(g)-1; ext=np.concatenate([g,g[-2:0:-1]])
 c=np.real(np.fft.fft(ext))[:n+1]/n;c[0]*=.5;c[n]*=.5;return c

def pmin(f):
 c=coeffs(f);rr=chebroots(chebder(c));rr=rr[np.abs(rr.imag)<1e-8].real;rr=rr[(rr>-1+1e-10)&(rr<1-1e-10)]
 cand=np.r_[-1.,rr];vals=chebval(cand,c);j=int(np.argmin(vals));return float(vals[j]),float((cand[j]+1)/2)

def setup(N,tf):
 p=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.,body_amplitude=.5,St_g=.5,ramp_cycles=0.)
 base=Parameters(Fr=10.,We=50.,theta0_deg=0.)
 w,prob=solve_steady(N,base);op=UnsteadySpectral(N,p,dealias=True);y0=op.state_from_steady(w,prob);op.volume_reference=op.volume(op.unpack(y0,0)[1],float(y0[-1]))
 sol=op.integrate(y0,tf,method='DOP853',rtol=1e-11,atol=1e-13,max_step=.01)
 return op,sol

def first_contact(op,sol,lo=10.6,hi=10.8):
 def h(t):return pmin(op.unpack(sol.sol(t),t)[1])[0]
 tt=np.linspace(lo,min(hi,sol.t[-1]),801);vv=np.array([h(float(t)) for t in tt]);idx=np.where((vv[:-1]>0)&(vv[1:]<=0))[0]
 if not len(idx):return None
 i=int(idx[0]);t=brentq(h,float(tt[i]),float(tt[i+1]),xtol=3e-13,rtol=5e-15);mn,e=pmin(op.unpack(sol.sol(t),t)[1]);return t,e

os.makedirs(os.path.join(ROOT,'figures'),exist_ok=True)

# Figure 2 replacement: global admissibility measure.
fig,ax=plt.subplots(figsize=(6.5,4.4))
for N,ls in ((16,':'),(24,'-.'),(32,'--'),(48,'-'),(64,'-')):
 op,sol=setup(N,11.0)
 tt=np.linspace(8.0,min(11.,sol.t[-1]),301)
 mm=[]
 for t in tt: mm.append(pmin(op.unpack(sol.sol(float(t)),float(t))[1])[0])
 label=f'$N={N}$'
 ax.plot(tt,mm,ls=ls,lw=1.25,label=label)
 fc=first_contact(op,sol)
 if fc is not None and N>=24:
  ax.plot(fc[0],0,'o',ms=3.5)
ax.axhline(0,lw=.8)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\min_{0\leq\eta<1} S(t,\eta)$')
ax.set_xlim(8,11)
ax.legend(frameon=False,ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(ROOT,'figures','fig2_global_admissibility.pdf'),bbox_inches='tight')
plt.close(fig)

# Figure 5 replacement: geometry approaching first interior touch-down.
op,sol=setup(96,10.755)
tc,ec=first_contact(op,sol,10.73,10.755)
times=(0.,8.,10.,10.5,10.70,tc)
fig,axes=plt.subplots(1,2,figsize=(7.0,3.25))
for t,ls in zip(times,('-', '--', ':', '-.', '--', '-')):
 m,S,u,v,L=op.unpack(sol.sol(float(t)),float(t));R=(1-op.eta)*S;z=L*op.eta
 label=(r'$t=t_c$' if abs(t-tc)<1e-9 else f'$t={t:g}$')
 axes[0].plot(z,R,ls=ls,lw=1.15,label=label)
 if t >= 10.5:
  axes[1].plot(z,R,ls=ls,lw=1.15,label=label)
zc=op.unpack(sol.sol(tc),tc)[4]*ec
axes[1].plot(zc,0,'o',ms=4,label=r'$z_c$')
axes[0].set_xlabel(r'$z$');axes[0].set_ylabel(r'$R$')
axes[0].set_xlim(0,13.4);axes[0].set_ylim(bottom=-.01)
axes[1].set_xlabel(r'$z$');axes[1].set_ylabel(r'$R$')
axes[1].set_xlim(11.9,13.25);axes[1].set_ylim(-.003,.08)
axes[0].legend(frameon=False,fontsize=7,ncol=2)
axes[1].legend(frameon=False,fontsize=7)
axes[0].text(.02,.96,'(a)',transform=axes[0].transAxes,va='top')
axes[1].text(.02,.96,'(b)',transform=axes[1].transAxes,va='top')
fig.tight_layout()
fig.savefig(os.path.join(ROOT,'figures','fig5_interior_touchdown.pdf'),bbox_inches='tight')
plt.close(fig)
print('tc',tc,'eta',ec,'zc',zc)
