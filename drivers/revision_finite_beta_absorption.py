"""Check the algebraic finite-beta constraint absorption.

R_i = R - beta m/(2R) = (1-eta) Sigma,
R = (R_i + sqrt(R_i^2 + 2 beta m))/2.
Thus 2R^2-beta m = 2 R R_i and vanishes identically at eta=1.
The paper may state this transformation for completeness while retaining
all reported unsteady calculations at beta=0.
"""
from __future__ import annotations
import os,sys,csv
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','src'))
from annular_thickness import ThickParameters,solve_steady_thick,reference_length,ThickUnsteady

rows=[]
for beta,N in ((0.05,48),(0.20,48)):
    p=ThickParameters(Fr=10,We=50,theta0_deg=0,Cpmax=1,pressure_ratio0=1.5,beta=beta)
    Lref=reference_length(p)
    w,prob=solve_steady_thick(N,p)
    r={'beta':beta,'N':N,'L_spectral':float(w[-1]),'L_reference':float(Lref),
       'abs_dL':float(abs(w[-1]-Lref)),'steady_tip_g':float(prob.tip_residual(w)),
       'steady_res_inf':float(np.max(np.abs(prob.residual(w))))}
    # A weak unsteady test verifies the algebraic identity along a trajectory.
    pu=ThickParameters(Fr=10,We=50,theta0_deg=0,Cpmax=1,pressure_ratio0=1.5,
                       beta=beta,amplitude=.01,St=.1,ramp_cycles=2.)
    op=ThickUnsteady(N,pu); y0=op.state_from_steady(w,prob)
    op.volume_reference=op.volume(op.unpack(y0,0.)[1],float(y0[-1]))
    ts=np.linspace(0,20,201)
    sol=op.integrate(y0,20.,t_eval=ts,rtol=1e-10,atol=1e-12)
    gs=np.array([op.tip_residual(sol.y[:,i],float(t)) for i,t in enumerate(sol.t)])
    r.update({'unsteady_success':bool(sol.success),'unsteady_t_end':float(sol.t[-1]),
              'max_abs_tip_g':float(np.max(np.abs(gs)))})
    rows.append(r); print(r,flush=True)
path=os.path.join(HERE,'..','data','revision_finite_beta_absorption.csv')
with open(path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('WROTE',path)
