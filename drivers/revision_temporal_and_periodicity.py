from __future__ import annotations
import os,sys,csv
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','src'))
sys.path.insert(0,HERE)
from revision_global_admissibility_precise import locate
from annular_spectral import Parameters,UnsteadySpectral,solve_steady

# Temporal tolerance/integrator convergence of the first interior contact.
rows=[]
for method,rtol,atol,ms in [
    ('DOP853',1e-8,1e-10,.02),
    ('DOP853',1e-10,1e-12,.02),
    ('DOP853',1e-12,1e-14,.01),
    ('Radau',1e-9,1e-11,.02),
]:
    print('TEMP',method,rtol,flush=True)
    r=locate(48,rtol=rtol,atol=atol,method=method,max_step=ms)
    rows.append({'method':method,'rtol':rtol,'atol':atol,'max_step':ms,'N':48,
                 't_touch':r['t_global'],'eta_touch':r['eta_global'],'S_tip':r['S_tip']})
path=os.path.join(HERE,'..','data','revision_temporal_convergence.csv')
with open(path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('WROTE',path)

# A=0.45, St_g=0.5: extend to t=100 (50 cycles), not merely t=40.
rows=[]
for N in (48,64):
    p=Parameters(Fr=10.,We=50.,theta0_deg=0.,Cpmax=1.,pressure_ratio0=1.,
                 body_amplitude=.45,St_g=.5,ramp_cycles=0.)
    base=Parameters(Fr=10.,We=50.,theta0_deg=0.)
    ww,prob=solve_steady(N,base)
    op=UnsteadySpectral(N,p,dealias=True)
    y0=op.state_from_steady(ww,prob)
    op.volume_reference=op.volume(op.unpack(y0,0)[1],float(y0[-1]))
    tf=100.;T=2.
    sol=op.integrate(y0,tf,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.02)
    phase=np.linspace(0,T,401,endpoint=False)
    curves=[]
    for cyc in (48,49):
        arr=[]
        for ph in phase:
            t=cyc*T+ph
            m,S,u,v,L=op.unpack(sol.sol(t),t)
            arr.append((L,op.pressure_coefficient(S,L),S[-1],float(S.min())))
        curves.append(np.array(arr))
    row={'N':N,'t_end':tf,'cycles_completed':50,'n_clamped':op.n_clamped,
         'success':bool(sol.success),'solver_t_end':float(sol.t[-1])}
    for j,name in enumerate(('L','Cpn','S_tip','Smin_nodes')):
        d=np.max(np.abs(curves[-1][:,j]-curves[-2][:,j]))
        scale=max(np.max(np.abs(curves[-1][:,j])),1e-15)
        row['cycle_diff_'+name]=float(d);row['cycle_rel_'+name]=float(d/scale)
    rows.append(row);print('PER',row,flush=True)
path=os.path.join(HERE,'..','data','revision_A045_periodicity.csv')
with open(path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('WROTE',path)
