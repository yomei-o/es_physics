import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, time, qcd_lib as Q
print("gammas ok:", Q.check_gammas())
rng=np.random.default_rng(0)
L,T=4,8
U=[np.broadcast_to(np.eye(3,dtype=complex),(L,L,L,T,3,3)).copy() for _ in range(4)]
eps=0.25; t0=time.time()
for i in range(60):
    a=Q.metropolis_sweep(U,5.7,eps,rng); eps*=1.05 if a>0.5 else 0.95
    if i%20==19: U=Q.reunitarize(U); print(f"sweep {i+1}: plaq={Q.plaquette(U):.4f} acc={a:.2f} eps={eps:.3f} ({time.time()-t0:.0f}s)")
phi=rng.normal(size=(L,L,L,T,4,3))+1j*rng.normal(size=(L,L,L,T,4,3)); psi=rng.normal(size=phi.shape)+1j*rng.normal(size=phi.shape)
g5=lambda v: np.einsum('st,...tb->...sb',Q.G5,v)
lhs=np.vdot(phi, g5(Q.dslash(U,g5(psi),0.164))); rhs=np.vdot(Q.dslash(U,phi,0.164),psi)
print("gamma5-hermiticity |lhs-rhs|/|lhs| =", abs(lhs-rhs)/abs(lhs))
t0=time.time(); S,res=Q.propagator(U,0.164); print(f"propagator: worst residual {res:.1e} ({time.time()-t0:.0f}s)")
cpi=Q.pion(S); print("pion C(t):", np.round(cpi/cpi[0],5))
Gp=(np.eye(4)+Q.G[3])/2
cn=Q.nucleon_full(S,S,Gp); print("nucleon C(t) real:", np.round(cn.real/cn.real[0],5), " imag/real max:", np.abs(cn.imag).max()/abs(cn.real[0]))
