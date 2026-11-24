# Verify Theorem 2 of "Information Update Geometry" (sign of curvature, direction of gravity)
import sympy as sp

x, y, z, t = sp.symbols('x y z t', real=True)
X = [x, y, z]


def christoffel(g, coords):
    ginv = sp.simplify(g.inv())
    n = len(coords)
    G = [[[0] * n for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                G[k][i][j] = sp.simplify(sum(ginv[k, l] * (sp.diff(g[l, i], coords[j]) + sp.diff(g[l, j], coords[i])
                                                          - sp.diff(g[i, j], coords[l])) for l in range(n)) / 2)
    return G


def ricci_scalar(g, coords):
    n = len(coords)
    G = christoffel(g, coords)
    ginv = sp.simplify(g.inv())
    Ric = sp.zeros(n)
    for i in range(n):
        for j in range(n):
            Ric[i, j] = sum(sp.diff(G[k][i][j], coords[k]) - sp.diff(G[k][i][k], coords[j])
                            + sum(G[k][k][l] * G[l][i][j] - G[k][j][l] * G[l][i][k] for l in range(n))
                            for k in range(n))
    return sp.simplify(sum(ginv[i, j] * Ric[i, j] for i in range(n) for j in range(n)))


print("=== (1) Eq.(7): sign of scalar curvature, metric g = sigma^-2 * delta, d=3 ===")
f = sp.Function('f')(x)          # f = log sigma, varying along x
sigma = sp.exp(f)
g = sp.eye(3) / sigma**2
R = ricci_scalar(g, X)
lap = sp.diff(f, x, 2)
grad2 = sp.diff(f, x)**2
print("R =", sp.simplify(R))
print("standard formula sigma^2[2(d-1) Lap log s - (d-1)(d-2)|grad log s|^2] matches:",
      sp.simplify(R - sigma**2 * (4 * lap - 2 * grad2)) == 0)
print("paper Eq.7 leading term: -(d-1) Lap(log sigma) = -2 f''  -> opposite sign to computed +4 f'' sigma^2")

print("\n=== (2) Theorem 2: spatial geodesic, sigma = sigma(x), moving along y ===")
s = sp.Function('s')(x)
G = christoffel(sp.eye(3) / s**2, X)
print("x'' = -Gamma^x_yy * (y')^2, Gamma^x_yy =", G[0][1][1])
print("-> x'' = -(s'/s)(y')^2 : drifts toward SMALLER sigma = HIGHER clock rate nu ~ 1/sigma")
print("   paper: mass = depression of nu (= larger sigma)  -> geodesics move AWAY from mass (repulsion)")

print("\n=== (3) Lorentzian version, weak field, PPN gamma ===")
# nu = 1 + eps(x), |eps|<<1 ; spatial metric g_ij = nu^2 delta (Eq.4 leading order, sigma = 1/nu)
# V = nu^a (any power law relating clock budget to clock rate)
a, eps0 = sp.symbols('a epsilon', real=True)
e = sp.Function('e')(x)
nu = 1 + e
V = nu**a
gL = sp.diag(-V, nu**2, nu**2, nu**2)
GL = christoffel(gL, [t, x, y, z])
acc = sp.series(-GL[1][0][0].subs(e, eps0 * sp.Symbol('u')), sp.Symbol('u'), 0, 2).removeO()
print("slow particle: x'' = -Gamma^x_tt =", sp.simplify(-GL[1][0][0]))
print("  to first order: x'' ~ -(a/2) e'(x)")
# Newtonian potential: g_tt = -(1+2Phi) -> Phi = a*e/2 ; PPN: g_ij = 1 - 2*gamma*Phi -> 2e = -2*gamma*a*e/2
gamma = sp.solve(sp.Eq(2, -2 * sp.Symbol('gam') * a / 2), sp.Symbol('gam'))[0]
print("  Phi = a*eps/2 ;  g_ij = 1+2eps = 1-2*gamma*Phi  ->  gamma =", gamma)
print("  mass = nu depression (eps<0): attraction needs Phi<0 -> a>0 -> gamma = -2/a < 0")
print("  light deflection = (1+gamma)/2 x GR value; Cassini measured gamma = 1 +/- 2e-5")
for av in [1, 2, 4]:
    gv = -sp.Rational(2, av)
    print(f"   a={av}: gamma={gv}, deflection = {(1+gv)/2} x GR")
print("  GR (gamma=1) requires a=-2 AND eps>0 at the mass, i.e. nu HIGHER at the mass -- contradicts the paper's premise")
