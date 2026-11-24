# C21: how typical is it for an observer to live near H t = 1 (the c_com t stationary point)?
# Observers ∝ cosmic star formation rate (Madau & Dickinson 2014: psi ∝ (1+z)^2.7 / (1 + ((1+z)/2.9)^5.6), formula from memory)
# shifted by a delay D (time from star formation to observers; Earth ~ 4.5 Gyr).  Future: SFR formula extrapolated to z<0 (declining).
# PRE-REGISTERED: for D = 4.5 Gyr, P(|Ht - 1| < 0.05) ~ 0.2, median Ht ~ 1.0  -> the coincidence is not very surprising for such observers.
import numpy as np

H0 = 67.4 / 977.792; Om, Or = 0.315, 9.1e-5; OL = 1 - Om - Or
lna = np.linspace(np.log(1e-4), np.log(30.0), 400001); a = np.exp(lna)
H = H0 * np.sqrt(Or / a**4 + Om / a**3 + OL)
t = np.concatenate([[0], np.cumsum(0.5 * (1/H[1:] + 1/H[:-1]) * np.diff(lna))])
z = 1/a - 1
psi = (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)
psi[z < -0.999] = 0
# SFR per unit cosmic time on a uniform t grid
tg = np.linspace(0.05, 200, 200001)
sfr = np.interp(tg, t, psi)
Ht_of_t = lambda tt: np.interp(tt, t, H * t)
print(f"today: t0 = {np.interp(1.0, a, t):.2f} Gyr, H0 t0 = {np.interp(1.0, a, H*t):.4f}")
for D in [0.0, 2.0, 4.5, 8.0]:
    tobs = tg + D
    w = sfr / sfr.sum()
    Ht = Ht_of_t(tobs)
    p05 = w[np.abs(Ht - 1) < 0.05].sum()
    p_le = w[Ht <= np.interp(1.0, a, H*t)].sum()          # fraction of observers who see Ht <= ours
    order = np.argsort(Ht); cdf = np.cumsum(w[order]); med = Ht[order][np.searchsorted(cdf, 0.5)]
    tmed = tobs[np.argsort(tobs)][np.searchsorted(np.cumsum(w[np.argsort(tobs)]), 0.5)]
    print(f"D={D:4.1f} Gyr: P(|Ht-1|<0.05) = {p05:.3f}, median Ht = {med:.3f}, median t_obs = {tmed:.1f} Gyr, "
          f"fraction seeing Ht <= ours (0.951) = {p_le:.3f}")
print("note: SFR beyond z<0 is an extrapolation; results for late times depend on it")

# network time (C2 correction): p_loc = 2 H T / a with T = ∫ a dtau.  Today p_loc = 1.054.
T = np.concatenate([[0], np.cumsum(0.5 * ((a/H)[1:] + (a/H)[:-1]) * np.diff(lna))])
ploc = 2 * H * T / a
p_today = np.interp(1.0, a, ploc)
print(f"\nnetwork-time exponent today p_loc = {p_today:.3f}")
for D in [0.0, 4.5, 8.0]:
    tobs = tg + D; w = sfr / sfr.sum()
    pl = np.interp(tobs, t, ploc)
    print(f"D={D:4.1f} Gyr: P(|p_loc-1|<0.05) = {w[np.abs(pl-1)<0.05].sum():.3f}, fraction seeing p_loc <= ours = {w[pl<=p_today].sum():.3f}")
