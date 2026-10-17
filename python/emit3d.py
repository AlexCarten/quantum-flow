import sys, math, subprocess, time
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
import emission as E
from emission import W, H, psi, velocity, ORANGE, BLUE, WHITE

FPS = 30; T = 38.0; NF = int(T * FPS)
N = 170000
rng = np.random.default_rng(3)
OMEGA_VIS = 2 * math.pi / 3.2          # one lobe rotation every 3.2 s
DT = OMEGA_VIS / FPS / E.DE             # atomic-time per frame

def prog_at(t):
    if t < 4: return 0.0
    if t > 31: return 1.0
    u = (t - 4) / 27
    return u * u * (3 - 2 * u)

def sample_grid(c1, c2, wt, n, L=24.0, G=120):
    g = np.linspace(-L, L, G); d = g[1] - g[0]
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    rho = np.abs(psi(X, Y, Z, c1, c2, wt))**2
    p = rho.ravel() / rho.sum()
    idx = rng.choice(p.size, n, p=p)
    pts = np.stack([X.ravel()[idx], Y.ravel()[idx], Z.ravel()[idx]], 1)
    return pts + (rng.random((n, 3)) - 0.5) * d

def vel(P, c1, c2, wt):
    V = velocity(P, c1, c2, wt)
    sp = np.linalg.norm(V, axis=1, keepdims=True)
    return V * np.minimum(1, 4.0 / (sp + 1e-9))

# static overlay (no progress bar fill)
base_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ovA = E.overlay(0.0, 0.0)  # includes empty bar
ovA_rgb, ovA_a = ovA[..., :3], ovA[..., 3:]
bar_y = H - 330 + 38

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

_n = np.exp(-(((xx - W / 2)**2 + (yy - H * 0.54)**2) / (2 * 6**2)))
NUC = _n[..., None]
NUG = np.clip(gaussian_filter(_n, 12)[..., None] * np.array([1.0, 0.08, 0.1]) * 3, 0, 1)
def cs(p): return math.sqrt(max(1e-4, p)), math.sqrt(max(1e-4, 1 - p))

c1, c2 = cs(0.0); wt = 0.0
P = E.sample(c1, c2, wt, N)
hist = [P.copy() for _ in range(4)]
age = np.ones(N) * 99
ref_ema = None
history_prog = []

proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                         "-pix_fmt", "yuv420p", "-movflags", "+faststart", sys.argv[1]], stdin=subprocess.PIPE)
t0 = time.time()
for f in range(NF):
    t = f / FPS
    pr = prog_at(t); history_prog.append(pr)
    c1, c2 = cs(pr)
    # advect (RK2, 2 substeps)
    h = DT / 2
    for _ in range(2):
        k1 = vel(P, c1, c2, wt)
        k2 = vel(P + k1 * h, c1, c2, wt + E.DE * h)
        P = P + k2 * h; wt += E.DE * h
    # gradual resampling to follow the decay
    if 0 < pr < 1 or (t > 31 and t < 34.5):
        m = int(N * 0.02)
        ii = rng.choice(N, m, replace=False)
        newp = sample_grid(c1, c2, wt, m)
        P[ii] = newp
        for hh in hist: hh[ii] = newp
        age[ii] = 0
    # respawn particles stuck in near-empty regions (numerical sheets) or with clipped speed
    Vr = velocity(P, c1, c2, wt)
    spr = np.linalg.norm(Vr, axis=1)
    rh = np.abs(psi(P[:, 0], P[:, 1], P[:, 2], c1, c2, wt))**2
    bad = (spr > 3.5) | (rh < 2e-4 * rh.max())
    nb = int(bad.sum())
    if nb:
        newp = sample_grid(c1, c2, wt, nb)
        P[bad] = newp
        for hh in hist: hh[bad] = newp
        age[bad] = 0
    age += 1
    hist = [P.copy()] + hist[:-1]
    V = vel(P, c1, c2, wt)
    sp = np.linalg.norm(V, axis=1); med = np.median(sp)
    sxy = np.sqrt(P[:, 0]**2 + P[:, 1]**2) + 1e-6
    vphi = (-P[:, 1] * V[:, 0] + P[:, 0] * V[:, 1]) / sxy
    slow = np.clip(1 - sp / (0.45 * med), 0, 1)**2
    col = np.where(vphi[:, None] >= 0, ORANGE, BLUE) * (1 - slow[:, None]) + WHITE * slow[:, None]
    fadein = np.clip(age / 12, 0, 1)
    el = 0.62 + 0.10 * math.sin(t * 0.17)
    yaw = 0.35 + t * 0.06
    se, ce = math.sin(el), math.cos(el); cy_, sy_ = math.cos(yaw), math.sin(yaw)
    rmax = 11.0 - 4.5 * pr
    scale = 0.50 * W / rmax
    cy0 = H * 0.54
    Dcam = 3.2 * rmax
    lam = 0.16 * rmax
    # cutaway wedge (front quadrant), decided on current position
    xr0 = P[:, 0] * cy_ + P[:, 1] * sy_; yr0 = -P[:, 0] * sy_ + P[:, 1] * cy_
    cutk = ~((xr0 > 0) & (yr0 < 0) & (P[:, 2] > 0))
    axisw = np.clip(np.sqrt(P[:, 0]**2 + P[:, 1]**2) / 1.2, 0, 1)**2
    S = 3
    IDX = []; WC = []
    for k in range(len(hist) - 1):
        A, B = hist[k], hist[k + 1]
        for j in range(S):
            al = j / S
            Q = A * (1 - al) + B * al
            tw = (1 - (k + al) / (len(hist) - 1)) ** 0.7 * fadein * axisw
            xr = Q[:, 0] * cy_ + Q[:, 1] * sy_; yr = -Q[:, 0] * sy_ + Q[:, 1] * cy_
            sx = xr; syv = yr * se + Q[:, 2] * ce
            c = -yr * ce + Q[:, 2] * se
            pf = Dcam / (Dcam - c)
            px = (W / 2 + sx * pf * scale).astype(np.int32); py = (cy0 - syv * pf * scale).astype(np.int32)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H) & cutk
            IDX.append(py[ok] * W + px[ok]); WC.append((tw[ok], c[ok], ok))
    idx = np.concatenate(IDX)
    tw = np.concatenate([w for w, _, _ in WC]); cz = np.concatenate([c for _, c, _ in WC])
    okm = [o for _, _, o in WC]
    wd = np.exp((cz - rmax * 1.5) / lam) * tw
    shade = 0.6 + 0.4 * np.clip((cz + rmax) / (2 * rmax), 0, 1)
    Wsum = np.bincount(idx, weights=wd, minlength=W * H)
    cov = np.bincount(idx, weights=tw, minlength=W * H)
    acc = np.empty((H, W, 3), np.float32)
    for ch in range(3):
        cc = np.concatenate([col[o, ch] for o in okm]) * shade
        acc[..., ch] = (np.bincount(idx, weights=wd * cc, minlength=W * H) / (Wsum + 1e-30)).reshape(H, W)
    cov = cov.reshape(H, W)
    ref = np.percentile(cov[::4, ::4][cov[::4, ::4] > 0], 97)
    ref_ema = ref if ref_ema is None else ref_ema * 0.9 + ref * 0.1
    alpha = 1 - np.exp(-(cov / ref_ema) ** 0.6 * 3.0)
    a = acc * alpha[..., None]
    small = a[::4, ::4]
    gs = gaussian_filter(small, sigma=(3, 3, 0))
    glow = np.repeat(np.repeat(gs, 4, 0), 4, 1)[:H, :W]
    img = np.clip(a * 1.35, 0, 1)
    img = 1 - (1 - img) * (1 - glow * 0.5)
    # light: spiral wavefronts in the orbit plane (ellipse from elevation)
    xs, ys = xx[::2, ::2], yy[::2, ::2]
    dx = (xs - W / 2) / scale; dyp = (cy0 - ys) / scale / se
    R = np.sqrt(dx**2 + dyp**2) + 1e-6; PH = np.arctan2(dyp, dx) + yaw
    lag = np.clip((R * scale / 260.0 * FPS).astype(np.int32), 0, len(history_prog) - 1)
    hp = np.array(history_prog)[::-1]
    pret = hp[lag]
    amp = 2 * np.sqrt(np.clip(pret * (1 - pret), 0, None))
    wave = np.cos(0.42 * R + PH - wt)**60 * np.clip((R - (rmax + 2)) / 4, 0, 1) * np.exp(-(R - rmax) / 30) * amp
    wave = np.repeat(np.repeat(wave, 2, 0), 2, 1)[:H, :W]
    wave = wave * (1 - alpha) ** 2
    img = 1 - (1 - img) * (1 - np.stack([wave, wave * 0.85, wave * 0.6], -1) * 0.45)
    img = 1 - (1 - img) * (1 - NUG)
    img = img * (1 - NUC) + np.array([1.0, 0.15, 0.18]) * NUC
    img = img * (1 - ovA_a) + ovA_rgb * ovA_a
    x1 = 60 + int((W - 120) * pr)
    if x1 > 62: img[bar_y:bar_y + 8, 60:x1] = np.array([1.0, 0.62, 0.0])
    fade = min(1, f / 20, (NF - 1 - f) / 25)
    img *= fade
    proc.stdin.write((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())
    if f % 60 == 0: print(f, round(time.time() - t0), flush=True)
proc.stdin.close(); proc.wait()
print("done", round(time.time() - t0))
