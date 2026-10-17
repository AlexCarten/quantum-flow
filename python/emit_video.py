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
    age += 1
    hist = [P.copy()] + hist[:-1]
    V = vel(P, c1, c2, wt)
    sp = np.linalg.norm(V, axis=1); med = np.median(sp)
    sxy = np.sqrt(P[:, 0]**2 + P[:, 1]**2) + 1e-6
    vphi = (-P[:, 1] * V[:, 0] + P[:, 0] * V[:, 1]) / sxy
    slow = np.clip(1 - sp / (0.45 * med), 0, 1)**2
    col = np.where(vphi[:, None] >= 0, ORANGE, BLUE) * (1 - slow[:, None]) + WHITE * slow[:, None]
    fadein = np.clip(age / 12, 0, 1)
    tilt = 0.95 + 0.18 * math.sin(t * 0.21)
    rmax = 11.0 - 4.5 * pr
    scale = 0.48 * W / rmax
    cy0 = H * 0.54
    S = 3
    IDX = []; WG = []
    ct, st = math.cos(tilt), math.sin(tilt)
    for k in range(len(hist) - 1):
        A, B = hist[k], hist[k + 1]
        for j in range(S):
            al = j / S
            Q = A * (1 - al) + B * al
            wgt = (1 - (k + al) / (len(hist) - 1)) ** 0.7 * fadein
            zz = Q[:, 2] * ct + Q[:, 1] * st
            px = (W / 2 + Q[:, 0] * scale).astype(np.int32); py = (cy0 - zz * scale).astype(np.int32)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            IDX.append(py[ok] * W + px[ok]); WG.append((wgt[ok], ok))
    idx = np.concatenate(IDX)
    acc = np.empty((H, W, 3), np.float32)
    for c in range(3):
        w = np.concatenate([wg * col[ok, c] for wg, ok in WG])
        acc[..., c] = np.bincount(idx, weights=w, minlength=W * H).reshape(H, W)
    lum = acc.max(-1)
    ref = np.percentile(lum[::4, ::4][lum[::4, ::4] > 0], 99.5)
    ref_ema = ref if ref_ema is None else ref_ema * 0.9 + ref * 0.1
    r = lum / ref_ema
    a = acc / (lum + 1e-6)[..., None] * (r ** 0.45)[..., None]
    small = a[::4, ::4]
    gs = gaussian_filter(small, sigma=(2.5, 2.5, 0))
    glow = np.repeat(np.repeat(gs, 4, 0), 4, 1)[:H, :W]
    img = 1 - np.exp(-(a * 2.6 + glow * 0.9))
    # light at quarter res
    xs, ys = xx[::2, ::2], yy[::2, ::2]
    dx = (xs - W / 2) / scale; dyp = (cy0 - ys) / scale / ct
    R = np.sqrt(dx**2 + dyp**2) + 1e-6; PH = np.arctan2(dyp, dx)
    lag = np.clip((R * scale / 260.0 * FPS).astype(np.int32), 0, len(history_prog) - 1)
    hp = np.array(history_prog)[::-1]
    pret = hp[lag]
    amp = 2 * np.sqrt(np.clip(pret * (1 - pret), 0, None))
    wave = np.cos(0.42 * R + PH - wt)**60 * np.clip((R - (rmax + 2)) / 4, 0, 1) * np.exp(-(R - rmax) / 30) * amp
    wave = np.repeat(np.repeat(wave, 2, 0), 2, 1)[:H, :W]
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
