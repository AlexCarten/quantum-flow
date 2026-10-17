"""Hydrogen orbitals with Pauli probability current, rendered as flowing particles.
Atomic units (hbar = m_e = a0 = 1). Spin-up electron.
Velocity field v = j / rho, with
  j_orb  = Im(psi* grad psi)          -> (m / s) phi_hat
  j_spin = 1/2 curl(rho z_hat)        -> -(1/2) d(rho)/ds phi_hat
so v_phi = m/s - 0.5 * d ln(rho)/ds  (pure azimuthal flow around z).
"""
import sys, subprocess, math
import numpy as np
from scipy.special import genlaguerre, lpmv, factorial
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1350, 30
rng = np.random.default_rng(7)

def radial(n, l, r):
    rho = 2 * r / n
    return rho**l * np.exp(-rho / 2) * genlaguerre(n - l - 1, 2 * l + 1)(rho)

def dens(n, l, m, r, ct):
    return radial(n, l, r)**2 * lpmv(abs(m), l, ct)**2

def sample(n, l, m, N):
    rmax = 2.2 * n * n + 6
    rg = np.linspace(1e-4, rmax, 6000)
    pr = rg**2 * radial(n, l, rg)**2
    cdf = np.cumsum(pr); cdf /= cdf[-1]
    r = np.interp(rng.random(N), cdf, rg)
    tg = np.linspace(1e-4, np.pi - 1e-4, 4000)
    pt = np.sin(tg) * lpmv(abs(m), l, np.cos(tg))**2
    cdt = np.cumsum(pt); cdt /= cdt[-1]
    th = np.interp(rng.random(N), cdt, tg)
    ph = rng.random(N) * 2 * np.pi
    s = r * np.sin(th); z = r * np.cos(th)
    return s, z, ph

def omega(n, l, m, s, z):
    h = 1e-3 * (1 + s)
    def lnr(ss):
        rr = np.sqrt(ss**2 + z**2)
        return np.log(dens(n, l, m, rr, z / rr) + 1e-300)
    dln = (lnr(s + h) - lnr(np.maximum(s - h, 1e-6))) / (2 * h)
    vphi = m / np.maximum(s, 1e-3) - 0.5 * dln
    return vphi / np.maximum(s, 1e-3)

ORANGE = np.array([1.0, 0.62, 0.0])
BLUE = np.array([0.114, 0.608, 0.941])
WHITE = np.array([1.0, 1.0, 1.0])
LETT = "spdfgh"

def font(sz, bold=False):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(p, sz)

def overlay(n, l, m, alpha):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    a = int(255 * alpha)
    d.text((70, 70), "HYDROGEN  ·  PAULI CURRENT", font=font(26), fill=(200, 200, 200, a))
    d.text((70, 110), f"{n}{LETT[l]}", font=font(110, True), fill=(255, 255, 255, a))
    d.text((72, 240), f"n = {n}    l = {l}    m = {m:+d}" if m else f"n = {n}    l = {l}    m = 0",
           font=font(34), fill=(220, 220, 220, a))
    y = H - 170
    d.text((70, y), "DIRECTION OF FLOW  (viewed from +z)", font=font(24), fill=(170, 170, 170, a))
    d.ellipse((72, y + 48, 92, y + 68), fill=(255, 158, 0, a)); d.text((106, y + 44), "counterclockwise", font=font(26), fill=(230, 230, 230, a))
    d.ellipse((72, y + 88, 92, y + 108), fill=(29, 155, 240, a)); d.text((106, y + 84), "clockwise", font=font(26), fill=(230, 230, 230, a))
    d.text((W - 300, H - 60), "@AlexCarten_", font=font(28, True), fill=(255, 255, 255, int(a * 0.7)))
    return np.asarray(im).astype(np.float32) / 255

def render_segment(proc, n, l, m, secs, N=220000, frame0=0):
    s, z, ph = sample(n, l, m, N)
    om = omega(n, l, m, s, z)
    med = np.median(np.abs(om))
    dphi = np.clip(om / med * 0.05, -0.22, 0.22)
    sign = np.sign(om)
    speed = np.clip(np.abs(om) / (2.5 * med), 0, 1)
    col = np.where(sign[:, None] > 0, ORANGE, BLUE) * (0.75 + 0.25 * speed[:, None]) \
          + WHITE * (0.18 * (1 - speed[:, None]))
    rr = np.sqrt(s**2 + z**2)
    rmax = np.percentile(rr, 97)
    scale = 0.42 * W / rmax
    frames = int(secs * FPS)
    ov_cache = {}
    trail = np.zeros((H, W, 3), np.float32)
    gain = 900.0 / N * (W * H / 1.4e6)
    for f in range(frames):
        ph = ph + dphi
        t = (frame0 + f) / FPS
        u = f / max(frames - 1, 1)
        tilt = 1.15 + 0.2 * math.sin(t * 0.35)
        yaw = t * 0.10
        x = s * np.cos(ph + yaw); y = s * np.sin(ph + yaw)
        yy = y * math.cos(tilt) - z * math.sin(tilt)
        zz = y * math.sin(tilt) + z * math.cos(tilt)
        # cutaway: slab in view depth shrinks in the middle of the segment
        cut = 1.0 - 0.82 * math.sin(math.pi * min(1, max(0, (u - 0.25) / 0.6)))**2
        px = (W / 2 + x * scale).astype(np.int32)
        py = (H / 2 + 40 - yy * scale).astype(np.int32)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (np.abs(zz) < cut * rmax * 1.2)
        idx = py[ok] * W + px[ok]
        acc = np.zeros((H, W, 3), np.float32)
        for c in range(3):
            acc[..., c].flat[:] += np.bincount(idx, weights=col[ok, c], minlength=W * H)
        trail = trail * 0.78 + acc
        glow = gaussian_filter(trail, sigma=(3, 3, 0))
        dens_boost = 1.0 / max(cut, 0.25) ** 0.6
        img = 1 - np.exp(-(trail * 0.9 + glow * 1.6) * gain * 38 * dens_boost)
        fade = min(1, f / 15, (frames - 1 - f) / 15)
        img *= fade
        bg = np.array([0.015, 0.018, 0.024], np.float32)
        img = bg + img * (1 - bg)
        key = round(fade, 2)
        if key not in ov_cache: ov_cache[key] = overlay(n, l, m, fade)
        ov = ov_cache[key]
        img = img * (1 - ov[..., 3:]) + ov[..., :3] * ov[..., 3:]
        proc.stdin.write((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())
    return frame0 + frames

if __name__ == "__main__":
    out = sys.argv[1]
    states = [(1, 0, 0), (3, 1, 1), (4, 2, 0), (5, 3, 1), (6, 4, -2)]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 7
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    f0 = 0
    for st in states:
        f0 = render_segment(proc, *st, secs, frame0=f0)
        print("done", st, flush=True)
    proc.stdin.close(); proc.wait()
