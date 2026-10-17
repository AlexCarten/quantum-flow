"""Hydrogen 2p(m=0) -> 1s emission: superposition c1*psi100 + c2*psi210*e^{-i w t}.
Atomic units. Velocity = Im(grad psi / psi) + spin term 1/2 (d_y ln rho, -d_x ln rho, 0)."""
import sys, math
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1080, 1920
import os
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts") + os.sep
ORANGE = np.array([1.0, 0.50, 0.03]); BLUE = np.array([0.10, 0.22, 0.95]); WHITE = np.array([1.0, 0.97, 0.9])
DE = 3 / 8  # Hartree, 2p-1s gap
rng = np.random.default_rng(11)

def psi(x, y, z, c1, c2, wt):
    r = np.sqrt(x*x + y*y + z*z) + 1e-9
    p1 = np.exp(-r) / math.sqrt(math.pi)
    p2 = (x - 1j * y) * np.exp(-r / 2) / (8 * math.sqrt(math.pi))
    return c1 * p1 + c2 * p2 * np.exp(-1j * wt)

def sample(c1, c2, wt, N, L=26.0, G=170):
    g = np.linspace(-L, L, G); d = g[1] - g[0]
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    rho = np.abs(psi(X, Y, Z, c1, c2, wt))**2
    p = rho.ravel() / rho.sum()
    idx = rng.choice(p.size, N, p=p)
    pts = np.stack([X.ravel()[idx], Y.ravel()[idx], Z.ravel()[idx]], 1)
    return pts + (rng.random((N, 3)) - 0.5) * d

def velocity(P, c1, c2, wt, h=1e-3):
    x, y, z = P.T
    p0 = psi(x, y, z, c1, c2, wt)
    rho0 = np.abs(p0)**2 + 1e-300
    v = np.zeros_like(P); gl = np.zeros_like(P)
    for k in range(3):
        e = np.zeros(3); e[k] = h
        pp = psi(*(P + e).T, c1, c2, wt); pm = psi(*(P - e).T, c1, c2, wt)
        g = (pp - pm) / (2 * h)
        v[:, k] = np.imag(g / p0)
        gl[:, k] = (np.abs(pp)**2 - np.abs(pm)**2) / (2 * h) / rho0
    v[:, 0] += 0.5 * gl[:, 1]; v[:, 1] -= 0.5 * gl[:, 0]
    return v

def F(name, sz): return ImageFont.truetype(name, sz)
INTER = "/usr/share/fonts/truetype/inter/Inter-Bold.ttf"
def inter(sz, w="Bold"):
    import glob
    c = (glob.glob(f"/usr/share/fonts/**/Inter-{w}.*", recursive=True) or glob.glob("/usr/share/fonts/**/Inter*.tt*", recursive=True)
         or glob.glob("C:/Windows/Fonts/arialbd.ttf") or glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True))
    return ImageFont.truetype(c[0], sz) if c else ImageFont.load_default()

def chalk_panel(prog):
    """Top-right blackboard inset: chalk Bohr-style sketch of the jump."""
    pw, ph = 330, 330
    im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, pw - 1, ph - 1), 22, fill=(22, 34, 28, 235), outline=(120, 110, 90, 255), width=4)
    cx, cy = 150, 175
    ch = (235, 235, 225, 235)
    for rad, w in ((48, 3), (100, 3)):
        for k in range(3):  # rough chalk circles
            j = rng.normal(0, 1.6, 2)
            d.ellipse((cx - rad + j[0], cy - rad + j[1], cx + rad + j[0], cy + rad + j[1]), outline=ch, width=1 + (k == 0))
    d.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill=(244, 33, 46, 255))
    # electron dropping 2 -> 1
    a = math.radians(-40); ex, ey = cx + 100 * math.cos(a), cy + 100 * math.sin(a)
    bx, by = cx + 48 * math.cos(a), cy + 48 * math.sin(a)
    d.line((ex, ey, bx + 8, by - 6), fill=ch, width=3)
    d.polygon([(bx, by), (bx + 16, by - 4), (bx + 6, by - 15)], fill=ch)
    d.ellipse((ex - 8, ey - 8, ex + 8, ey + 8), fill=(255, 158, 0, 255))
    # photon squiggle
    pts = [(ex + 10 + t * 2.2, ey - 30 - t * 1.4 + 9 * math.sin(t * 0.45)) for t in range(0, 70)]
    d.line(pts, fill=(255, 210, 120, 240), width=3)
    d.text((22, 14), "what school draws", font=F(FD + "Caveat-Bold.ttf", 34), fill=ch)
    d.text((215, 70), "light", font=F(FD + "Caveat.ttf", 34), fill=(255, 210, 120, 240))
    d.text((cx + 108, cy + 20), "n=2", font=F(FD + "Caveat.ttf", 28), fill=ch)
    d.text((cx + 10, cy + 40), "n=1", font=F(FD + "Caveat.ttf", 28), fill=ch)
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    return im

def overlay(prog, wt):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rectangle((60, 84, 110, 89), fill=(255, 158, 0, 255))
    d.text((60, 104), "HYDROGEN / LIGHT EMISSION", font=inter(24), fill=(235, 235, 235, 255))
    d.text((58, 150), "2p → 1s", font=inter(118, "Bold"), fill=(255, 255, 255, 255))
    d.rounded_rectangle((60, 300, 560, 352), 8, fill=(255, 255, 255, 22))
    d.text((78, 310), "ΔE = 10.2 eV    λ = 121.6 nm", font=inter(28, "Medium"), fill=(235, 235, 235, 255))
    im.alpha_composite(chalk_panel(prog), (W - 390, 70))
    # transition bar
    y = H - 330
    d.text((60, y), "ELECTRON IN 2p", font=inter(22), fill=(200, 200, 200, 255))
    d.text((W - 60 - d.textlength("1s", font=inter(22)), y), "1s", font=inter(22), fill=(200, 200, 200, 255))
    d.rounded_rectangle((60, y + 38, W - 60, y + 46), 4, fill=(255, 255, 255, 40))
    d.rounded_rectangle((60, y + 38, 60 + (W - 120) * prog, y + 46), 4, fill=(255, 158, 0, 255))
    y = H - 230
    d.text((60, y), "DIRECTION OF FLOW", font=inter(22), fill=(235, 235, 235, 255))
    d.text((60, y + 30), "VIEWED FROM +Z", font=inter(18, "Medium"), fill=(150, 150, 150, 255))
    for i, (c, t) in enumerate((((255, 144, 13), "ORANGE / COUNTERCLOCKWISE"), ((26, 56, 242), "BLUE / CLOCKWISE"), ((255, 248, 230), "WHITE / SLOW"))):
        yy = y + 72 + i * 34
        d.ellipse((62, yy + 4, 76, yy + 18), fill=c + (255,)); d.text((92, yy), t, font=inter(21, "Medium"), fill=(220, 220, 220, 255))
    d.text((W - 60 - d.textlength("@AlexCarten_", font=inter(30)), H - 90), "@AlexCarten_", font=inter(30), fill=(255, 255, 255, 190))
    return np.asarray(im).astype(np.float32) / 255

def render_frame(prog, wt, N=170000, trail=9, yaw=0.0, tilt=0.95):
    c2 = math.sqrt(max(1e-4, 1 - prog)); c1 = math.sqrt(max(1e-4, prog))
    P = sample(c1, c2, wt, N)
    V = velocity(P, c1, c2, wt)
    sp = np.linalg.norm(V, axis=1); med = np.median(sp)
    sxy = np.sqrt(P[:, 0]**2 + P[:, 1]**2) + 1e-6
    vphi = (-P[:, 1] * V[:, 0] + P[:, 0] * V[:, 1]) / sxy
    slow = np.clip(1 - sp / (0.45 * med), 0, 1)**2
    col = np.where(vphi[:, None] >= 0, ORANGE, BLUE) * (1 - slow[:, None]) + WHITE * slow[:, None]
    rmax = 11.0; scale = 0.48 * W / rmax
    cy0 = H * 0.54
    acc = np.zeros((H, W, 3), np.float32)
    Vn = V / (sp[:, None] + 1e-9) * np.clip(sp / med, 0.3, 2.0)[:, None] * 0.07
    for k in range(trail):
        Q = P - Vn * k
        x = Q[:, 0]; y = Q[:, 1]
        zz = Q[:, 2] * math.cos(tilt) + y * math.sin(tilt)
        px = (W / 2 + x * scale).astype(np.int32); py = (cy0 - zz * scale).astype(np.int32)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        idx = py[ok] * W + px[ok]; wgt = (1 - k / trail) ** 0.7
        for c in range(3):
            acc[..., c].flat[:] += np.bincount(idx, weights=col[ok, c] * wgt, minlength=W * H)
    lum = acc.max(-1)
    ref = np.percentile(lum[lum > 0], 99.5)
    a = acc / ref
    a = a / (lum / ref + 1e-6)[..., None] * (np.clip(lum / ref, 0, None) ** 0.45)[..., None]
    glow = gaussian_filter(a, sigma=(10, 10, 0))
    img = 1 - np.exp(-(a * 2.6 + glow * 0.9))
    # circularly polarized light from the rotating dipole: spiral wavefronts in the orbit plane
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (xx - W / 2) / scale; dyp = (cy0 - yy) / scale / math.cos(tilt)
    R = np.sqrt(dx**2 + dyp**2) + 1e-6; PH = np.arctan2(dyp, dx)
    wave = np.cos(0.42 * R + PH - wt)**60 * np.clip((R - 13) / 4, 0, 1) * np.exp(-(R - 11) / 30) * (1 - prog)**0.5
    light = np.stack([wave, wave * 0.85, wave * 0.6], -1) * 0.4
    img = 1 - (1 - img) * (1 - light)
    nuc = np.exp(-(((xx - W / 2)**2 + (yy - cy0)**2) / (2 * 6**2)))
    nug = gaussian_filter(nuc, 12)[..., None] * np.array([1.0, 0.08, 0.1]) * 3
    img = 1 - (1 - img) * (1 - np.clip(nug, 0, 1))
    img = img * (1 - nuc[..., None]) + np.array([1.0, 0.15, 0.18]) * nuc[..., None]
    ov = overlay(prog, wt)
    img = img * (1 - ov[..., 3:]) + ov[..., :3] * ov[..., 3:]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

if __name__ == "__main__":
    out = sys.argv[1]
    Image.fromarray(render_frame(0.5, math.pi / 4)).save(out)
