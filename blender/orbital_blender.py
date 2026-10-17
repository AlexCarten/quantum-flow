"""
ORBITAL STUDIO for Blender 5.x  (@AlexCarten_)
Hydrogen orbital + Pauli probability current, rendered as millions of lit particles.

HOW TO USE
1. Blender -> Scripting tab -> Open -> this file -> Run Script (triangle button).
2. Look at the result in the viewport (Numpad 0 = camera view, Z -> Rendered).
3. Render -> Render Animation (Ctrl+F12). The MP4 goes to the OUTPUT folder below.

Change the settings in the CONFIG block and run the script again.
"""
import math, os
import numpy as np
import bpy

# ======================= CONFIG =======================
STATE        = (6, 1, 1)     # (n, l, m)   try (5,4,-1), (4,3,2), (3,2,1), (2,1,1), (1,0,0)
PARTICLES    = 3_000_000     # 1-6 million. More = denser, slower
SECONDS      = 12
FPS          = 30
FLOW_SPEED   = 1.0           # how fast the probability flows
CAM_ELEV_DEG = 28            # camera height: 0 = side view, 90 = from the top
CAM_START_DEG= 35            # where the camera starts around the atom
ORBIT_DEG    = 40            # how far the camera travels around the atom during the clip
ZOOM         = 1.15          # >1 = closer
CUT          = "upper"       # "upper" wedge, "full" wedge to the core, or "none"
WEDGE_DEG    = 90            # width of the cut
POINT_SIZE   = 1.0           # particle size multiplier
MOTION_BLUR  = True          # turns particles into short streaks
SAMPLES      = 64            # Cycles samples (denoiser is on)
RES_SCALE    = 100           # % of 1080x1920 (use 50 for fast tests)
FLY_IN       = True          # camera flies through the cut, into the nucleus
FLY_HOLD_SEC = 1.5           # seconds before the flight starts
FLY_END      = 1.2           # final camera distance (in atom radii). Smaller = deeper inside
OUTPUT       = os.path.join(os.path.expanduser("~"), "OrbitalRenders")
# ======================================================

ORANGE_LO, ORANGE_HI = np.array([0.40, 0.12, 0.01]), np.array([1.00, 0.55, 0.10])
BLUE_LO,   BLUE_HI   = np.array([0.02, 0.03, 0.35]), np.array([0.15, 0.30, 1.00])
WHITE                = np.array([1.00, 0.95, 0.85])
HOT                  = np.array([1.00, 0.08, 0.02])
LETT = "spdfghik"
rng = np.random.default_rng(7)

# ---------------- physics (atomic units) ----------------
def laguerre(k, a, x):
    if k == 0: return np.ones_like(x)
    L0, L1 = np.ones_like(x), 1 + a - x
    for i in range(1, k):
        L0, L1 = L1, ((2 * i + 1 + a - x) * L1 - (i + a) * L0) / (i + 1)
    return L1

def legendre(l, m, x):
    s = np.sqrt(np.clip(1 - x * x, 0, None))
    pmm = np.ones_like(x)
    for i in range(1, m + 1): pmm = pmm * (-(2 * i - 1)) * s
    if l == m: return pmm
    pm1 = x * (2 * m + 1) * pmm
    if l == m + 1: return pm1
    for ll in range(m + 2, l + 1):
        pmm, pm1 = pm1, (x * (2 * ll - 1) * pm1 - (ll + m - 1) * pmm) / (ll - m)
    return pm1

def radial(n, l, r):
    rho = 2 * r / n
    return rho**l * np.exp(-rho / 2) * laguerre(n - l - 1, 2 * l + 1, rho)

def density(n, l, m, s, z):
    r = np.sqrt(s * s + z * z) + 1e-12
    return radial(n, l, r)**2 * legendre(l, abs(m), z / r)**2

def sample(n, l, m, N):
    rg = np.linspace(1e-4, 2.4 * n * n + 8, 8000)
    cr = np.cumsum(rg**2 * radial(n, l, rg)**2); cr /= cr[-1]
    tg = np.linspace(1e-4, math.pi - 1e-4, 4000)
    ct = np.cumsum(np.sin(tg) * legendre(l, abs(m), np.cos(tg))**2); ct /= ct[-1]
    r = np.interp(rng.random(N), cr, rg)
    th = np.interp(rng.random(N), ct, tg)
    ph = rng.random(N) * 2 * math.pi
    s = np.maximum(r * np.sin(th), 1e-3); z = r * np.cos(th)
    r97 = np.interp(0.97, cr, rg)
    return s, z, ph, r97

def flow(n, l, m, s, z):
    """angular velocity d(phi)/dt from the Pauli current: v_phi = m/s - 1/2 d ln(rho)/ds"""
    h = 1e-3 * (1 + s)
    dl = (np.log(density(n, l, m, s + h, z) + 1e-300) - np.log(density(n, l, m, np.maximum(s - h, 1e-6), z) + 1e-300)) / (2 * h)
    return (m / s - 0.5 * dl) / s

def colors(n, l, m, s, z, om, r97):
    med = np.median(np.abs(om))
    rho = density(n, l, m, s, z)
    k = np.clip((rho / np.percentile(rho, 95))**0.25, 0, 1)[:, None]
    ccw = (om >= 0)[:, None]
    base = np.where(ccw, ORANGE_LO + (ORANGE_HI - ORANGE_LO) * k, BLUE_LO + (BLUE_HI - BLUE_LO) * k)
    slow = (np.clip(1 - np.abs(om) / (0.5 * med), 0, 1)**2)[:, None]
    hot = (np.clip(1 - np.sqrt(s * s + z * z) / (0.14 * r97), 0, 1)**1.5)[:, None]
    base = base * (1 - hot) + HOT * hot
    col = base * (1 - slow) + WHITE * slow
    return col, np.clip(om / med * 0.55, -3, 3)

# ---------------- scene ----------------
def clear_scene():
    for ob in list(bpy.data.objects): bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.node_groups):
        for b in list(coll):
            if b.users == 0: coll.remove(b)

def make_points(s, z, ph, om, col):
    N = len(s)
    me = bpy.data.meshes.new("orbital_points")
    me.vertices.add(N)
    co = np.zeros((N, 3), np.float32); co[:, 0] = s * np.cos(ph); co[:, 1] = s * np.sin(ph); co[:, 2] = z
    me.vertices.foreach_set("co", co.ravel())
    for name, arr in (("s", s), ("z", z), ("phi0", ph), ("om", om)):
        a = me.attributes.new(name, 'FLOAT', 'POINT'); a.data.foreach_set("value", arr.astype(np.float32))
    c = me.attributes.new("col", 'FLOAT_COLOR', 'POINT')
    rgba = np.ones((N, 4), np.float32); rgba[:, :3] = col
    c.data.foreach_set("color", rgba.ravel())
    me.update()
    ob = bpy.data.objects.new("Orbital", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob

def particle_material():
    mat = bpy.data.materials.new("OrbitalParticles"); mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "col"; attr.attribute_type = 'GEOMETRY'
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.22
    bsdf.inputs["Roughness"].default_value = 0.55
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def emission_material(name, rgb, strength):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1); em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat

def geo_nodes(ob, cam, mat, radius, cut_half, cut_z, d0):
    ng = bpy.data.node_groups.new("OrbitalFlow", 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N, L = ng.nodes, ng.links
    gi = N.new("NodeGroupInput"); go = N.new("NodeGroupOutput")
    def attr(name):
        a = N.new("GeometryNodeInputNamedAttribute"); a.data_type = 'FLOAT'; a.inputs["Name"].default_value = name; return a.outputs["Attribute"]
    def math(op, a, b=None, c=None):
        m = N.new("ShaderNodeMath"); m.operation = op
        for i, v in enumerate((a, b, c)):
            if v is None: continue
            if isinstance(v, (int, float)): m.inputs[i].default_value = v
            else: L.new(v, m.inputs[i])
        return m.outputs[0]
    t = N.new("GeometryNodeInputSceneTime").outputs["Seconds"]
    s, z, ph0, om = attr("s"), attr("z"), attr("phi0"), attr("om")
    ang = math('ADD', ph0, math('MULTIPLY', om, math('MULTIPLY', t, FLOW_SPEED)))
    comb = N.new("ShaderNodeCombineXYZ")
    L.new(math('MULTIPLY', s, math('COSINE', ang)), comb.inputs[0])
    L.new(math('MULTIPLY', s, math('SINE', ang)), comb.inputs[1])
    L.new(z, comb.inputs[2])
    setp = N.new("GeometryNodeSetPosition")
    L.new(gi.outputs[0], setp.inputs["Geometry"]); L.new(comb.outputs[0], setp.inputs["Position"])
    geo = setp.outputs["Geometry"]
    info = N.new("GeometryNodeObjectInfo"); info.inputs["Object"].default_value = cam
    if cut_half > 0:
        sep = N.new("ShaderNodeSeparateXYZ"); L.new(info.outputs["Location"], sep.inputs[0])
        cam_az = math('ARCTAN2', sep.outputs[1], sep.outputs[0])
        wrap = N.new("ShaderNodeMath"); wrap.operation = 'WRAP'
        L.new(math('SUBTRACT', ang, cam_az), wrap.inputs[0]); wrap.inputs[1].default_value = math_pi; wrap.inputs[2].default_value = -math_pi
        inwedge = math('LESS_THAN', math('ABSOLUTE', wrap.outputs[0]), cut_half)
        above = math('GREATER_THAN', z, cut_z)
        sel = math('MULTIPLY', inwedge, above)
        dele = N.new("GeometryNodeDeleteGeometry"); dele.domain = 'POINT'
        L.new(geo, dele.inputs["Geometry"]); L.new(sel, dele.inputs["Selection"])
        geo = dele.outputs[0]
    m2p = N.new("GeometryNodeMeshToPoints"); m2p.mode = 'VERTICES'
    L.new(geo, m2p.inputs["Mesh"])
    # constant on-screen size: radius grows with distance from the camera
    pos = N.new("GeometryNodeInputPosition")
    dist = N.new("ShaderNodeVectorMath"); dist.operation = 'DISTANCE'
    L.new(pos.outputs[0], dist.inputs[0]); L.new(info.outputs["Location"], dist.inputs[1])
    L.new(math('MULTIPLY', dist.outputs["Value"], radius / d0), m2p.inputs["Radius"])
    sm = N.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = mat
    L.new(m2p.outputs["Points"], sm.inputs["Geometry"])
    L.new(sm.outputs[0], go.inputs[0])
    mod = ob.modifiers.new("OrbitalFlow", 'NODES'); mod.node_group = ng

math_pi = math.pi

def setup_gpu():
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                devs = [d for d in prefs.devices if d.type == backend]
                if devs:
                    for d in prefs.devices: d.use = d.type == backend
                    bpy.context.scene.cycles.device = 'GPU'
                    print("Rendering on", backend, [d.name for d in devs])
                    return
            except Exception:
                pass
    except Exception:
        pass
    print("No GPU found, rendering on CPU")

def build():
    n, l, m = STATE
    scene = bpy.context.scene
    clear_scene()
    print("Sampling", STATE, PARTICLES, "particles…")
    s, z, ph, r97 = sample(n, l, m, PARTICLES)
    om = flow(n, l, m, s, z)
    col, omv = colors(n, l, m, s, z, om, r97)
    ob = make_points(s, z, ph, omv, col)

    # camera on a pivot that orbits the atom
    pivot = bpy.data.objects.new("CameraPivot", None); scene.collection.objects.link(pivot)
    camd = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Camera", camd); scene.collection.objects.link(cam)
    D = 3.2 * r97 / ZOOM
    el = math.radians(CAM_ELEV_DEG)
    cam.location = (D * math.cos(el), 0, D * math.sin(el)); cam.parent = pivot
    tr = cam.constraints.new('TRACK_TO'); tr.target = pivot; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
    camd.sensor_fit = 'HORIZONTAL'; camd.angle = 2 * math.atan(1.02 * r97 / (D * ZOOM) * ZOOM)
    camd.clip_start = D * 0.02; camd.clip_end = D * 6
    camd.shift_y = 0.05
    scene.camera = cam
    frames = int(SECONDS * FPS)
    pivot.rotation_euler = (0, 0, math.radians(CAM_START_DEG)); pivot.keyframe_insert("rotation_euler", frame=1)
    pivot.rotation_euler = (0, 0, math.radians(CAM_START_DEG + ORBIT_DEG)); pivot.keyframe_insert("rotation_euler", frame=frames)
    try:
        for fc in pivot.animation_data.action.fcurves:
            for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'
    except Exception:
        try:
            for layer in pivot.animation_data.action.layers:
                for strip in layer.strips:
                    for cb in strip.channelbags:
                        for fc in cb.fcurves:
                            for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'
        except Exception:
            pass

    if FLY_IN:
        start = cam.location.copy()
        cam.keyframe_insert("location", frame=1)
        cam.keyframe_insert("location", frame=1 + int(FLY_HOLD_SEC * FPS))
        cam.location = start * (FLY_END * r97 / D)
        cam.keyframe_insert("location", frame=frames)
        camd.clip_start = r97 * 0.002

    cut_half = 0 if CUT == "none" else math.radians(WEDGE_DEG) / 2
    cut_z = -1e9 if CUT == "full" else 0.0
    geo_nodes(ob, cam, particle_material(), 0.0032 * r97 * POINT_SIZE, cut_half, cut_z, D)

    # nucleus
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.004 * r97, location=(0, 0, 0))
    nuc = bpy.context.active_object; nuc.name = "Nucleus"
    nuc.data.materials.append(emission_material("Nucleus", (1.0, 0.1, 0.05), 12))

    # lights: key light above the camera side, cool rim from behind
    def area(name, loc, energy, size, rgb):
        ld = bpy.data.lights.new(name, 'AREA'); ld.energy = energy; ld.size = size; ld.color = rgb
        lo = bpy.data.objects.new(name, ld); lo.location = loc; lo.parent = pivot; scene.collection.objects.link(lo)
        t = lo.constraints.new('TRACK_TO'); t.target = pivot; t.track_axis = 'TRACK_NEGATIVE_Z'; t.up_axis = 'UP_Y'
    sc = (r97 / 10.0)**2
    area("KeyLight", (D * 0.7, D * 0.6, D * 0.9), 2200 * sc, r97 * 0.8, (1.0, 0.92, 0.82))
    area("RimLight", (-D * 0.9, -D * 0.3, D * 0.3), 900 * sc, r97 * 1.2, (0.75, 0.82, 1.0))

    world = scene.world or bpy.data.worlds.new("World"); scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg: bg.inputs[0].default_value = (0, 0, 0, 1); bg.inputs[1].default_value = 0.0

    # render settings
    scene.render.engine = 'CYCLES'
    setup_gpu()
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1080, 1920
    scene.render.resolution_percentage = RES_SCALE
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.use_motion_blur = MOTION_BLUR
    scene.render.motion_blur_shutter = 1.0
    try: scene.view_settings.view_transform = 'Standard'; scene.view_settings.look = 'None'
    except Exception: pass
    scene.view_settings.exposure = 0.0
    os.makedirs(OUTPUT, exist_ok=True)
    name = f"orbital_{n}{LETT[l]}_m{m:+d}"
    scene.render.filepath = os.path.join(OUTPUT, name + "_")
    ims = scene.render.image_settings
    try:
        ims.media_type = 'VIDEO'
    except Exception:
        pass
    ims.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    print(f"Ready: {name}, {frames} frames -> {OUTPUT}")

build()
