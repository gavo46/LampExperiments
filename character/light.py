"""
Gives the lamp an actual emitted light that can be recolored/dimmed at
runtime to convey emotion.

--- What investigation found ---
mujoco.mj_name2id(..., mjOBJ_MATERIAL, "fixture_light") returns nothing
useful because there's nothing to find: loading
robot/dummy_lamp_5dof.urdf produces model.nmat == 0. MuJoCo's URDF
importer never creates <material> assets at all - every named URDF
<material> gets flattened straight into a baked geom_rgba instead.

Digging further with mujoco.MjSpec (which exposes the pre-compile
model tree) turned up the rest of the picture:

  - Each <link>'s several <visual> tags collapse into a *single* geom,
    shaped like that link's <collision> geometry but colored with
    whichever <visual> came *last* in that link. lamp_head_link's last
    <visual> used "fixture_light", and lamp_head_link is rigidly fixed
    to head_gimbal_link, so mujoco's fusestatic compiler pass merges it
    in - that's why the compiled geom on head_gimbal_link (there are
    two; the later one) carries the fixture_light color. It's shaped
    like the *entire* lamp-head housing capsule, not a small bulb
    disc, since it borrowed the collision shape.
  - light_emitter_link - the link the URDF's own comments describe as
    the semantic marker for where a simulator adapter should put a
    light - has a <visual> but no <collision>, and MuJoCo's URDF
    conversion only ever produces a geom from a link's <collision>.
    Visual-only links compile to zero geoms. It never had anything to
    grab in the first place.
  - Adding a brand-new *geom* to an already-existing URDF-derived body
    via MjSpec (spec.body(...).add_geom(...)) does not survive
    spec.compile() in this MuJoCo build - confirmed by testing: ngeom
    doesn't change and the geom can't be found by name afterward.
    Lights added the same way do survive, which is why this module can
    add a dedicated light but not a small dedicated bulb geom to go
    with it. (An earlier version worked around that by recoloring the
    existing housing geom instead - see below for why that was dropped.)
  - The obvious mujoco.mjtLightType.mjLIGHT_POINT type (a real physical
    point light) compiles fine, shows up correctly in the mjvScene, but
    renders with *zero* visible effect - confirmed by rendering
    identical scenes with it on/off/full-blast red and diffing pixels
    (always 0 difference). A plain mjLIGHT_SPOT with cutoff=180
    (spot's default type, just opened all the way into a sphere) lights
    the scene correctly. So: use SPOT, not POINT, for anything you
    actually want visible.
  - model.light_intensity (the newer physically-based brightness
    scalar) also measurably does nothing here - toggling it 0 to 100
    produced identical renders. Brightness has to come from the
    magnitude of diffuse/ambient/specular instead.

--- What this module does ---
"The lamp's light" is a real mjOBJ_LIGHT (SPOT type, cutoff=180 so it
acts like an omnidirectional point light), added via MjSpec at
model-build time. It's positioned where light_emitter_link would have
been - that offset is derived from the URDF's own fixed-joint offsets
via MjSpec, not hardcoded - and parented to head_gimbal_link so it
moves with the lamp head. This actually illuminates the scene, and its
color/intensity is what set_target()/update() below drive.

An earlier version of this module also recolored the housing geom that
happened to inherit the "fixture_light" baked color (see above - it's
the *entire* lamp-head housing capsule, not a small bulb). That made
the whole housing change color along with the mood, which looked
wrong, so LampLight now only ever touches the light - every geom stays
at whatever color the URDF importer baked into it.

Because building the light requires augmenting the model *before* it's
compiled, load the model via build_model() below instead of calling
mujoco.MjModel.from_xml_path() directly.
"""

import mujoco
import numpy as np

URDF_PATH = "robot/dummy_lamp_5dof.urdf"

LIGHT_NAME = "lamp_light"
LIGHT_HOST_BODY = "head_gimbal_link"        # real, non-fused body the light rides on
LIGHT_ORIGIN_BODY = "light_emitter_link"    # URDF's semantic marker for where the bulb sits

# Always-on floor so the lamp is never fully dark.
AMBIENT_COLOR = np.array([1.0, 0.95, 0.76])
AMBIENT_INTENSITY = 0.35

DEFAULT_SMOOTHING = 0.15  # per-frame interpolation factor toward the target (0-1)


def _local_offset(body, ancestor_name):
    """
    Walk up `body`'s ancestors in the MjSpec tree, summing local .pos,
    until `ancestor_name` is reached. Returns the accumulated offset in
    that ancestor's frame. Raises if a non-identity rotation is found
    along the way (the plain sum would then be wrong) or if the
    ancestor isn't actually an ancestor.
    """
    offset = np.zeros(3)
    node = body
    while node is not None and node.name != ancestor_name:
        if not np.allclose(node.quat, [1, 0, 0, 0]):
            raise ValueError(
                f"non-identity rotation on {node.name!r} - can't sum offsets naively"
            )
        offset += np.array(node.pos)
        node = node.parent

    if node is None:
        raise ValueError(f"{ancestor_name!r} is not an ancestor of {body.name!r}")

    return offset


def build_model(urdf_path=URDF_PATH):
    """
    Load `urdf_path` and add a real light near where the URDF's
    light_emitter_link marker sits, parented to head_gimbal_link so it
    moves with the lamp head. Returns a compiled MjModel; use
    LampLight(model) afterward to drive it at runtime.
    """
    spec = mujoco.MjSpec.from_file(urdf_path)

    host = spec.body(LIGHT_HOST_BODY)
    if host is None:
        raise RuntimeError(f"expected a {LIGHT_HOST_BODY!r} body in {urdf_path}")

    origin_body = spec.body(LIGHT_ORIGIN_BODY)
    if origin_body is None:
        raise RuntimeError(f"expected a {LIGHT_ORIGIN_BODY!r} body in {urdf_path}")

    offset = _local_offset(origin_body, LIGHT_HOST_BODY)

    # SPOT with cutoff=180 and exponent=0, not POINT - see module
    # docstring for why POINT silently renders as nothing here.
    host.add_light(
        name=LIGHT_NAME,
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        pos=list(offset),
        dir=[1, 0, 0],
        cutoff=180,
        exponent=0,
        active=1,
        castshadow=0,
        diffuse=list(AMBIENT_COLOR * AMBIENT_INTENSITY),
        ambient=list(AMBIENT_COLOR * AMBIENT_INTENSITY * 0.3),
        specular=[0.0, 0.0, 0.0],
    )

    return spec.compile()


class LampLight:
    """
    Runtime handle for the lamp's emitted light.

    Call set_target(color, intensity) whenever the desired emotional
    color changes - as rarely or often as you like, e.g. once per mode
    change - and update(model) once per sim-loop frame to ease the
    displayed color toward that target and write it into the model.
    Only the light itself is touched - every geom (including the
    housing) stays at its baked URDF color.
    """

    def __init__(self, model, smoothing=DEFAULT_SMOOTHING):
        self.light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, LIGHT_NAME)
        if self.light_id < 0:
            raise RuntimeError("model wasn't built with character.light.build_model()")

        self.smoothing = smoothing
        floor = AMBIENT_COLOR * AMBIENT_INTENSITY
        self._current = floor.copy()
        self._target = floor.copy()

    def set_target(self, color, intensity=1.0):
        """
        color: (r, g, b) in 0..1 - the emotion color to aim for.
        intensity: overall brightness multiplier. Blended with the
        default ambient floor, so the lamp never goes fully dark even
        at intensity=0.

        This is the hook to plug an emotion -> color/intensity mapping
        into - call it from wherever that decision gets made (e.g.
        alongside character/behavior.py's mode -> pose choice).
        """
        color = np.clip(np.asarray(color, dtype=float), 0.0, 1.0)
        floor = AMBIENT_COLOR * AMBIENT_INTENSITY
        self._target = np.maximum(color * intensity, floor)

    def update(self, model):
        """Call once per sim-loop frame. Eases toward the target and
        writes the result into the model's light. Geoms are never
        touched - the housing stays whatever color the URDF baked in."""
        self._current += (self._target - self._current) * self.smoothing

        model.light_diffuse[self.light_id] = self._current
        model.light_ambient[self.light_id] = self._current * 0.3
