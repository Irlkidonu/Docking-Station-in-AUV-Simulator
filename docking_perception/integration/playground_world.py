"""Build the rich interactive world around the powered Heavy plant.

The survey scenery remains the visible environment, but the vehicle and dock
come from the qualified dynamic Heavy world.  In particular this function must
never reintroduce the playground's kinematic, pose-written vehicle.
"""
from __future__ import annotations

import copy
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path


#: Measured worst-case radius of the Heavy in the throat plane: the corner of a
#: vertical thruster nacelle at y=0.2464, z=-0.0284 (link thr_body_vfl). The
#: hull box alone is only 0.169 half-width, so the nacelles -- not the body --
#: set the aperture the dock must provide. The Heavy is not modified.
HEAVY_THROAT_RADIUS_M = 0.2480
#: Chosen throat inner radius: the SMALLEST opening that clears the Heavy with
#: a useful margin, because the dock's apparent size is also a perception
#: input. The frozen network reads range from how large the dock looks, so
#: widening the collar makes it judge the dock nearer than it is. Measured on
#: the composed world: at a 0.310 m throat (radial scale 1.259 over the
#: validated dock) the network returned a consistent 0.66-0.76 of true range
#: against 0.90-0.94 before the resize, and the vehicle halted 0.147 m short of
#: the capture plane. Capture needs that ratio above ~0.70, so the opening is
#: kept as close to the validated appearance as a positive clearance allows.
THROAT_INNER_RADIUS_M = 0.265
#: Resulting physical clearance, documented rather than implied.
THROAT_CLEARANCE_M = THROAT_INNER_RADIUS_M - HEAVY_THROAT_RADIUS_M   # +0.017 m
#: Radial thickness of the collar ring segments, unchanged from the validated
#: dock; only their radius, tangential length and axial depth move.
COLLAR_RADIAL_THICKNESS_M = 0.06
COLLAR_CENTRE_RADIUS_M = THROAT_INNER_RADIUS_M + COLLAR_RADIAL_THICKNESS_M / 2
#: Axial span of the collar, dock frame. The validated ring was 0.06 m deep at
#: x=-0.02, ending at x=+0.01, while the vehicle parks with its nose at about
#: x=0: the hull reached the mouth but was never inside the structure, so the
#: capture ray at x=+0.10 asserted 0.09 m before any entry had happened.
#: The ring is extended forward to x=+0.07 -- the smallest depth that puts a
#: capture plane behind the measured nose position -- and no other dock feature
#: is touched, because rescaling the funnel and rim is what moved perception.
COLLAR_AXIAL_SPAN_M = (-0.05, 0.07)
#: Where the capture ray sits, dock frame. It must lie INSIDE the collar, so
#: that breaking it means the hull is radially enclosed by dock structure and
#: not merely crossing a plane in front of the mouth. V6 parks the 0.457 m
#: Heavy with its nose measured between -0.001 m and +0.033 m across centered
#: and offset approaches, so the ring is extended forward only far enough to
#: put a capture plane behind the nose in both cases -- 0.06 m of extra depth,
#: against the 0.12 m that was shown to move the network's range estimate.
CAPTURE_PLANE_X_M = 0.05
#: Scale applied to the dock's other radial features so the station stays one
#: coherent object: funnel mouth, rim LEDs, guidance lasers, strobes.
_RADIAL_SCALE = COLLAR_CENTRE_RADIUS_M / 0.27
#: Funnel plate offset, chosen so the mouth's narrow end meets the collar ring
#: instead of necking inside it, which would obstruct the widened throat.
FUNNEL_OFFSET_M = 0.50


def _widen_dock_throat(dock) -> None:
    """Open the collar so the Heavy can physically enter it.

    The validated collar has a 0.240 m inner radius against a 0.248 m vehicle:
    an 8 mm interference, so insertion was geometrically impossible and every
    "successful" dock stopped with the nose at the collar mouth. This widens
    the dock -- visible and collision geometry together -- and leaves the
    vehicle, the controller and the capture logic alone. Nothing is made
    non-colliding, nothing is hidden, and no pose is snapped.
    """
    front, back = COLLAR_AXIAL_SPAN_M[1], COLLAR_AXIAL_SPAN_M[0]
    axial_depth = front - back
    axial_centre = (front + back) / 2

    # The ring visual stays a filled cylinder, which is NOT what its collision
    # is -- that is an eight-segment annulus with an open 0.265 m bore.
    #
    # Redrawing it as the ring it collides as was tried and reverted. The
    # frozen network reads the dock from this camera, and the bright orange
    # emissive face is a large part of what it sees: with the bore opened the
    # terminal approach yawed 23 degrees off axis during DOCK_APPROACH and the
    # run ended CRASHED on funnel_collision at x=0.71, where the run before it
    # -- identical but for this one visual -- reached DOCKED. The disagreement
    # is cosmetic and it is documented here rather than removed, because the
    # perception domain this demonstrator is qualified against includes it.
    ring = _one(dock, "link[@name='collar']/visual[@name='ring']", "collar ring")
    _one(ring, "geometry/cylinder/radius", "ring radius").text = (
        f"{COLLAR_CENTRE_RADIUS_M + COLLAR_RADIAL_THICKNESS_M / 2:.6f}")
    _one(ring, "geometry/cylinder/length", "ring length").text = f"{axial_depth:.6f}"
    pose = ring.find("pose").text.split()
    pose[0] = f"{axial_centre:.6f}"
    ring.find("pose").text = " ".join(pose)

    for segment in dock.findall("link[@name='collar_collision']/collision"):
        values = segment.find("pose").text.split()
        y, z = float(values[1]), float(values[2])
        scale = COLLAR_CENTRE_RADIUS_M / 0.27
        values[0] = f"{axial_centre:.6f}"
        values[1] = f"{y * scale:.6f}"
        values[2] = f"{z * scale:.6f}"
        segment.find("pose").text = " ".join(values)
        size = _one(segment, "geometry/box/size", "segment size")
        radial, _, tangential = size.text.split()
        size.text = (f"{axial_depth:.6f} {radial} "
                     f"{float(tangential) * scale:.6f}")

    if os.environ.get("DOCK_DEV_OPEN_COLLAR") or os.environ.get("DOCK_CLEAN_DOCK"):
        _open_collar_bore(dock, ring)

    # The funnel and the rim furniture are deliberately NOT rescaled.
    #
    # They are the dock's largest visual features, and the frozen network reads
    # range from apparent size. Measured on this world: enlarging them (funnel
    # 0.44 -> 0.50, rim scaled with the collar, ring deepened 0.06 -> 0.18)
    # drove the network's range estimate from 0.90-0.94 of truth down to
    # 0.64-0.76, and shrinking the throat back from 0.310 to 0.285 did not
    # recover it -- so the throat radius was not what moved the estimate.
    # Neither has to move for the Heavy to fit: the funnel's narrow end sits at
    # radius 0.280 and the rim LEDs at 0.283, both already outside the widened
    # 0.265 m throat, so the aperture stays monotonically convergent.


#: Where the capture arms are hinged, dock frame. The historical assembly hung
#: its shoulders at (-0.28, +/-0.11) -- BEHIND the collar and INSIDE the throat
#: -- and reached forward to pads at (0.18, +/-0.185). Every member joining the
#: two therefore had to cross the collar plane. That was harmless while the
#: collar was a decorative ring (the source station gives link `collar` a
#: visual and no collision at all), but the qualified dock makes the collar
#: solid, and a swept-volume check of the authored arm against it shows the
#: forearm entering the ring wall by up to 29 mm: closure stalls at 0.45 rad
#: with the pads still 84 mm off the hull, which is exactly the reported
#: "arms stop against the disk instead of gripping".
#:
#: The obstruction is structural, not a tuning error. For a hinge at
#: (-0.28, 0.11) a point at hinge-radius rho crosses the collar band
#: x in [-0.05, 0.07] at y = 0.11 + sqrt(rho^2 - d^2), d in [0.23, 0.35], so it
#: clears the 0.265-0.325 m ring only for rho < 0.277 (passes inside the
#: throat) or rho > 0.411 (passes outside the collar). A pad at rho = 0.466
#: and a shoulder at rho = 0 cannot be joined without crossing the gap
#: between. No joint limit, angle or pad size removes that.
#:
#: The arms are therefore re-hinged FORWARD of the collar and OUTBOARD of the
#: vehicle, where the whole mechanism lives in free water: x = 0.14 is past the
#: collar's front face at 0.07 and short of the funnel's narrow end at 0.50,
#: and y = +/-0.40 is outside the collar's 0.325 m outer radius and outside the
#: 0.2464 m thruster-nacelle envelope. The dock, its throat radius, the capture
#: plane, the break beam and the Heavy are all untouched.
ARM_HINGE_M = (0.14, 0.40, 0.075)
#: Shoulder angle that opens the arms. Swept against the Heavy's approach
#: corridor (hull envelope plus the four vertical nacelles, extruded along the
#: approach axis) the closest arm point clears it by 129 mm here, against the
#: 209 mm of the old fully-splayed pose: open enough to be unambiguous, no
#: longer flung backwards past the dock.
ARM_OPEN_RAD = 1.30
#: The arms work in a plane 75 mm above the vehicle's centreline. On the
#: centreline the pads swept through the vertical thruster nacelles, which
#: occupy |z| <= 0.028; lifted, the pad's lower edge passes 7 mm over them and
#: its upper edge stays 2 mm inside the hull's 0.127 m half-height, so the jaw
#: closes onto hull side plating and nothing else.
ARM_PLANE_Z_M = 0.075
#: Where the hull mesh has to sit so that it is the Heavy, and not a hull
#: floating beside it. Measured from the asset with its COLLADA node transforms
#: resolved, the mesh occupies 0.4576 x 0.3442 x 0.2552 m about a centre of
#: (-0.0066, 0.0000, -0.0612) -- the Heavy's own contact envelope is
#: 0.457 x 0.338 x 0.254, so it is the same vehicle at the same scale, and it
#: is exactly symmetric in y.
#:
#: The composition previously reused the playground's visual offset,
#: (0.0672, -0.0798, -0.0265). That offset is correct THERE because the
#: playground vehicle's link origin is not its centre. The Heavy's origin IS
#: its centre, so the same numbers put the drawn hull 80 mm to starboard of the
#: vehicle that was actually flying: the camera at y=0 appeared off-centre, and
#: the starboard jaw closed 80 mm inside the picture of the hull while the port
#: jaw closed in open water. Both jaws were on the real envelope the whole
#: time -- the picture was what had moved.
HULL_MESH_OFFSET_M = (0.0066, 0.0, 0.0612)
#: The playground authored its vehicle furniture against ITS offset, so every
#: piece is carried by the difference between the two, keeping each exactly
#: where it was authored to sit on the hull. This matters most for the lamp
#: arm, which reaches 0.27 m outboard: left uncarried it would stand 0.35 m
#: out and foul the collar ring on entry.
_FURNITURE_SHIFT_M = tuple(HULL_MESH_OFFSET_M[i] - o
                           for i, o in enumerate((0.0672, -0.0798, -0.0265)))
#: The forward camera cluster is the exception. The playground draws its camera
#: housing at its own link's y=0, which is 80 mm off ITS hull's centreline --
#: harmless on a vehicle that never docks, and the reason the camera looked
#: off-centre here. The Heavy's camera is physically on the axis it docks
#: along, so the housing and the coaxial lamp that flank it are put back on
#: that axis rather than inheriting the playground's lateral quirk.
_ON_AXIS_FURNITURE = ("camera_housing",)
#: Substituted rather than inherited, so these are already written in the
#: Heavy's own frame and must not be carried by the playground difference.
_HEAVY_FRAME_FURNITURE = ("lamp_l", "lamp_r", "lumen_l", "lumen_r")



def _open_collar_bore(dock, ring) -> None:
    """DEVELOPMENT ONLY, off unless DOCK_DEV_OPEN_COLLAR is set.

    The collar's collision is an eight-segment annulus with an open bore; its
    visual is a solid emissive cylinder covering that bore. Below about 0.45 m
    the camera sits ~0.12 m from the dock origin and that orange face fills the
    frame: the processed image collapses to three unique pixel values and is
    byte-identical across +/-80 mm of lateral offset, so the visual lateral
    channel carries no information at all in the region where capture is
    decided.

    This redraws the ring as the same eight segments the collision already
    uses, with the ring's own material. The structure a camera should see -- an
    annular rim of the right radius, thickness and depth -- is preserved
    exactly; only the bore, which is physically empty, stops being painted.

    Nothing else moves: collision geometry, the capture break-beam, the funnel,
    the rim furniture, the jaws and the latch are untouched, and with the flag
    unset the emitted SDF is byte-identical to the validated world.

    This was tried once before and reverted (see
    test_collar_ring_visual_is_left_as_qualified): with the FROZEN network,
    opening the bore sent the terminal approach 23 degrees off axis and the run
    ended CRASHED on funnel_collision. That network reads the dock largely from
    this emissive face and has an inverted lateral slope of -0.21..-0.35. The
    promoted candU checkpoint was trained on the rich world and behaves
    differently, which is why the question is worth re-opening -- but the prior
    result stands as the reason this is a flag and not a change.
    """
    collar = _one(dock, "link[@name='collar']", "collar link")
    material = ring.find("material")
    collar.remove(ring)
    for index, segment in enumerate(
            dock.findall("link[@name='collar_collision']/collision")):
        visual = ET.Element("visual")
        visual.set("name", f"ring_segment_{index:02d}")
        pose = ET.SubElement(visual, "pose")
        pose.text = segment.find("pose").text
        geometry = ET.SubElement(visual, "geometry")
        box = ET.SubElement(geometry, "box")
        ET.SubElement(box, "size").text = segment.findtext("geometry/box/size")
        if material is not None:
            visual.append(copy.deepcopy(material))
        ET.SubElement(visual, "visibility_flags").text = "1"
        collar.append(visual)


#: The dock's non-physical green cues, as counted in the validated world.
#: Visual-only in every case except the grabber status lamps, whose collision
#: bodies (physical_status_l / physical_status_r) are deliberately left in place.
NONPHYSICAL_DOCK_CUES = {
    "lasers": ("docking_station", "guidance_lasers",
               ("beam_tl", "beam_tr", "beam_bl", "beam_br", "axis_laser")),
    "leds": ("docking_station", "led_markers",
             ("led_tl", "led_tr", "led_bl", "led_br")),
    "status": ("dock_grabber", "base_link", ("status_l", "status_r")),
    "beacons": ("docking_station", "beacons",
                ("transponder", "strobe_l", "strobe_r")),
    #: The transponder alone. The two strobes flank the funnel mouth and are
    #: kept: they read as real navigation lights on the structure, where the
    #: transponder is a lone emissive ball floating above the collar.
    "transponder": ("docking_station", "beacons", ("transponder",)),
}


def _strip_nonphysical_dock_cues(world, groups=None) -> int:
    """DEVELOPMENT ONLY, off unless DOCK_DEV_PLAIN_DOCK is set.

    Removes the eleven emissive green cues the dock carries: five guidance
    "lasers" (four 3 m beams and a 4 m axis beam, radius 8-12 mm), four LED
    marker spheres on the dock face, and the grabber's two status lamps. None
    of it is physically plausible underwater at these ranges -- a collimated
    beam is not visible side-on in clear water, and the markers are emissive
    rather than lit -- yet all of it is bright, saturated and geometrically
    fixed relative to the throat, which makes it exactly the kind of cue a
    network will anchor on.

    What is NOT touched: every collision body, including
    physical_status_l/physical_status_r under the grabber's status lamps; the
    FLS and break-beam sensors, which read collision geometry and never saw
    these visuals at all; the funnel, ring, rim and all real structure; the
    jaws, latch and capture mechanics; and the 40 `growth` cylinders on the
    seabed, which are muted non-emissive scenery rather than dock cues.

    DOCK_DEV_STRIP_CUES selects which groups go: any comma-separated subset
    of {"lasers", "leds", "status"}, or "all" (the default when unset). This
    exists so the groups can be ablated one at a time against an otherwise
    identical world, which is the only way to tell whether a network has learnt
    the dock or merely these cues.

    Returns the number of visuals removed, so the caller can assert the count.
    """
    if groups is None:
        selection = (os.environ.get("DOCK_DEV_STRIP_CUES") or "all").lower()
        groups = (tuple(NONPHYSICAL_DOCK_CUES) if selection == "all"
                  else tuple(g.strip() for g in selection.split(",") if g.strip()))
    unknown = [g for g in groups if g not in NONPHYSICAL_DOCK_CUES]
    if unknown:
        raise ValueError(f"unknown cue group(s) {unknown}; "
                         f"choose from {sorted(NONPHYSICAL_DOCK_CUES)} or 'all'")
    removed = 0
    for group in groups:
        model_name, link_name, visual_names = NONPHYSICAL_DOCK_CUES[group]
        link = world.find(
            f"model[@name='{model_name}']/link[@name='{link_name}']")
        if link is None:
            continue
        for visual in list(link.findall("visual")):
            if visual.get("name") in visual_names:
                link.remove(visual)
                removed += 1
    return removed


def _brace_the_funnel(world) -> None:
    """Give the funnel visible structural support.

    The funnel flares sit at x = 0.75 in the dock frame while the collar is at
    x = 0.01, with nothing drawn between them: the mouth hangs unsupported in
    the water, which looks wrong and -- more to the point -- gives a network no
    structural context tying the mouth back to the throat it leads into.

    Three braces are added: two side arms from the collar's outer rim out to
    the funnel's narrow end at y = +/-0.335, and one lower arm doing the same
    beneath. They start just outside the collar's 0.325 m outer radius and stay
    outside the bore, so they are clear of the approach corridor the vehicle
    flies down.

    VISUAL ONLY, deliberately. No collision body is added, because collision
    geometry near the throat would change capture mechanics, and the brief for
    this domain is that collisions, break-beam, jaws and latch stay exactly as
    validated. They are placed where a real brace would go and where they
    cannot foul the vehicle, so the appearance is honest even though the
    physics is unchanged.
    """
    import math
    support = world.find(
        "model[@name='docking_station']/link[@name='support']")
    if support is None:
        return
    struts = (("brace_port", (0.07, 0.335, 0.0), (0.52, 0.40, 0.0)),
              ("brace_stbd", (0.07, -0.335, 0.0), (0.52, -0.40, 0.0)),
              ("brace_lower", (0.07, 0.0, -0.335), (0.52, 0.0, -0.40)))
    for name, a, b in struts:
        d = tuple(bi - ai for ai, bi in zip(a, b))
        length = math.sqrt(sum(v * v for v in d))
        mid = tuple((ai + bi) / 2.0 for ai, bi in zip(a, b))
        yaw = math.atan2(d[1], d[0])
        pitch = math.acos(max(-1.0, min(1.0, d[2] / length)))
        visual = ET.Element("visual")
        visual.set("name", name)
        ET.SubElement(visual, "pose").text = (
            f"{mid[0]:.6f} {mid[1]:.6f} {mid[2]:.6f} 0 {pitch:.6f} {yaw:.6f}")
        geometry = ET.SubElement(visual, "geometry")
        cylinder = ET.SubElement(geometry, "cylinder")
        ET.SubElement(cylinder, "radius").text = "0.020000"
        ET.SubElement(cylinder, "length").text = f"{length:.6f}"
        material = ET.SubElement(visual, "material")
        ET.SubElement(material, "ambient").text = "0.28 0.30 0.32 1"
        ET.SubElement(material, "diffuse").text = "0.45 0.47 0.50 1"
        ET.SubElement(visual, "visibility_flags").text = "1"
        support.append(visual)

def _shift_pose(element, shift) -> None:
    """Translate one authored pose, leaving its orientation alone."""
    pose = element.find("pose")
    if pose is None:
        pose = ET.SubElement(element, "pose")
        pose.text = "0 0 0 0 0 0"
    values = pose.text.split()
    while len(values) < 6:
        values.append("0")
    for axis in range(3):
        values[axis] = f"{float(values[axis]) + shift[axis]:.6f}"
    pose.text = " ".join(values)


def _vehicle_dress(playground_vehicle):
    """The playground vehicle's appearance: everything except the hull mesh.

    Its camera housing, the two lamps and the arm that carries the off-axis
    one, the lidar head and the acoustic head. These are what make it
    recognisable as that vehicle rather than a generic hull, and both lamp
    beams come with them.
    """
    link = _one(playground_vehicle, "link", "playground vehicle link")
    # Everything except the hull mesh itself and the off-axis lamp boom.
    #
    # That boom reaches 0.27 m outboard of a hull whose half width is 0.172 m.
    # On a vehicle that never docks it is scenery; on this one it stands out
    # past the hull exactly where the collar throat is, its head sits inside
    # the ring wall once the vehicle is captured, and to an operator watching
    # an approach it reads as an arm fouling the funnel. It is cosmetic -- it
    # carries no collision and the onboard cameras are masked from it -- so it
    # is dropped rather than explained.
    boom = {"lamp_arm_clamp", "lamp_arm_seg1", "lamp_arm_knuckle",
            "lamp_arm_seg2", "lamp_offaxis", "lamp_coaxial"}
    dress = [copy.deepcopy(visual) for visual in link.findall("visual")
             if visual.get("name") not in boom | {"body"}]
    # The lamps are NOT carried across. The playground lights its scene with an
    # asymmetric pair -- one beside the camera at y=+0.02, one out on the boom
    # at y=-0.35 -- which is fine on a vehicle that flies a survey and reads
    # nothing from the image. This one estimates its lateral position from what
    # its camera sees, and lighting a docking target harder from one side than
    # the other biases exactly that estimate. A matched pair flanking the
    # camera is substituted instead, aimed down-forward as the playground's
    # are, so the beams point where an operator expects and the target is lit
    # evenly about the axis the vehicle docks along.
    aim = _one(link, "light[@name='lamp_coaxial_light']", "playground lamp")
    lamps = []
    for side in (1.0, -1.0):
        lamp = copy.deepcopy(aim)
        lamp.set("name", f"lumen_{'l' if side > 0 else 'r'}")
        _one(lamp, "pose", "lamp pose").text = f"0.2494 {side * 0.11:.4f} 0.0197 0 0 0"
        _one(lamp, "direction", "lamp direction").text = "0.939 0 -0.343"
        lamps.append(lamp)
        housing = ET.Element("visual", {"name": f"lamp_{'l' if side > 0 else 'r'}"})
        _set_geometry(housing, _sphere((0.2494, side * 0.11, 0.0197), 0.021))
        material = ET.SubElement(housing, "material")
        for channel, value in (("ambient", "0.85 0.83 0.72 1"),
                               ("diffuse", "0.95 0.92 0.80 1"),
                               ("emissive", "0.85 0.83 0.72 1")):
            ET.SubElement(material, channel).text = value
        dress.append(housing)
    return dress, lamps


def _dress_as_playground_vehicle(heavy, base_link, dress) -> None:
    """Make the powered Heavy look like the vehicle it replaced.

    The Heavy's mass, collision envelope, hydrodynamics and eight-thruster
    plant are what this demonstrator needs and none of them are touched. Its
    APPEARANCE is not: composed as authored it was the recognisable hull mesh
    with eight bare thruster cylinders protruding past it, lit by two spot
    lamps aimed dead ahead along +X. The playground vehicle carries its lamps
    on an arm, aimed 20 degrees down -- which is why the reference simulator
    lights the seabed in front of the vehicle and this one lit the water
    column ahead of it.

    The lamp poses line up because the hull mesh is transplanted with the same
    visual offset it has in the playground, so the furniture authored against
    that mesh lands exactly where it was authored to sit.
    """
    visuals, lamps = dress
    for element in list(visuals) + list(lamps):
        if element.get("name") in _HEAVY_FRAME_FURNITURE:
            continue
        _shift_pose(element, _FURNITURE_SHIFT_M)
        if element.get("name") in _ON_AXIS_FURNITURE:
            _shift_pose(element, (0.0, -_FURNITURE_SHIFT_M[1], 0.0))
    for visual in visuals:
        # GUI layer only, like the hull mesh: a camera housing must not appear
        # in the view of the camera inside it.
        if visual.find("visibility_flags") is None:
            ET.SubElement(visual, "visibility_flags").text = "2"
        base_link.append(visual)
    # The Heavy's own lamps are replaced rather than added to, so the scene is
    # lit by two beams as in the reference and not by four.
    for light in heavy.findall(".//light"):
        for parent in heavy.iter():
            if light in list(parent):
                parent.remove(light)
                break
    for lamp in lamps:
        base_link.append(lamp)
    # The bare thruster primitives are the Heavy observation world's stand-in
    # for a hull it had no mesh for. With the mesh present they are visible
    # duplicates sticking out through it. Visual only: the links, their mass,
    # their collisions and the thruster plugins that drive them are untouched.
    for link in heavy.findall("link"):
        if not link.get("name", "").startswith("thr_"):
            continue
        for visual in link.findall("visual"):
            link.remove(visual)


def _add_operator_camera(base_link) -> None:
    """A display-only camera in the playground's own optical format.

    The Transformer's camera is a 1920x1080, 110-degree docking sensor. Shown
    to an operator it is neither the view the rich world was authored for nor
    cheap to render, and it carries no water column: the medium is applied
    downstream, in the supervisor, on its way into the network. The panel was
    therefore being handed a clean, extremely wide frame that looked nothing
    like the playground at any turbidity setting.

    This adds a second, small camera in the playground's published format --
    640x480, 60-degree, 15 Hz -- at the Heavy's real camera station. It feeds
    the operator display only; the qualified docking sensor above it keeps its
    format, its rate, its mask and its topic, so the Transformer's input domain
    is bit-for-bit what it was validated on.
    """
    sensor = ET.SubElement(base_link, "sensor",
                           {"name": OPERATOR_CAMERA_SENSOR, "type": "camera"})
    for tag, value in (("always_on", "1"), ("update_rate", "15"),
                       ("topic", OPERATOR_CAMERA_TOPIC),
                       # Same station as the docking camera, tilted a little
                       # down so the seabed is in frame on transit the way the
                       # playground's own camera framed it.
                       ("pose", "0.23 0 -0.01 0 0.2 0")):
        ET.SubElement(sensor, tag).text = value
    camera = ET.SubElement(sensor, "camera")
    ET.SubElement(camera, "horizontal_fov").text = "1.047"
    image = ET.SubElement(camera, "image")
    for tag, value in (("width", "640"), ("height", "480"), ("format", "R8G8B8")):
        ET.SubElement(image, tag).text = value
    clip = ET.SubElement(camera, "clip")
    for tag, value in (("near", "0.05"), ("far", "60")):
        ET.SubElement(clip, tag).text = value
    # Layer 1 only: like the docking camera it must not see the cosmetic hull
    # shell that surrounds it (layer 2) or the break-beam target (layer 4).
    ET.SubElement(camera, "visibility_mask").text = "1"


#: Sensor and topic the operator display is fed from. Separate from the
#: Transformer's camera so neither can perturb the other.
OPERATOR_CAMERA_SENSOR = "operator_camera"
OPERATOR_CAMERA_TOPIC = "/bluerov2_heavy/camera_view"


def _strut(start, end, radius):
    """A cylinder spanning two points in the arms' working plane."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    centre = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2,
              (start[2] + end[2]) / 2)
    return {"kind": "cylinder", "radius": radius, "length": length,
            "pose": (f"{centre[0]:.6f} {centre[1]:.6f} {centre[2]:.6f} "
                     f"0 1.570796 {math.atan2(dy, dx):.6f}")}


def _mirror(part):
    """The starboard counterpart of a port part: y negated, so is the yaw."""
    out = dict(part)
    values = part["pose"].split()
    values[1] = f"{-float(values[1]):.6f}"
    if len(values) == 6:
        values[5] = f"{-float(values[5]):.6f}"
    out["pose"] = " ".join(values)
    return out


def _sphere(centre, radius):
    return {"kind": "sphere", "radius": radius,
            "pose": f"{centre[0]:.6f} {centre[1]:.6f} {centre[2]:.6f} 0 0 0"}


def _box(centre, size):
    return {"kind": "box", "size": size,
            "pose": f"{centre[0]:.6f} {centre[1]:.6f} {centre[2]:.6f} 0 0 0"}


def _arm_parts():
    """The port arm, authored in its CLOSED pose, dock frame.

    Every dimension is checked in test_playground_docking_grabber.py against
    the collar, the funnel, the Heavy at its measured capture pose and the
    approach corridor, over the whole travel -- not only at the end stops.
    """
    hinge = ARM_HINGE_M
    elbow = (0.235, 0.215, ARM_PLANE_Z_M - 0.013)
    jaw = (0.150, 0.196, ARM_PLANE_Z_M - 0.020)
    return {
        "p_shoulder": _sphere(hinge, 0.034),
        "p_upper": _strut(hinge, elbow, 0.026),
        "p_elbow": _sphere(elbow, 0.026),
        "p_fore": _strut(elbow, jaw, 0.020),
        "p_wrist": _sphere(jaw, 0.024),
        # The jaw face. Its inner surface sits at y = 0.167 against a hull half
        # width of 0.169, so at the closed stop it is already 2 mm into the
        # plating and the servo's remaining travel becomes clamping force.
        "p_pad": _box((0.195, 0.181, ARM_PLANE_Z_M + 0.005), (0.16, 0.028, 0.09)),
        # Forward hook, ahead of the front nacelle station, so a captured hull
        # cannot slide out along the approach axis.
        "p_lip": _box((0.112, 0.198, ARM_PLANE_Z_M + 0.005), (0.03, 0.05, 0.09)),
    }


def _pylon_parts():
    """Fixed structure carrying the hinges from the dock post to the mouth.

    It leaves the housing behind the collar, goes outboard while still behind
    it, and only then runs forward at y = 0.40 -- outside the 0.325 m collar
    and clear of the funnel, which starts at x = 0.50.
    """
    hinge = ARM_HINGE_M
    return {
        "pylon_root": _strut((-0.30, 0.10, ARM_PLANE_Z_M),
                             (-0.12, 0.40, ARM_PLANE_Z_M), 0.030),
        "pylon_rail": _strut((-0.12, 0.40, ARM_PLANE_Z_M),
                             (0.16, 0.40, ARM_PLANE_Z_M), 0.030),
        "pylon_boss": {"kind": "cylinder", "radius": 0.045, "length": 0.09,
                       "pose": f"{hinge[0]:.6f} {hinge[1]:.6f} {hinge[2]:.6f} 0 0 0"},
    }


def _set_geometry(element, part) -> None:
    """Give one visual the pose and primitive named by ``part``."""
    pose = element.find("pose")
    if pose is None:
        pose = ET.Element("pose")
        element.insert(0, pose)
    pose.text = part["pose"]
    geometry = element.find("geometry")
    if geometry is None:
        geometry = ET.SubElement(element, "geometry")
    geometry.clear()
    shape = ET.SubElement(geometry, part["kind"])
    if part["kind"] == "sphere":
        ET.SubElement(shape, "radius").text = f"{part['radius']:.6f}"
    elif part["kind"] == "cylinder":
        ET.SubElement(shape, "radius").text = f"{part['radius']:.6f}"
        ET.SubElement(shape, "length").text = f"{part['length']:.6f}"
    else:
        ET.SubElement(shape, "size").text = " ".join(
            f"{v:.6f}" for v in part["size"])


def _remount_capture_arms(grabber) -> None:
    """Re-hinge the capture arms clear of the collar, and make them clamp.

    Two faults are corrected together, because they have one cause. The arms
    could not close -- their forearms fouled the now-solid collar 0.45 rad from
    the stop -- and the pose they were parked in was the outward joint limit,
    2.0 rad from closed, because that was the only place the old geometry was
    reliably clear of the vehicle. Moving the hinges in front of the collar
    removes the obstruction, and with it the reason for the exaggerated open
    pose.

    Nothing outside this assembly moves: not the dock, not the throat radius,
    not the capture plane, not the break beam, not the Heavy.
    """
    base = _one(grabber, "link[@name='base_link']", "grabber base")
    pattern = _one(base, "visual[@name='housing']", "housing visual")
    for side, sign in (("port", 1.0), ("stbd", -1.0)):
        for name, part in _pylon_parts().items():
            visual = ET.SubElement(base, "visual",
                                   {"name": f"{name}_{side}"})
            _set_geometry(visual, part if sign > 0 else _mirror(part))
            visual.append(copy.deepcopy(_one(pattern, "material", "material")))

    for link_name, prefix, sign in (("arm_port", "p", 1.0),
                                    ("arm_stbd", "s", -1.0)):
        link = _one(grabber, f"link[@name='{link_name}']", link_name)
        for name, part in _arm_parts().items():
            visual = _one(link, f"visual[@name='{prefix}{name[1:]}']",
                          f"{link_name} {name}")
            _set_geometry(visual, part if sign > 0 else _mirror(part))

    hinge = ARM_HINGE_M
    for joint_name, sign in (("sh_port", 1.0), ("sh_stbd", -1.0)):
        joint = _one(grabber, f"joint[@name='{joint_name}']", joint_name)
        pose = joint.find("pose")
        if pose is None:
            pose = ET.SubElement(joint, "pose")
        pose.text = (f"{hinge[0]:.6f} {sign * hinge[1]:.6f} "
                     f"{hinge[2]:.6f} 0 0 0")
        # SDFormat defaults a joint pose to the CHILD frame, which would apply
        # the open link rotation below a second time. State the base frame.
        pose.set("relative_to", "base_link")
        axis = _one(joint, "axis", f"{joint_name} axis")
        lower, upper = -ARM_OPEN_RAD - 0.15, 0.10
        if sign < 0:
            lower, upper = -0.10, ARM_OPEN_RAD + 0.15
        _one(axis, "limit/lower", f"{joint_name} lower").text = f"{lower:.2f}"
        _one(axis, "limit/upper", f"{joint_name} upper").text = f"{upper:.2f}"
        # Closing rate, and why it is slow.
        #
        # The jaw meets the hull at its own tip speed: this angular limit times
        # the 0.226 m jaw arm. The authored 6 rad/s belonged to arms that swung
        # 0.2 rad in free water. At 1.5 rad/s these arrived at 0.34 m/s and the
        # first one to touch visibly smacked the vehicle into line -- it worked,
        # and it looked like a collision rather than a capture. 0.5 rad/s puts
        # the pad on the plating at 0.11 m/s, so the same correction happens as
        # a push instead of an impact. It costs about 2.6 s of travel and
        # changes nothing about the force available once contact is made.
        _one(axis, "limit/velocity", f"{joint_name} velocity").text = "0.5"
        # Damping to match: an underdamped jaw would bounce off the hull it
        # just touched and hit it again.
        damping = axis.find("dynamics/damping")
        if damping is None:
            dynamics = axis.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(axis, "dynamics")
            damping = ET.SubElement(dynamics, "damping")
        damping.text = "2.0"
        # Effort, and why it is small.
        #
        # A position servo commands an ANGLE. It has no idea where the vehicle
        # actually is, and the vehicle does not park perfectly centred -- a
        # measured 47 mm to port on the qualifying run, well inside the
        # validated terminal envelope. At that offset the port jaw's commanded
        # angle lies 54 mm INSIDE the hull, and with the authored 500 N.m the
        # servo simply drove it there: one jaw buried in the vehicle, the other
        # hanging in open water. The remedy is not a better angle -- there
        # isn't one, for every possible offset -- it is to let contact win.
        # 24 N.m over a 0.226 m jaw arm is about 106 N of clamp: firm on the
        # plating, comfortably more than the 55 N the vehicle's own station
        # keeping can resist -- so the jaw still wins and still centres it --
        # and nowhere near enough to push through the hull.
        _one(axis, "limit/effort", f"{joint_name} effort").text = "24"

        # Spawn open, so joint coordinate zero is the parked pose and closing
        # is a real -1.30 rad rotation of a real body. For a hinge h and angle
        # q the child-link origin is h - R(q)h; this is SDF initialisation
        # derived from geometry, not a runtime pose write.
        angle = sign * ARM_OPEN_RAD
        hx, hy = hinge[0], sign * hinge[1]
        rx = hx * math.cos(angle) - hy * math.sin(angle)
        ry = hx * math.sin(angle) + hy * math.cos(angle)
        link = _one(grabber, f"link[@name='arm_{joint_name[3:]}']",
                    f"arm_{joint_name[3:]}")
        link_pose = link.find("pose")
        if link_pose is None:
            link_pose = ET.Element("pose")
            link.insert(0, link_pose)
        link_pose.text = (f"{hx - rx:.6f} {hy - ry:.6f} 0 0 0 {angle:.6f}")

    for plugin in grabber.findall(
            "plugin[@filename='gz-sim-joint-position-controller-system']"):
        # The command ceiling is what actually caps the torque; the axis limit
        # above only bounds it. Both are lowered together, and the gain with
        # them, so the jaw arrives against the hull rather than at it.
        for tag, value in (("p_gain", "3.0"), ("i_gain", "0.05"),
                           ("d_gain", "1.5"), ("cmd_max", "24"),
                           ("cmd_min", "-24")):
            element = plugin.find(tag)
            if element is None:
                element = ET.SubElement(plugin, tag)
            element.text = value


def _one(parent, xpath: str, label: str):
    found = parent.findall(xpath)
    if len(found) != 1:
        raise ValueError(f"expected one {label}, found {len(found)}")
    return found[0]


def build_playground_docking_world(
    playground_sdf: Path,
    heavy_sdf_text: str,
    capture_station_sdf: Path,
    *,
    world_name: str = "mode_aware_survey_docking",
) -> str:
    """Return powered Heavy physics plus the familiar survey scenery."""
    # The existing playground is authoritative: scene, seabed, lighting,
    # fish, vegetation, wreck, pipeline and GUI systems stay byte-for-byte as
    # authored. Replace only its explicitly kinematic vehicle with the powered
    # Heavy and add the docking assembly.
    root = ET.parse(playground_sdf).getroot()
    world = _one(root, "world", "survey world")
    old_vehicle = _one(world, "model[@name='bluerov2']", "kinematic vehicle")
    world.remove(old_vehicle)
    # Its MOTION is what must not come back. Its APPEARANCE is the reference
    # this demonstrator is supposed to look like, so the lamp arms, camera
    # housing, sensor heads and the two lamp beams are kept and transplanted
    # onto the Heavy below.
    playground_dress = _vehicle_dress(old_vehicle)
    heavy_root = ET.fromstring(heavy_sdf_text)
    heavy_world = _one(heavy_root, "world", "Heavy source world")
    dock = copy.deepcopy(_one(heavy_world, "model[@name='docking_station']",
                              "validated dock"))
    heavy = copy.deepcopy(_one(heavy_world, "model[@name='bluerov2_heavy']",
                               "powered Heavy"))
    # Keep the dock at the same clear horizontal site used by the frozen Heavy
    # observation worlds.  Relocating it to (-10, 9) placed the terminal fan
    # inside dense survey scenery: the GPU-lidar's nearest physical returns
    # became background/seabed dominated and the camera saw an occluded, dark
    # target.  Translation to the rich world's -17 m operating depth preserves
    # the validated relative sensor/dock geometry without changing the scene.
    dock.find("pose").text = "0 0 -17 0 0 0"
    _widen_dock_throat(dock)
    # Spawn is overridable for engineering validation only; the default is the
    # qualified start. A lateral-offset case must be created HERE rather than
    # by teleporting a running vehicle: onboard position is dead-reckoned from
    # the DVL, which cannot observe a pose that is simply moved, so a teleport
    # silently destroys the navigation solution instead of testing the docking.
    heavy.find("pose").text = os.environ.get("DOCK_HEAVY_SPAWN", "-10 -9 -17 0 0 0")
    # Still water is the baseline, stated explicitly rather than left implied.
    # gz-sim's Hydrodynamics system evaluates its drag terms on the vehicle's
    # velocity RELATIVE to this flow, so publishing a current on /ocean_current
    # is a genuine hydrodynamic disturbance on the hull -- not an offset added
    # to any reported state -- and it acts hardest exactly when the vehicle is
    # not station-keeping against it.
    hydro = _one(heavy, "plugin[@filename='gz-sim-hydrodynamics-system']",
                 "Heavy hydrodynamics")
    if hydro.find("default_current") is None:
        ET.SubElement(hydro, "default_current").text = "0 0 0"

    world.append(dock); world.append(heavy)

    # Keep the familiar BlueROV appearance while retaining the Heavy's mass,
    # collision envelope, hydrodynamics and eight-thruster plant.
    body = _one(heavy, "link[@name='base_link']/visual[@name='body']", "body visual")
    body.find("geometry").clear()
    mesh = ET.SubElement(body.find("geometry"), "mesh")
    ET.SubElement(mesh, "uri").text = (
        "model://uuv_mode_aware_navigation/models/meshes/bluerov2.dae")
    pose = body.find("pose")
    if pose is None:
        pose = ET.SubElement(body, "pose")
    pose.text = f"{HULL_MESH_OFFSET_M[0]:.4f} {HULL_MESH_OFFSET_M[1]:.4f} " \
                f"{HULL_MESH_OFFSET_M[2]:.4f} 0 0 0"
    # The mesh carries its own materials. An SDF <material> on the visual
    # overrides all of them, which is why the hull rendered as one flat blue
    # shape instead of the vehicle. The playground's own body visual declares
    # no material for exactly this reason.
    material = body.find("material")
    if material is not None:
        body.remove(material)
    # The recognisable DAE extends to x~=0.399 m after this visual transform,
    # while the physically declared camera/FLS origins are x=0.23/0.24 m. It
    # therefore surrounds the sensors and self-occludes their render views.
    # Real housings cannot see their own exterior shell. Put only this cosmetic
    # mesh on a GUI-visible layer and mask it from the onboard render sensors;
    # collision, mass, geometry, sensor extrinsics and GUI appearance remain.
    ET.SubElement(body, "visibility_flags").text = "2"
    base_link = _one(heavy, "link[@name='base_link']", "Heavy base link")
    camera_sensor = _one(base_link, "sensor[@name='camera']", "Heavy camera")
    ET.SubElement(_one(camera_sensor, "camera", "camera parameters"),
                  "visibility_mask").text = "1"
    fls_sensor = _one(base_link,
                      "sensor[@name='oculus_m750d_intersections']", "Heavy FLS")
    ET.SubElement(_one(fls_sensor, "lidar", "FLS parameters"),
                  "visibility_mask").text = "1"
    # A camera sensor with <visualize>true</visualize> makes Gazebo draw a live
    # preview quad in the 3D scene, hanging off the vehicle. That flat panel --
    # showing seabed and vegetation, roughly a third of a metre across -- is
    # the large square artefact. It is a GUI debug aid, not part of the
    # vehicle: the sensor, its format, rate, mask and topic are untouched.
    for sensor in base_link.findall("sensor[@type='camera']"):
        visualize = sensor.find("visualize")
        if visualize is None:
            visualize = ET.SubElement(sensor, "visualize")
        visualize.text = "false"
    _add_operator_camera(base_link)
    _dress_as_playground_vehicle(heavy, base_link, playground_dress)

    # Diagnostic-only stream for every collision involving the Heavy contact
    # envelope. It is never consumed by operational autonomy.
    contact_sensor = ET.SubElement(base_link, "sensor", {
        "name": "all_environment_contacts", "type": "contact"})
    ET.SubElement(contact_sensor, "always_on").text = "true"
    ET.SubElement(contact_sensor, "update_rate").text = "100"
    ET.SubElement(contact_sensor, "topic").text = "/bluerov2_heavy/contact/all"
    contact = ET.SubElement(contact_sensor, "contact")
    ET.SubElement(contact, "collision").text = "contact_envelope"
    # A non-colliding optical target co-located with the physical Heavy
    # envelope. The collar beam sees this layer only, so static dock/scenery
    # geometry cannot self-trigger it. It changes no mass or contact geometry.
    capture_target = ET.SubElement(base_link, "visual", {
        "name": "capture_beam_target"})
    ET.SubElement(capture_target, "visibility_flags").text = "4"
    target_geometry = ET.SubElement(capture_target, "geometry")
    target_box = ET.SubElement(target_geometry, "box")
    ET.SubElement(target_box, "size").text = "0.457 0.338 0.254"
    # Presentation only. visibility_flags gates cameras that set a restrictive
    # mask; the GUI user camera uses the default full mask and MinimalScene
    # exposes no visibility_mask, so nothing in the flag scheme can hide this
    # box from an operator. Rendered opaque it sits over the hull mesh and the
    # vehicle reads as a grey slab. Full transparency removes it from the GUI
    # while leaving the geometry, its layer and the break beam untouched.
    ET.SubElement(capture_target, "transparency").text = "1.0"
    # Transparency governs the colour written to the frame buffer, not whether
    # the mesh is entered into the shadow pass: ogre2 still rasterises this box
    # into the directional light's depth map, so the invisible slab went on
    # throwing a hard rectangular shadow across the seabed that read to an
    # operator as a large black artefact following the vehicle. It carries no
    # illumination role of its own -- it exists solely as a layer-4 return for
    # the collar break beam, which is a gpu_lidar and never consults shadows --
    # so it is taken out of the shadow pass as well.
    ET.SubElement(capture_target, "cast_shadows").text = "false"
    target_material = ET.SubElement(capture_target, "material")
    for channel in ("ambient", "diffuse", "specular"):
        ET.SubElement(target_material, channel).text = "0 0 0 0"

    # Reuse only the capture-arm model from the historical visual asset. Never
    # import its world, vehicle, rocks, markers, or experimental fixtures.
    capture_world = _one(ET.parse(capture_station_sdf).getroot(), "world",
                         "capture-station source world")
    grabber = _one(capture_world, "model[@name='dock_grabber']",
                   "capture-arm assembly")
    final_grabber = copy.deepcopy(grabber)
    final_grabber.find("pose").text = "0 0 -17 0 0 0"
    _remount_capture_arms(final_grabber)
    # The historical arm model was visual-only. Mirror each structural visual
    # as collision geometry so the visible hooks are the physical hooks.
    for link in final_grabber.findall("link"):
        for visual in link.findall("visual"):
            collision = ET.SubElement(link, "collision", {
                "name": "physical_" + visual.get("name", "part")})
            if visual.find("pose") is not None:
                collision.append(copy.deepcopy(visual.find("pose")))
            collision.append(copy.deepcopy(visual.find("geometry")))
        if link.get("name") in {"arm_port", "arm_stbd"}:
            sensor = ET.SubElement(link, "sensor", {
                "name": f"{link.get('name')}_all_contacts", "type": "contact"})
            ET.SubElement(sensor, "always_on").text = "true"
            ET.SubElement(sensor, "update_rate").text = "100"
            ET.SubElement(sensor, "topic").text = (
                f"/dock_grabber/contact/{link.get('name')}")
            contact = ET.SubElement(sensor, "contact")
            for collision in link.findall("collision"):
                ET.SubElement(contact, "collision").text = collision.get("name")
    latch = ET.SubElement(final_grabber, "plugin", {
        "filename": "gz-sim-detachable-joint-system",
        "name": "gz::sim::systems::DetachableJoint"})
    for tag, value in (("parent_link", "base_link"),
                       ("child_model", "bluerov2_heavy"),
                       ("child_link", "base_link"),
                       ("detach_topic", "/dock/latch/detach"),
                       ("attach_topic", "/dock/latch/attach"),
                       ("output_topic", "/dock/latch/state")):
        ET.SubElement(latch, tag).text = value
    world.append(final_grabber)

    # The playground's own lighting is authoritative and is left exactly as
    # authored: one shadow-casting key light (filtered_daylight) and one fill
    # (water_column_fill).
    #
    # An earlier revision also imported the Heavy observation world's lights,
    # on the reasoning that the camera interface had been qualified under them.
    # Measured against the reference playground that is what made the composed
    # scene look wrong: it added surface_light -- a SECOND shadow-casting
    # directional light -- so every object in the world threw two hard shadows
    # at different angles, and caustic_fill, a near-horizontal wash from
    # (10, 0, -10) that lit the seabed from the side. Neither exists in the
    # simulator this is meant to reproduce.
    #
    # This changes the illumination the docking camera sees. It is a renderer
    # configuration change only -- no image is post-processed, no threshold and
    # no weight moves -- but it is a genuine change to the optical domain and
    # is called out rather than absorbed silently.

    # The dock confirmation ray is operational input, separate from scoring.
    #
    # Its plane now lies INSIDE the collar ring (x=-0.05..+0.07), so a hull
    # breaking this ray is radially enclosed by dock structure rather than
    # merely crossing a plane in front of the mouth. What also had to move is
    # length -- the old origin at y=-0.235 sat in mid-air inside a 0.310 m
    # throat and the 0.47 m ray no longer reached the far side. The origin now
    # lies on the collar's inner surface and the ray spans the full throat
    # diameter, so "beam broken" keeps its physical meaning at the new size.
    collar = _one(world, "model[@name='docking_station']/link[@name='collar']",
                  "dock collar")
    beam = ET.SubElement(collar, "sensor", {
        "name": "capture_break_beam", "type": "gpu_lidar"})
    for tag, value in (("pose", f"{CAPTURE_PLANE_X_M:.6f} "
                                f"{-THROAT_INNER_RADIUS_M:.6f} 0 0 0 1.57079632679"),
                       ("always_on", "true"), ("update_rate", "50"),
                       ("topic", "/dock/capture_break_beam")):
        ET.SubElement(beam, tag).text = value
    lidar = ET.SubElement(beam, "lidar")
    ET.SubElement(lidar, "visibility_mask").text = "4"
    scan = ET.SubElement(lidar, "scan")
    for axis in ("horizontal", "vertical"):
        node = ET.SubElement(scan, axis)
        for tag, value in (("samples", "1"), ("resolution", "1"),
                           ("min_angle", "0"), ("max_angle", "0")):
            ET.SubElement(node, tag).text = value
    distance = ET.SubElement(lidar, "range")
    for tag, value in (("min", "0.01"),
                       ("max", f"{2 * THROAT_INNER_RADIUS_M:.6f}"),
                       ("resolution", "0.001")):
        ET.SubElement(distance, tag).text = value

    # Operational contact sensors identify forbidden structure independently
    # of scoring truth. Each sensor is attached to the collision-only link it
    # observes and publishes the colliding dock region.
    for link_name in ("support_collision", "funnel_collision", "collar_collision"):
        link = _one(world, f"model[@name='docking_station']/link[@name='{link_name}']",
                    link_name)
        sensor = ET.SubElement(link, "sensor", {
            "name": f"{link_name}_contact", "type": "contact"})
        ET.SubElement(sensor, "always_on").text = "true"
        ET.SubElement(sensor, "update_rate").text = "100"
        ET.SubElement(sensor, "topic").text = f"/dock/contact/{link_name}"
        contact = ET.SubElement(sensor, "contact")
        for collision in link.findall("collision"):
            ET.SubElement(contact, "collision").text = collision.get("name")

    # Add only sensor systems missing from the playground. Physics, lighting,
    # GUI, scenery and the vehicle motion model all remain its own.
    existing_systems = {p.get("filename") for p in world.findall("plugin")}
    for plugin in heavy_world.findall("plugin"):
        filename = plugin.get("filename", "")
        # A declared sensor is inert without its system. The altimeter was
        # missing here, so the depth sensor existed on the vehicle and silently
        # published nothing -- verified by its topic being absent from a live
        # run. Same silent-failure mode uuv_sim_physics documents for the IMU.
        # The powered Heavy's open-frame displacement bodies are inert unless
        # its scoped buoyancy system is carried into the composed playground.
        # Omitting it made the physical vehicle sink to the seabed even though
        # the source Heavy world is neutrally buoyant by construction.
        required = {"gz-sim-buoyancy-system", "gz-sim-imu-system",
                    "gz-sim-dvl-system", "gz-sim-altimeter-system",
                    "gz-sim-contact-system"}
        if filename in required and filename not in existing_systems:
            world.insert(0, copy.deepcopy(plugin))
            existing_systems.add(filename)
    if "gz-sim-contact-system" not in existing_systems:
        world.insert(0, ET.Element("plugin", {
            "filename": "gz-sim-contact-system",
            "name": "gz::sim::systems::Contact"}))

    # All ordinary rendering stays on layer 1 (the cosmetic Heavy shell is
    # already GUI-only layer 2). Layer 4 is reserved solely for the physical
    # capture target above.
    for visual in world.findall(".//visual"):
        if visual.get("name") == "capture_beam_target":
            continue
        flags = visual.find("visibility_flags")
        if flags is None:
            ET.SubElement(visual, "visibility_flags").text = "1"

    # The playground's GUI block is inherited wholesale, ImageDisplay included.
    # Its /uuv/camera_degraded topic is kept exactly as authored: this
    # demonstrator supplies that topic itself, from OPERATOR_CAMERA_TOPIC
    # through the particulate-water model, so the panel shows the
    # onboard view after the water column rather than the clean render. An
    # earlier revision pointed it straight at the 1920x1080 Transformer camera;
    # that fed the panel an undegraded 110-degree image that no longer looked
    # like the playground, and its permanent subscription held the docking
    # camera rendering from launch. No navigation-demonstrator node or bridge is involved.

    if os.environ.get("DOCK_CLEAN_DOCK"):
        # The cleaned dock domain. The nonphysical orange collar face is opened
        # (above) and twelve marker visuals go: five guidance-laser beams, the
        # four green LED markers on the collar face, the lone white transponder
        # ball above the collar, and the grabber's two green status lamps.
        #
        # The two white strobes flanking the funnel mouth are KEPT -- they read
        # as real navigation lights mounted on the structure rather than as
        # free-floating markers.
        #
        # What remains is structure: the collar ring and rim, the funnel flares
        # and their new braces, the support post and baseplate, and the
        # grabber's housing, pylons and arms. Every collision body stays,
        # including the two under the removed status lamps.
        #
        # The point is that a network trained here has far less to anchor on
        # than the dock's actual geometry.
        #
        # The beams are removed because a controlled ablation showed candU had
        # anchored on them rather than on the dock: deleting them collapsed the
        # lateral response to slope ~0.00 at 0.50 and 0.60 m, against a render
        # noise floor of 0.0-0.1 mm. Deleting the LED markers instead changed
        # nothing at all, which rules out a generic domain-shift explanation.
        removed = _strip_nonphysical_dock_cues(
            world, groups=("lasers", "leds", "transponder", "status"))
        if removed != 12:
            raise RuntimeError(
                f"cleaned dock expected to remove 12 marker visuals, removed {removed}")
        if os.environ.get("DOCK_ABLATE_STROBES"):
            # Shortcut-learning check for the cleaned-dock model. The two white
            # strobes at the funnel mouth are the only emissive markers left;
            # if the new model has merely swapped one artificial anchor for
            # another, removing them will collapse its lateral response the way
            # removing the guidance beams collapsed candU's.
            gone = _strip_nonphysical_dock_cues(world, groups=("beacons",))
            if gone != 2:
                raise RuntimeError(
                    f"strobe ablation expected 2 visuals, removed {gone}")
        _brace_the_funnel(world)
    elif os.environ.get("DOCK_DEV_PLAIN_DOCK"):
        _strip_nonphysical_dock_cues(world)

    world.set("name", world_name)
    ET.register_namespace("gz", "http://gazebosim.org/schema")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)
