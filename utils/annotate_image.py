import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import os


ROUTE_CROSSWALK_STRIPE_COUNT = 7
ROUTE_CROSSWALK_STRIPE_WIDTH_CM = 45.0
ROUTE_CROSSWALK_BAND_LENGTH_CM = 420.0
ROUTE_CROSSWALK_PAINT_Z_CM = 3.0
WAYPOINT_GROUND_Z_CM = 20.0


def project_waypoints_to_image(
    waypoints,
    camera_location,
    camera_rotation,
    camera_fov,
    width,
    height,
):
    """Project executable waypoints through the policy-camera model."""
    std_camera_pos = convert_ue_to_std_location(camera_location)
    if camera_rotation is None or len(camera_rotation) != 3:
        camera_rotation = (0.0, 0.0, 0.0)
    try:
        rotation = get_rotation_matrix(*camera_rotation)
    except Exception:
        rotation = np.eye(3)

    projections = []
    for waypoint in waypoints:
        if hasattr(waypoint, 'x') and hasattr(waypoint, 'y'):
            ue_waypoint = (
                float(waypoint.x),
                float(waypoint.y),
                WAYPOINT_GROUND_Z_CM,
            )
        elif isinstance(waypoint, (list, tuple)) and len(waypoint) >= 2:
            ue_waypoint = (
                float(waypoint[0]),
                float(waypoint[1]),
                WAYPOINT_GROUND_Z_CM,
            )
        else:
            projections.append(None)
            continue
        projections.append(project_point(
            convert_ue_to_std_location(ue_waypoint),
            std_camera_pos,
            rotation,
            camera_fov,
            width,
            height,
        ))
    return projections


def annotate_image(
    image,
    waypoints,
    camera_location,
    camera_rotation,
    camera_fov,
    route_crosswalks=None,
):
    """
    Annotate image with waypoints projected from 3D world coordinates.
    
    Args:
        image: numpy array image in BGR format (from unrealcv decode)
        waypoints: list of Vector objects with x, y coordinates
        camera_location: tuple of (x, y, z) camera position in UE coordinates
        camera_rotation: tuple of (pitch, yaw, roll) camera rotation in STANDARD format (already converted)
        camera_fov: horizontal field of view in degrees
    
    Returns:
        annotated_image: PIL Image with waypoints drawn on it
    """
    # Convert numpy array to PIL Image for drawing
    if len(image.shape) == 3 and image.shape[2] == 3:
        # BGR to RGB conversion
        image_rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)
    else:
        # If already RGB or grayscale
        pil_image = Image.fromarray(image)
    
    # Get image dimensions
    width, height = pil_image.size
    
    # Create drawing object
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font, fall back to default if not available
    try:
        # Try different font paths for different operating systems
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "C:/Windows/Fonts/arial.ttf",      # Windows
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, size=min(24, max(12, height // 30)))
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # Convert UE coordinates to standard coordinates
    std_camera_pos = convert_ue_to_std_location(camera_location)
    
    if camera_rotation is None or len(camera_rotation) != 3:
        camera_rotation = (0.0, 0.0, 0.0)
    
    try:
        std_cam_rot_matrix = get_rotation_matrix(*camera_rotation)
    except Exception:
        std_cam_rot_matrix = np.eye(3)

    _draw_route_crosswalks(
        draw,
        route_crosswalks or [],
        std_camera_pos,
        std_cam_rot_matrix,
        camera_fov,
        width,
        height,
    )
    
    waypoint_pixels = project_waypoints_to_image(
        waypoints,
        camera_location,
        camera_rotation,
        camera_fov,
        width,
        height,
    )
    for i, projection in enumerate(waypoint_pixels):
        if projection is None:
            continue

        u, v = projection
        marker_size = max(12, min(30, height // 40))
        draw.ellipse(
            [(u - marker_size, v - marker_size),
             (u + marker_size, v + marker_size)],
            fill="red",
            outline="white",
            width=2,
        )

        label = f"{i + 1}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = u - text_width // 2
        text_y = v - text_height // 2
        draw.text((text_x, text_y), label, fill="white", font=font)

    return pil_image


def _draw_route_crosswalks(
    draw,
    route_crosswalks,
    std_camera_pos,
    std_cam_rot_matrix,
    camera_fov,
    width,
    height,
):
    """Draw perspective zebra paint from the route graph into the RGB view.

    The model and the recorded first-person demo consume this same annotated
    frame. This is a visual correction for packaged road arms that omit paint;
    it never modifies route geometry, traffic state, or action execution.
    """
    enabled = os.environ.get(
        'SIMWORLD_RENDER_ROUTE_CROSSWALKS',
        '1',
    ).strip().lower() not in {'0', 'false', 'no', 'off'}
    if not enabled:
        return

    # The production path renders each stripe as one persistent, static,
    # non-colliding UE actor.  Never paint a second synthetic copy into model
    # inputs: the old overlay existed only in decision/output images and
    # disappeared from raw action frames, making the crossing visibly flash.
    ue_actor_markings = os.environ.get(
        'SIMWORLD_RENDER_ROUTE_CROSSWALK_UE_ACTORS',
        '1',
    ).strip().lower() in {'1', 'true', 'yes', 'on'}
    if ue_actor_markings:
        return

    projected_stripes = []
    camera_ue = (
        -float(std_camera_pos[2]),
        float(std_camera_pos[0]),
        float(std_camera_pos[1]),
    )
    for crosswalk in route_crosswalks:
        start = crosswalk.start
        end = crosswalk.end
        dx = float(end.x - start.x)
        dy = float(end.y - start.y)
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        axis_x, axis_y = dx / length, dy / length
        perpendicular_x, perpendicular_y = -axis_y, axis_x
        half_width = ROUTE_CROSSWALK_STRIPE_WIDTH_CM / 2.0
        half_band = ROUTE_CROSSWALK_BAND_LENGTH_CM / 2.0

        for index in range(ROUTE_CROSSWALK_STRIPE_COUNT):
            fraction = (index + 1.0) / (
                ROUTE_CROSSWALK_STRIPE_COUNT + 1.0
            )
            center_x = float(start.x) + dx * fraction
            center_y = float(start.y) + dy * fraction
            corners = []
            for along, across in (
                (-half_width, -half_band),
                (half_width, -half_band),
                (half_width, half_band),
                (-half_width, half_band),
            ):
                ue_corner = (
                    center_x + axis_x * along + perpendicular_x * across,
                    center_y + axis_y * along + perpendicular_y * across,
                    ROUTE_CROSSWALK_PAINT_Z_CM,
                )
                projection = project_point(
                    world_point_std=convert_ue_to_std_location(ue_corner),
                    camera_pos_std=std_camera_pos,
                    camera_rot_matrix_std=std_cam_rot_matrix,
                    fov=camera_fov,
                    width=width,
                    height=height,
                    clip_to_frame=False,
                )
                if projection is None:
                    corners = []
                    break
                corners.append(projection)
            if len(corners) != 4:
                continue
            coordinate_limit = 10 * max(width, height)
            if any(
                abs(u) > coordinate_limit or abs(v) > coordinate_limit
                for u, v in corners
            ):
                continue
            distance = math.hypot(
                center_x - camera_ue[0],
                center_y - camera_ue[1],
            )
            projected_stripes.append((distance, corners))

    for _, corners in sorted(projected_stripes, reverse=True):
        draw.polygon(
            corners,
            fill=(238, 238, 224),
            outline=(118, 118, 108),
        )


def deg2rad(deg):
    """Convert degree to radian"""
    return deg * math.pi / 180

def get_rotation_matrix(pitch, yaw, roll):
    """
    Calculate rotation matrix from standard aviation euler angles (Pitch, Yaw, Roll).
    Note: Input should be converted to standard coordinate system.
    """
    pitch, yaw, roll = deg2rad(pitch), deg2rad(yaw), deg2rad(roll)
    
    # Rotate around X axis (Pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])
    # Rotate around Y axis (Yaw)
    Ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])
    # Rotate around Z axis (Roll)
    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])
    
    # Standard rotation order: ZYX
    return Rz @ Ry @ Rx

def project_point(
    world_point_std,
    camera_pos_std,
    camera_rot_matrix_std,
    fov,
    width,
    height,
    clip_to_frame=True,
):
    """
    Project a single 3D world coordinate point to 2D screen coordinates.
    All input parameters must be in the standard coordinate system.
    """
    T = np.array(camera_pos_std)
    relative_pos = np.array(world_point_std) - T
    
    # Use the inverse of the rotation matrix (i.e., transpose) to convert the point from the world coordinate system to the camera coordinate system
    cam_coords = camera_rot_matrix_std.T @ relative_pos

    # In the camera coordinate system, +Z is backward, so Z must be negative to indicate in front of the camera
    if cam_coords[2] >= 0:
        return None  # The point is behind the camera or on the camera plane, invisible

    # Calculate the focal length based on the horizontal FOV (in pixels)
    focal_length = 0.5 * width / math.tan(deg2rad(fov / 2))
    
    # Perspective projection
    # The camera looks towards the -Z direction, so use -cam_coords[2]
    x_proj = cam_coords[0] / -cam_coords[2]
    y_proj = cam_coords[1] / -cam_coords[2]
    
    # Convert to screen pixel coordinates
    # Standard-camera X is UE +Y at zero yaw, which UE renders on the
    # visual right (left-handed world), so it maps to increasing image X.
    # Verified against lit UE renders of spawned markers.
    u = focal_length * x_proj + width / 2
    v = -focal_length * y_proj + height / 2  # Y axis flip
    
    # Check if the point is within the screen range (optional)
    if clip_to_frame and not (0 <= u < width and 0 <= v < height):
        return None

    return int(u), int(v)


def convert_ue_to_std_location(ue_location):
    """Convert UE location coordinates to standard coordinates"""
    x_ue, y_ue, z_ue = ue_location
    x_std = y_ue
    y_std = z_ue
    z_std = -x_ue
    return (x_std, y_std, z_std)

def convert_ue_to_std_euler(ue_rotation):
    """Convert UE rotation (Yaw, Pitch, Roll) to standard euler angles (Pitch, Yaw, Roll)"""
    yaw_ue, pitch_ue, roll_ue = ue_rotation
    yaw_std = yaw_ue
    pitch_std = pitch_ue
    roll_std = -roll_ue # UE's Roll is around X (front) axis, standard is around Z (front) axis, opposite direction
    return (pitch_std, yaw_std, roll_std)


def parse_ue_string_values(location_str, rotation_str):
    """
    Parse UE string values into numerical values.
    
    Args:
        location_str: string like '0.000 0.000 -0.000' (x y z)
        rotation_str: string like '0.000 0.000 -0.000' (pitch yaw roll) 
    
    Returns:
        tuple: (camera_location, camera_rotation)
            - camera_location: tuple of (x, y, z) floats
            - camera_rotation: tuple of (pitch, yaw, roll) floats  
    """
    try:
        # Parse location string "x y z" -> (x, y, z)
        location_parts = location_str.strip().split()
        if len(location_parts) >= 3:
            camera_location = (
                float(location_parts[0]), 
                float(location_parts[1]), 
                float(location_parts[2])
            )
        else:
            camera_location = (0.0, 0.0, 0.0)
            
        # Parse rotation string "pitch yaw roll" -> (pitch, yaw, roll)
        rotation_parts = rotation_str.strip().split()
        if len(rotation_parts) >= 3:
            camera_rotation = (
                float(rotation_parts[0]), 
                -float(rotation_parts[1]), 
                float(rotation_parts[2])
            )
        else:
            camera_rotation = (0.0, 0.0, 0.0)
            
        
        return camera_location, camera_rotation
        
    except (ValueError, AttributeError) as e:
        print(f"Warning: Failed to parse UE values: {e}")
        print(f"Location: {location_str}")
        print(f"Rotation: {rotation_str}") 
        # Return default values
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
