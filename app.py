import os
from collections import deque

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(page_title="ADAS Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

FONT_PATH = "Oswald-VariableFont_wght.ttf"
ADAS_CLASSES = [0, 2, 3, 5, 7, 9, 11]  # person, car, motorcycle, bus, truck
COLLISION_DISTANCE_M = 15.0
LANE_DEVIATION_PX = 30
FRAME_SIZE = (854, 480)

PALETTE = {
    "critical":  {"bgr": (0,  30,  220), "rgb": (220, 30,   0)},
    "in_lane":   {"bgr": (0,  165, 255), "rgb": (255, 165,  0)},
    "off_lane":  {"bgr": (180, 180, 40), "rgb": (40,  180, 180)},
    "lane_line": {"bgr": (50,  230, 50), "rgb": (50,  230,  50)},
    "hud_bg":    {"bgr": (18,  18,  18), "rgb": (18,  18,   18)},
    "lane_fill": {"bgr": (0,  160, 240), "rgb": (240, 160,  0)},
    "lane_ok":   {"bgr": (0,  200, 80),  "rgb": (80,  200,  0)},
    "lane_warn": {"bgr": (0,   50, 220), "rgb": (220, 50,   0)},
}

VEHICLE_WIDTHS_M = {"CAR": 1.8, "TRUCK": 2.5, "BUS": 3.0, "MOTORCYCLE": 0.8}
ASSUMED_FOCAL_LENGTH = 750  # px, calibrated against the test footage


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

st.session_state.setdefault("playing", False)
st.session_state.setdefault("current_frame", 0)
st.session_state.setdefault("total_frames", 0)
st.session_state.setdefault("seek_to", None)
st.session_state.setdefault("last_bev", None)
st.session_state.setdefault("last_stats", None)


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")


@st.cache_data
def load_font(size, path=FONT_PATH):
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------

def to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_text(image, text, xy, font, fill=(255, 255, 255), shadow=True, shadow_fill=(0, 0, 0), anchor="lt"):
    draw = ImageDraw.Draw(image)
    x, y = xy
    if shadow:
        draw.text((x + 2, y + 2), text, font=font, fill=shadow_fill, anchor=anchor)
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)
    return image


def draw_box(image, x1, y1, x2, y2, color, thickness=2):
    draw = ImageDraw.Draw(image)
    for i in range(thickness):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    return image


def draw_label_pill(image, text, x, y, font, text_color=(255, 255, 255), bg_color=(220, 30, 0),
                     pad_x=6, pad_y=3):
    draw = ImageDraw.Draw(image)
    bbox = font.getbbox(text)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img_w, img_h = image.size

    pill_h = text_h + pad_y * 2
    x1, x2 = max(0, x), min(max(0, x) + text_w + pad_x * 2, img_w - 1)
    y2 = y - 2
    y1 = y2 - pill_h
    if y1 < 0:
        y1 = max(0, y + 2)
        y2 = y1 + pill_h

    draw.rounded_rectangle([x1, y1, x2, y2], radius=4, fill=bg_color)
    draw.text((x1 + pad_x - bbox[0], y1 + pad_y - bbox[1]), text, font=font, fill=text_color)
    return image


# ----------------------------------------------------------------------------
# Distance + lane detection
# ----------------------------------------------------------------------------

def estimate_distance_m(pixel_width, class_name):
    real_width = VEHICLE_WIDTHS_M.get(class_name, 1.8)
    pixel_width = max(pixel_width, 1)
    return (real_width * ASSUMED_FOCAL_LENGTH) / pixel_width


def trapezoid_roi_mask(image, horizon_ratio=0.55, hood_ratio=1.0, top_spread=0.10, bottom_spread=0.90):
    """Mask out everything but a trapezoid in front of the vehicle, anchored to
    the bottom corners of the frame, narrowing toward the horizon."""
    h, w = image.shape[:2]
    mask = np.zeros_like(image)

    top_y, bottom_y = int(h * horizon_ratio), int(h * hood_ratio)
    cx = w // 2
    top_half, bottom_half = int((w * top_spread) / 2), int((w * bottom_spread) / 2)

    trapezoid = np.array([[
        (cx - bottom_half, bottom_y),
        (cx - top_half, top_y),
        (cx + top_half, top_y),
        (cx + bottom_half, bottom_y),
    ]])
    cv2.fillPoly(mask, trapezoid, 255)
    return cv2.bitwise_and(image, image, mask=mask)


def find_lane_segments(frame):
    """Return weighted (slope, intercept, length) candidates for the left and
    right lane boundaries based on Hough line detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    roi = trapezoid_roi_mask(edges)

    lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi / 180, threshold=30,
                             minLineLength=40, maxLineGap=150)

    left_candidates, right_candidates = [], []
    width = frame.shape[1]
    left_limit, right_limit = width * 0.85, width * 0.15

    if lines is None:
        return left_candidates, right_candidates

    for (x1, y1, x2, y2), in lines.reshape(-1, 1, 4):
        if x1 == x2:
            continue
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        length = np.hypot(x2 - x1, y2 - y1)

        if -2.5 < slope < -0.5 and x1 < left_limit and x2 < left_limit:
            left_candidates.append((slope, intercept, length))
        elif 0.5 < slope < 2.5 and x1 > right_limit and x2 > right_limit:
            right_candidates.append((slope, intercept, length))

    return left_candidates, right_candidates


def weighted_average_line(candidates):
    weights = [c[2] for c in candidates]
    return np.average(candidates, axis=0, weights=weights)


# ----------------------------------------------------------------------------
# Bird's-eye view rendering (pseudo-3D chase view)
# ----------------------------------------------------------------------------

def lateral_position_in_lane(obj_x, obj_y, lane_eqs):
    """Project an object's screen position into a ratio across the detected
    lane: 0 = on the left lane line, 1 = on the right lane line. Falls back
    to lane center if the lane geometry is degenerate at this y."""
    m1, b1, m2, b2 = lane_eqs

    if abs(m1) < 1e-3 or abs(m2) < 1e-3:
        return 0.5

    lane_left_x = (obj_y - b1) / m1
    lane_right_x = (obj_y - b2) / m2
    lane_width = lane_right_x - lane_left_x

    if lane_width <= 20:
        return 0.5

    ratio = (obj_x - lane_left_x) / lane_width
    return float(np.clip(ratio, -0.6, 1.6))  # allow slight overshoot for adjacent-lane traffic


def project_to_chase_view(obj_x, obj_y, distance_m, lane_eqs, view):
    """Map a detection's screen-space foot point + estimated distance into
    the chase-view canvas, using real lane geometry when available."""
    max_depth_m = view["max_depth_m"]
    depth_t = min(distance_m, max_depth_m) / max_depth_m  # 0 = at car, 1 = horizon

    # Perspective depth: objects further away sit higher (toward horizon_y)
    # and the available lane width narrows toward the vanishing point.
    cy = view["ego_y"] + (view["horizon_y"] - view["ego_y"]) * (depth_t ** 0.9)

    half_width_here = view["half_width_at"](depth_t)
    center_x = view["center_x"]

    if lane_eqs:
        ratio = lateral_position_in_lane(obj_x, obj_y, lane_eqs)
        # 0 -> left edge of ego lane, 1 -> right edge of ego lane
        cx = center_x + (ratio - 0.5) * 2 * half_width_here
    else:
        dx_px = obj_x - (FRAME_SIZE[0] / 2)
        dx_m = (dx_px * distance_m) / ASSUMED_FOCAL_LENGTH
        cx = center_x + (dx_m / view["lane_half_width_m"]) * half_width_here

    cx = max(view["road_left_at"](depth_t) + 8, min(view["road_right_at"](depth_t) - 8, cx))
    scale = 0.35 + 0.65 * (1 - depth_t)  # nearer objects render larger
    return cx, cy, scale


def draw_chase_sedan(draw, x, y, scale, color, is_ego=False):
    w, h = 40 * scale, 76 * scale
    draw.rounded_rectangle([x - w / 2, y - h, x + w / 2, y], radius=8 * scale, fill=color)
    draw.polygon([(x - w * 0.35, y - h * 0.72), (x + w * 0.35, y - h * 0.72),
                   (x + w * 0.4, y - h * 0.42), (x - w * 0.4, y - h * 0.42)], fill=(22, 27, 34))
    draw.polygon([(x - w * 0.35, y - h * 0.32), (x + w * 0.35, y - h * 0.32),
                   (x + w * 0.4, y - h * 0.1), (x - w * 0.4, y - h * 0.1)], fill=(22, 27, 34))
    draw.rectangle([x - w * 0.45, y - 6 * scale, x - w * 0.2, y], fill=(255, 60, 60))
    draw.rectangle([x + w * 0.2, y - 6 * scale, x + w * 0.45, y], fill=(255, 60, 60))
    if is_ego:
        draw.rectangle([x - w * 0.45, y - h, x - w * 0.2, y - h + 6 * scale], fill=(190, 250, 255))
        draw.rectangle([x + w * 0.2, y - h, x + w * 0.45, y - h + 6 * scale], fill=(190, 250, 255))


def draw_chase_truck(draw, x, y, scale, color):
    w, h = 50 * scale, 112 * scale
    cab_h = h * 0.25
    draw.rounded_rectangle([x - w / 2, y - h + cab_h, x + w / 2, y], radius=2 * scale, fill=color)
    draw.rounded_rectangle([x - w * 0.4, y - h, x + w * 0.4, y - h + cab_h], radius=4 * scale, fill=(155, 165, 175))
    draw.rectangle([x - w * 0.35, y - h + 2 * scale, x + w * 0.35, y - h + cab_h - 4 * scale], fill=(22, 27, 34))
    draw.rectangle([x - w / 2 + 2 * scale, y - 8 * scale, x - w * 0.2, y], fill=(255, 60, 60))
    draw.rectangle([x + w * 0.2, y - 8 * scale, x + w / 2 - 2 * scale, y], fill=(255, 60, 60))


def build_chase_view(size, curve_offset):
    w, h = size
    horizon_y = h * 0.16
    ego_y = h - 56
    center_x = w / 2 + curve_offset * 0.4

    road_half_top, road_half_bottom = w * 0.16, w * 0.62

    def road_half_at(t):
        return road_half_top + (road_half_bottom - road_half_top) * (1 - t)

    return {
        "horizon_y": horizon_y,
        "ego_y": ego_y,
        "center_x": center_x,
        "max_depth_m": 60.0,
        "lane_half_width_m": 1.85,
        "half_width_at": lambda t: road_half_at(t) / 3,  # ego-lane half-width
        "road_left_at": lambda t: center_x - road_half_at(t),
        "road_right_at": lambda t: center_x + road_half_at(t),
    }


def render_bev(detections, turn_status, departure_status, lane_eqs, size=(360, 460)):
    w, h = size
    bev = Image.new("RGBA", (w, h), color=(10, 12, 17, 255))
    draw = ImageDraw.Draw(bev)

    curve_offset = 26 if "LEFT" in turn_status else (-26 if "RIGHT" in turn_status else 0)
    view = build_chase_view(size, curve_offset)
    horizon_y, ego_y, center_x = view["horizon_y"], view["ego_y"], view["center_x"]

    # --- Sky / ground gradient -------------------------------------------------
    sky_top, sky_bottom = (16, 19, 26), (24, 28, 38)
    for y in range(0, int(horizon_y)):
        t = y / max(horizon_y, 1)
        col = tuple(int(sky_top[i] + (sky_bottom[i] - sky_top[i]) * t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=col + (255,))

    ground_top, ground_bottom = (26, 30, 38), (15, 17, 23)
    for y in range(int(horizon_y), h):
        t = (y - horizon_y) / max(h - horizon_y, 1)
        col = tuple(int(ground_top[i] + (ground_bottom[i] - ground_top[i]) * t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=col + (255,))

    # --- Road surface as a perspective trapezoid --------------------------------
    steps = 48
    for i in range(steps):
        t0, t1 = i / steps, (i + 1) / steps
        y0 = horizon_y + (ego_y - horizon_y) * t0
        y1 = horizon_y + (ego_y - horizon_y) * t1
        hw0, hw1 = view["road_left_at"](1 - t0), view["road_left_at"](1 - t1)
        rw0, rw1 = view["road_right_at"](1 - t0), view["road_right_at"](1 - t1)
        shade = 30 + int(10 * t0)
        draw.polygon([(hw0, y0), (rw0, y0), (rw1, y1), (hw1, y1)], fill=(shade, shade + 3, shade + 8, 255))

    # Road edge lines
    edge_pts_l = [(view["road_left_at"](1 - i / steps), horizon_y + (ego_y - horizon_y) * (i / steps)) for i in range(steps + 1)]
    edge_pts_r = [(view["road_right_at"](1 - i / steps), horizon_y + (ego_y - horizon_y) * (i / steps)) for i in range(steps + 1)]
    draw.line(edge_pts_l, fill=(150, 160, 175, 255), width=3)
    draw.line(edge_pts_r, fill=(150, 160, 175, 255), width=3)

    # Dashed lane dividers (left/right of ego lane), perspective-scaled
    for divider_ratio in (-1 / 3, 1 / 3):
        dash_on = True
        steps_dash = 36
        for i in range(steps_dash):
            t0, t1 = i / steps_dash, (i + 1) / steps_dash
            if dash_on:
                y0 = horizon_y + (ego_y - horizon_y) * t0
                y1 = horizon_y + (ego_y - horizon_y) * t1
                half0 = view["half_width_at"](1 - t0)
                half1 = view["half_width_at"](1 - t1)
                x0 = center_x + divider_ratio * 3 * half0
                x1 = center_x + divider_ratio * 3 * half1
                width_px = max(1, int(3 * (1 - t0)))
                draw.line([(x0, y0), (x1, y1)], fill=(120, 200, 255, 200), width=width_px)
            dash_on = not dash_on

    # --- Ego vehicle -------------------------------------------------------------
    draw_chase_sedan(draw, w / 2, ego_y, 1.0, (50, 210, 255), is_ego=True)

    # --- Detected vehicles --------------------------------------------------------
    label_font = load_font(13)
    ordered = sorted(detections, key=lambda d: -d[4])  # draw farthest first

    for class_name, obj_x, obj_y, is_critical, distance_m, _track_id in ordered:
        cx, cy, scale = project_to_chase_view(obj_x, obj_y, distance_m, lane_eqs, view)

        color = (255, 70, 70) if is_critical else (255, 185, 60)
        if "TRUCK" in class_name or "BUS" in class_name:
            draw_chase_truck(draw, cx, cy, scale, color)
        else:
            draw_chase_sedan(draw, cx, cy, scale, color)

        label = f"{distance_m:.1f}m"
        bbox = label_font.getbbox(label)
        tx, ty = cx + 18 * scale, cy - 14 * scale
        draw.rounded_rectangle(
            [tx - 4, ty - 2, tx + (bbox[2] - bbox[0]) + 4, ty + (bbox[3] - bbox[1]) + 4],
            radius=4, fill=(18, 20, 26, 210),
        )
        draw.text((tx, ty), label, font=label_font, fill=(190, 235, 255))

    # --- Departure warning frame --------------------------------------------------
    if "DEPARTING" in departure_status:
        draw.rectangle([0, 0, w - 1, h - 1], outline=(255, 50, 65, 220), width=6)

    # --- Vignette -------------------------------------------------------------------
    for i in range(36):
        alpha = int(200 * (1 - i / 36))
        draw.line([(0, i), (w, i)], fill=(10, 12, 17, alpha), width=1)

    return bev.convert("RGB")


# ----------------------------------------------------------------------------
# Lane geometry for the current frame
# ----------------------------------------------------------------------------

class LaneState:
    """Tracks smoothed left/right lane lines across frames."""

    def __init__(self, history_len=10):
        self.left_history = deque(maxlen=history_len)
        self.right_history = deque(maxlen=history_len)

    def update(self, frame):
        left_candidates, right_candidates = find_lane_segments(frame)
        if left_candidates:
            self.left_history.append(weighted_average_line(left_candidates))
        if right_candidates:
            self.right_history.append(weighted_average_line(right_candidates))

    def current_lines(self):
        if not self.left_history or not self.right_history:
            return None
        left = np.mean(self.left_history, axis=0)
        right = np.mean(self.right_history, axis=0)
        return left[0], left[1], right[0], right[1]  # m1, b1, m2, b2


def build_lane_polygon(lane_eqs, height):
    m1, b1, m2, b2 = lane_eqs
    y1 = height
    lx1, rx1 = int((y1 - b1) / m1), int((y1 - b2) / m2)

    target_y2 = int(height * 0.60)
    if abs(m1 - m2) > 1e-3:
        ix = (b2 - b1) / (m1 - m2)
        iy = m1 * ix + b1
        if iy > target_y2:
            target_y2 = int(iy + 25)
    target_y2 = min(target_y2, int(height * 0.80))

    ly2 = ry2 = target_y2
    lx2, rx2 = int((ly2 - b1) / m1), int((ry2 - b2) / m2)

    polygon = np.array([[lx1, y1], [lx2, ly2], [rx2, ry2], [rx1, y1]], dtype=np.int32)
    points = {"lx1": lx1, "ly1": y1, "lx2": lx2, "ly2": ly2, "rx1": rx1, "ry1": y1, "rx2": rx2, "ry2": ry2}
    return polygon, points


def classify_road_state(lane_eqs, polygon_points, img_center):
    m1, _, m2, _ = lane_eqs
    turn_status = "Straight Road"
    if abs(m1) < 0.7 and m2 < 1.0:
        turn_status = "CURVE AHEAD: LEFT <"
    elif abs(m1) > 1.3 and m2 > 0.7:
        turn_status = "CURVE AHEAD: RIGHT >"

    lane_center = (polygon_points["lx1"] + polygon_points["rx1"]) // 2
    deviation = lane_center - img_center
    if deviation < -LANE_DEVIATION_PX:
        departure_status = "! DEPARTING RIGHT !"
    elif deviation > LANE_DEVIATION_PX:
        departure_status = "! DEPARTING LEFT !"
    else:
        departure_status = "LANE KEEP ASSIST: OK"

    return turn_status, departure_status


def draw_lane_overlay(frame, polygon, polygon_points):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], PALETTE["lane_fill"]["bgr"])
    frame = cv2.addWeighted(frame, 0.95, overlay, 0.05, 0)  # faint ghost lane fill

    p = polygon_points
    cv2.line(frame, (p["lx1"], p["ly1"]), (p["lx2"], p["ly2"]), PALETTE["lane_line"]["bgr"], 5)
    cv2.line(frame, (p["rx1"], p["ry1"]), (p["rx2"], p["ry2"]), PALETTE["lane_line"]["bgr"], 5)
    cv2.line(frame, (p["lx2"], p["ly2"]), (p["rx2"], p["ry2"]), PALETTE["lane_line"]["bgr"], 3)
    return frame


# ----------------------------------------------------------------------------
# Object detection + annotation
# ----------------------------------------------------------------------------

def is_inside_lane(box_x, y1, y2, box_height, lane_polygon):
    p1, p2 = (box_x, y2), (box_x, int(y1 + box_height * 0.75))
    return any(cv2.pointPolygonTest(lane_polygon, p, False) >= 0 for p in (p1, p2))


def annotate_detections(pil_frame, results, model_names, lane_polygon, font_sm):
    detections = []
    collision_alert = False

    if results.boxes.id is None:
        return detections, collision_alert

    boxes = results.boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        class_name = model_names[int(boxes.cls[i])].upper()
        track_id = int(boxes.id[i])

        distance_m = estimate_distance_m(x2 - x1, class_name)
        label = f"{class_name} #{track_id} | {distance_m:.1f}m"

        center_x, foot_y, box_height = (x1 + x2) // 2, y2, y2 - y1
        is_critical = False

        if lane_polygon is not None and is_inside_lane(center_x, y1, y2, box_height, lane_polygon):
            if distance_m < COLLISION_DISTANCE_M:
                collision_alert, is_critical = True, True
                draw_box(pil_frame, x1, y1, x2, y2, PALETTE["critical"]["rgb"], 4)
                draw_label_pill(pil_frame, f"CRITICAL: {label}", x1, y1, font_sm, bg_color=PALETTE["critical"]["rgb"])
            else:
                draw_box(pil_frame, x1, y1, x2, y2, PALETTE["in_lane"]["rgb"], 2)
                draw_label_pill(pil_frame, label, x1, y1, font_sm, bg_color=PALETTE["in_lane"]["rgb"])
        elif lane_polygon is not None:
            draw_box(pil_frame, x1, y1, x2, y2, PALETTE["off_lane"]["rgb"], 1)
            draw_label_pill(pil_frame, label, x1, y1, font_sm, text_color=(20, 20, 20), bg_color=PALETTE["off_lane"]["rgb"])
        else:
            draw_box(pil_frame, x1, y1, x2, y2, PALETTE["off_lane"]["rgb"], 1)

        detections.append((class_name, center_x, foot_y, is_critical, distance_m, track_id))

    return detections, collision_alert


def draw_hud(frame, turn_status, departure_status, collision_alert, font_md, font_xl):
    height, width = frame.shape[:2]

    hud = frame.copy()
    cv2.rectangle(hud, (0, 0), (width, 90), PALETTE["hud_bg"]["bgr"], -1)
    frame = cv2.addWeighted(frame, 0.65, hud, 0.35, 0)

    pil_frame = to_pil(frame)
    draw_text(pil_frame, f"ROAD: {turn_status}", (20, 14), font_md)

    departure_color = PALETTE["lane_warn"]["rgb"] if "!" in departure_status else PALETTE["lane_ok"]["rgb"]
    draw_text(pil_frame, departure_status, (20, 50), font_md, fill=departure_color)

    if collision_alert:
        draw_text(pil_frame, "BRAKE NOW", (width - 200, 30), font_xl, fill=(255, 50, 50))

    return to_cv2(pil_frame)


# ----------------------------------------------------------------------------
# Frame pipeline
# ----------------------------------------------------------------------------

def nearest_distance(detections):
    if not detections:
        return None
    return min(d[4] for d in detections)


def process_frame(frame, model, lane_state, fonts):
    frame = cv2.resize(frame, FRAME_SIZE)
    height, width = frame.shape[:2]
    img_center = width // 2

    # Run detection on the clean, unmodified frame first so lane overlays
    # never affect what YOLO sees.
    results = model.track(frame, classes=ADAS_CLASSES, conf=0.45, persist=True,
                           tracker="bytetrack.yaml", verbose=False)[0]

    lane_state.update(frame)
    lane_eqs = lane_state.current_lines()

    lane_polygon = None
    turn_status, departure_status = "Straight Road", "System Ready"

    if lane_eqs:
        lane_polygon, polygon_points = build_lane_polygon(lane_eqs, height)
        frame = draw_lane_overlay(frame, lane_polygon, polygon_points)
        turn_status, departure_status = classify_road_state(lane_eqs, polygon_points, img_center)

    pil_frame = to_pil(frame)
    detections, collision_alert = annotate_detections(
        pil_frame, results, model.names, lane_polygon, fonts["sm"]
    )
    frame = to_cv2(pil_frame)
    frame = draw_hud(frame, turn_status, departure_status, collision_alert, fonts["md"], fonts["xl"])

    bev_image = render_bev(detections, turn_status, departure_status, lane_eqs)

    stats = {
        "turn_status": turn_status,
        "departure_status": departure_status,
        "collision_alert": collision_alert,
        "object_count": len(detections),
        "nearest_m": nearest_distance(detections),
        "lane_detected": lane_eqs is not None,
    }
    return frame, bev_image, stats


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --chassis: #0B0E14;
    --panel: #11161F;
    --panel-border: #1C2330;
    --accent: #39E0C4;
    --alert: #FF4B5C;
    --muted: #7C8699;
    --text: #E8ECF4;
}

.stApp { background-color: var(--chassis); }

[data-testid="stHeader"] { background-color: transparent; }

.block-container { padding-top: 1.5rem; max-width: 1400px; }

h1, h2, h3 { font-family: 'Oswald', sans-serif; letter-spacing: 0.02em; }

.console-title {
    font-family: 'Oswald', sans-serif;
    font-weight: 600;
    font-size: 1.7rem;
    letter-spacing: 0.12em;
    color: var(--text);
    text-transform: uppercase;
    margin-bottom: 0.1rem;
}

.console-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    margin-bottom: 1.1rem;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 0.9rem 1rem;
}

.panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.gauge {
    border-bottom: 1px solid var(--panel-border);
    padding: 0.55rem 0;
}
.gauge:last-child { border-bottom: none; }

.gauge-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.15rem;
}

.gauge-value {
    font-family: 'Oswald', sans-serif;
    font-weight: 500;
    font-size: 1.45rem;
    color: var(--text);
    line-height: 1.15;
}

.gauge-value.accent { color: var(--accent); }
.gauge-value.alert { color: var(--alert); }
.gauge-value.ok { color: #5FE08C; }

.gauge-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    margin-left: 0.25rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
}

.stSlider { padding-top: 0.3rem; }
.stSlider [data-baseweb="slider"] { padding-bottom: 0.2rem; }

div[data-testid="stImage"] img { border-radius: 6px; }

.stButton button {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    border-radius: 5px;
}
.stButton button:hover {
    border-color: var(--accent);
    color: var(--accent);
}

.stTextInput input {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    border-radius: 5px;
}

[data-testid="stExpander"] {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)


def render_telemetry_rail(stats, current_frame, total_frames):
    if stats is None:
        stats = {
            "turn_status": "STANDBY",
            "departure_status": "SYSTEM IDLE",
            "collision_alert": False,
            "object_count": 0,
            "nearest_m": None,
            "lane_detected": False,
        }

    dep = stats["departure_status"]
    dep_class = "alert" if "!" in dep else ("ok" if "OK" in dep else "")

    nearest = f'{stats["nearest_m"]:.1f}' if stats["nearest_m"] is not None else "—"
    nearest_class = "alert" if (stats["nearest_m"] is not None and stats["nearest_m"] < COLLISION_DISTANCE_M) else "accent"

    lane_text = "LOCKED" if stats["lane_detected"] else "SEARCHING"
    lane_class = "ok" if stats["lane_detected"] else ""

    progress_pct = (current_frame / total_frames * 100) if total_frames else 0

    html = f"""
    <div class="panel">
        <div class="panel-label">Telemetry</div>
        <div class="gauge">
            <div class="gauge-label">Lane Keep Assist</div>
            <div class="gauge-value {dep_class}">{dep}</div>
        </div>
        <div class="gauge">
            <div class="gauge-label">Road Geometry</div>
            <div class="gauge-value">{stats["turn_status"]}</div>
        </div>
        <div class="gauge">
            <div class="gauge-label">Nearest Object</div>
            <div class="gauge-value {nearest_class}">{nearest}<span class="gauge-unit">m</span></div>
        </div>
        <div class="gauge">
            <div class="gauge-label">Tracked Objects</div>
            <div class="gauge-value">{stats["object_count"]}</div>
        </div>
        <div class="gauge">
            <div class="gauge-label">Lane Lock</div>
            <div class="gauge-value {lane_class}">{lane_text}</div>
        </div>
        <div class="gauge">
            <div class="gauge-label">Playback Position</div>
            <div class="gauge-value">{progress_pct:.0f}<span class="gauge-unit">%</span></div>
        </div>
    </div>
    """
    return html


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.markdown('<div class="console-title">ADAS Telemetry Console</div>', unsafe_allow_html=True)
st.markdown('<div class="console-subtitle">LANE TRACKING · OBJECT DETECTION · COLLISION WARNING</div>', unsafe_allow_html=True)

with st.expander("Video source", expanded=False):
    video_path = st.text_input("File path", "videos/test5.mp4", label_visibility="collapsed")
    if st.button("Load video"):
        st.session_state.current_frame = 0
        st.session_state.seek_to = 0
        st.session_state.playing = False
        cap = cv2.VideoCapture(video_path)
        st.session_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

if st.session_state.total_frames == 0:
    cap = cv2.VideoCapture(video_path)
    st.session_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

total_frames = max(st.session_state.total_frames, 1)

video_col, bev_col, rail_col = st.columns([2.4, 1, 0.9], gap="medium")

with video_col:
    st.markdown('<div class="panel-label">Forward Camera</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()

    slider_value = st.slider(
        "Scrub",
        min_value=0,
        max_value=total_frames - 1,
        value=min(st.session_state.current_frame, total_frames - 1),
        label_visibility="collapsed",
        key="frame_slider",
        disabled=st.session_state.playing,
    )
    if slider_value != st.session_state.current_frame:
        st.session_state.current_frame = slider_value
        st.session_state.seek_to = slider_value
        st.session_state.playing = False

    ctrl_a, ctrl_b, ctrl_c = st.columns(3)
    if ctrl_a.button("▶  Play", use_container_width=True):
        st.session_state.playing = True
    if ctrl_b.button("‖  Pause", use_container_width=True):
        st.session_state.playing = False
    if ctrl_c.button("⟲  Restart", use_container_width=True):
        st.session_state.current_frame = 0
        st.session_state.seek_to = 0
        st.session_state.playing = False

with bev_col:
    st.markdown('<div class="panel-label">Bird\'s-Eye View</div>', unsafe_allow_html=True)
    bev_placeholder = st.empty()
    if st.session_state.last_bev is not None:
        bev_placeholder.image(st.session_state.last_bev, use_container_width=True)

with rail_col:
    rail_placeholder = st.empty()
    rail_placeholder.markdown(
        render_telemetry_rail(st.session_state.last_stats, st.session_state.current_frame, total_frames),
        unsafe_allow_html=True,
    )

def render_static_frame(video_path, frame_index, total_frames, video_placeholder, bev_placeholder, rail_placeholder):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, raw_frame = cap.read()
    cap.release()
    if not success:
        return

    model = load_model()
    fonts = {"sm": load_font(18), "md": load_font(22), "xl": load_font(36)}
    lane_state = LaneState()

    frame, bev_image, stats = process_frame(raw_frame, model, lane_state, fonts)
    video_placeholder.image(frame, channels="BGR", use_container_width=True)
    bev_placeholder.image(bev_image, use_container_width=True)
    st.session_state.last_bev = bev_image
    st.session_state.last_stats = stats
    rail_placeholder.markdown(
        render_telemetry_rail(stats, frame_index, total_frames),
        unsafe_allow_html=True,
    )


if not st.session_state.playing:
    render_static_frame(video_path, st.session_state.current_frame, total_frames,
                         video_placeholder, bev_placeholder, rail_placeholder)

if st.session_state.playing:
    model = load_model()
    fonts = {"sm": load_font(18), "md": load_font(22), "xl": load_font(36)}
    lane_state = LaneState()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)

    while cap.isOpened() and st.session_state.playing:
        success, raw_frame = cap.read()
        if not success:
            st.session_state.current_frame = 0
            st.session_state.playing = False
            break

        st.session_state.current_frame += 1
        frame, bev_image, stats = process_frame(raw_frame, model, lane_state, fonts)

        st.session_state.last_bev = bev_image
        st.session_state.last_stats = stats

        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        bev_placeholder.image(bev_image, use_container_width=True)
        rail_placeholder.markdown(
            render_telemetry_rail(stats, st.session_state.current_frame, total_frames),
            unsafe_allow_html=True,
        )

    cap.release()
    st.rerun()