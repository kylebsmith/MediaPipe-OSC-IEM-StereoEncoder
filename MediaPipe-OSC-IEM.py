#!/usr/bin/env python3
"""
MediaPipe-OSC-IEM.py
--------------------
Kyle Smith

Hand-tracking only -> OSC out for IEM StereoEncoder.

- Draws a bounding box and centroid square for the detected hands.

Install:
    pip install opencv-python mediapipe python-osc numpy

Default OSC ports:
- Right hand → 9002
- Left hand  → 9003

Use --port-right and --port-left to override.

"""
import argparse, time, math, sys
from typing import Optional, Tuple
import numpy as np
import cv2
from pythonosc.udp_client import SimpleUDPClient

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("mediapipe is not installed. Run: pip install mediapipe\n" + str(e))


def ema(prev: Optional[float], new: float, alpha: float) -> float:
    return new if prev is None else (1.0 - alpha) * prev + alpha * new


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def map_cartesian(x: float, y: float, elev_max: float) -> Tuple[float, float]:
    """x -> azimuth (±180), y -> elevation (±elev_max)."""
    x = clamp01(x); y = clamp01(y)
    az = (2.0 * x - 1.0) * 180.0
    el = (0.5 - y) * (2.0 * elev_max)
    return az, el


def map_polar(x: float, y: float, elev_max: float) -> Tuple[float, float]:
    """
    Polar: angle around center -> azimuth, radius -> elevation.
    Move in a circle around the screen center to spin the source; distance from center sets elevation.
    """
    x = clamp01(x); y = clamp01(y)
    # center coordinate in [-1,1] with +cy being FRONT (screen top)
    cx = 2.0 * (x - 0.5)
    cy = 2.0 * (0.5 - y)
    # 0° at FRONT (positive cy), +90° at LEFT (negative cx), -90° at RIGHT (positive cx)
    angle_deg = -math.degrees(math.atan2(cx, cy))  # CCW positive, front=0
    r = min(1.0, math.hypot(cx, cy))
    az = angle_deg
    el = r * elev_max
    return az, el


def map_orbit(x: float, y: float, elev_max: float, prev_angle: Optional[float]) -> Tuple[float, float, float]:
    """
    Orbit: drive azimuth by *continuous* phase; moving horizontally advances phase,
    vertical sets elevation. Feels like a jog wheel.
    """
    x = clamp01(x); y = clamp01(y)
    speed = (x - 0.5) * 180.0  # deg per frame (scaled further by send rate)
    angle = (prev_angle if prev_angle is not None else 0.0) + speed
    angle = (angle + 180.0) % 360.0 - 180.0
    el = (0.5 - y) * (2.0 * elev_max)
    return angle, el, angle


def hand_bbox_from_landmarks(lm, w: int, h: int) -> Tuple[int, int, int, int]:
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    xmin = max(0, min(xs)) * w
    xmax = min(1, max(xs)) * w
    ymin = max(0, min(ys)) * h
    ymax = min(1, max(ys)) * h
    x = int(xmin); y = int(ymin)
    return x, y, int(xmax - xmin), int(ymax - ymin)


def hand_openness(lm) -> float:
    """Return a crude openness metric in [0,1] using fingertips vs wrist distance.
    0 ~ closed fist, 1 ~ open hand.
    """
    # landmark indices per MediaPipe Hands
    WRIST = 0
    TIPS = [8, 12, 16, 20]  # index, middle, ring, pinky tips
    wrist = lm[WRIST]
    # Normalized distances in [0,1] image space
    dists = []
    for t in TIPS:
        tip = lm[t]
        d = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2) ** 0.5
        dists.append(d)
    # Normalize by approximate hand size: distance wrist->index MCP (5) + wrist->pinky MCP (17)
    # Fallback to sum of two distances, small epsilon to avoid div0.
    base = 1e-6
    if len(lm) > 17:
        base = ((lm[5].x - wrist.x)**2 + (lm[5].y - wrist.y)**2) ** 0.5
        base += ((lm[17].x - wrist.x)**2 + (lm[17].y - wrist.y)**2) ** 0.5
        base = max(base, 1e-3)
    openness = sum(dists) / (len(dists) * base)
    # Clip to [0,1.5] then scale to [0,1]
    openness = max(0.0, min(1.5, openness)) / 1.5
    return openness


def main():
    ap = argparse.ArgumentParser(description="MediaPipe HAND -> IEM StereoEncoder via OSC")
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9002, help="Match StereoEncoder 'Listen to port'")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mirror", action="store_true", default=True,
                    help="Horizontally flip camera *before* tracking (default: on)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false",
                    help="Disable horizontal flip before tracking")
    ap.add_argument("--show", action="store_true", default=True, help="Show camera with overlay (on by default)")
    ap.add_argument("--alpha", type=float, default=0.35, help="EMA smoothing")
    ap.add_argument("--rate-limit", type=float, default=60.0, help="Max send rate (Hz)")
    ap.add_argument("--elev-max", type=float, default=60.0, help="Max |elevation| in degrees")
    ap.add_argument("--map", choices=["cartesian", "polar", "orbit"], default="cartesian",
                    help="Mapping strategy from (x,y) to (az, el)")
    ap.add_argument("--width-addr", default=None,
                    help="Optional OSC address to send width/spread (e.g., /StereoEncoder/width)")
    ap.add_argument("--width-scale", type=float, default=1.0,
                    help="Scale for width mapping from hand speed (try 0.5..2.0)")
    ap.add_argument("--az-offset", type=float, default=0.0,
                    help="Azimuth offset in degrees applied after mapping (e.g., 90 to make front=0)")
    ap.add_argument("--invert-az", action="store_true", default=True,
                    help="Invert azimuth sign after mapping (default: on)")
    ap.add_argument("--no-invert-az", dest="invert_az", action="store_false",
                    help="Disable azimuth inversion (use if rotation direction now feels wrong)")
    ap.add_argument("--width-mode", choices=["deg","norm"], default="deg",
                    help="Send width in degrees (0..180) or normalized (0..1)")
    ap.add_argument("--width-source", choices=["openness","speed","radius"], default="openness",
                    help="What drives width: hand openness (default), motion speed, or polar radius from center")
    ap.add_argument("--width-curve", type=float, default=1.0,
                    help="Apply a curve to width input: out = in**curve (1.0 = linear)")
    ap.add_argument("--width-alpha", type=float, default=0.3,
                    help="EMA smoothing for outgoing width value (0..1)")
    ap.add_argument("--hand", choices=["right", "left", "both"], default="both",
                    help="Which hand to track (right, left, or both)")
    ap.add_argument("--port-right", type=int, default=None,
                    help="OSC port for RIGHT hand (default: --port if set, else 9002)")
    ap.add_argument("--port-left", type=int, default=9003,
                    help="OSC port for LEFT hand (default: 9003)")
    # --repeats, --no-show, --bundle removed
    args = ap.parse_args()

    # Resolve per-hand ports (back-compat: --port applies to RIGHT if provided)
    port_right = args.port if args.port_right is None else args.port_right
    if port_right is None:
        port_right = 9002
    client_right = SimpleUDPClient(args.ip, port_right)
    client_left  = SimpleUDPClient(args.ip, args.port_left)
    # (Removed UDP send buffer hack)

    # Prefer AVFoundation on macOS (darwin) to avoid GUI/permission quirks
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        backend = "AVFOUNDATION"
    else:
        cap = cv2.VideoCapture(args.camera)
        backend = "DEFAULT"
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    min_dt = 1.0 / max(1e-3, args.rate_limit)

    # Per-hand state (independent smoothing per side, with last_send for pacing)
    state = {
        "right": {"sx":None, "sy":None, "saz":None, "sel":None, "prev_angle":None,
                   "prev_sx":None, "prev_sy":None, "speed_ema":None, "w_smooth":None,
                   "bbox":None, "open_metric":None, "radius_n":0.0, "last_send":0.0},
        "left":  {"sx":None, "sy":None, "saz":None, "sel":None, "prev_angle":None,
                   "prev_sx":None, "prev_sy":None, "speed_ema":None, "w_smooth":None,
                   "bbox":None, "open_metric":None, "radius_n":0.0, "last_send":0.0},
    }

    print(f"[INFO] OSC RIGHT -> {args.ip}:{port_right} | LEFT -> {args.ip}:{args.port_left} | map={args.map} (q quits)")
    print(f"[INFO] Camera opened on index {args.camera} via backend: {backend}")

    if args.show:
        try:
            cv2.namedWindow("Hand -> IEM OSC (q to quit)", cv2.WINDOW_NORMAL)
        except cv2.error:
            pass
        try:
            cv2.resizeWindow("Hand -> IEM OSC (q to quit)", 960, 540)
            cv2.moveWindow("Hand -> IEM OSC (q to quit)", 100, 80)
            print("[INFO] Preview window created. If you don't see it, check macOS Screen/Desktop Spaces and privacy perms.")
        except cv2.error:
            pass

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # Give a friendly message once, then break
                print("[WARN] No frames from camera. Try a different --camera index and check macOS Camera permissions for PyCharm/Python.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = hands.process(rgb)
            # --- dual-hand selection and processing ---
            # Reset HUD containers each frame (drawing will read from state)
            state["right"]["bbox"] = None
            state["left"]["bbox"] = None

            # Determine indices for each hand, corrected for mirror to match POV
            indices = {"right": None, "left": None}
            if res.multi_hand_landmarks and res.multi_handedness:
                for i, hnd in enumerate(res.multi_handedness):
                    raw = hnd.classification[0].label.lower()  # 'right' or 'left' from model
                    pov = ("left" if raw == "right" else "right") if args.mirror else raw
                    if pov in indices and indices[pov] is None:
                        indices[pov] = i

            # Which sides to process this frame
            sides = [args.hand] if args.hand in ("right","left") else ["right","left"]

            def process_side(side: str, idx: int):
                st = state[side]
                lm = res.multi_hand_landmarks[idx].landmark

                st["open_metric"] = hand_openness(lm)

                # centroid (normalized)
                x_norm = float(np.mean([p.x for p in lm]))
                y_norm = float(np.mean([p.y for p in lm]))
                st["sx"] = ema(st["sx"], x_norm, args.alpha)
                st["sy"] = ema(st["sy"], y_norm, args.alpha)

                # bbox for HUD
                bx, by, bw, bh = hand_bbox_from_landmarks(lm, w, h)
                st["bbox"] = (bx, by, bw, bh)

                # radius from center (for width-source=radius)
                cx_n = 2.0 * (st["sx"] - 0.5)
                cy_n = 2.0 * (0.5 - st["sy"])
                st["radius_n"] = min(1.0, math.hypot(cx_n, cy_n))

                # mapping
                if args.map == "cartesian":
                    az_raw, el_raw = map_cartesian(st["sx"], st["sy"], args.elev_max)
                elif args.map == "polar":
                    az_raw, el_raw = map_polar(st["sx"], st["sy"], args.elev_max)
                else:
                    az_raw, el_raw, st["prev_angle"] = map_orbit(st["sx"], st["sy"], args.elev_max, st["prev_angle"])

                # offset/invert
                az_raw = ((az_raw + args.az_offset + 180.0) % 360.0) - 180.0
                if args.invert_az:
                    az_raw = -az_raw

                st["saz"] = ema(st["saz"], az_raw, args.alpha)
                st["sel"] = ema(st["sel"], el_raw, args.alpha)

                # motion speed (for width-source=speed)
                if st["prev_sx"] is not None and st["prev_sy"] is not None:
                    dx = st["sx"] - st["prev_sx"]
                    dy = st["sy"] - st["prev_sy"]
                    inst_speed = math.hypot(dx, dy)
                    st["speed_ema"] = ema(st["speed_ema"], inst_speed, 0.2)
                st["prev_sx"], st["prev_sy"] = st["sx"], st["sy"]

                # choose per-hand client
                client = client_right if side == "right" else client_left

                # per-hand pacing
                now = time.monotonic()
                if now - st["last_send"] >= min_dt:
                    # width mapping
                    width_addr = args.width_addr or "/StereoEncoder/width"

                    if args.width_source == "openness":
                        win = 0.0 if st["open_metric"] is None else max(0.0, min(1.0, float(st["open_metric"])))
                    elif args.width_source == "speed":
                        base = 0.0 if st["speed_ema"] is None else float(st["speed_ema"]) * 5.0
                        win = max(0.0, min(1.0, base))
                    else:  # radius
                        win = max(0.0, min(1.0, float(st["radius_n"])) )

                    if args.width_curve != 1.0:
                        try: win = pow(win, float(args.width_curve))
                        except Exception: pass

                    wout = 180.0 * win if args.width_mode == "deg" else win
                    st["w_smooth"] = wout if st["w_smooth"] is None else ((1.0 - args.width_alpha) * st["w_smooth"] + args.width_alpha * wout)

                    # send OSC messages (one each)
                    try:
                        client.send_message("/StereoEncoder/azimuth", float(st["saz"]))
                        client.send_message("/StereoEncoder/elevation", float(st["sel"]))
                        client.send_message(width_addr, float(st["w_smooth"]))
                    except Exception:
                        pass

                    st["last_send"] = now

            # Run per requested side
            for side in sides:
                idx = indices.get(side)
                if idx is not None:
                    process_side(side, idx)

            # draw overlay
            if args.show:
                # crosshair
                cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
                cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

                hud_hand = args.hand if not args.mirror else ("left" if args.hand == "right" else "right")
                cv2.putText(frame, f"hand={hud_hand}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
                cv2.putText(frame, f"mirror={'on' if args.mirror else 'off'}", (10, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

                # Draw RIGHT hand (green/yellow) and LEFT hand (magenta/cyan)
                for side, colors in (("right", ((0,255,0),(0,255,255))), ("left", ((255,0,255),(255,255,0)))):
                    st = state[side]
                    if st["bbox"] is not None:
                        bx, by, bw, bh = st["bbox"]
                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), colors[0], 2)
                    if st["sx"] is not None and st["sy"] is not None:
                        cx, cy = int(st["sx"] * w), int(st["sy"] * h)
                        cv2.rectangle(frame, (cx - 8, cy - 8), (cx + 8, cy + 8), colors[1], 2)
                        cv2.putText(frame, f"{side}: (x={st['sx']:.2f}, y={st['sy']:.2f})", (10, 24 if side=='right' else 44),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[0], 2)
                    if st["saz"] is not None and st["sel"] is not None:
                        cv2.putText(frame, f"{side}: az={st['saz']:6.1f} el={st['sel']:6.1f}", (10, 70 if side=='right' else 96),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[1], 2)
                    if st["open_metric"] is not None:
                        # width debug
                        if args.width_source == "openness":
                            win_dbg = max(0.0, min(1.0, float(st["open_metric"])) )
                        elif args.width_source == "speed":
                            base = st["speed_ema"] if st["speed_ema"] is not None else 0.0
                            win_dbg = max(0.0, min(1.0, base * 5.0))
                        else:
                            win_dbg = max(0.0, min(1.0, float(st["radius_n"])) )
                        out_dbg = 180.0 * win_dbg if args.width_mode == "deg" else win_dbg
                        if st["w_smooth"] is not None:
                            cv2.putText(frame, f"{side}: width~{st['w_smooth']:.2f}{'deg' if args.width_mode=='deg' else ''}", (10, 122 if side=='right' else 148),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[0], 2)

                cv2.imshow("Hand -> IEM OSC (q to quit)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    pass  # allow graceful quit via Ctrl+C too
            else:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    pass

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()