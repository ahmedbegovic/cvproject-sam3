import cv2
import numpy as np
import supervision as sv
import pickle
import json
from pathlib import Path
from tqdm import tqdm

# Config
TARGET_PLAYER_ID = None  
HOME = Path.cwd()
DATA_PATH = HOME / "output" / "tracking_data.pkl"
OUTPUT_DIR = HOME / "output"
TARGET_VIDEO = OUTPUT_DIR / "basketball_analytics_view.mp4"
TARGET_MAP = OUTPUT_DIR / "basketball_tactical_map.mp4"
HEATMAP_PATH = OUTPUT_DIR / "basketball_heatmap.png"
JSON_PATH = OUTPUT_DIR / "match_probabilities.json"

# Constants
MAP_W, MAP_H = 470, 500 
COLORS = {
    'bg': (175, 200, 225), 'lines': (255, 255, 255), 'border': (255, 255, 255),
    0: (150, 100, 0), 1: (50, 120, 20), 2: (0, 140, 255) # Team 0, Team 1, Ball
}
SRC_PTS = np.array([[550, 250], [1275, 305], [1260, 690], [50, 530]], dtype=np.float32)
DST_PTS = np.array([[0, 0], [MAP_W, 0], [MAP_W, MAP_H], [0, MAP_H]], dtype=np.float32)
H_MATRIX, _ = cv2.findHomography(SRC_PTS, DST_PTS)

def get_label(cid):
    return {0: "Spurs", 1: "OKC", 2: "Ball"}.get(cid, "Unknown")

def transform_points(points):
    if len(points) == 0: return []
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_MATRIX).reshape(-1, 2)

def draw_court(img):
    img[:] = COLORS['bg']
    c, t = COLORS['lines'], 2
    cv2.rectangle(img, (0, 0), (MAP_W, MAP_H), c, t)
    cv2.rectangle(img, (0, 170), (190, 330), c, t)
    cv2.circle(img, (190, 250), 60, c, t)
    cv2.circle(img, (40, 250), 5, (0, 100, 200), -1) 
    cv2.line(img, (40, 220), (40, 280), c, t)
    cv2.ellipse(img, (470, 250), (60, 60), 0, 90, 270, c, t)
    cv2.line(img, (0, 30), (129, 30), c, t)
    cv2.line(img, (0, 470), (129, 470), c, t)
    cv2.ellipse(img, (40, 250), (237, 237), 0, -68, 68, c, t)
    return img

def generate_heatmap(points_list):
    base = draw_court(np.full((MAP_H, MAP_W, 3), COLORS['bg'], dtype=np.uint8))
    if not points_list: return base
    
    overlay = np.zeros((MAP_H, MAP_W), dtype=np.float32)
    for px, py in np.vstack(points_list):
        if 0 <= px < MAP_W and 0 <= py < MAP_H:
            overlay[int(py), int(px)] += 1
            
    overlay = cv2.GaussianBlur(overlay, (31, 31), 0)
    if overlay.max() > 0: overlay = overlay / overlay.max() * 255
    colored = cv2.applyColorMap(overlay.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(base, 0.6, colored, 0.4, 0)

# Execution
with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

detections_dict = data['detections']
vid_info = data['video_info']

traj_points, all_history, json_data = [], [], []
box_an = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS, thickness=2)
lbl_an = sv.LabelAnnotator(color_lookup=sv.ColorLookup.CLASS, text_color=sv.Color.BLACK)

with sv.VideoSink(str(TARGET_VIDEO), vid_info) as sink_main, \
     sv.VideoSink(str(TARGET_MAP), sv.VideoInfo(width=MAP_W, height=MAP_H, fps=vid_info.fps)) as sink_map:
    
    frame_gen = sv.get_video_frames_generator(str(data['source_path']))
    
    for i, frame in tqdm(enumerate(frame_gen), total=vid_info.total_frames):
        dets = detections_dict.get(i, sv.Detections.empty())
        
        if TARGET_PLAYER_ID is not None:
            dets = dets[(dets.tracker_id == TARGET_PLAYER_ID) | (dets.class_id == 2)]
        
        map_coords = []
        if len(dets) > 0:
            feet = np.column_stack(((dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2, dets.xyxy[:, 3]))
            map_coords = transform_points(feet)

        frame_objs = []
        for idx, (mx, my) in enumerate(map_coords):
            tid, cid, conf = int(dets.tracker_id[idx]), int(dets.class_id[idx]), float(dets.confidence[idx])
            
            frame_objs.append({"id": tid, "team": get_label(cid), "probability": round(conf, 4)})

            if TARGET_PLAYER_ID is not None and tid == TARGET_PLAYER_ID:
                traj_points.append((int(mx), int(my)))
            elif TARGET_PLAYER_ID is None and cid != 2:
                all_history.append([mx, my])

        json_data.append({"frame": i, "timestamp": round(i / vid_info.fps, 3), "objects": frame_objs})

        # Draw Map
        t_map = draw_court(np.full((MAP_H, MAP_W, 3), COLORS['bg'], dtype=np.uint8))
        if TARGET_PLAYER_ID and len(traj_points) > 1:
            cv2.polylines(t_map, [np.array(traj_points)], False, (0, 0, 255), 2)

        for pt_idx, point in enumerate(map_coords):
            x, y = int(point[0]), int(point[1])
            cid = dets.class_id[pt_idx]
            if 0 <= x < MAP_W and 0 <= y < MAP_H:
                if cid != 2: cv2.circle(t_map, (x, y), 12, COLORS['border'], -1)
                cv2.circle(t_map, (x, y), 8 if cid == 2 else 9, COLORS[cid], -1)

        sink_main.write_frame(lbl_an.annotate(
            box_an.annotate(frame.copy(), dets), dets, labels=[f"ID {tid}" for tid in dets.tracker_id]
        ))
        sink_map.write_frame(t_map)

if TARGET_PLAYER_ID is None:
    cv2.imwrite(str(HEATMAP_PATH), generate_heatmap(all_history))

with open(JSON_PATH, 'w') as f:
    json.dump(json_data, f, indent=2)

print("Processing complete.")
