import numpy as np
import supervision as sv
import pickle
from pathlib import Path
from tqdm import tqdm
from sam3.model_builder import build_sam3_video_predictor

# Config
HOME = Path.cwd()
SOURCE_VIDEO_PATH = HOME / "sam3" / "sources" / "basketball_game.mp4"
DATA_OUTPUT_PATH = HOME / "output" / "tracking_data.pkl"
DATA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 0: Spurs (White), 1: OKC (Red), 2: Ball
PROMPTS = ["Players in white jersey", "Players in red jersey", "basketball"]

def sam3_to_sv_detections(sam3_outputs, class_id_override):
    if not sam3_outputs or len(sam3_outputs.get("out_obj_ids", [])) == 0:
        return sv.Detections.empty()
    
    object_ids = sam3_outputs["out_obj_ids"]
    masks = sam3_outputs["out_binary_masks"].astype(bool)
    raw_probs = sam3_outputs.get("out_probs", [1.0] * len(object_ids))
    
    return sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),
        mask=masks,
        tracker_id=np.array(object_ids, dtype=int),
        confidence=np.array(raw_probs, dtype=float),
        class_id=np.full(len(object_ids), class_id_override, dtype=int)
    )

def filter_ball_detections(detections):
    if len(detections) == 0: return detections
    w = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    h = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    h[h == 0] = 1e-6
    ar = w / h
    area = w * h
    
    # Geometric and confidence filters
    valid = (ar > 0.5) & (ar < 1.8) & (area > 50) & (area < 3000) & \
            (detections.confidence > 0.45) & (detections.confidence < 0.70)
    return detections[valid]

# Main
video_predictor = build_sam3_video_predictor()
video_info = sv.VideoInfo.from_video_path(str(SOURCE_VIDEO_PATH))
merged_detections = {}

for class_id, prompt_text in enumerate(PROMPTS):
    print(f"Processing Class {class_id}: {prompt_text}")
    resp = video_predictor.handle_request(request=dict(type="start_session", resource_path=str(SOURCE_VIDEO_PATH)))
    sess_id = resp["session_id"]
    
    video_predictor.handle_request(request=dict(type="add_prompt", session_id=sess_id, frame_index=0, text=prompt_text))
    stream = video_predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=sess_id))
    
    for resp in tqdm(stream, total=video_info.total_frames):
        f_idx = resp["frame_index"]
        raw_dets = sam3_to_sv_detections(resp.get("outputs", {}), class_id)
        
        if class_id == 2:
            raw_dets = filter_ball_detections(raw_dets)
            
        merged_detections[f_idx] = sv.Detections.merge([merged_detections.get(f_idx, sv.Detections.empty()), raw_dets])
            
    video_predictor.handle_request(request=dict(type="close_session", session_id=sess_id))

video_predictor.shutdown()

with open(DATA_OUTPUT_PATH, 'wb') as f:
    pickle.dump({"detections": merged_detections, "video_info": video_info, "source_path": SOURCE_VIDEO_PATH}, f)
print(f"Data saved to {DATA_OUTPUT_PATH}")