import numpy as np

def frame_metrics(y, pred, data_mask, block_size=32):
    yt, yp = y[data_mask], pred[data_mask]
    errors = yt != yp
    ber = float(errors.mean()) if len(errors) else 0.0
    blocks = [errors[i:i+block_size] for i in range(0, len(errors), block_size)]
    bler = float(np.mean([b.any() for b in blocks])) if blocks else 0.0
    return {"ber": ber, "bler": bler, "throughput": 1.0 - bler}

def adaptation_speed(rows, shift_frame, threshold=0.25):
    post = [r for r in rows if r["frame"] >= shift_frame]
    for i in range(len(post)):
        window = post[i:i+5]
        if len(window) == 5 and np.mean([r["bler"] for r in window]) <= threshold:
            return int(post[i]["frame"] - shift_frame)
    return None

