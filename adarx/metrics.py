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


def classes_to_bits(classes, bits_per_symbol=4):
    c = np.asarray(classes, dtype=np.int64)
    shifts = np.arange(bits_per_symbol-1, -1, -1)
    return ((c[:, None] >> shifts) & 1).astype(np.int8)


def real_sdr_metrics(y, pred, block_size=32):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    ser = float(np.mean(y != pred))
    yb = classes_to_bits(y)
    pb = classes_to_bits(pred)
    bit_error_matrix = yb != pb
    ber = float(np.mean(bit_error_matrix))
    symbol_has_bit_error = np.any(bit_error_matrix, axis=1)
    blocks = [symbol_has_bit_error[i:i+block_size] for i in range(0, len(symbol_has_bit_error), block_size)]
    bler = float(np.mean([b.any() for b in blocks])) if blocks else 0.0
    return {"ber": ber, "ser": ser, "bler": bler, "throughput": 1.0 - bler}
