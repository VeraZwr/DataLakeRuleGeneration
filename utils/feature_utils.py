def normalize_feature(value, min_val, max_val):
    if value is None or min_val == max_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
