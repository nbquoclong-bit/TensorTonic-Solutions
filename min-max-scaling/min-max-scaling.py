def min_max_scaling(data):
    """Scale each column of the data matrix to the [0, 1] range."""
    if not data or not data[0]:
        return []

    num_rows = len(data)
    num_cols = len(data[0])

    # 1. Compute min and max for each column
    min_cols = [min(data[r][c] for r in range(num_rows)) for c in range(num_cols)]
    max_cols = [max(data[r][c] for r in range(num_rows)) for c in range(num_cols)]

    scaled_data = []

    # 2. Build the scaled matrix row by row
    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            range_val = max_cols[c] - min_cols[c]
            # Handle constant columns (avoid division by zero)
            val = (
                (data[r][c] - min_cols[c]) / range_val
                if range_val != 0
                else 0.0
            )
            row.append(val)
        scaled_data.append(row)

    return scaled_data