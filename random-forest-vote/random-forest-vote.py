import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Ép kiểu về mảng NumPy, nếu là số đơn lẻ sẽ tự biến thành mảng 1 phần tử
    predictions = np.atleast_1d(predictions)
    
    # TRƯỜNG HỢP 1: Chỉ có 1 mẫu dữ liệu (Mảng 1 chiều chứa các dự đoán từ các cây)
    if predictions.ndim == 1:
        vals, counts = np.unique(predictions, return_counts=True)
        return vals[np.argmax(counts)]  # Trả về 1 giá trị chiến thắng duy nhất
        
    # TRƯỜNG HỢP 2: Có nhiều mẫu dữ liệu (Mảng 2 chiều kích thước: n_trees x n_samples)
    elif predictions.ndim == 2:
        votes = []
        # Duyệt qua từng mẫu (duyệt theo cột)
        for i in range(predictions.shape[1]):
            sample_preds = predictions[:, i]
            vals, counts = np.unique(sample_preds, return_counts=True)
            votes.append(vals[np.argmax(counts)])
        return votes