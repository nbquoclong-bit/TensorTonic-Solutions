def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    cos, L = 0.0, 0.0
    cos = sum(a*b for a,b in zip(x1,x2))/(math.sqrt(sum(a*a for a in x1)) * math.sqrt(sum(a*a for a in x2)))
    L = 1.0-cos if label == 1.0 else max(0.0, cos-margin)
    return L