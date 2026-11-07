import torch
import torch.nn.functional as F


def response_kd_loss(z_s, z_t, y, alpha=0.5, beta=0.5, tau=4.0, ignore_index=255):
    """Response-based KD loss.
    z_s, z_t: (N, C, H, W) student and teacher logits. y: (N, H, W) labels."""
    gd_loss = F.cross_entropy(z_s, y, ignore_index=ignore_index)
    teacher_prob = F.softmax(z_t / tau, dim=1)
    student_prob = F.log_softmax(z_s/tau, dim=1)
    ts_loss = - torch.sum(teacher_prob * student_prob, dim=1)
    # Ignore where pixel is 255
    mask = (y != 255)
    loss = ts_loss[mask].mean()

    return alpha * gd_loss + beta * loss



def feature_kd_loss(f_s, f_t):
    """Feature-based KD loss: 1 - cosine similarity, averaged over all positions.
    f_s: (N, C, h, w) projected student features. f_t: (N, C, H, W) teacher features."""
    H, W = f_t.shape[2:]
    resized_f_s = F.interpolate(
        f_s,
        size=(H, W),
        mode='bilinear'
    )

    cos_sim = F.cosine_similarity(resized_f_s, f_t, dim=1)
    # Average all pixels
    loss = (1 - cos_sim).mean()
    return loss
