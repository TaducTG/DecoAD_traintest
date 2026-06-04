import cv2
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Thay thế cython_bbox bằng hàm Numpy thuần bên dưới
# from cython_bbox import bbox_overlaps as bbox_ious
from trackers.utils import kalman_filter
import time

def bbox_ious(bboxes1, bboxes2):
    """
    Tính IoU giữa 2 danh sách bounding boxes bằng Numpy thay thế cho cython_bbox
    bboxes1: shape (N, 4) -> [x1, y1, x2, y2]
    bboxes2: shape (M, 4) -> [x1, y1, x2, y2]
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

    bboxes1 = np.array(bboxes1, dtype=np.float32)
    bboxes2 = np.array(bboxes2, dtype=np.float32)

    # Giải nén tọa độ [x1, y1, x2, y2]
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # Tính diện tích từng box
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    # Tính tọa độ vùng giao nhau (Intersection)
    xi1 = np.maximum(x11, x21.T)
    yi1 = np.maximum(y11, y21.T)
    xi2 = np.minimum(x12, x22.T)
    yi2 = np.minimum(y12, y22.T)

    # Chiều rộng và chiều cao vùng giao nhau
    wi = np.maximum(0.0, xi2 - xi1)
    hi = np.maximum(0.0, yi2 - yi1)
    intersection = wi * hi

    # Tính Union
    union = area1 + area2.T - intersection

    return intersection / (union + 1e-6)


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    Simple linear assignment
    :type cost_matrix: np.ndarray
    :type thresh: float
    :return: matches, unmatched_a, unmatched_b
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    indices = np.column_stack((row_ind, col_ind))

    return _indices_to_matches(cost_matrix, indices, thresh)


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    # Đã sửa np.float thành float chuẩn để tránh lỗi deprecation của Numpy
    ious_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious_matrix.size == 0:
        return ious_matrix

    # Gọi hàm nội bộ dùng Numpy vừa định nghĩa ở trên
    ious_matrix = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious_matrix


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    # Đã sửa np.float thành float chuẩn
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    for i, track in enumerate(tracks):
        cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1, -1), det_features, metric))
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]