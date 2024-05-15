import numpy as np
import cv2 as cv
from itertools import combinations

CLASSES = {

}

def classificator(graph):
    vertexes = {g[0] for g in graph.keys()} + {g[1] for g in graph.keys()}
    connections = np.ones(len(vertexes))
    for key, value in vertexes.items():
        connections[key[0], key[1]] = value
    connections += connections.T

    D = np.diag(np.sum(connections, axis=1))
    Laplasian = D - connections
    singulars = np.linalg.svdvals(Laplasian)

    # TODO my best

    raise NotImplemented()

def get_graph(Matrix: np.array) -> tuple[np.array, dict[dict[int, int], float]]:
    local_curve_kernel = np.ones((15, 15)) 
    local_curve_kernel[1:-1, 1:-1] = 0
    skeleton = np.where(Matrix[15:-15, 15:-15] > 0)

    skeleton_base = []
    for idx, idy in np.stack(skeleton, axis=-1) + 15:
        point_local = np.stack(np.where(Matrix[idx-7:idx+8, idy-7:idy+8] * local_curve_kernel > 0), axis=-1)
        if len(point_local) != 2:
            skeleton_base.append((idx, idy))
            continue

        end1 = point_local[0]
        end2 = point_local[1]

        A = (end1[0] - idx) ** 2 + (end1[1] - idy) ** 2
        B = (end2[0] - idx) ** 2 + (end2[1] - idy) ** 2
        C = (end1[0] - end2[0]) ** 2 + (end1[1] - end2[1]) ** 2
        angle = np.arccos((C - A - B) / (2 * np.sqrt(A * B)))

        if angle < 5 * np.pi / 6:
            skeleton_base.append((idx, idy))


    point_canvas = np.zeros_like(Matrix)
    for idx, idy in skeleton_base:
        cv.circle(point_canvas, (idy, idx), 20, 255, -1)
    _, _, _, centroids = cv.connectedComponentsWithStats(point_canvas)

    point_canvas = np.zeros_like(Matrix)
    points = []
    for center in centroids:
        c = (int(center[0]), int(center[1]))
        points.append(c)
        cv.circle(point_canvas, c, 20, 255, -1)
    points = np.array(points)

    connectivity = {}
    point_indexes = combinations(range(len(points)), 2)
    set_of_all_indexes = set(range(len(points)))
    k = 30
    for a_index, b_index in point_indexes:
        other_indexes = set_of_all_indexes - {a_index, b_index}
        a = points[a_index]
        b = points[b_index]

        canvas_without_other_dots = Matrix.copy()
        cv.circle(canvas_without_other_dots, a, 15, 255, -1)
        cv.circle(canvas_without_other_dots, b, 15, 255, -1)

        mask_over_all_other_points = np.zeros_like(Matrix)
        for i in other_indexes:
            cv.circle(mask_over_all_other_points, points[i], 15, 255, -1)
        canvas_without_other_dots[mask_over_all_other_points > 0] = 0

        _, labels, _, _ = cv.connectedComponentsWithStats(canvas_without_other_dots)
        connectivity[(a_index, b_index)] = np.linalg.norm(a - b) if labels[a[1], a[0]] == labels[b[1], b[0]] else 0

    result_canvas = Matrix.copy()
    for key, value in connectivity.items():
        if value > 0.:
            cv.circle(result_canvas, points[key[0]], 20, 255, -1)
            cv.circle(result_canvas, points[key[1]], 20, 255, -1)
            cv.line(result_canvas, points[key[0]], points[key[1]], 180, 10)

    return result_canvas, connectivity