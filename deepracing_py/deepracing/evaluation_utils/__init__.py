import numpy as np
from shapely.geometry import Point as ShapelyPoint, MultiPoint#, Point2d as ShapelyPoint2d
from shapely.geometry.polygon import Polygon
from shapely.geometry import LinearRing

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    return idx.reshape(-1,2).astype(np.int64)
def lapMetrics(positions : np.ndarray, timestamps : np.ndarray, innerboundary_poly : Polygon, outerboundary_poly : Polygon):
    inside_violations = np.array([innerboundary_poly.contains(ShapelyPoint(positions[i,0], positions[i,2])) for i in range(positions.shape[0])])
    inner_contiguous_regions = contiguous_regions(inside_violations)
    inside_violation_regions = np.column_stack([inner_contiguous_regions, -1*np.ones([inner_contiguous_regions.shape[0], 1], dtype=np.int64)])
    outside_violations = np.array([not outerboundary_poly.contains(ShapelyPoint(positions[i,0], positions[i,2])) for i in range(positions.shape[0])])
    outer_contiguous_regions = contiguous_regions(outside_violations)
    outside_violation_regions = np.column_stack([outer_contiguous_regions, 1*np.ones([outer_contiguous_regions.shape[0], 1], dtype=np.int64)])
    
    all_violation_regions = np.row_stack([inside_violation_regions, outside_violation_regions])
    delta_poly : Polygon = outerboundary_poly.difference(innerboundary_poly)

    I = np.argsort(all_violation_regions[:,0])
    all_violation_regions = all_violation_regions[I]

    t0 = timestamps[0]
    tbf = []
    bfmaxdists = []
    bfmeandists = []
    bftimes = []
    for i in range(all_violation_regions.shape[0]):
        region = all_violation_regions[i]
        tf = timestamps[region[0]]
        dt = tf - t0
        tbf.append(dt)
        t0 = timestamps[region[1]]
        idx = np.arange(region[0], region[1]+1, step=1, dtype=np.int64)
        points = positions[idx]
        timestampslocal = timestamps[idx]
        bftimes.append(timestampslocal[-1] - timestampslocal[0])
        # if region[2]<0:
        #     distances = np.array([innerboundary_poly.exterior.distance(ShapelyPoint(points[j,0], points[j,2])) for j in range(points.shape[0])])
        # else:
        #     distances = np.array([outerboundary_poly.distance(ShapelyPoint(points[j,0], points[j,2])) for j in range(points.shape[0])])
        distances = np.array([delta_poly.distance(ShapelyPoint(points[j,0], points[j,2])) for j in range(points.shape[0])])
        bfmeandists.append(float(np.mean(distances)))
        bfmaxdists.append(float(np.max(distances)))

    return all_violation_regions, {"number_boundary_failures" : all_violation_regions.shape[0], "time_between_failures": tbf, "boundary_failure_max_distances" : bfmaxdists, "boundary_failure_mean_distances" : bfmeandists, "boundary_failure_times" : bftimes}