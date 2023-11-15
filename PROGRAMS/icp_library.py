import numpy as np
import numpy.linalg as la
import calibration_library as cal

def find_closest_point(point, vertices, triangles):
    """
    Find the closest point on the mesh surface defined by vertices and triangles to the given point.
    Parameters:
    point: Given one point as a NumPy array.
    vertices: Coordinates of vertices as a NumPy array.
    triangles: Indices of vertex coordinates for each triangle as a NumPy array.

    return the closest point to the given point on the surface mesh
    """
    num_triangles = triangles.shape[1]
    c_ij = np.zeros([3, num_triangles])
    S = np.zeros([3, 2])

    for i in range(num_triangles):
        # p, q, r = vertices[:, triangles[:, i]]
        p = vertices[:, int(triangles[0][i])]
        q = vertices[:, int(triangles[1][i])]
        r = vertices[:, int(triangles[2][i])]
        print(p,q,r)
        for j in range(3):
            S[j][0] = q[j] - p[j]
            S[j][1] = r[j] - p[j]

        b = point - p
        soln = la.lstsq(S, b, rcond=None)
        l = soln[0][0]
        m = soln[0][1]
        mid = p + l * (q - p) + m * (r - p)

        if l >= 0 and m >= 0 and l + m <= 1:
            c_star = mid
        elif l < 0:
            c_star = project_on_segment(mid , r, p)
        elif m < 0:
            c_star = project_on_segment(mid , p, q)
        else:  # l + m > 1
            c_star = project_on_segment(mid , q, r)

        c_ij[:, i] = c_star[:]

    distance = la.norm(point - c_ij[:, 0])
    minpoint = c_ij[:, 0]

    for i in range((c_ij).shape[1]):
        dist = la.norm(point - c_ij[:, i])

        if dist < distance:
            distance = dist
            closest_point = c_ij[: ,i]

    return minpoint

def project_on_segment(c, p, q):
    """
    Project point c onto the line segment defined by endpoints p and q.

    Parameters:
    c: The point to project as a NumPy array.
    p, q: Endpoints of the line segment as NumPy arrays.

    Returns:
    c_star: The projected point on the line segment.
    """
    # Compute the scalar projection of c onto the line defined by p and q
    lambda_ = np.dot(c - p, q - p) / np.dot(q - p, q - p)

    # Clamp lambda to lie within the segment
    lambda_seg = max(0, min(lambda_, 1))

    # Compute the actual projection point on the segment
    c_star = p + lambda_seg * (q - p)

    return c_star


def calc_difference(c_k_points, d_k_points):
    """
    Calculates the Euclidean distance between corresponding points in two point clouds.

    Parameters:
    c_k_points: Closest point on surface to tip in each frame
    d_k_points: Position of tip in each frame

    Returns: 1D array with distances between each pair of corresponding points.
    """
    dist = np.zeros(np.shape(c_k_points)[0])
    for i in range(np.shape(c_k_points)[0]):

        dist[i] = np.linalg.norm(d_k_points[i, :] - c_k_points[i, :])

    return dist


# tip transform related part

def find_tip_positions(a_frames, b_frames, led_a, led_b, tip_a):
    """
    Finds the positions of the tip of pointer A with respect to the B rigid body for each sample frame.
    param a_frames: frames of A pointers
    param b_frames: frames of B rigid body points
    param led_a: Position of LEDs on A pointer in calibration frame
    param tip_a: Position of tip with respect to A body coordinate
    param led_b: Position of LEDs on B rigid body in calibration frame

    return: tip_positions_b: Tip positions with respect to B rigid body in each data frame

    """
    tip_positions_b = np.zeros([3, len(a_frames)])

    for i in range(len(a_frames)):
        registration_a = cal.setRegistration.calculate_3d_transformation(led_a, a_frames[i])
        registration_b = cal.setRegistration.calculate_3d_transformation(led_b, b_frames[i])
        combined_registration = registration_b.inv.compose(registration_a)
        transformed_tip_a = cal.setRegistration.apply_transformation_single_pt(tip_a, combined_registration)
        for j in range(3):
            tip_positions_b[j][i] = transformed_tip_a[j]

    return tip_positions_b


def transform_tip_positions(tip_positions, frame_transformation):
    """
    Transforms tip positions with the given frame transformation. The func. will be useful in PA#4, In PA#3 we assume F_reg is I.
    param tip_positions: Array containing positions of tip with respect to rigid body B
    param frame_transformation: Frame transformation

    return: transformed_tip_positions: Transformed array of points

    """
    for i in range(np.shape(tip_positions.data)[1]):

        tip_positions.data[:, i] = cal.setRegistration.apply_transformation_single_pt(tip_positions, frame_transformation)

    return tip_positions
