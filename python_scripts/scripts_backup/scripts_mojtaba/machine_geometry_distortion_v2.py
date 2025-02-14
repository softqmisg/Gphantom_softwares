
"""
Changes are made on January 19, 2025. Dr Baghemofidi has requested slice-wise distortion map visualization

"""

import os
import json
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, feature
from scipy.ndimage import gaussian_filter, correlate
from scipy.spatial import cKDTree
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#############################
#      Utility Functions    #
#############################

class NumpyEncoder(json.JSONEncoder):
    """Helper for saving NumPy arrays in JSON."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        elif isinstance(obj, (np.integer)):
            return int(obj)  # Convert NumPy integer to Python int
        elif isinstance(obj, (np.floating)):
            return float(obj)  # Convert NumPy float to Python float
        elif isinstance(obj, (np.bool_)):
            return bool(obj)  # Convert NumPy boolean to Python bool
        else:
            return super(NumpyEncoder, self).default(obj)

def convert_ndarray_to_list(data):
    """
    Recursively convert numpy arrays to lists in a nested dictionary or list.
    Useful for JSON serialization.
    """
    if isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

#############################
#   Image I/O and Loading   #
#############################

def load_nifti_image(file_path):
    """
    Loads a NIfTI image using SimpleITK and returns:
      - image (NumPy array, shape: [z, y, x])
      - pixel_spacing (tuple of floats: (x_spacing, y_spacing, z_spacing))
      - isocenter (world coordinates of the physical center of the image)
      - sitk_image (the original SimpleITK Image object)
    """
    nifti_data = sitk.ReadImage(file_path)
    image = sitk.GetArrayFromImage(nifti_data)            # shape: (z, y, x)
    spacing = nifti_data.GetSpacing()                     # (x_spacing, y_spacing, z_spacing) in SITK
    origin = np.array(nifti_data.GetOrigin())             # physical origin
    direction = np.array(nifti_data.GetDirection()).reshape(3, 3)

    image_dimensions = image.shape  # [z_dim, y_dim, x_dim]

    # Compute the physical center (isocenter) in world coordinates
    center_voxel = np.array([dim // 2 for dim in image_dimensions])
    physical_center = origin + direction.dot(center_voxel * spacing)

    return image, spacing, physical_center, nifti_data

#############################
#      Preprocessing        #
#############################

def clip_intensity(image, intensity_range):
    """
    Clamps (clips) the intensities of 'image' to [min_val, max_val].
    """
    min_val, max_val = intensity_range
    return np.clip(image, min_val, max_val)

def remove_outside_regions(image):
    """
    Uses Otsu thresholding + largest connected component to identify phantom region.
    Returns:
      - masked_image
      - mask (boolean)
      - coordinates of the largest connected component
    """
    thresh = filters.threshold_otsu(image)
    binary_image = image > thresh

    # Remove small objects
    binary_image = morphology.remove_small_objects(binary_image, min_size=500)

    # Label connected components
    labeled_image = measure.label(binary_image)
    regions = measure.regionprops(labeled_image)

    if not regions:
        logging.error("No regions found after thresholding and removing small objects.")
        return image, np.ones_like(image, dtype=bool), [], [], []

    # Find the largest region, assuming it's the phantom
    largest_region = max(regions, key=lambda r: r.area)

    # Create a mask for the largest region
    mask = np.zeros_like(image, dtype=bool)
    coords = largest_region.coords  # shape: (N, 3) with (z, y, x) indexing
    mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    idx_x, idx_y, idx_z = np.where(mask)
    idx_x, idx_y, idx_z = np.unique(idx_x), np.unique(idx_y), np.unique(idx_z)
    mask[idx_x[0]:idx_x[-1], idx_y[0]:idx_y[-1], idx_z[0]:idx_z[-1]] = 1
    # Apply mask
    masked_image = image * mask
    dim_z, dim_y, dim_x = coords[:, 0], coords[:, 1], coords[:, 2]
    return masked_image, mask, dim_z, dim_y, dim_x

#############################
#   Grid Cross Detection    #
#############################

def detect_plus_like_cross_sections(image_3d, image_type='CT'):
    """
    Detects "+"-like cross-sections in a 3D image.
    Uses image_type to adjust parameters accordingly.
    Returns an array of centroid coordinates in (z, y, x).
    """
    logging.info(f"Detecting grid intersections for {image_type} image...")

    # Smoothing parameters can be adjusted based on image type
    if image_type == 'CT':
        # blurred = gaussian_filter(image_3d, sigma=2)
        # Enhance grid lines: assuming grid lines are darker
        enhanced = filters.sobel(image_3d)
    else:  # MRI
        # blurred = gaussian_filter(image_3d, sigma=1)
        # Enhance grid lines: assuming grid lines are brighter
        enhanced = filters.scharr(image_3d)

    # Detect peaks which might correspond to grid intersections
    coordinates = feature.peak_local_max(
        enhanced,
        min_distance=5,  # Minimum number of pixels separating peaks
        threshold_abs=np.percentile(enhanced, 99),
        exclude_border=True
    )

    logging.info(f"Detected {len(coordinates)} potential grid intersections in {image_type} image.")

    # Optionally, refine detections using additional criteria
    # For example, validate local cross-like structures around each peak

    return coordinates  # in (z, y, x)

#############################
#     Matching Points       #
#############################

def compute_common_distances(points_img1, points_img2, spacing1, spacing2, tolerance_mm=5.0, verbose=False):
    """
    Compute 1-to-1 matches between two sets of 3D points using spatial tolerance.
    Spacing is used to convert voxel distances to physical distances.
    Returns:
      - common_points_img1
      - common_points_img2
    """
    # Convert voxel coordinates to physical space
    points1_mm = points_img1 * spacing1[::-1]  # (z, y, x) * (z, y, x)
    points2_mm = points_img2 * spacing2[::-1]

    tree2 = cKDTree(points2_mm)
    matches = tree2.query_ball_point(points1_mm, r=tolerance_mm)

    common1, common2 = [], []
    used_indices_img2 = set()

    for i, neighbors in enumerate(matches):
        if neighbors:
            # Find the closest neighbor not already used
            distances = np.linalg.norm(points2_mm[neighbors] - points1_mm[i], axis=1)
            sorted_indices = np.argsort(distances)
            for idx in sorted_indices:
                neighbor = neighbors[idx]
                if neighbor not in used_indices_img2:
                    common1.append(points_img1[i])
                    common2.append(points_img2[neighbor])
                    used_indices_img2.add(neighbor)
                    break  # Move to next point in img1

    if verbose:
        logging.info(f"Found {len(common1)} matched points.")

    return np.array(common1), np.array(common2)

#############################
#   Distortion Computation  #
#############################

def compute_distortions(common_img1, common_img2, spacing1, spacing2):
    """
    Computes distortions between matched points.
    Returns arrays of distortions in x, y, z, and total distance.
    """
    # Convert voxel to physical coordinates
    common1_mm = common_img1 * spacing1[::-1]
    common2_mm = common_img2 * spacing2[::-1]

    distortions = common2_mm - common1_mm  # (dz, dy, dx)
    dist_x = distortions[:, 2]
    dist_y = distortions[:, 1]
    dist_z = distortions[:, 0]
    dist_r = np.linalg.norm(distortions, axis=1)

    return dist_x, dist_y, dist_z, dist_r, common1_mm, common2_mm

def summarize_distortions(dist_x, dist_y, dist_z, dist_r, threshold_mm=5.0):
    """
    Generates summary statistics for distortions.
    """
    summary = {
        'dist_x': {
            'mean': float(np.mean(dist_x)),
            'median': float(np.median(dist_x)),
            'std': float(np.std(dist_x)),
            'max': float(np.max(dist_x)),
            'min': float(np.min(dist_x)),
            'percentage_above_threshold': float(np.sum(np.abs(dist_x) > threshold_mm) / len(dist_x) * 100)
        },
        'dist_y': {
            'mean': float(np.mean(dist_y)),
            'median': float(np.median(dist_y)),
            'std': float(np.std(dist_y)),
            'max': float(np.max(dist_y)),
            'min': float(np.min(dist_y)),
            'percentage_above_threshold': float(np.sum(np.abs(dist_y) > threshold_mm) / len(dist_y) * 100)
        },
        'dist_z': {
            'mean': float(np.mean(dist_z)),
            'median': float(np.median(dist_z)),
            'std': float(np.std(dist_z)),
            'max': float(np.max(dist_z)),
            'min': float(np.min(dist_z)),
            'percentage_above_threshold': float(np.sum(np.abs(dist_z) > threshold_mm) / len(dist_z) * 100)
        },
        'dist_r': {
            'mean': float(np.mean(dist_r)),
            'median': float(np.median(dist_r)),
            'std': float(np.std(dist_r)),
            'max': float(np.max(dist_r)),
            'min': float(np.min(dist_r)),
            'percentage_above_threshold': float(np.sum(dist_r > threshold_mm) / len(dist_r) * 100)
        }
    }
    return summary

#############################
#   Visualization Functions #
#############################

def plot_distortions(dist_x, dist_y, dist_z, dist_r, output_dir):
    """
    Plots histograms of distortions and saves the figures.
    """
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(dist_x, bins=50, color='r', alpha=0.7)
    plt.title('Distortion X')
    plt.xlabel('mm')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    plt.hist(dist_y, bins=50, color='g', alpha=0.7)
    plt.title('Distortion Y')
    plt.xlabel('mm')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 3)
    plt.hist(dist_z, bins=50, color='b', alpha=0.7)
    plt.title('Distortion Z')
    plt.xlabel('mm')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 4)
    plt.hist(dist_r, bins=50, color='k', alpha=0.7)
    plt.title('Total Distortion')
    plt.xlabel('mm')
    plt.ylabel('Frequency')

    plt.tight_layout()
    hist_path = os.path.join(output_dir, 'distortion_histograms.png')
    plt.savefig(hist_path)
    plt.close()
    logging.info(f"Saved distortion histograms to {hist_path}")

def visualize_matched_points(common_img1, common_img2, ct_image, mri_image, output_dir):
    """
    Creates a 3D scatter plot of matched points overlayed on CT and MRI images.
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 7))

    # CT image plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(common_img1[:, 2], common_img1[:, 1], common_img1[:, 0],
                c='r', marker='o', label='CT Points')
    ax1.set_title('CT Matched Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # MRI image plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(common_img2[:, 2], common_img2[:, 1], common_img2[:, 0],
                c='b', marker='^', label='MRI Points')
    ax2.set_title('MRI Matched Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'matched_points_scatter.png')
    plt.savefig(scatter_path)
    plt.close()
    logging.info(f"Saved matched points scatter plot to {scatter_path}")

#############################
#   Slice-Based Distortion  #
#############################

def compute_slice_distortions(common_img1, common_img2, spacing1, spacing2,
                              threshold_mm=None, output_file=None, percentage_threshold=1.0):
    """
    Computes distortions (dx, dy, dz, total) between corresponding points in CT and MRI.
    Both 'common_img1' and 'common_img2' are in voxel coordinates (z,y,x).
    We use the pixel_spacing for distances.
    Returns a dict of slice distortions:
      slice_index -> {
         'dist_x': [...],
         'dist_y': [...],
         'dist_z': [...],
         'total_dist': [...],
         'ct_points': [...],
         'mri_points': [...],
         'ct_points_mm': [...],
         'mri_points_mm': [...],
         'summary': {...}
      }
    """
    # Convert voxel coords to mm
    ct_xyz_mm = common_img1 * spacing1[::-1]  # (z, y, x) * (z, y, x)
    mri_xyz_mm = common_img2 * spacing2[::-1]

    # Calculate distortions
    distortions = mri_xyz_mm - ct_xyz_mm  # [dz, dy, dx]
    dist_x = distortions[:, 2]
    dist_y = distortions[:, 1]
    dist_z = distortions[:, 0]
    dist_r = np.linalg.norm(distortions, axis=1)

    # Assign distortions to slices based on CT z-coordinate
    slice_distortions = {}
    for i in range(len(dist_x)):
        if threshold_mm and dist_r[i] > threshold_mm:
            continue  # Skip distortions exceeding the threshold

        slice_idx = int(round(common_img1[i, 0]))  # z voxel index from CT

        if slice_idx not in slice_distortions:
            slice_distortions[slice_idx] = {
                'dist_x': [],
                'dist_y': [],
                'dist_z': [],
                'total_dist': [],
                'ct_points': [],
                'mri_points': [],
                'ct_points_mm': [],
                'mri_points_mm': []
            }

        slice_distortions[slice_idx]['dist_x'].append(float(dist_x[i]))
        slice_distortions[slice_idx]['dist_y'].append(float(dist_y[i]))
        slice_distortions[slice_idx]['dist_z'].append(float(dist_z[i]))
        slice_distortions[slice_idx]['total_dist'].append(float(dist_r[i]))
        slice_distortions[slice_idx]['ct_points'].append(common_img1[i].tolist())
        slice_distortions[slice_idx]['mri_points'].append(common_img2[i].tolist())
        slice_distortions[slice_idx]['ct_points_mm'].append(ct_xyz_mm[i].tolist())
        slice_distortions[slice_idx]['mri_points_mm'].append(mri_xyz_mm[i].tolist())

    # Compute summary statistics per slice
    for slice_idx, values in slice_distortions.items():
        arr_dx = np.array(values['dist_x'])
        arr_dy = np.array(values['dist_y'])
        arr_dz = np.array(values['dist_z'])
        arr_dr = np.array(values['total_dist'])

        summary = {
            'dist_x': {
                'mean': float(np.mean(arr_dx)),
                'median': float(np.median(arr_dx)),
                'max': float(np.max(arr_dx)),
                'min': float(np.min(arr_dx)),
                'percentage_above_threshold': float(
                    np.sum(np.abs(arr_dx) > percentage_threshold) / len(arr_dx) * 100
                )
            },
            'dist_y': {
                'mean': float(np.mean(arr_dy)),
                'median': float(np.median(arr_dy)),
                'max': float(np.max(arr_dy)),
                'min': float(np.min(arr_dy)),
                'percentage_above_threshold': float(
                    np.sum(np.abs(arr_dy) > percentage_threshold) / len(arr_dy) * 100
                )
            },
            'dist_z': {
                'mean': float(np.mean(arr_dz)),
                'median': float(np.median(arr_dz)),
                'max': float(np.max(arr_dz)),
                'min': float(np.min(arr_dz)),
                'percentage_above_threshold': float(
                    np.sum(np.abs(arr_dz) > percentage_threshold) / len(arr_dz) * 100
                )
            },
            'dist_r': {
                'mean': float(np.mean(arr_dr)),
                'median': float(np.median(arr_dr)),
                'max': float(np.max(arr_dr)),
                'min': float(np.min(arr_dr)),
                'percentage_above_threshold': float(np.sum(arr_dr > percentage_threshold) / len(arr_dr) * 100)
            }
        }
        slice_distortions[slice_idx]['summary'] = summary

    # Convert for JSON
    sorted_slice_distortions = dict(sorted(slice_distortions.items(), key=lambda x: x[0]))
    sorted_slice_distortions_list = convert_ndarray_to_list(sorted_slice_distortions)

    # Optionally save
    if output_file:
        out_json = os.path.join(output_file, "distortions_per_slice.json")
        with open(out_json, 'w') as f:
            json.dump(sorted_slice_distortions_list, f, indent=4)
        logging.info(f"[INFO] Distortion per-slice stats saved to {out_json}")

    return sorted_slice_distortions

def generate_distortion_plot_from_isocenter(common_img1_mm, distortions, isocenter, component='total_dist', save_path=None):
    """
    Creates a scatter plot of 'component' vs. distance-from-isocenter, saves data to JSON.
    'isocenter' is expected in mm: (x, y, z).
    'common_img1_mm' are the CT matched points in mm.
    'distortions' are the corresponding distortion values.
    """
    isocenter_x, isocenter_y, isocenter_z = isocenter
    distances_from_iso = []
    distortions_component = []

    for i in range(len(common_img1_mm)):
        pt_mm = common_img1_mm[i]
        dx, dy, dz = distortions[i]
        if component == 'dist_x':
            comp = dx
        elif component == 'dist_y':
            comp = dy
        elif component == 'dist_z':
            comp = dz
        else:
            # 'total_dist'
            comp = np.sqrt(dx**2 + dy**2 + dz**2)

        # distance from iso
        r = np.sqrt(
            (pt_mm[0] - isocenter_x)**2 +
            (pt_mm[1] - isocenter_y)**2 +
            (pt_mm[2] - isocenter_z)**2
        )
        distances_from_iso.append(float(r))
        distortions_component.append(float(comp))

    # Prepare dictionary
    distortion_data = {
        'distance_from_isocenter': distances_from_iso,
        'distortion': distortions_component
    }

    # Save to JSON
    if save_path is not None:
        out_json = os.path.join(save_path, f'distortion_scatter_{component}.json')
        with open(out_json, 'w') as f:
            json.dump(convert_ndarray_to_list(distortion_data), f, indent=4)
        logging.info(f"[INFO] Distortion scatter data ({component}) saved to {out_json}")

    return distortion_data

#############################
#   Creating Dist. Maps     #
#############################

def create_and_save_all_distortion_maps(sorted_slice_distortions,
                                        reference_sitk_image,
                                        output_dir,
                                        apply_gaussian_smoothing=True,
                                        sigma=1.0,
                                        truncate=1.0,
                                        mask=None):
    """
    Create 4 separate 3D distortion maps: dx, dy, dz, dr (in mm),
    for each voxel location. Each volume is saved as both .npy and .nii.gz.

    NOTE: We rely on 'ct_points' in (z,y,x) voxel coords, and
          'dist_x', 'dist_y', 'dist_z', 'total_dist' from the dictionary.
    """
    mask = mask.astype(int)
    # Grab reference shape (z, y, x) from reference_sitk_image
    ref_array = sitk.GetArrayFromImage(reference_sitk_image)
    shape_z, shape_y, shape_x = ref_array.shape

    # Prepare empty volumes
    map_dx = np.zeros_like(ref_array, dtype=np.float64)
    map_dy = np.zeros_like(ref_array, dtype=np.float64)
    map_dz = np.zeros_like(ref_array, dtype=np.float64)
    map_dr = np.zeros_like(ref_array, dtype=np.float64)

    total_points = 0
    assigned_points = 0

    # Fill in
    for slice_idx, vals in sorted_slice_distortions.items():
        pts_vox = vals['ct_points']     # each is [z_vox, y_vox, x_vox]
        arr_dx = vals['dist_x']
        arr_dy = vals['dist_y']
        arr_dz = vals['dist_z']
        arr_dr = vals['total_dist']

        for i, pt_vox in enumerate(pts_vox):
            z, y, x = pt_vox
            # cast to int
            z, y, x = int(round(z)), int(round(y)), int(round(x))
            total_points += 1
            if (0 <= z < shape_z) and (0 <= y < shape_y) and (0 <= x < shape_x):
                map_dx[z, y, x] = arr_dx[i]
                map_dy[z, y, x] = arr_dy[i]
                map_dz[z, y, x] = arr_dz[i]
                map_dr[z, y, x] = arr_dr[i]
                assigned_points += 1

    logging.info(f"[INFO] Assigned {assigned_points} / {total_points} matched-distortion points to the voxel grid.")

    # Optionally smooth
    if apply_gaussian_smoothing:
        map_dx = gaussian_filter(map_dx, sigma=sigma, truncate=truncate)
        map_dy = gaussian_filter(map_dy, sigma=sigma, truncate=truncate)
        map_dz = gaussian_filter(map_dz, sigma=sigma, truncate=truncate)
        map_dr = gaussian_filter(map_dr, sigma=sigma, truncate=truncate)
        logging.info(f"[INFO] Applied Gaussian smoothing (sigma={sigma}, truncate={truncate}).")

    # Prepare SITK images
    dx_sitk = sitk.GetImageFromArray(map_dx)
    dy_sitk = sitk.GetImageFromArray(map_dy)
    dz_sitk = sitk.GetImageFromArray(map_dz)
    dr_sitk = sitk.GetImageFromArray(map_dr)
    mask_sitk = sitk.GetImageFromArray(mask)

    # Copy geometry
    dx_sitk.CopyInformation(reference_sitk_image)
    dy_sitk.CopyInformation(reference_sitk_image)
    dz_sitk.CopyInformation(reference_sitk_image)
    dr_sitk.CopyInformation(reference_sitk_image)
    mask_sitk.CopyInformation(reference_sitk_image)

    # Save all as NIfTI + NPY
    def _save_nii_npy(array_data, sitk_img, out_prefix):
        """Saves array_data to .npy and .nii.gz with given prefix."""
        npy_path = os.path.join(output_dir, f"{out_prefix}.npy")
        np.save(npy_path, array_data)
        nii_path = os.path.join(output_dir, f"{out_prefix}.nii.gz")
        sitk.WriteImage(sitk_img, nii_path)
        logging.info(f"[INFO] Saved {out_prefix}.npy and {out_prefix}.nii.gz")

    _save_nii_npy(map_dx, dx_sitk, "distortion_dx")
    _save_nii_npy(map_dy, dy_sitk, "distortion_dy")
    _save_nii_npy(map_dz, dz_sitk, "distortion_dz")
    _save_nii_npy(map_dr, dr_sitk, "distortion_dr")
    _save_nii_npy(mask, mask_sitk, "mask")

#############################
#     Slice-Based JSON      #
#############################

def save_distortions_per_slice_json(sorted_slice_distortions, output_dir):
    """
    Saves distortions_per_slice.json similar to the original code.
    """
    out_json = os.path.join(output_dir, "distortions_per_slice.json")
    with open(out_json, 'w') as f:
        json.dump(convert_ndarray_to_list(sorted_slice_distortions), f, indent=4)
    logging.info(f"[INFO] Distortions per slice saved to {out_json}")

def save_distortion_scatter_json(common_img1_mm, distortions, isocenter, output_dir):
    """
    Saves distortion_scatter_{component}.json files similar to the original code.
    """
    components = ['dist_x', 'dist_y', 'dist_z', 'total_dist']
    for component in components:
        generate_distortion_plot_from_isocenter(common_img1_mm, distortions, isocenter,
                                                component=component, save_path=output_dir)

def save_grid_position_json(common_img1_vox, output_dir):
    """
    Saves grid_position.json similar to the original code.
    'common_img1_vox' is in (z,y,x) voxel coordinates.
    """
    centers = common_img1_vox
    z_coords = centers[:, 0]
    y_coords = centers[:, 1]
    x_coords = centers[:, 2]
    series_metadata = {
        "z_coords": z_coords.tolist(),
        "y_coords": y_coords.tolist(),
        "x_coords": x_coords.tolist(),
    }
    out_json = os.path.join(output_dir, "grid_position.json")
    with open(out_json, 'w') as json_file:
        json.dump(series_metadata, json_file, indent=4, cls=NumpyEncoder)
    logging.info(f"[INFO] Saved grid positions to {out_json}")


#############################
#           MAIN            #
#############################

def main(ct_file_path, mri_file_path, output_dir,
         ct_intensity_range=(0, 100),
         mri_intensity_range=(200, 1000),
         tolerance_mm=2.0,
         threshold_mm=2.0):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1) Load CT and MRI
    logging.info("Loading CT and MRI images...")
    ct_image, ct_spacing, ct_isocenter, ct_sitk = load_nifti_image(ct_file_path)
    mri_image, mri_spacing, mri_isocenter, mri_sitk = load_nifti_image(mri_file_path)

    # 2) Clip intensities
    logging.info("Clipping intensities...")
    ct_clipped = clip_intensity(ct_image, ct_intensity_range)
    mri_clipped = clip_intensity(mri_image, mri_intensity_range)

    # 3) Mask out outside region from MRI and apply same mask to CT
    logging.info("Removing outside regions...")
    mri_phantom, mask, _, _, _ = remove_outside_regions(mri_clipped)
    ct_phantom = ct_clipped * mask


    # 4) Detect cross-sections
    ct_intersections_vox = detect_plus_like_cross_sections(ct_phantom, image_type='CT')
    mri_intersections_vox = detect_plus_like_cross_sections(mri_phantom, image_type='MRI')

    logging.info(f"CT intersections detected: {ct_intersections_vox.shape[0]}")
    logging.info(f"MRI intersections detected: {mri_intersections_vox.shape[0]}")

    # 5) Match points (in voxel space) with a certain tolerance
    logging.info("Matching grid intersection points between CT and MRI...")
    ct_matched_vox, mri_matched_vox = compute_common_distances(
        ct_intersections_vox,
        mri_intersections_vox,
        spacing1=ct_spacing,
        spacing2=mri_spacing,
        tolerance_mm=tolerance_mm,
        verbose=True
    )

    if len(ct_matched_vox) == 0:
        logging.error("No matched points found. Exiting.")
        return

    # 6) Compute distortions
    logging.info("Computing distortions between matched points...")
    dist_x, dist_y, dist_z, dist_r, ct_matched_mm, mri_matched_mm = compute_distortions(
        ct_matched_vox,
        mri_matched_vox,
        spacing1=ct_spacing,
        spacing2=mri_spacing
    )

    distortion_summary = summarize_distortions(dist_x, dist_y, dist_z, dist_r, threshold_mm=threshold_mm)

    # Save distortion summary
    summary_path = os.path.join(output_dir, 'distortion_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(distortion_summary, f, indent=4)
    logging.info(f"Saved distortion summary to {summary_path}")

    # 7) Generate and save visualizations
    plot_distortions(dist_x, dist_y, dist_z, dist_r, output_dir)
    visualize_matched_points(ct_matched_vox, mri_matched_vox, ct_image, mri_image, output_dir)

    # 8) Save matched points and distortions
    matched_points = {
        'CT_matched_points_mm': ct_matched_mm.tolist(),
        'MRI_matched_points_mm': mri_matched_mm.tolist(),
        'dist_x_mm': dist_x.tolist(),
        'dist_y_mm': dist_y.tolist(),
        'dist_z_mm': dist_z.tolist(),
        'dist_r_mm': dist_r.tolist()
    }
    matched_points_path = os.path.join(output_dir, 'matched_points.json')
    with open(matched_points_path, 'w') as f:
        json.dump(matched_points, f, indent=4, cls=NumpyEncoder)
    logging.info(f"Saved matched points and distortions to {matched_points_path}")

    # 9) Compute and save slice-based distortions
    logging.info("Computing slice-based distortions...")

    sorted_slice_distortions = compute_slice_distortions(
        common_img1=ct_matched_vox,
        common_img2=mri_matched_vox,
        spacing1=ct_spacing,
        spacing2=mri_spacing,
        threshold_mm=threshold_mm,
        output_file=output_dir,
        percentage_threshold= 1.0,
    )

    # 10) Generate and save distortion scatter JSON files
    logging.info("Generating distortion scatter JSON files...")
    distortions = np.stack((dist_x, dist_y, dist_z), axis=1)
    save_distortion_scatter_json(
        common_img1_mm=ct_matched_mm,
        distortions=distortions,
        isocenter=ct_isocenter,
        output_dir=output_dir
    )

    # 11) Save grid positions
    logging.info("Saving grid positions...")
    save_grid_position_json(
        common_img1_vox=ct_matched_vox,
        output_dir=output_dir
    )

    # 12) Create and save distortion maps
    logging.info("Creating 3D distortion maps (dx, dy, dz, dr)...")
    create_and_save_all_distortion_maps(
        sorted_slice_distortions=sorted_slice_distortions,
        reference_sitk_image=ct_sitk,
        output_dir=output_dir,
        apply_gaussian_smoothing=True,
        sigma=1.0,
        truncate=1.0,
        mask=mask,
    )

    # 13) (Optional) Compute mutual nearest distances and log stats
    dist_x_mutual, dist_y_mutual, dist_z_mutual, dist_r_mutual = compute_mutual_nearest_distances(
        ct_matched_vox,
        mri_matched_vox,
        ct_spacing,
        mri_spacing,
        threshold_mm=threshold_mm
    )
    # Log some stats
    logging.info("[INFO] Overall distortion stats (mutual nearest):")
    logging.info(f"  dx: mean={dist_x_mutual.mean():.3f}, std={dist_x_mutual.std():.3f} (mm)")
    logging.info(f"  dy: mean={dist_y_mutual.mean():.3f}, std={dist_y_mutual.std():.3f} (mm)")
    logging.info(f"  dz: mean={dist_z_mutual.mean():.3f}, std={dist_z_mutual.std():.3f} (mm)")
    logging.info(f"  dr: mean={dist_r_mutual.mean():.3f}, std={dist_r_mutual.std():.3f} (mm)")

    # 14) Save the intersection points found in CT (for reference)
    logging.info("Saving grid positions (voxel coordinates)...")
    save_grid_position_json(
        common_img1_vox=ct_matched_vox,
        output_dir=output_dir
    )

    logging.info("Processing complete.")

def compute_mutual_nearest_distances(ct_points_vox, mri_points_vox, ct_pixel_spacing, mri_pixel_spacing, threshold_mm=None):
    """
    (Optional) Another approach to compute dx, dy, dz, and total distances for matched points.
    Returns arrays: dist_x, dist_y, dist_z, total_distances.
    """
    ct_xyz_mm = ct_points_vox * ct_pixel_spacing[::-1]
    mri_xyz_mm = mri_points_vox * mri_pixel_spacing[::-1]

    tree_ct = cKDTree(ct_xyz_mm)
    tree_mri = cKDTree(mri_xyz_mm)

    dist_ct_to_mri, idx_ct_to_mri = tree_ct.query(mri_xyz_mm)
    dist_mri_to_ct, idx_mri_to_ct = tree_mri.query(ct_xyz_mm)

    matched_dx = []
    matched_dy = []
    matched_dz = []
    matched_dr = []

    for i_mri, (d1, i_ct) in enumerate(zip(dist_ct_to_mri, idx_ct_to_mri)):
        if idx_mri_to_ct[i_ct] == i_mri:
            pt_ct = ct_xyz_mm[i_ct]
            pt_mri = mri_xyz_mm[i_mri]
            dx = pt_mri[0] - pt_ct[0]
            dy = pt_mri[1] - pt_ct[1]
            dz = pt_mri[2] - pt_ct[2]
            dr = np.sqrt(dx**2 + dy**2 + dz**2)
            if (threshold_mm is None) or (dr <= threshold_mm):
                matched_dx.append(dx)
                matched_dy.append(dy)
                matched_dz.append(dz)
                matched_dr.append(dr)

    return (np.array(matched_dx),
            np.array(matched_dy),
            np.array(matched_dz),
            np.array(matched_dr))

#############################
#   JSON Saving Functions   #
#############################

def save_distortions_per_slice_json(sorted_slice_distortions, output_dir):
    """
    Saves distortions_per_slice.json similar to the original code.
    """
    out_json = os.path.join(output_dir, "distortions_per_slice.json")
    with open(out_json, 'w') as f:
        json.dump(convert_ndarray_to_list(sorted_slice_distortions), f, indent=4)
    logging.info(f"[INFO] Distortions per slice saved to {out_json}")

def save_distortion_scatter_json(common_img1_mm, distortions, isocenter, output_dir):
    """
    Saves distortion_scatter_{component}.json files similar to the original code.
    """
    components = ['dist_x', 'dist_y', 'dist_z', 'total_dist']
    for component in components:
        generate_distortion_plot_from_isocenter(common_img1_mm, distortions, isocenter,
                                                component=component, save_path=output_dir)

def save_grid_position_json(common_img1_vox, output_dir):
    """
    Saves grid_position.json similar to the original code.
    'common_img1_vox' is in (z,y,x) voxel coordinates.
    """
    centers = common_img1_vox
    z_coords = centers[:, 0]
    y_coords = centers[:, 1]
    x_coords = centers[:, 2]
    series_metadata = {
        "z_coords": z_coords.tolist(),
        "y_coords": y_coords.tolist(),
        "x_coords": x_coords.tolist(),
    }
    out_json = os.path.join(output_dir, "grid_position.json")
    with open(out_json, 'w') as json_file:
        json.dump(series_metadata, json_file, indent=4, cls=NumpyEncoder)
    logging.info(f"[INFO] Saved grid positions to {out_json}")

#############################
#           MAIN            #
#############################

if __name__ == "__main__":
    # Example usage; adjust paths as necessary
    root = "/media/jdi/ssd2/Data/grant_nih_2023/grid_phantom_for_part2/registered_t1w_v02"
    ct_file_path = os.path.join(root, "ct.nii.gz")
    mri_file_path = os.path.join(root, "registered_t1w.nii.gz")

    output_dir = "/media/jdi/ssd2/Data/grant_nih_2023/Gphantom3_May18_2024/MR_04_29_2024_tilted_90/" \
                 "separated_data_Jan19_2025/stats"
    os.makedirs(output_dir, exist_ok=True)

    main(ct_file_path, mri_file_path, output_dir=output_dir,
         ct_intensity_range=(0, 100),
         mri_intensity_range=(200, 1000),
         tolerance_mm=5.0,
         threshold_mm=5.0)
