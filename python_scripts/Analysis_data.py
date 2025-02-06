import sys
import os
import shutil
import dicom2nifti
import pydicom
import json 
import datetime
from pathlib import Path


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import point
from skimage import filters, measure, morphology
from scipy.spatial import KDTree
from scipy.ndimage import sobel, correlate, gaussian_filter
import json
import SimpleITK as sitk
from skimage.filters.rank import percentile
import os

from scipy.spatial import cKDTree
                

"""
    Database queery Templates
"""
from Database import DataBase


template_query_read_directory="SELECT * FROM studies WHERE id={id}"
template_query_write_save_status="UPDATE studies SET \
                                          status=\"{status}\"\
                                          WHERE id={id}"

template_query_write_save_scatter="UPDATE studies SET \
                                          scatter=\"{scatter}\"\
                                          WHERE id={id}"

template_query_write_save_curve="UPDATE studies SET \
                                          curve=\"{curve}\"\
                                          WHERE id={id}"

CT_reference=""
# BaseAddress="/home/mehdi/Documents/Gphantom2/python_scripts/dicoms"
BaseAddress="E:/PostDoc/__Works/Dr.Fatemi/G_phantom/Software/Gphantom2/python_scripts/dicoms"

db_host="127.0.0.1"
db_port=3306
db_name="phantom_db_v2"
db_user="root"
db_password=""


class NumpyEncoder(json.JSONEncoder):
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

'''
    Class of analysis CT/MRI. Find geometry distortion and its statistics
'''        
class analysisData:
    def __init__(self,mydatabase,currentId_mri,ct_nifti_path=None,mri_nifti_path=None):
        self.mydatabase=mydatabase
        self.currentId_mri=currentId_mri
        self.start_analysis(ct_nifti_path,mri_nifti_path)
     ##################################################################################
    '''
        Image Tools function
    '''
    ##################################################################################
    def convert_ndarray_to_list(self,data):
        """
        Recursively convert numpy arrays to lists in a nested dictionary or list.
        """
        if isinstance(data, dict):
            return {key: self.convert_ndarray_to_list(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_ndarray_to_list(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    ##################################################################################
    # Function to load NIfTI images and extract the 3D volume
    def load_nifti_image(self,file_path):
        nifti_data = sitk.ReadImage(file_path)
        image = sitk.GetArrayFromImage(nifti_data)
        pixel_spacing = nifti_data.GetSpacing()  # Extract 3D pixel spacing (in mm)

        # Get image dimensions (z, y, x) order for the numpy array
        image_dimensions = image.shape

        # Get the origin of the image (world coordinates of voxel (0,0,0))
        origin = np.array(nifti_data.GetOrigin())

        # Calculate the center of the image in voxel coordinates (x, y, z order)
        image_center_voxel = np.array(image_dimensions[::-1]) / 2.0

        # Calculate the physical isocenter (world coordinates)
        isocenter = origin + image_center_voxel * np.array(pixel_spacing)


        return image, pixel_spacing, isocenter
    ##################################################################################
    # Clip intensity of the CT and MRI images separately
    def clip_intensity(self,image, intensity_range):
        min_val, max_val = intensity_range
        clipped_image = np.clip(image, min_val, max_val)
        return clipped_image 
    ##################################################################################
    # Remove regions outside the squared phantom
    def remove_outside_regions(self,image):
        # Otsu thresholding to create binary mask
        thresh = filters.threshold_otsu(image)
        binary_image = image > thresh

        # Label connected components and remove small objects
        labeled_image = measure.label(binary_image)
        regions = measure.regionprops(labeled_image)

        # Find the largest region, assuming it's the phantom
        largest_region = max(regions, key=lambda r: r.area)
        # Create a mask for the largest region
        mask = np.zeros_like(image, dtype=bool)
        mask[largest_region.coords[:, 0], largest_region.coords[:, 1], largest_region.coords[:, 2]] = True
        dim_x, dim_y, dist_z = largest_region.coords[:, 0], largest_region.coords[:, 1], largest_region.coords[:, 2]
        # Apply mask to the image
        phantom_image = image * mask
        return phantom_image, mask, dim_x, dim_y, dist_z
    ################################################################################## 
    '''
        Analysis tools
    '''
    ##################################################################################    
    def detect_plus_like_cross_sections_3D_MRI(self,image_3d):
        # image_3d = gaussian_filter(image_3d, sigma=3)


        # Create a 3D "+"-like kernel (cross shape)
        plus_kernel = np.zeros((9, 9, 9))  # 7x7x7 kernel size, adjust as needed
        center = 4  # Center of the kernel (middle of the 7x7x7 cube)

        # Define the "+" structure in the x, y, and z directions
        plus_kernel[center, :, center] = 1  # Vertical line in the y-axis
        plus_kernel[:, center, center] = 1  # Vertical line in the x-axis
        plus_kernel[center, center, :] = 1  # Vertical line in the z-axis

        # Perform cross-correlation between the image and the "+" kernel
        response = correlate(image_3d, plus_kernel)

        # Threshold the response to keep only strong matches (high correlation values)
        response_thresholded = response > np.percentile(response,
                                                        99)  # Keeping top 1% of responses, adjust as necessary

        # Remove small objects to clean up noise
        response_clean = morphology.remove_small_objects(response_thresholded, min_size=3)

        # Use labeled image to find coordinates of the intersecting regions
        labeled_plus = measure.label(response_clean)
        intersection_props = measure.regionprops(labeled_plus)

        # Get the coordinates of the centers of the "+" like cross-sections
        centers = np.array([prop.centroid for prop in intersection_props])
        return centers
    ##################################################################################
    def detect_plus_like_cross_sections_3D_CT(self,image_3d):
        # image_3d = gaussian_filter(image_3d, sigma=7)
        # Create a 3D "+"-like kernel (cross shape)
        plus_kernel = np.zeros((5, 5, 5))  # 7x7x7 kernel size, adjust as needed
        center = 2  # Center of the kernel (middle of the 7x7x7 cube)

        # Define the "+" structure in the x, y, and z directions (inverted kernel for detecting low-intensity regions)
        plus_kernel[center, :, center] = -1  # Vertical line in the y-axis (negative values for dark areas)
        plus_kernel[:, center, center] = -1  # Vertical line in the x-axis
        plus_kernel[center, center, :] = -1  # Vertical line in the z-axis

        # Fill surroundings with positive values to ensure the "+" structure is detected as darker than surroundings
        plus_kernel[plus_kernel == 0] = 1

        # Perform cross-correlation between the image and the "+" kernel
        response = correlate(image_3d, plus_kernel)

        # Threshold the response to keep only strong matches (high correlation values)
        response_thresholded = response > np.percentile(response,
                                                        99)  # Keeping top 1% of responses, adjust as necessary

        # Remove small objects to clean up noise
        response_clean = morphology.remove_small_objects(response_thresholded, min_size=5)

        # Use labeled image to find coordinates of the intersecting regions
        labeled_plus = measure.label(response_clean)
        intersection_props = measure.regionprops(labeled_plus)

        # Get the coordinates of the centers of the "+" like cross-sections
        centers = np.array([prop.centroid for prop in intersection_props])
        return centers
    ##################################################################################
    def generate_distortion_plot_from_isocenter(self,sorted_slice_distortions, isocenter, component='total_dist',
                                                save_path=None):
        """
        Generates a distortion plot for the specified component relative to the distance from the isocenter
        and saves the data in a dictionary.

        :param sorted_slice_distortions: Dictionary of distortions with statistical summaries.
        :param isocenter: Tuple representing the coordinates of the isocenter (x, y, z) in mm.
        :param component: The distortion component to plot ('dist_x', 'dist_y', 'dist_z', 'total_dist').
        :param save_path: File path to save the plot image.

        :return: Dictionary with 'distance_from_isocenter' and 'distortion' values.
        """
        # Initialize lists to store distances and distortions
        distances_from_isocenter = []
        distortions = []

        # Unpack isocenter coordinates
        isocenter_x, isocenter_y, isocenter_z = isocenter

        # Iterate through each slice and gather data
        for slice_index, values in sorted_slice_distortions.items():
            # Iterate through each distortion point in the current slice
            for ct_point, distortion in zip(values['ct_points'], values[component]):
                # Calculate the distance from the specified isocenter
                distance_from_isocenter = np.sqrt(
                    (ct_point[0] - isocenter_x) ** 2 +
                    (ct_point[1] - isocenter_y) ** 2 +
                    (ct_point[2] - isocenter_z) ** 2
                )

                # Append the calculated distance and the distortion to the lists
                distances_from_isocenter.append(distance_from_isocenter)
                distortions.append(distortion)

        # Create the dictionary to save the data
        distortion_data = {
            'distance_from_isocenter': distances_from_isocenter,
            'distortion': distortions
        }

        # Convert data to a JSON-serializable format
        distortion_data_list = self.convert_ndarray_to_list(distortion_data)
        if save_path:
        # Save the data dictionary to a JSON file
            with open(os.path.join(save_path, 'distortion_scatter.json'), 'w') as json_file:
                json.dump(distortion_data_list, json_file, indent=4)
            # print(f"Distortion data saved to distortion_data.json")
        self.mydatabase.write_query(template_query_write_save_scatter.format(id=self.currentId_mri,
                                                                                    scatter=distortion_data_list
                                                                                    )
                                    )

        # Return the data dictionary
        return distortion_data
    ##################################################################################
    def save_scatter(self,centers, target_path):
        centers = np.array(centers)
        z_coords = centers[:, 2]
        y_coords = centers[:, 1]
        x_coords = centers[:, 0]
        series_metadata = {
            "z_coords": z_coords,
            "y_coords": y_coords,
            "x_coords": x_coords,
        }
        # data_serializable = {k: convert_numpy_to_native(v) if isinstance(v, (np.ndarray, np.generic)) else v for k, v in
        #                      series_metadata.items()}
        #
        with open(os.path.join(target_path, "grid_position.json"), 'w') as json_file:
            json.dump(series_metadata, json_file, indent=4, cls=NumpyEncoder)
    ##################################################################################
    '''
        Analysis
    '''  
    ##################################################################################    
    def compute_common_distances(self,points_img1, points_img2, tolerance=1e-5, verbose=False):
        """
        Compute the Euclidean distance between common 3D points in two sets of points.

        Parameters:
        points_img1, points_img2: numpy arrays of shape (N, 3) representing 3D points.
        tolerance: a small value or a tuple/list of three values to handle matching tolerance.
        verbose: if True, print detailed information about the matching process.

        Returns:
        distances: a list of distances between the common points.
        common_points_img1, common_points_img2: the matched 3D points from both sets.
        """
        # Ensure inputs are 2D arrays
        points_img1 = np.atleast_2d(points_img1)
        points_img2 = np.atleast_2d(points_img2)



        # Check if both arrays have the correct number of columns (3 for 3D points)
        if points_img1.shape[1] != 3 or points_img2.shape[1] != 3:
            raise ValueError("Each point set must have a shape (N, 3) for 3D points.")

        if points_img1.size == 0 or points_img2.size == 0:
            raise ValueError("One or both of the point sets are empty.")

        # Ensure tolerance is a tuple of length 3 or a single float
        if isinstance(tolerance, (float, int)):
            tolerance = (tolerance, tolerance, tolerance)
        elif len(tolerance) != 3:
            raise ValueError("Tolerance must be a single float or a tuple of three values.")

        if verbose:
            print(f"Matching with tolerance: {tolerance}")

        # Use a KD-tree for efficient nearest-neighbor search
        tree = cKDTree(points_img2)

        # Match points from img1 to img2
        common_points_img1 = []
        common_points_img2 = []

        for i, point1 in enumerate(points_img1):
            distances, indexes = tree.query(point1, k=1, distance_upper_bound=max(tolerance))
            if distances <= max(tolerance):
                common_points_img1.append(point1)
                common_points_img2.append(points_img2[indexes])

        if len(common_points_img1) == 0:
            raise ValueError("No common points found within the given tolerance.")

        # Convert to numpy arrays for further computation
        common_points_img1 = np.array(common_points_img1)
        common_points_img2 = np.array(common_points_img2)

        if verbose:
            print(f"Matched {len(common_points_img1)} points.")

        return common_points_img1, common_points_img2
    ##################################################################################
    def compute_slice_distortions(self,ct_centers,
                                mri_centers,
                                ct_pixel_spacing,
                                mri_pixel_spacing,
                                threshold=None,
                                output_file=None,
                                percentage_threshold=1.0):
        """
        Computes distortions between corresponding points in CT and MRI scans in x, y, and z axes.
        Keeps distortions within their corresponding image slice for 3D visualization.

        :param ct_centers: Array of CT centers (in pixel coordinates).
        :param mri_centers: Array of MRI centers (in pixel coordinates).
        :param ct_pixel_spacing: Pixel spacing for CT (x, y, z).
        :param mri_pixel_spacing: Pixel spacing for MRI (x, y, z).
        :param threshold: Maximum allowed distance (in mm) to consider points as corresponding. If None, no threshold is applied.

        :return: Dictionary of distortions with keys as slice indices and values as lists of distortions in x, y, z, and total distance.
        """
        # Convert CT and MRI pixel coordinates to mm by multiplying with pixel spacing
        set1 = ct_centers * np.array(ct_pixel_spacing)
        set2 = mri_centers * np.array(mri_pixel_spacing)

        # Build KDTree for both point sets
        tree1 = KDTree(set1, leafsize=3)
        tree2 = KDTree(set2, leafsize=3)

        # Find nearest neighbors from set1 to set2
        distances_set1_to_set2, indices_set1_to_set2 = tree1.query(set2)

        # Find nearest neighbors from set2 to set1
        distances_set2_to_set1, indices_set2_to_set1 = tree2.query(set1)

        # Dictionary to hold distortions for each slice
        slice_distortions = {}

        # Iterate through set1 and set2 points to keep only mutual nearest neighbors
        for i, (dist1, idx1) in enumerate(zip(distances_set1_to_set2, indices_set1_to_set2)):
            # Ensure the match is mutual
            if indices_set2_to_set1[idx1] == i:
                # Get the corresponding points
                point_set1 = set1[idx1]
                point_set2 = set2[i]

                # Calculate the distances along x, y, z axes
                dist_x = abs(point_set1[0] - point_set2[0])
                dist_y = abs(point_set1[1] - point_set2[1])
                dist_z = abs(point_set1[2] - point_set2[2])

                # Calculate the total Euclidean distance
                total_dist = np.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z ** 2)

                # Apply the distance threshold if needed
                if threshold is None or total_dist <= threshold:
                    # Determine the slice index (e.g., use z-coordinate to determine the slice index)
                    slice_index = int(round(point_set1[2] / ct_pixel_spacing[2]))  # Assuming z is the slice index

                    # Initialize the slice index if not present
                    if slice_index not in slice_distortions:
                        slice_distortions[slice_index] = {
                            'dist_x': [],
                            'dist_y': [],
                            'dist_z': [],
                            'total_dist': [],
                            'ct_points_mm': [],  # Store CT points in mm
                            'ct_points': [],  # Store CT points
                            'mri_points': [],  # Store MRI points
                            'mri_points_mm': []  # Store MRI points in mm
                        }

                    # Store distortions and corresponding points in the slice dictionary
                    slice_distortions[slice_index]['dist_x'].append(dist_x)
                    slice_distortions[slice_index]['dist_y'].append(dist_y)
                    slice_distortions[slice_index]['dist_z'].append(dist_z)
                    slice_distortions[slice_index]['total_dist'].append(total_dist)

                    slice_distortions[slice_index]['ct_points'].append(np.round(point_set1 / ct_pixel_spacing).astype(int))
                    slice_distortions[slice_index]['ct_points_mm'].append(point_set1)

                    slice_distortions[slice_index]['mri_points'].append(np.round(point_set2 / ct_pixel_spacing).astype(int))
                    slice_distortions[slice_index]['mri_points_mm'].append(point_set2)

        # Sort slice_distortions dictionary by slice index
        sorted_slice_distortions = dict(sorted(slice_distortions.items()))

        # Calculate mean, median, max, min, and percentage above threshold for each item in each slice
        for slice_index, values in sorted_slice_distortions.items():
            # Calculate summary statistics
            summary = {
                'dist_x': {
                    'mean': np.mean(values['dist_x']),
                    'median': np.median(values['dist_x']),
                    'max': np.max(values['dist_x']),
                    'min': np.min(values['dist_x']),
                    'percentage_above_threshold': (np.sum(np.array(values['dist_x']) > percentage_threshold) / len(
                        values['dist_x'])) * 100
                },
                'dist_y': {
                    'mean': np.mean(values['dist_y']),
                    'median': np.median(values['dist_y']),
                    'max': np.max(values['dist_y']),
                    'min': np.min(values['dist_y']),
                    'percentage_above_threshold': (np.sum(np.array(values['dist_y']) > percentage_threshold) / len(
                        values['dist_y'])) * 100
                },
                'dist_z': {
                    'mean': np.mean(values['dist_z']),
                    'median': np.median(values['dist_z']),
                    'max': np.max(values['dist_z']),
                    'min': np.min(values['dist_z']),
                    'percentage_above_threshold': (np.sum(np.array(values['dist_z']) > percentage_threshold) / len(
                        values['dist_z'])) * 100
                },
                'total_dist': {
                    'mean': np.mean(values['total_dist']),
                    'median': np.median(values['total_dist']),
                    'max': np.max(values['total_dist']),
                    'min': np.min(values['total_dist']),
                    'percentage_above_threshold': (np.sum(np.array(values['total_dist']) > percentage_threshold) / len(
                        values['total_dist'])) * 100
                }
            }
            # Update summary in the dictionary
            sorted_slice_distortions[slice_index]['summary'] = summary
            # Convert numpy arrays to lists before saving to JSON
        sorted_slice_distortions_list = self.convert_ndarray_to_list(sorted_slice_distortions)

        # Save the sorted dictionary to a file if output_file is provided
        if output_file:
            with open(os.path.join(output_file, "select_description3.json"), 'w') as json_file:
                json.dump(sorted_slice_distortions_list, json_file, indent=4)
            print(f"Sorted distortions saved to {output_file}")
        
        self.mydatabase.write_query(template_query_write_save_curve.format(id=self.currentId_mri,
                                                                                    curve=sorted_slice_distortions_list
                                                                                    )
                                    )

        return sorted_slice_distortions
    ##################################################################################
    def compute_mutual_nearest_distances(self,ct_centers, mri_centers, ct_pixel_spacing, mri_pixel_spacing,
                                        threshold=None):
        """
        Computes distances between corresponding points in CT and MRI scans. If points are too far apart (above a threshold), they are ignored.

        :param ct_centers: Array of CT centers (in pixel coordinates).
        :param mri_centers: Array of MRI centers (in pixel coordinates).
        :param ct_pixel_spacing: Pixel spacing for CT (x, y, z).
        :param mri_pixel_spacing: Pixel spacing for MRI (x, y, z).
        :param distance_threshold: Maximum allowed distance (in mm) to consider points as corresponding. If None, no threshold is applied.

        :return: Arrays of distances in x, y, z directions, and total distances for corresponding points.
        """
        # Convert CT and MRI pixel coordinates to mm by multiplying with pixel spacing
        set1 = ct_centers * np.array(ct_pixel_spacing)
        set2 = mri_centers * np.array(mri_pixel_spacing)
        # Build KDTree for both point sets
        # Build KDTree for both point sets
        tree1 = KDTree(set1, leafsize=3)
        tree2 = KDTree(set2, leafsize=3)

        # Find nearest neighbors from set1 to set2
        distances_set1_to_set2, indices_set1_to_set2 = tree1.query(set2)

        # Find nearest neighbors from set2 to set1
        distances_set2_to_set1, indices_set2_to_set1 = tree2.query(set1)

        matched_distances_x = []
        matched_distances_y = []
        matched_distances_z = []
        total_distances = []
        matched_points_set1 = []
        matched_points_set2 = []

        # Iterate through set1 and set2 points to keep only mutual nearest neighbors
        for i, (dist1, idx1) in enumerate(zip(distances_set1_to_set2, indices_set1_to_set2)):
            # Ensure the match is mutual
            if indices_set2_to_set1[idx1] == i:
                # Get the corresponding points
                point_set1 = set1[idx1]
                point_set2 = set2[i]

                # Calculate the distances along x, y, z axes
                dist_x = abs(point_set1[0] - point_set2[0])
                dist_y = abs(point_set1[1] - point_set2[1])
                dist_z = abs(point_set1[2] - point_set2[2])

                # Calculate the total Euclidean distance
                total_dist = np.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z ** 2)

                # Apply the distance threshold if needed
                if threshold is None or dist_x <= threshold:
                    matched_distances_x.append(dist_x)
                    matched_distances_y.append(dist_y)
                    matched_distances_z.append(dist_z)
                    total_distances.append(total_dist)
                    matched_points_set1.append(point_set1)
                    matched_points_set2.append(point_set2)

        return (np.array(matched_distances_x),
                np.array(matched_distances_y),
                np.array(matched_distances_z),
                np.array(total_distances)
                )
    ##################################################################################
    '''
    Start Analysis
    '''
    ##################################################################################    
    def start_analysis(self,ct_file_path,mri_file_path):
        # Load CT and MRI NIfTI images (full 3D volumes)
        try:
            if(ct_file_path!=""):
                ct_image, ct_pixel_spacing, origin_ct = self.load_nifti_image(ct_file_path)
        except Exception as e:
            print(f"Unexpected error  in loading CT:'{ct_file_path}': {e}")
        try:
            mri_image, mri_pixel_spacing, origin_mr = self.load_nifti_image(mri_file_path)
        except Exception as e:
            print(f"Unexpected error  in loading MRI:'{mri_file_path}': {e}")            
        # Clip intensity of CT and MRI separately
        clipped_ct = self.clip_intensity(ct_image, (0, 100))  # Clip CT between 0 and 100
        clipped_mri = self.clip_intensity(mri_image, (0, 1000))  # Clip MRI between 0 and 1000
        # Remove regions outside the squared phantom
        mri_phantom, mask, dim_x, dim_y, dim_z = self.remove_outside_regions(clipped_mri)
        ct_phantom = clipped_ct * mask
        print(f"{mask.shape = }")
        # Detect cross sections in CT and MRI (in 3D)
        ct_intersections = self.detect_plus_like_cross_sections_3D_MRI(ct_phantom)
        mri_intersections = self.detect_plus_like_cross_sections_3D_MRI(mri_phantom)
        print(f"{ct_intersections.shape = }")
        print(f"{mri_intersections.shape = }")
        # Compute common points
        print("Computing the grids' center ...")
        common_points_img1, common_points_img2 = self.compute_common_distances(ct_intersections,
                                                                        mri_intersections,
                                                                        tolerance=(2, 2, 2),
                                                                        verbose=True)
        # self.save_scatter(ct_intersections, target_path=None)

        print('Computer the slice distortion ...')
        distortions_per_slice = self.compute_slice_distortions(common_points_img1,
                                                        common_points_img2,
                                                        ct_pixel_spacing,
                                                        mri_pixel_spacing,
                                                        threshold=None,
                                                        output_file=None)
        self.generate_distortion_plot_from_isocenter(distortions_per_slice, isocenter=origin_ct, save_path = None)
        print('Save distortion stats at JSON file ...')
        dist_x, dist_y, dist_z, total_distances = self.compute_mutual_nearest_distances(common_points_img1,
                                                                                common_points_img2,
                                                                                ct_pixel_spacing,
                                                                                mri_pixel_spacing,
                                                                                threshold=None)

'''
    Class of registeration CT/MRI. MRI registered to CT
'''
class registerImages:
    def __init__(self,ct_nifti_path=None,mri_nifti_path=None,registered_mri_nifti_path=None):
        print("=======================")
        print(ct_nifti_path)
        print(mri_nifti_path)
        print(registered_mri_nifti_path)
        print("=======================")
        self.start_register(ct_nifti_path,mri_nifti_path,registered_mri_nifti_path)
    '''
        Registeration Tools
    '''
    ##################################################################################
    def read_image(self,file_path):
        """Read the volumetric image file."""
        return sitk.ReadImage(file_path)
    ##################################################################################
    def check_image_compatibility(self,fixed_image, moving_image):
        """Ensure the images have the same dimension and type."""
        if fixed_image.GetDimension() != moving_image.GetDimension():
            raise ValueError(
                f"Fixed image and moving image have different dimensions: {fixed_image.GetDimension()} vs {moving_image.GetDimension()}")

        if fixed_image.GetPixelID() != moving_image.GetPixelID():
            print(f"Warning: Fixed image and moving image have different types. Casting moving image to fixed image type.")
            moving_image = sitk.Cast(moving_image, fixed_image.GetPixelID())

        return moving_image
    
    ##################################################################################
    def register_images(self,fixed_image, moving_image):
        """Register the moving_image to the fixed_image using mutual information."""

        # Ensure compatibility
        moving_image = self.check_image_compatibility(fixed_image, moving_image)

        # Initialize the registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings
        registration_method.SetOptimizerAsGradientDescent(learningRate=.1, numberOfIterations=1000,
                                                        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the initial alignment
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                            moving_image,
                                                            sitk.Euler3DTransform(),
                                                            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Multi-resolution framework
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Execute the registration
        final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                    sitk.Cast(moving_image, sitk.sitkFloat32))

        # Print out the final metric value
        print("Final metric value: {0}".format(registration_method.GetMetricValue()))
        print("Optimizer's stopping condition, {0}".format(registration_method.GetOptimizerStopConditionDescription()))

        return final_transform


    ##################################################################################
    def resample_image(self,moving_image, transform, reference_image):
        """Resample the moving image using the final transformation."""
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)

        return resampler.Execute(moving_image)
    ##################################################################################
    '''
        Start Registeration
    '''
    ##################################################################################
    def start_register(self,ct_file_path,mri_file_path,registered_mri_nifti_path):
        fixed_image_path=ct_file_path
        moving_image_path=mri_file_path
        # Read the images
        fixed_image = self.read_image(fixed_image_path)
        moving_image = self.read_image(moving_image_path)

        # Register the moving image to the fixed image
        final_transform = self.register_images(fixed_image, moving_image)

        # Resample the moving image using the final transform
        registered_moving_image = self.resample_image(moving_image, final_transform, fixed_image)

        # Save the registered image
        sitk.WriteImage(registered_moving_image, registered_mri_nifti_path)


'''
    Find Name of files and call rgisteration the analysis
'''       
def main():
    print(len(sys.argv))
    id_ct=0
    id_mri=0
    if(len(sys.argv)<2):
        print("Error: Syntax")
        return 
    elif (len(sys.argv)==2):
        id_mri=sys.argv[1]
    else:
        id_ct=int(sys.argv[1])
        id_mri=int(sys.argv[2])
    ##############################################################################
    ### Find Path and Name of Files
    mydatabase=DataBase(host=db_host,port=db_port,database=db_name,user=db_user,password=db_password)
    currentId_ct=id_ct
    currentId_mri=id_mri
    print('current ct Id:',currentId_ct)
    print('current mri Id:',currentId_mri)
    if (currentId_ct != 0):
        readid_ct=mydatabase.read_query(template_query_read_directory.format(id=currentId_ct))
        ct_Directory=readid_ct[0]['directory']
        ct_nifti_Directory=os.path.join(Path(ct_Directory),'Nifti')
        # ct_nifti_Directory=f'{ct_nifti_Directory}'.replace('\\','\\\\')
        ct_nifti_files=[f for f in os.listdir(ct_nifti_Directory) if (f.endswith('.nii.gz')and (not f.startswith('registeredImage_')))]
        ct_nifti_file=os.path.join(ct_nifti_Directory,ct_nifti_files[0])

    else:
        ct_nifti_file=CT_reference
    ct_nifti_file=f'{ct_nifti_file}'.replace('\\','\\\\')
    print(">>>>>>CT File:",ct_nifti_file)  

    readid_mri=mydatabase.read_query(template_query_read_directory.format(id=currentId_mri))
    mri_Directory=readid_mri[0]['directory']
    mri_nifti_Directory=os.path.join(Path(mri_Directory),'Nifti')
    # mri_nifti_Directory=f'{mri_nifti_Directory}'.replace('\\','\\\\')
    mri_nifti_files=[f for f in os.listdir(mri_nifti_Directory) if (f.endswith('.nii.gz')and (not f.startswith('registeredImage_')))]
    mri_nifti_file=os.path.join(mri_nifti_Directory,mri_nifti_files[0])
    mri_nifti_file=f'{mri_nifti_file}'.replace('\\','\\\\')
    print(">>>>>>MRI File:",mri_nifti_file)  
    ##############################################################################
    ### Write start of analysis in DB
    mydatabase.write_query(template_query_write_save_status.format(id=currentId_mri,
                                                                                status="analyzing"
                                                                                )
                                    )
    ##############################################################################
    ###  Registeration
    registered_mri_nifti_file=os.path.join(mri_nifti_Directory,'registeredImage_'+mri_nifti_files[0])
    registered_mri_nifti_file=f'{registered_mri_nifti_file}'.replace('\\','\\\\')
    registerImages(ct_nifti_path=ct_nifti_file,mri_nifti_path=mri_nifti_file,registered_mri_nifti_path=registered_mri_nifti_file)
    print("Registeration was done!!!!")
    ##############################################################################
    ###  Analysis
    analysisData(mydatabase,currentId_mri,ct_nifti_path=ct_nifti_file,mri_nifti_path=registered_mri_nifti_file)
    print("Analysis was done!!!!")
    ##############################################################################
    ### Write end of analysis in DB
    mydatabase.write_query(template_query_write_save_status.format(id=currentId_mri,
                                                                                status="analyzed"
                                                                                )
                                )        
    mydatabase.close()        
    ##################################################################################

if __name__=="__main__":
    main()
    sys.exit()
