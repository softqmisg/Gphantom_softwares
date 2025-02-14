import sys
import os
import shutil
import dicom2nifti
import pydicom
import json 
import datetime
from pathlib import Path
import logging

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import point
from skimage import filters, measure, morphology,feature
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

template_query_write_save_scatters={'dist_x':"UPDATE studies SET \
                                          dist_x=\"{dist}\"\
                                          WHERE id={id}",
                                    'dist_y':"UPDATE studies SET \
                                          dist_y=\"{dist}\"\
                                          WHERE id={id}",
                                    'dist_z':"UPDATE studies SET \
                                          dist_z=\"{dist}\"\
                                          WHERE id={id}",                                          
                                    'dist_total':"UPDATE studies SET \
                                          dist_total=\"{dist}\"\
                                          WHERE id={id}",                                          
                                    }
#previously curve
template_query_write_save_distortions="UPDATE studies SET \
                                          distortions=\"{distortions}\"\
                                          WHERE id={id}"

template_query_write_save_gridpositions="UPDATE studies SET \
                                          gridspos=\"{grids}\"\
                                          WHERE id={id}"

template_query_write_save_statistics="UPDATE studies SET \
                                          stats=\"{statistics}\"\
                                          WHERE id={id}"

template_query_write_save_histfigpath="UPDATE studies SET \
                                          histfigpath=\"{histpath}\"\
                                          WHERE id={id}"

template_query_write_save_matchfigpath="UPDATE studies SET \
                                          matchfigpath=\"{matchpath}\"\
                                          WHERE id={id}"

template_query_write_save_matchpointpath="UPDATE studies SET \
                                          matchpointpath=\"{matchpointpath}\"\
                                          WHERE id={id}"

template_query_write_save_map3Ddistpath="UPDATE studies SET \
                                          map3Ddistpath=\"{map3Ddistpath}\"\
                                          WHERE id={id}"

CT_reference="E:/PostDoc/__Works/Dr.Fatemi/G_phantom/Software/dicoms/CT_reference/13_125.nii.gz"
#CT_reference="/var/www/html/backend2/public/13_125.nii.gz"

BaseAddress="E:/PostDoc/__Works/Dr.Fatemi/G_phantom/Software/dicoms"
# BaseAddress="/home/mehdi/Documents/Gphantom2/python_scripts/dicoms"

db_host="127.0.0.1"
db_port=3306
db_name="phantom_db_v2" #"phantom"
db_user="root" #"debian-sys-maint"
db_password="" #"Vmz8JKxbWusTLhio"

##################################################################################
'''
    Helper for saving NumPy arrays in JSON
'''
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
##################################################################################
'''
    Class of analysis CT/MRI. Find geometry distortion and its statistics
'''        
class analysisData:
    def __init__(self,mydatabase,currentId_mri,mri_analyze_Directory,ct_nifti_path=None,mri_nifti_path=None):
        self.mydatabase=mydatabase
        self.currentId_mri=currentId_mri
        self.analyze_Directory=mri_analyze_Directory
        self.start_analysis(ct_nifti_path,mri_nifti_path,
                            ct_intensity_range=(0, 1000),
                            mri_intensity_range=(0, 100),
                            tolerance_mm=5.0,
                            threshold_mm=5.0
                            )
     ##################################################################################
    '''
        Image Tools function
    '''
     ##########################
    
    #############################
    #      Utility Functions    #
    #############################    
    
    def convert_ndarray_to_list(self,data):
        """
         Recursively convert numpy arrays to lists in a nested dictionary or list.
         Useful for JSON serialization.
        """
        if isinstance(data, dict):
            return {key: self.convert_ndarray_to_list(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_ndarray_to_list(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    #############################
    # Function to load NIfTI    #
    # images and extract the 3D #
    # volume  Image I/O and     #
    # Loading                   #
    ############################# 
        
    def load_nifti_image(self,file_path):
        """
        Loads a NIfTI image using SimpleITK and returns:
        - image (NumPy array, shape: [z, y, x])
        - pixel_spacing (tuple of floats: (x_spacing, y_spacing, z_spacing))
        - isocenter (world coordinates of the physical center of the image)
        - sitk_image (the original SimpleITK Image object)
        """
        nifti_data = sitk.ReadImage(file_path)
        image = sitk.GetArrayFromImage(nifti_data)
        spacing = nifti_data.GetSpacing()  # Extract 3D pixel spacing (in mm)
        origin = np.array(nifti_data.GetOrigin()) # Get the origin of the image (world coordinates of voxel (0,0,0))
        direction = np.array(nifti_data.GetDirection()).reshape(3, 3)
        image_dimensions = image.shape # Get image dimensions (z, y, x) order for the numpy array
        # Compute the physical center (isocenter) in world coordinates
        # Calculate the center of the image in voxel coordinates (x, y, z order)
        center_voxel = np.array([dim // 2 for dim in image_dimensions])
        #image_center_voxel = np.array(image_dimensions[::-1]) / 2.0
        # Calculate the physical isocenter (world coordinates)
        #isocenter = origin + image_center_voxel * np.array(pixel_spacing)
        physical_center = origin + direction.dot(center_voxel * spacing)

        return image, spacing, physical_center, nifti_data
    ##################################################################################
    """
          Preprocessing        
    """
    #############################
    # Clip intensity of the CT  #
    # and MRI images separately #
    #############################
    
    def clip_intensity(self,image, intensity_range):
        """
        Clamps (clips) the intensities of 'image' to [min_val, max_val].
        """        
        max=np.amax(image)
        min=np.amin(image)
        logging.info(f"Max:{max} Min:{min}")
        min_val, max_val = intensity_range
        clipped_image = np.clip(image, min_val, max_val)
        return clipped_image 
    
    #############################
    # Remove regions outside    #
    # the squared phantom       #
    #############################

    def remove_outside_regions(self,image):
        """
        Uses Otsu thresholding + largest connected component to identify phantom region.
        Returns:
        - masked_image
        - mask (boolean)
        - coordinates of the largest connected component
        """
        # Otsu thresholding to create binary mask
        thresh = filters.threshold_otsu(image)
        binary_image = image > thresh
        # Remove small objects
        binary_image = morphology.remove_small_objects(binary_image, min_size=500)

        # Label connected components and remove small objects
        labeled_image = measure.label(binary_image)
        regions = measure.regionprops(labeled_image)

        if not regions:
            logging.error("No regions found after thresholding and removing small objects!")
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
    ################################################################################## 
    '''
        Analysis tools
    '''
    #############################
    #   Grid Cross Detection    #
    #############################    
    
    def detect_plus_like_cross_sections(self,image_3d, image_type='CT'):
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
    ##################################################################################
    '''
        Analysis
    '''  
    #############################
    #     Matching Points       #
    #############################
    
    def compute_common_distances(self,points_img1, points_img2,
                                  spacing1, spacing2,
                                    tolerance_mm=5.0, verbose=False):
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
    
    def compute_distortions(self,common_img1, common_img2, spacing1, spacing2):
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
    #############################
    def summarize_distortions(self,dist_x, dist_y, dist_z, dist_r, threshold_mm=5.0):
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
        # save distortion summary JSON
        self.mydatabase.write_query(
            template_query_write_save_statistics.format(
                id=self.currentId_mri,
                statistics=distortion_summary
            )
        )        
        return summary

    #############################
    #   Visualization Functions #
    #############################

    def plot_distortions(self,dist_x, dist_y, dist_z, dist_r, output_dir):
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
        return hist_path
    
    def visualize_matched_points(self,common_img1, common_img2, ct_image, mri_image, output_dir):
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
        return scatter_path
    
    #############################
    #   Slice-Based Distortion  #
    #############################    

    def compute_slice_distortions(self,common_img1, common_img2, spacing1, spacing2,
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
        sorted_slice_distortions_list = self.convert_ndarray_to_list(sorted_slice_distortions)

        # Optionally save
        if output_file:
            out_json = os.path.join(output_file, "distortions_per_slice.json")
            with open(out_json, 'w') as f:
                json.dump(sorted_slice_distortions_list, f, indent=4)
            logging.info(f"[INFO] Distortion per-slice stats saved to {out_json}")
        self.mydatabase.write_query(template_query_write_save_distortions.format(id=self.currentId_mri,
                                                                                distortions=sorted_slice_distortions_list
                                                                                )
                                    )
        return sorted_slice_distortions    
    #############################
    def generate_distortion_plot_from_isocenter(self,common_img1_mm, distortions, isocenter, component='total_dist', save_path=None):
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
                # 'dist_total'
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
                json.dump(self.convert_ndarray_to_list(distortion_data), f, indent=4)
            logging.info(f"[INFO] Distortion scatter data ({component}) saved to {out_json}")
        self.mydatabase.write_query(template_query_write_save_scatters[component].format(id=self.currentId_mri,
                                                                                        dist=self.convert_ndarray_to_list(distortion_data)
                                                                                        )
                                    )  
        # Return the data dictionary
        return distortion_data
    #############################
    def compute_mutual_nearest_distances(self,ct_points_vox, mri_points_vox, ct_pixel_spacing, mri_pixel_spacing, threshold_mm=None):
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

    ##################################################################################
    '''
        JSON Saving Functions
    '''  
    #############################
    #   Creating Dist. Maps     #
    #############################
    
    def create_and_save_all_distortion_maps(self,sorted_slice_distortions,
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
            logging.info(f"Saved {out_prefix}.npy and {out_prefix}.nii.gz")

        _save_nii_npy(map_dx, dx_sitk, "distortion_dx")
        _save_nii_npy(map_dy, dy_sitk, "distortion_dy")
        _save_nii_npy(map_dz, dz_sitk, "distortion_dz")
        _save_nii_npy(map_dr, dr_sitk, "distortion_dr")
        _save_nii_npy(mask, mask_sitk, "mask")
        Map3DdistPathJSON={
            "distortion_dx":os.path.join(output_dir, "distortion_dx.nii.gz"),
            "distortion_dy":os.path.join(output_dir, "distortion_dy.nii.gz"),
            "distortion_dz":os.path.join(output_dir, "distortion_dz.nii.gz"),
            "distortion_dr":os.path.join(output_dir, "distortion_dr.nii.gz"),
            "mask":os.path.join(output_dir, "mask.nii.gz"),
        }
        self.mydatabase.write_query(
            template_query_write_save_map3Ddistpath.format(
                id=self.currentId_mri,
                map3Ddistpath=Map3DdistPathJSON
            )
        )        
    
    #############################
    #     Slice-Based JSON      #
    #############################

    def save_distortion_scatter_json(self,common_img1_mm, distortions, isocenter, output_dir):
        """
        Saves distortion_scatter_{component}.json files similar to the original code.
        """
        components = ['dist_x', 'dist_y', 'dist_z', 'dist_total']
        for component in components:
            self.generate_distortion_plot_from_isocenter(common_img1_mm, distortions, isocenter,
                                                    component=component, save_path=output_dir)
    #############################
    def save_grid_position_json(self,common_img1_vox, output_dir):
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
        self.mydatabase.write_query(template_query_write_save_gridpositions.format(id=self.currentId_mri,
                                                                                    grids=series_metadata
                                                                                    )
                                    )  
    ##############################
    def save_distortions_per_slice_json(self,sorted_slice_distortions, output_dir):
        """
        Saves distortions_per_slice.json similar to the original code.
        """
        out_json = os.path.join(output_dir, "distortions_per_slice.json")
        with open(out_json, 'w') as f:
            json.dump(self.convert_ndarray_to_list(sorted_slice_distortions), f, indent=4)
        logging.info(f"[INFO] Distortions per slice saved to {out_json}")

    ##################################################################################
    '''
    Start Analysis
    '''
    ##############################
    def start_analysis(self,ct_file_path,mri_file_path,
         ct_intensity_range=(0, 100),
         mri_intensity_range=(200, 1000),
         tolerance_mm=2.0,
         threshold_mm=2.0):

        # 1) Load CT and MRI NIfTI images (full 3D volumes)
        logging.info("1. Loading CT and MRI images...")
        try:
            if(ct_file_path!=""):
                ct_image, ct_spacing, ct_isocenter, ct_sitk = self.load_nifti_image(ct_file_path)
                logging.info(f'ct_pacing:{ct_spacing}')
        except Exception as e:
            logging.error(f"Unexpected error  in loading CT:'{ct_file_path}': {e}")
        try:
            mri_image, mri_spacing, mri_isocenter, mri_sitk = self.load_nifti_image(mri_file_path)
            logging.info(f'mri_spacing:{mri_spacing}')
        except Exception as e:
            logging.error(f"Unexpected error  in loading MRI:'{mri_file_path}': {e}")            

        # 2) Clip intensity of CT and MRI separately
        logging.info("2. Clipping intensities...")
        clipped_ct = self.clip_intensity(ct_image, ct_intensity_range)  # Clip CT between 0 and 100
        clipped_mri = self.clip_intensity(mri_image, mri_intensity_range)  # Clip MRI between 0 and 1000

        # 3) Mask out outside region from MRI and apply same mask to CT
        logging.info("3. Removing outside regions...")
        mri_phantom, mask, _, _, _ = self.remove_outside_regions(clipped_mri)
        ct_phantom = clipped_ct * mask

        # 4) Detect cross-sections
        logging.info("4. Detect cross-sections...")
        ct_intersections_vox = self.detect_plus_like_cross_sections(ct_phantom, image_type='CT')
        mri_intersections_vox = self.detect_plus_like_cross_sections(mri_phantom, image_type='MRI')
        logging.info(f"CT intersections detected: {ct_intersections_vox.shape[0]}")
        logging.info(f"MRI intersections detected: {mri_intersections_vox.shape[0]}")

        # 5) Match points (in voxel space) with a certain tolerance
        logging.info("5. Matching grid intersection points between CT and MRI...")

        ct_matched_vox, mri_matched_vox = self.compute_common_distances(
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
        logging.info("6. Computing distortions between matched points...")
        dist_x, dist_y, dist_z, dist_r, ct_matched_mm, mri_matched_mm = self.compute_distortions(
            ct_matched_vox,
            mri_matched_vox,
            spacing1=ct_spacing,
            spacing2=mri_spacing
        )        
        self.summarize_distortions(dist_x, dist_y, dist_z, dist_r, threshold_mm=threshold_mm)
        logging.info(f"Saved distortion summary to statistics  in DB")

        # 7) Generate and save visualizations
        logging.info("7. Generate and save visualizations...")
        histfigpath=self.plot_distortions(dist_x, dist_y, dist_z, dist_r, self.analyze_Directory)
        # save distortion summary JSON
        self.mydatabase.write_query(
            template_query_write_save_histfigpath.format(
                id=self.currentId_mri,
                histfigpath=histfigpath
            )
        )
        
        matchfigpath=self.visualize_matched_points(ct_matched_vox, mri_matched_vox, ct_image, mri_image, self.analyze_Directory)
        # save distortion summary JSON
        self.mydatabase.write_query(
            template_query_write_save_matchfigpath.format(
                id=self.currentId_mri,
                matchfigpath=matchfigpath
            )
        )
        
        # 8) Save matched points and distortions

        matched_points = {
            'CT_matched_points_mm': ct_matched_mm.tolist(),
            'MRI_matched_points_mm': mri_matched_mm.tolist(),
            'dist_x_mm': dist_x.tolist(),
            'dist_y_mm': dist_y.tolist(),
            'dist_z_mm': dist_z.tolist(),
            'dist_r_mm': dist_r.tolist()
        }
        matched_points_path = os.path.join(self.analyze_Directory, 'matched_points.json')
        with open(matched_points_path, 'w') as f:
            json.dump(matched_points, f, indent=4, cls=NumpyEncoder)
        self.mydatabase.write_query(
            template_query_write_save_matchpointpath.format(
                id=self.currentId_mri,
                matchpointpath=matched_points_path
            )
        )
        logging.info(f"8. Saved matched points and distortions to {matched_points_path}")

        # 9) Compute and save slice-based distortions
        logging.info("9. Computing slice-based distortions...")

        sorted_slice_distortions=self.compute_slice_distortions(
            common_img1=ct_matched_vox,
            common_img2=mri_matched_vox,
            spacing1=ct_spacing,
            spacing2=mri_spacing,
            threshold_mm=threshold_mm,
            output_file=self.analyze_Directory,
            percentage_threshold= 1.0,
            )
        # 10) Generate and save distortion scatter JSON files
        logging.info("10. Generating distortion scatter JSON files...")
        distortions = np.stack((dist_x, dist_y, dist_z), axis=1)
        self.save_distortion_scatter_json(
            common_img1_mm=ct_matched_mm,
            distortions=distortions,
            isocenter=ct_isocenter,
            output_dir=self.analyze_Directory
            )
        # 11) Save grid positions
        logging.info("11. Saving grid positions...")
        self.save_grid_position_json(
            common_img1_vox=ct_matched_vox,
            output_dir=self.analyze_Directory
        )
        # 12) Create and save distortion maps
        logging.info("12. Creating 3D distortion maps (dx, dy, dz, dr)...")
        self.create_and_save_all_distortion_maps(
            sorted_slice_distortions=sorted_slice_distortions,
            reference_sitk_image=ct_sitk,
            output_dir=self.analyze_Directory,
            apply_gaussian_smoothing=True,
            sigma=1.0,
            truncate=1.0,
            mask=mask,
        )
        # 13) (Optional) Compute mutual nearest distances and log stats
        logging.info("13. Compute mutual nearest distances and log stats...")
        dist_x_mutual, dist_y_mutual, dist_z_mutual, dist_r_mutual = self.compute_mutual_nearest_distances(
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

        # # 14) Save the intersection points found in CT (for reference)
        # logging.info("Saving grid positions (voxel coordinates)...")
        # self.save_grid_position_json(
        #     common_img1_vox=ct_matched_vox,
        #     output_dir=self.analyze_Directory
        # )

        logging.info("Processing complete.")

##############################################################################################################
'''
    Class of registeration CT/MRI. MRI registered to CT
'''
class registerImages:
    def __init__(self,ct_nifti_path=None,mri_nifti_path=None,registered_mri_nifti_path=None):
        # print("=======================")
        # print(ct_nifti_path)
        # print(mri_nifti_path)
        # print(registered_mri_nifti_path)
        # print("=======================")
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
##############################################################################################################
'''
    Find Name of files and call rgisteration the analysis
'''       
def main(id=-1):
    logging.basicConfig(level=logging.NOTSET,format='%(asctime)s:%(levelname)s: %(message)s')
    id_ct=0
    id_mri=0
    if(len(sys.argv)<2 and id==-1):
        logging.error("Syntax: Analysis_data.py [<id_ct>] <id_mri>")
        return 
    elif (len(sys.argv)==2):
        id_mri=sys.argv[1]
    elif (len(sys.argv)>2):
        id_ct=int(sys.argv[1])
        id_mri=int(sys.argv[2])
    else :
        id_mri=id
    ##############################################################################
    ### Find Path and Name of Files
    mydatabase=DataBase(host=db_host,port=db_port,database=db_name,user=db_user,password=db_password)
    currentId_ct=id_ct
    currentId_mri=id_mri
    logging.info(f'current ct Id:{currentId_ct}')
    logging.info(f'current mri Id:{currentId_mri}')
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
    logging.info(f">>>>>>CT File:{ct_nifti_file}")  

    readid_mri=mydatabase.read_query(template_query_read_directory.format(id=currentId_mri))
    mri_Directory=readid_mri[0]['directory']
    mri_nifti_Directory=os.path.join(Path(mri_Directory),'Nifti')
    # mri_nifti_Directory=f'{mri_nifti_Directory}'.replace('\\','\\\\')
    mri_nifti_files=[f for f in os.listdir(mri_nifti_Directory) if (f.endswith('.nii.gz')and (not f.startswith('registeredImage_')))]
    mri_nifti_file=os.path.join(mri_nifti_Directory,mri_nifti_files[0])
    mri_nifti_file=f'{mri_nifti_file}'.replace('\\','\\\\')
    logging.info(f">>>>>>MRI File: {mri_nifti_file}")  
    mri_analyze_Directory=os.path.join(Path(mri_Directory),'analyze')
    os.makedirs(mri_analyze_Directory,exist_ok=True)
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
    logging.info("A. Registeration was done!!!!")
    ##############################################################################
    ###  Analysis
    analysisData(mydatabase,currentId_mri,mri_analyze_Directory,ct_nifti_path=ct_nifti_file,mri_nifti_path=registered_mri_nifti_file)
    logging.info("B. Analysis was done!!!!")
    ##############################################################################
    ### Write end of analysis in DB
    mydatabase.write_query(template_query_write_save_status.format(id=currentId_mri,
                                                                                status="analyzed"
                                                                                )
                                )        
    mydatabase.close()        
    ##################################################################################

if __name__=="__main__":
    main(93)
    sys.exit()
