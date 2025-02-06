import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import logging
from skimage.measure import block_reduce

def load_and_interpolate_distortion(name, slice_number, direction, M, distortion_maps_dir, output_dir, visualize=True):
    """
    Loads a distortion map, slices it in the specified direction and slice number,
    performs block-wise averaging interpolation with block size M, and saves the interpolated surface.

    Parameters:
    - name (str): Distortion map identifier ('_dr', '_dx', '_dy', '_dz').
    - slice_number (int): The index of the slice to extract.
    - direction (str): The slicing direction ('axial', 'sagittal', or 'coronal').
    - M (int): The block size for interpolation (e.g., 5).
    - distortion_maps_dir (str): Directory containing distortion maps.
    - output_dir (str): Directory to save the interpolated surface.
    - visualize (bool): If True, saves a heatmap of the interpolated surface.

    Returns:
    - interpolated_surface (np.ndarray): The interpolated 2D surface.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Validate direction
    valid_directions = ['axial', 'sagittal', 'coronal']
    if direction.lower() not in valid_directions:
        logging.error(f"Invalid direction '{direction}'. Choose from {valid_directions}.")
        return None

    # Determine file paths
    # Assuming distortion maps are saved as 'distortion_dr.npy' and 'distortion_dr.nii.gz', etc.
    base_name = f"distortion{name}.npy"
    nii_name = f"distortion{name}.nii.gz"
    npy_path = os.path.join(distortion_maps_dir, base_name)
    npy_mask_path = os.path.join(distortion_maps_dir, 'mask.npy')
    nii_path = os.path.join(distortion_maps_dir, nii_name)
    nii_mask_path = os.path.join(distortion_maps_dir, 'mask.npy')

    # Load the distortion map
    if os.path.exists(npy_path):
        distortion_map = np.load(npy_path)
        mask = np.load(npy_mask_path)
        logging.info(f"Loaded distortion map from {npy_path}")
    elif os.path.exists(nii_path):
        sitk_image = sitk.ReadImage(nii_path)
        distortion_map = sitk.GetArrayFromImage(sitk_image)

        sitk_mask = sitk.ReadImage(nii_mask_path)
        mask = sitk.GetArrayFromImage(sitk_mask)
        logging.info(f"Loaded distortion map from {nii_path}")
    else:
        logging.error(f"Distortion map '{name}' not found in '{distortion_maps_dir}'.")
        return None

    # Determine slicing axis
    if direction.lower() == 'axial':
        axis = 0  # Slicing along z-axis
    elif direction.lower() == 'coronal':
        axis = 1  # Slicing along y-axis
    else:  # 'sagittal'
        axis = 2  # Slicing along x-axis

    # Validate slice_number
    if slice_number < 0 or slice_number >= distortion_map.shape[axis]:
        logging.error(
            f"slice_number {slice_number} is out of bounds for direction '{direction}' with size {distortion_map.shape[axis]}.")
        return None

    # Extract the 2D slice
    if axis == 0:
        mask_2d = mask[slice_number, :, :]
        idx1, idx2 = np.where(mask_2d)
        idx1, idx2 = np.unique(idx1), np.unique(idx2)

        slice_2d = distortion_map[slice_number, idx1[0]:idx1[-1], idx2[0]:idx2[-1]]  # Axial
    elif axis == 1:
        mask_2d = mask[:, slice_number, :]
        idx1, idx2 = np.where(mask_2d)
        idx1, idx2 = np.unique(idx1), np.unique(idx2)
        slice_2d = distortion_map[idx1[0]:idx1[-1], slice_number, idx2[0]:idx2[-1]]  # Coronal
    else:
        mask_2d = mask[:, :, slice_number]
        idx1, idx2 = np.where(mask_2d)
        idx1, idx2 = np.unique(idx1), np.unique(idx2)
        slice_2d = distortion_map[idx1[0]:idx1[-1], idx2[0]:idx2[-1], slice_number]  # Sagittal

    logging.info(f"Extracted slice number {slice_number} in '{direction}' direction with shape {slice_2d.shape}.")

    # Function to pad the array so that its dimensions are divisible by M
    def pad_to_multiple(arr, M):
        """
        Pads a 2D array so that both dimensions are multiples of M.
        Padding is done by reflecting the array at the borders.

        Parameters:
        - arr (np.ndarray): 2D input array.
        - M (int): Block size.

        Returns:
        - padded_arr (np.ndarray): Padded 2D array.
        """
        pad_rows = (M - arr.shape[0] % M) if arr.shape[0] % M != 0 else 0
        pad_cols = (M - arr.shape[1] % M) if arr.shape[1] % M != 0 else 0
        if pad_rows > 0 or pad_cols > 0:
            padded_arr = np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode='reflect')
            logging.info(f"Padded slice from {arr.shape} to {padded_arr.shape} to make dimensions divisible by {M}.")
            return padded_arr
        else:
            logging.info("No padding needed.")
            return arr

    # Pad the slice if necessary
    slice_padded = pad_to_multiple(slice_2d, M)

    # Perform block-wise averaging using skimage's block_reduce for robustness
    try:
        interpolated_surface = block_reduce(slice_padded, block_size=(M, M), func=np.mean)
        logging.info(
            f"Performed block-wise averaging with block size {M}. Interpolated surface shape: {interpolated_surface.shape}")
    except Exception as e:
        logging.error(f"Error during block-wise averaging: {e}")
        return None

    # # Save the interpolated surface
    # interpolated_filename = f"interpolated{name}_slice{slice_number}_{direction}_M{M}.npy"
    # interpolated_path = os.path.join(output_dir, interpolated_filename)
    # np.save(interpolated_path, interpolated_surface)
    # logging.info(f"Saved interpolated surface to {interpolated_path}")

    # Optionally visualize the interpolated surface
    if visualize:
        plt.figure(figsize=(8, 6))
        plt.imshow(interpolated_surface, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Distortion (mm)')

        for i in range(interpolated_surface.shape[0]):
            for j in range(interpolated_surface.shape[1]):
                value = interpolated_surface[i, j]

                # Choose text color based on distortion magnitude for better visibility
                text_color = 'white' if value > 0 else 'yellow'
                # Add text with a semi-transparent background for readability
                plt.text(
                    j, i, f"{value:.2f}",
                    ha='center', va='center',
                    color=text_color,
                    fontsize=6,
                    bbox=dict(facecolor='black', alpha=0.3, edgecolor='none')
                )
        plt.xticks([])
        plt.yticks([])
        heatmap_filename = f"interpolated{name}_slice{slice_number}_{direction}_M{M}.png"
        heatmap_path = os.path.join(output_dir, heatmap_filename)
        plt.savefig(heatmap_path) # , pad_inches=0, tight_layout=True, dpi=300
        plt.close()
        logging.info(f"Saved interpolated surface heatmap to {heatmap_path}")

    return interpolated_surface


# Define paths
distortion_maps_dir = "/media/jdi/ssd2/Data/grant_nih_2023/Gphantom3_May18_2024/MR_04_29_2024_tilted_90/separated_data_Jan19_2025/stats"
output_dir = "/media/jdi/ssd2/Data/grant_nih_2023/Gphantom3_May18_2024/MR_04_29_2024_tilted_90/separated_data_Jan19_2025/interpolated_distortions"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Parameters
name = '_dr'            # Choose from '_dr', '_dx', '_dy', '_dz'
slice_number = 5      # Example slice number
direction = 'axial'    # Choose from 'axial', 'sagittal', 'coronal'
M = 20                  # Block size for interpolation

# Call the function
interpolated_surface = load_and_interpolate_distortion(
    name=name,
    slice_number=slice_number,
    direction=direction,
    M=M,
    distortion_maps_dir=distortion_maps_dir,
    output_dir=output_dir,
    visualize=True
)


# if __name__ == '__main__':
#     path = "/media/jdi/ssd2/Data/grant_nih_2023/Gphantom3_May18_2024/MR_04_29_2024_tilted_90/separated_data_Jan19_2025/stats/mask.nii.gz"
#     mask = sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))
#     idx_x, idx_y, idx_z = np.where(mask)
#     idx_x, idx_y, idx_z = np.unique(idx_x), np.unique(idx_y), np.unique(idx_z)
#     mask[idx_x[0]:idx_x[-1], idx_y[0]:idx_y[-1], idx_z[0]:idx_z[-1]] = 1
#     plt.imshow(mask[90], cmap='gray')
#
#     plt.show()
# The interpolated_surface is now a 2D NumPy array representing the interpolated distortion surface.
