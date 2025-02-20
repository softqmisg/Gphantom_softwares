import sys
import os
import numpy as np
from pathlib import Path
import logging
import json
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

"""
    Database queery Templates
"""
from Database import DataBase
template_query_read_directory="SELECT directory FROM studies WHERE id={id}"
template_query_read_map3Ddistpath="SELECT map3Ddistpath FROM studies WHERE id={id}"
template_query_write_save_tempfile="""
                                          UPDATE studies SET 
                                          tempfile = %s
                                          WHERE id = %s
                                          """


BaseAddress="E:/PostDoc/__Works/Dr.Fatemi/G_phantom/Software/dicoms"

db_host="127.0.0.1"
db_port=3306
db_name="phantom_db_v2" #"phantom"
db_user="root" #"debian-sys-maint"
db_password="" #"Vmz8JKxbWusTLhio"



def load_and_interpolate_distortion(dist_name, slice_number, direction, M, distortionfile_name,maskfile_name, output_dir, visualize=True):
    """
    Loads a distortion map, slices it in the specified direction and slice number,
    performs block-wise averaging interpolation with block size M, and saves the interpolated surface.

    Parameters:
    - dist_name (str): Distortion map identifier ('_dr', '_dx', '_dy', '_dz').
    - slice_number (int): The index of the slice to extract.
    - direction (str): The slicing direction ('axial', 'sagittal', or 'coronal').
    - M (int): The block size for interpolation (e.g., 5).
    - distortionfile_name (str): File containing distortion Map.
    - maskfile_name (str): File containing distortion Mask.
    - output_dir (str): Directory to save the interpolated surface.
    - visualize (bool): If True, saves a heatmap of the interpolated surface.

    Returns:
    - interpolated_surface (np.ndarray): The interpolated 2D surface.
    - heatmap_path
    """

    # Validate direction
    valid_directions = ['axial', 'sagittal', 'coronal']
    if direction.lower() not in valid_directions:
        logging.error(f"Invalid direction '{direction}'. Choose from {valid_directions}.")
        return None

    # Determine file paths
    npy_path = ""
    npy_mask_path = ""

    nii_path = distortionfile_name
    nii_mask_path = maskfile_name

    # Load the distortion map
    if npy_path!="":
        distortion_map = np.load(npy_path)
        mask = np.load(npy_mask_path)
        logging.info(f"Loaded distortion map(npy) from {npy_path}")
    elif os.path.exists(nii_path):
        sitk_image = sitk.ReadImage(nii_path)
        distortion_map = sitk.GetArrayFromImage(sitk_image)

        sitk_mask = sitk.ReadImage(nii_mask_path)
        mask = sitk.GetArrayFromImage(sitk_mask)
        logging.info(f"Loaded distortion map(nii) from {nii_path}")
    else:
        logging.error(f"Distortion map '{dist_name}' not found in '{nii_path}'.")
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
        logging.info(idx1)
        logging.info(idx2)
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


    # Optionally visualize the interpolated surface
    plt.figure(figsize=(8, 6))
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
    if visualize:
        plt.imshow(interpolated_surface, cmap='hot', interpolation='nearest')
    heatmap_filename = f"interpolated{dist_name}_slice{slice_number}_{direction}_M{M}.png"
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    plt.savefig(heatmap_path) # , pad_inches=0, tight_layout=True, dpi=300
    plt.close()
    logging.info(f"Saved interpolated surface heatmap to {heatmap_path}")

    return interpolated_surface,heatmap_path

# Call the function
def main(id=-1,dir=0,slice_number=5,dist_id=0,block_size=20):
    # Configure logging
    dist_components=['_dx','_dy','_dz','_dr']
    directions=['axial','sagittal','coronal']

    logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s: %(message)s')

    if(len(sys.argv)<6 and id ==-1):
        logging.error("Syntax:\n"
                      "Generate_plot_Slice.py <id_mri> <dir> <slice_number> <dist_id> <block_size> "
                    )
    elif (len(sys.argv)>=6):
        id_mri=int(sys.argv[1])
        dir=int(sys.argv[2])
        slice_number=int(sys.argv[3])
        dist_id=int(sys.argv[4])
        block_size=int(sys.argv[5])
    else:
        id_mri=id

    direction=directions[dir]
    dist_component=dist_components[dist_id]
    ################################
    # Find Path and Name of Files  #
    ################################
    mydatabase=DataBase(host=db_host,port=db_port,database=db_name,user=db_user,password=db_password)

    readid_mri=mydatabase.read_query(template_query_read_directory.format(id=id_mri))
    mri_Directory=json.loads(readid_mri[0]['directory'])
    mri_temp_Directory=os.path.join(Path(mri_Directory),'temp')
    os.makedirs(mri_temp_Directory,exist_ok=True)

    readquery=mydatabase.read_query(template_query_read_map3Ddistpath.format(id=id_mri))
    Map3DdistPathJSON=json.loads(readquery[0]['map3Ddistpath'])

    MaskFileName= Map3DdistPathJSON['mask']
    MaskFileName=f'{MaskFileName}'.replace('\\','\\\\')
    logging.info(MaskFileName)

    DistortionFileName=Map3DdistPathJSON['distortion'+dist_component]
    DistortionFileName=f'{DistortionFileName}'.replace('\\','\\\\')
    logging.info(DistortionFileName)
    
    ###################################
    # Generate Image of dist Heat map #
    ###################################
    interpolated_surface,heatmap_path = load_and_interpolate_distortion(
        
        dist_name=dist_component,
        slice_number=slice_number,
        direction=direction,
        M=block_size,
        distortionfile_name=DistortionFileName,
        maskfile_name=MaskFileName,
        output_dir=mri_temp_Directory,
        visualize=False
    )
    mydatabase.write_queryparam(template_query_write_save_tempfile,
                            (json.dumps(heatmap_path) ,id_mri)
                            )
   ###################################
    # close atabase                  #
    ##################################
    mydatabase.close()        
    return

    
if __name__=="__main__":
    main(99)
    sys.exit()
