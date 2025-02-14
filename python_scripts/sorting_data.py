import sys
import os
import shutil
import dicom2nifti
import dicom2nifti.exceptions
import pydicom
from collections import defaultdict
from dicom2nifti.exceptions import ConversionValidationError
import json 
import datetime

#dicom2nifti               2.4.11
#pydicom                   2.4.4
#pip install mysql-connector-python

"""
    Database queery Templates
"""
from Database import DataBase

template_query_read_directory="SELECT * FROM uploads WHERE id={id}"
template_query_write_save_info_directory="INSERT  INTO studies (userId,info,directory,status,created_at) VALUES ({userId},\"{metadata}\",\"{directory}\",\"non\",\"{created_at}\")"

# BaseAddress="/home/mehdi/Documents/Gphantom2/python_scripts/dicoms"
BaseAddress="E:/PostDoc/__Works/Dr.Fatemi/G_phantom/Software/dicoms"
db_host="127.0.0.1"
db_port=3306
db_name="phantom_db_v2"
db_user="root"
db_password=""

class sortData:
    def __init__(self,id, dir_out="outs",move:bool=True):
        """

        :param dir_src: directory of the dicom files
        :param dir_out: directory to save the separated dicom files
        :param move: to remove or keep the source file (default: False --> to keep the source images)
        """

        self.mydatabase=DataBase(host=db_host,port=db_port,database=db_name,user=db_user,password=db_password)
        self.currentId=id
        readid=self.mydatabase.read_query(template_query_read_directory.format(id=self.currentId))
        print(readid[0]['directory'])
        Directory=readid[0]['directory']
        self.dir_src=os.path.join(BaseAddress,Directory)
        self.dir_out = os.path.join(self.dir_src,dir_out)
        self.currentUserId=readid[0]['userId']
        print("input:",self.dir_src)
        print("output:",self.dir_out)
        self.move = move

        self.separate_data()



    @staticmethod
    def is_dicom_file(filename):
        try:
            # Attempt to read the file's metadata to check if it is a valid DICOM file
            pydicom.dcmread(filename, stop_before_pixels=True)
            return True
        except pydicom.errors.InvalidDicomError:
            return False

    def load_dicom_directory(self, directory_path):
        # Get all files in the directory
        all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)]

        # Filter only DICOM files
        dicom_files = [f for f in all_files if self.is_dicom_file(f)]
        
        # Initialize a dictionary to store the DICOM datasets by SeriesDescription
        dicom_series = defaultdict(list)
        for file in dicom_files:
            # Read each DICOM file
            ds = pydicom.dcmread(file)

            # Use the SeriesDescription (or SequenceName if SeriesDescription is not available) to group the images
            sequence_name = getattr(ds, 'SeriesDescription', None) or getattr(ds, 'SequenceName', None)
            if not sequence_name:
                sequence_name = 'UnknownSequence'  # Use a default name if neither attribute is available

            dicom_series[sequence_name].append(file)

        return dicom_series

    def move_files_to_series_directories(self, dicom_series, base_output_dir:str, move:bool):

        # Create the base output directory if it doesn't exist
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

        for series_uid, files in dicom_series.items():
            # Create a directory for each series
            series_dir = os.path.join(base_output_dir, series_uid, "dicom")
            if not os.path.exists(series_dir):
                os.makedirs(series_dir)
                print('generate folder:',series_dir)

            # Move each file to the corresponding series directory
            if move:
                for file in files:
                    shutil.move(file, series_dir)
            else:
                for file in files:
                    shutil.copy(file, series_dir)

    def extract_dicom_metadata(self, dicom_file):
        """Extract relevant metadata from a DICOM file."""
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        # print(ds)
        # print(ds.ImageOrientationPatient)
        # exit()
        metadata = {
            "SelectScan": ds.SeriesDescription if 'SeriesDescription' in ds else "Unknown",
            "ImageFolderName": ds.SeriesInstanceUID if 'SeriesInstanceUID' in ds else "Unknown",
            "Modality": ds.Modality if 'Modality' in ds else "Unknown",
            # "Orientation": ds.ImageOrientationPatient if 'ImageOrientationPatient' in ds else "Unknown",
            "SeriesDescription": ds.SeriesDescription if 'SeriesDescription' in ds else "Unknown",
            "SeriesDate": ds.SeriesDate if 'SeriesDate' in ds else "Unknown",
            "SequenceName": ds.SequenceName if 'SequenceName' in ds else "Unknown",
            "Bandwidth": ds.PixelBandwidth if 'PixelBandwidth' in ds else "Unknown",
            "FieldStrength": ds.MagneticFieldStrength if 'MagneticFieldStrength' in ds else "Unknown",
            "GradientStrengthmT/m": ds.GradientOutputType if 'GradientOutputType' in ds else "Unknown",
            "Encoding": ds.ScanningSequence if 'ScanningSequence' in ds else "Unknown",
            "Manufacturer": ds.Manufacturer if 'Manufacturer' in ds else "Unknown",
            "Model": ds.ManufacturerModelName if 'ManufacturerModelName' in ds else "Unknown",
            "EngineVersion": ds.SoftwareVersions if 'SoftwareVersions' in ds else "Unknown",
            "NumberofSlices": ds.ImagesInAcquisition if 'ImagesInAcquisition' in ds else "Unknown",
            "SeriesUID": ds.SeriesInstanceUID if 'SeriesInstanceUID' in ds else "Unknown",
            "StationName": ds.StationName if 'StationName' in ds else "Unknown"
        }
        return metadata

    def dicom_to_nifti_by_series(self, input_dir, output_dir, min_slices=3):
        """
        Convert DICOM files to NIfTI format, separating them by MRI sequence.
        Save metadata in JSON files alongside the NIfTI files.

        Parameters:
            input_dir (str): Directory containing DICOM files.
            output_dir (str): Directory where NIfTI files will be saved.
            min_slices (int): Minimum number of slices required to convert a series.
        """
        series_dict = {}

        # Traverse the input directory to find and group DICOM files by SeriesInstanceUID
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # try:
                # Read the DICOM file
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                # # Skip files without pixel data
                # if not hasattr(ds, 'PixelData'):
                #     print(f"Skipping non-image DICOM file: {file_path}")
                #     continue

                # Get the SeriesInstanceUID and SeriesDescription to group files
                series_uid = ds.SeriesInstanceUID
                series_description = ds.SeriesDescription if 'SeriesDescription' in ds else series_uid

                # Skip localizers based on SeriesDescription or ImageType
                if 'LOCALIZER' in series_description.upper() or \
                        (hasattr(ds, 'ImageType') and 'LOCALIZER' in ds.ImageType):
                    print(f"Skipping localizer series: {series_description}")
                    continue

                if series_uid not in series_dict:
                    series_dict[series_uid] = {
                        "description": series_description,
                        "files": [],
                        "metadata": self.extract_dicom_metadata(file_path)  # Save metadata from the first file
                    }
                series_dict[series_uid]["files"].append(file_path)
                # except Exception as e:
                #     print(f"Could not read DICOM file {file_path}: {e}")


        # Process each series separately
        for series_uid, series_info in series_dict.items():
            series_files = series_info["files"]
            series_description = series_info["description"]
            series_metadata = series_info["metadata"]

            # Check if series has sufficient slices
            if len(series_files) < min_slices:
                print(f"Skipping series '{series_description}' due to insufficient slices ({len(series_files)}).")
                continue

            # Create output subdirectory for this series
            series_output_name = os.path.join(output_dir, series_description)
            series_output_dir=series_output_name.replace(' ','_')
            os.makedirs(series_output_dir, exist_ok=True)
            
            series_Niftioutput_dir = os.path.join(series_output_dir,'Nifti')
            os.makedirs(series_Niftioutput_dir, exist_ok=True)

            # # Temporary directory for the DICOM series files
            # temp_series_dir = os.path.join(series_output_dir, 'temp_series')
            # os.makedirs(temp_series_dir, exist_ok=True)

            # Copy the selected series files to the temporary directory
            temp_file_path=""
            filename_index=1
            for file_path in series_files:
                file_name = os.path.basename(file_path)
                temp_file_path = os.path.join(series_output_dir, f'{filename_index}.dcm')
                # os.symlink(file_path, temp_file_path)  # Create a symbolic link to avoid copying
                shutil.move(file_path, temp_file_path)
                filename_index=filename_index+1

            try:
                # Convert DICOM directory to NIfTI
                print("Out Nifti:",series_Niftioutput_dir)
                dicom2nifti.convert_directory(series_output_dir, series_Niftioutput_dir, compression=True, reorient=True)
                # dicom2nifti.dicom_series_to_nifti(series_output_dir, os.path.join(series_Niftioutput_dir, f"{series_description}.nii"), reorient_nifti=True)
            # except dicom2nifti.exceptions.ConversionValidationError as e:
            #     print(f"Could not convert series '{series_description}': {e}")
            #     return
            except Exception as e:
                print(f"Unexpected error converting series '{series_description}': {e}")
                return

            """
                update studies Table with directory and metafile

            """
            metafile=self.extract_dicom_metadata(temp_file_path)
            str_dir=f'{series_output_dir}'.replace('\\','\\\\')

            self.mydatabase.write_query(template_query_write_save_info_directory.format(userId=self.currentUserId,
                                                                                        metadata=metafile,
                                                                                        directory=str_dir,
                                                                                        created_at=datetime.datetime.now()
                                                                                        ))
        print("===========================")

    def separate_data(self):
        """

        :param dir_src: Directory where dicom files are located.
        :param dir_out: Directory where separated dicom  files will be stored.
        :param move: Move it or copy: default is COPY.
        :return:
        """
        # Load DICOM files grouped by series
        dicom_series = self.load_dicom_directory(self.dir_src)
        self.dicom_to_nifti_by_series(self.dir_src, self.dir_out)

        # self.move_files_to_series_directories(dicom_series, self.dir_out, self.move)
        self.mydatabase.close()



def main():
    # print(pydicom.__version__)
    # sorting = sortData(id=1)

    print(len(sys.argv))
    if(len(sys.argv)<2):
        print("syntax: \n "
              "python sorting_data.py <id>")
    else:
        print("start sorting")
        sorting = sortData(id=int(sys.argv[1])) #.separate_data()


if __name__=="__main__":
    main()
    sys.exit()