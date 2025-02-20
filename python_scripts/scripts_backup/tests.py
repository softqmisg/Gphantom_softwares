import pydicom
import json
dicom_file="E:\PostDoc\__Works\Dr.Fatemi\G_phantom\Software\dicoms\MRI_Ax_T1_SE_2\outs\Ax_T1_SE\\1.dcm"
# dicom_file=f'{dicom_file}'.replace('\\','\\\\')

ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)

# metadata = {
#     "SelectScan": ds.SeriesDescription if 'SeriesDescription' in ds else "Unknown",
#     "ImageFolderName": ds.SeriesInstanceUID if 'SeriesInstanceUID' in ds else "Unknown",
#     "Modality": ds.Modality if 'Modality' in ds else "Unknown",
#     "SeriesDescription": ds.SeriesDescription if 'SeriesDescription' in ds else "Unknown",
#     "SeriesDate": ds.SeriesDate if 'SeriesDate' in ds else "Unknown",
#     "SequenceName": ds.SequenceName if 'SequenceName' in ds else "Unknown",
#     "Bandwidth": ds.PixelBandwidth if 'PixelBandwidth' in ds else "Unknown",
#     "FieldStrength": ds.MagneticFieldStrength if 'MagneticFieldStrength' in ds else "Unknown",
#     "GradientStrengthmT/m": ds.GradientOutputType if 'GradientOutputType' in ds else "Unknown",
#     "Encoding": ds.ScanningSequence if 'ScanningSequence' in ds else "Unknown",
#     "Manufacturer": ds.Manufacturer if 'Manufacturer' in ds else "Unknown",
#     "Model": ds.ManufacturerModelName if 'ManufacturerModelName' in ds else "Unknown",
#     "EngineVersion": ds.SoftwareVersions if 'SoftwareVersions' in ds else "Unknown",
#     "NumberofSlices": ds.ImagesInAcquisition if 'ImagesInAcquisition' in ds else "Unknown",
#     "SeriesUID": ds.SeriesInstanceUID if 'SeriesInstanceUID' in ds else "Unknown",
#     "StationName": ds.StationName if 'StationName' in ds else "Unknown"
# }


"""
    Custom JSON Encoder
"""
class DICOMEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pydicom.multival.MultiValue):
            return list(obj)  # Convert MultiValue to list
        elif isinstance(obj, pydicom.valuerep.PersonName):
            return str(obj)  # Convert PersonName to string
        elif isinstance(obj, pydicom.valuerep.DSfloat):
            return float(obj)  # Convert DSfloat to float
        elif isinstance(obj, pydicom.valuerep.IS):
            return int(obj)  # Convert IS to int
        # Add more custom handling as needed
        return super().default(obj)
    


metadata={"EngineVersion": ds.SoftwareVersions if 'SoftwareVersions' in ds else "Unknown"}

# print(metadata)
print( json.dumps(metadata, cls=DICOMEncoder, indent=4))


metadata={
            'SelectScan': 'Ax T1 SE',
            'ImageFolderName': '1.2.840.113619.2.312.6945.296904.16109.1739798501.839',
            'Modality': 'MR',
            'SeriesDescription': 'Ax T1 SE',
            'SeriesDate': '20250217',
            'SequenceName': 'Unknown',
            'Bandwidth': '122.07',
            'FieldStrength': '1.5',
            'GradientStrengthmT/m': 'Unknown',
            'Encoding': 'SE',
            'Manufacturer': 'GE MEDICAL SYSTEMS',
            'Model': 'Signa HDxt',
            'EngineVersion': ['24', 'LX', 'MR Software release:HD16.0_V03_1638.a'],
            'NumberofSlices': '73',
            'SeriesUID': '1.2.840.113619.2.312.6945.296904.16109.1739798501.839',
            'StationName': 'mr1hdx'
        }