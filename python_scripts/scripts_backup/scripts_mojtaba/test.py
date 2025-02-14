import SimpleITK as sitk
file_path='E:\\PostDoc\\__Works\\Dr.Fatemi\\G_phantom\\Software\\Gphantom2\\python_scripts\\dicoms\\test_dicom\\outs\\t1_mprage_sag_p2_iso_90deg\\Nifti\\4001_t1_mprage_sag_p2_iso_90deg.nii.gz'
print(file_path)
sitk.ReadImage(file_path)