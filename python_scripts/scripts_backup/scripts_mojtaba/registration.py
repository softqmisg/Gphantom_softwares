"""
This code perform registration well but requires Nifti data
"""
import SimpleITK as sitk


def read_image(file_path):
    """Read the volumetric image file."""
    return sitk.ReadImage(file_path)


def check_image_compatibility(fixed_image, moving_image):
    """Ensure the images have the same dimension and type."""
    if fixed_image.GetDimension() != moving_image.GetDimension():
        raise ValueError(
            f"Fixed image and moving image have different dimensions: {fixed_image.GetDimension()} vs {moving_image.GetDimension()}")

    if fixed_image.GetPixelID() != moving_image.GetPixelID():
        print(f"Warning: Fixed image and moving image have different types. Casting moving image to fixed image type.")
        moving_image = sitk.Cast(moving_image, fixed_image.GetPixelID())

    return moving_image


def register_images(fixed_image, moving_image):
    """Register the moving_image to the fixed_image using mutual information."""

    # Ensure compatibility
    moving_image = check_image_compatibility(fixed_image, moving_image)

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


def resample_image(moving_image, transform, reference_image):
    """Resample the moving image using the final transformation."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    return resampler.Execute(moving_image)

# File paths for the fixed (CT) and moving (MRI) images
fixed_image_path = "/media/jdi/ssd2/Data/grant_nih_2023/grid_phantom_for_part2/nifti/ct.nii.gz"
moving_image_path = "/media/jdi/ssd2/Data/grant_nih_2023/grid_phantom_for_part2/nifti/t1w.nii.gz"

# Read the images
fixed_image = read_image(fixed_image_path)
moving_image = read_image(moving_image_path)

# Register the moving image to the fixed image
final_transform = register_images(fixed_image, moving_image)

# Resample the moving image using the final transform
registered_moving_image = resample_image(moving_image, final_transform, fixed_image)

# Save the registered image
sitk.WriteImage(registered_moving_image, "/media/jdi/ssd2/Data/grant_nih_2023/grid_phantom_for_part2/nifti/registered_MRI_image.nii.gz")


