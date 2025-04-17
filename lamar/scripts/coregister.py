"""
coregister - Image Registration for Aligning Neuroimaging Data

Part of the micaflow processing pipeline for neuroimaging data.

This module performs comprehensive image registration between two images using the
Advanced Normalization Tools (ANTs) SyNRA algorithm, which combines rigid, affine,
and symmetric normalization transformations. It aligns a moving image with a fixed
reference space, enabling spatial normalization of neuroimaging data for group analysis,
multimodal integration, or atlas-based analyses.

Features:
--------
- Combined rigid, affine, and SyN nonlinear registration in one step
- Bidirectional transformation capability (forward and inverse)
- Option to save all transformation components for later application
- Uses ANTs' powerful SyNRA algorithm for optimal accuracy
- Preserves header information in the registered output images

API Usage:
---------
micaflow coregister
    --fixed-file <path/to/reference.nii.gz>
    --moving-file <path/to/source.nii.gz>
    --output <path/to/registered.nii.gz>
    [--warp-file <path/to/warp.nii.gz>]
    [--affine-file <path/to/affine.mat>]
    [--rev-warp-file <path/to/reverse_warp.nii.gz>]
    [--rev-affine-file <path/to/reverse_affine.mat>]

Python Usage:
-----------
>>> from micaflow.scripts.coregister import ants_linear_nonlinear_registration
>>> ants_linear_nonlinear_registration(
...     fixed_file="mni152.nii.gz",
...     moving_file="subject_t1w.nii.gz",
...     out_file="registered_t1w.nii.gz",
...     warp_file="warp.nii.gz",
...     affine_file="affine.mat",
...     rev_warp_file="reverse_warp.nii.gz",
...     rev_affine_file="reverse_affine.mat"
... )

"""
import ants
import argparse
import shutil
import sys
from colorama import init, Fore, Style

init()


def print_help():
    """Print a help message with examples."""
    # ANSI color codes
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL

    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                      IMAGE COREGISTRATION                      ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}

    This script performs linear (rigid + affine) and nonlinear (SyN) registration 
    between two images using ANTs. The registration aligns the moving image to 
    match the fixed reference image space.

    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow coregister {GREEN}[options]{RESET}

    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--fixed-file{RESET}   : Path to the fixed/reference image (.nii.gz)
      {YELLOW}--moving-file{RESET}  : Path to the moving image to be registered (.nii.gz)
      {YELLOW}--output{RESET}       : Output path for the registered image (.nii.gz)

    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--warp-file{RESET}      : Path to save the forward warp field (.nii.gz)
      {YELLOW}--affine-file{RESET}    : Path to save the forward affine transform (.mat)
      {YELLOW}--rev-warp-file{RESET}  : Path to save the reverse warp field (.nii.gz)
      {YELLOW}--rev-affine-file{RESET}: Path to save the reverse affine transform (.mat)

    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}

    {BLUE}# Register a moving image to a fixed image{RESET}
    micaflow coregister {YELLOW}--fixed-file{RESET} mni152.nii.gz {YELLOW}--moving-file{RESET} subject_t1w.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz {YELLOW}--warp-file{RESET} warp.nii.gz {YELLOW}--affine-file{RESET} affine.mat

    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} The registration performs SyNRA transformation (rigid+affine+SyN)
    {MAGENTA}•{RESET} Forward transforms convert from moving space to fixed space
    {MAGENTA}•{RESET} Reverse transforms convert from fixed space to moving space
    {MAGENTA}•{RESET} The transforms can be applied to other images using apply_warp
    """
    print(help_text)


def ants_linear_nonlinear_registration(
        fixed_file,
        moving_file,
        out_file="registered_image.nii",
        warp_file=None,
        affine_file=None,
        rev_warp_file=None,
        rev_affine_file=None,
        registration_method="SyNRA",
        second_moving_file=None,
        second_fixed_file=None,
        metric_weights=[1.0, 1.0]  # Weights for each image pair
):
    """Perform registration using single or multiple image pairs.
    
    When second_moving_file and second_fixed_file are provided, performs
    multi-channel registration that optimizes alignment for both image pairs
    simultaneously, generating a single set of transforms.
    """
    # Load primary images
    fixed = ants.image_read(fixed_file)
    moving = ants.image_read(moving_file)
    
    # Check if we're doing multi-channel registration
    if second_moving_file and second_fixed_file:
        moving2 = ants.image_read(second_moving_file)
        fixed2 = ants.image_read(second_fixed_file)
        
        print("Performing multi-channel registration with both image pairs...")
        
        # Set up multi-metric registration
        fixed_images = [fixed, fixed2]
        moving_images = [moving, moving2]
        metrics = ['mattes', 'mattes']  # Mutual information for both pairs
        
        # We need to normalize weights
        norm_weights = [w / sum(metric_weights) for w in metric_weights]
        
        # Configure the multi-metric registration
        multivariate_extras = {
            'metrics': metrics,
            'fixed_images': fixed_images,
            'moving_images': moving_images,
            'weights': norm_weights
        }
        
        # Run registration with multi-metric approach
        transforms = ants.registration(
            fixed=fixed,  # Primary fixed image (needed but multimetric takes precedence)
            moving=moving,  # Primary moving image
            type_of_transform=registration_method,
            multivariate_extras=multivariate_extras
        )
        
        # Apply transforms to both moving images
        registered = ants.apply_transforms(
            fixed=fixed, 
            moving=moving,
            transformlist=transforms["fwdtransforms"],
            interpolator="nearestNeighbor"
        )
        
        # Save the primary registered image
        ants.image_write(registered, out_file)
        print(f"Registration complete. Saved registered image as {out_file}")
        
    else:
        # Standard single-channel registration
        transforms = ants.registration(
            fixed=fixed, 
            moving=moving, 
            type_of_transform=registration_method
        )
        
        # Apply transforms
        registered = ants.apply_transforms(
            fixed=fixed, 
            moving=moving, 
            transformlist=transforms["fwdtransforms"],
            interpolator="nearestNeighbor"
        )
        
        # Save the registered image
        ants.image_write(registered, out_file)
        print(f"Registration complete. Saved registered image as {out_file}")

    # Save transform files as before
    if warp_file:
        shutil.copyfile(transforms["fwdtransforms"][0], warp_file)
        print(f"Saved warp field as {warp_file}")
    if affine_file:
        shutil.copyfile(transforms["fwdtransforms"][1], affine_file)
        print(f"Saved affine transform as {affine_file}")
    if rev_warp_file:
        shutil.copyfile(transforms["invtransforms"][1], rev_warp_file)
        print(f"Saved reverse warp field as {rev_warp_file}")
    if rev_affine_file:
        shutil.copyfile(transforms["invtransforms"][0], rev_affine_file)
        print(f"Saved reverse affine transform as {rev_affine_file}")


def main():
    """Entry point for command-line use"""
    parser = argparse.ArgumentParser(description="Coregistration tool")
    # Existing arguments
    parser.add_argument("--fixed-file", required=True, help="Primary fixed image file path")
    parser.add_argument("--moving-file", required=True, help="Primary moving image file path")
    parser.add_argument("--output", required=True, help="Output image file path")
    parser.add_argument("--registration-method", default="SyNRA", help="Registration method")
    parser.add_argument("--affine-file", help="Affine transformation file path")
    parser.add_argument("--warp-file", help="Warp field file path")
    parser.add_argument("--rev-warp-file", help="Reverse warp field file path")
    parser.add_argument("--rev-affine-file", help="Reverse affine transformation file path")
    
    # New arguments for multi-channel registration
    parser.add_argument("--second-fixed-file", help="Secondary fixed image file path")
    parser.add_argument("--second-moving-file", help="Secondary moving image file path")
    parser.add_argument("--metric-weights", nargs=2, type=float, default=[1.0, 1.0], 
                        help="Weights for each image pair in registration (default: 1.0 1.0)")
    
    args = parser.parse_args()
    
    ants_linear_nonlinear_registration(
        fixed_file=args.fixed_file,
        moving_file=args.moving_file,
        out_file=args.output,
        registration_method=args.registration_method,
        affine_file=args.affine_file,
        warp_file=args.warp_file,
        rev_warp_file=args.rev_warp_file,
        rev_affine_file=args.rev_affine_file,
        second_fixed_file=args.second_fixed_file,
        second_moving_file=args.second_moving_file,
        metric_weights=args.metric_weights
    )

if __name__ == "__main__":
    main()