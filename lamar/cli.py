#!/usr/bin/env python3
"""
LaMAR: Label Augmented Modality Agnostic Registration
Command-line interface
"""

import argparse
import sys
import os
from lamar.scripts.lamar import lamareg
from lamar.scripts import synthseg, coregister, apply_warp
from colorama import init, Fore, Style

init()


def print_cli_help():
    """Print a comprehensive help message for the LAMAR CLI."""
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
    ║                             LAMAR                              ║
    ║             Label Augmented Modality Agnostic Registration     ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}

    LAMAR provides contrast-agnostic registration between different MRI modalities
    by using SynthSeg's brain parcellation to enable robust alignment between images
    with different contrasts (e.g., T1w to T2w, FLAIR to T1w, DWI to T1w).

    {CYAN}{BOLD}────────────────────────── WORKFLOWS ──────────────────────────{RESET}
    
    {BLUE}1. FULL REGISTRATION PIPELINE{RESET}
      Parcellate both input images, register them, and apply the transformation:
      lamar {GREEN}register{RESET} [options]
      
    {BLUE}2. GENERATE WARPFIELD ONLY{RESET}
      Create warpfields without applying them to the input image:
      lamar {GREEN}generate-warpfield{RESET} [options]
      
    {BLUE}3. APPLY EXISTING WARPFIELD{RESET}
      Apply previously created warpfields to an input image:
      lamar {GREEN}apply-warpfield{RESET} [options]
      
    {BLUE}4. DIRECT TOOL ACCESS{RESET}
      Run individual components directly:
      lamar {GREEN}synthseg{RESET} [options]     : Run SynthSeg brain parcellation
      lamar {GREEN}coregister{RESET} [options]   : Run ANTs coregistration
      lamar {GREEN}apply-warp{RESET} [options]   : Apply transformations
      lamar {GREEN}dice-compare{RESET} [options] : Calculate Dice similarity coefficient

    {CYAN}{BOLD}──────────────────── FULL REGISTRATION ────────────────────{RESET}
    
    {BLUE}# Required Arguments:{RESET}
      {YELLOW}--moving{RESET} PATH         : Input image to be registered
      {YELLOW}--fixed{RESET} PATH          : Reference image (target space)
      {YELLOW}--output{RESET} PATH         : Output registered image
      {YELLOW}--moving-parc{RESET} PATH    : Path for moving image parcellation
      {YELLOW}--fixed-parc{RESET} PATH     : Path for fixed image parcellation
      {YELLOW}--registered-parc{RESET} PATH: Path for registered parcellation
      {YELLOW}--affine{RESET} PATH         : Path for affine transformation
      {YELLOW}--warpfield{RESET} PATH      : Path for warp field
      {YELLOW}--inverse-warpfield{RESET} P : Path for inverse warp field
      {YELLOW}--inverse-affine{RESET} PATH : Path for inverse affine transformation
      
    {BLUE}# Optional Arguments:{RESET}
      {YELLOW}--registration-method{RESET} STR : Registration method (default: SyNRA)
      {YELLOW}--synthseg-threads{RESET} N      : SynthSeg threads (default: 1)
      {YELLOW}--ants-threads{RESET} N          : ANTs threads (default: 1)
      {YELLOW}--qc-csv{RESET} PATH             : Path for QC Dice score CSV file

    {CYAN}{BOLD}────────────────── GENERATE WARPFIELD ────────────────────{RESET}
    
    Same arguments as full registration, but without {YELLOW}--output{RESET}
    
    {CYAN}{BOLD}─────────────────── APPLY WARPFIELD ──────────────────────{RESET}
    
    {BLUE}# Required Arguments:{RESET}
      {YELLOW}--moving{RESET} PATH      : Input image to transform
      {YELLOW}--fixed{RESET} PATH       : Reference space image
      {YELLOW}--output{RESET} PATH      : Output registered image
      {YELLOW}--warpfield{RESET} PATH   : Path to warp field
      {YELLOW}--affine{RESET} PATH      : Path to affine transformation
      
    {BLUE}# Optional Arguments:{RESET}
      {YELLOW}--ants-threads{RESET} N   : ANTs threads (default: 1)

    {CYAN}{BOLD}─────────────────── EXAMPLE USAGE ───────────────────────{RESET}

    {BLUE}# Register DWI to T1w:{RESET}
    lamar {GREEN}register{RESET} {YELLOW}--moving{RESET} sub-001_dwi.nii.gz {YELLOW}--fixed{RESET} sub-001_T1w.nii.gz \\
      {YELLOW}--output{RESET} sub-001_dwi_in_T1w.nii.gz {YELLOW}--moving-parc{RESET} sub-001_dwi_parc.nii.gz \\
      {YELLOW}--fixed-parc{RESET} sub-001_T1w_parc.nii.gz {YELLOW}--registered-parc{RESET} sub-001_dwi_reg_parc.nii.gz \\
      {YELLOW}--affine{RESET} dwi_to_T1w_affine.mat {YELLOW}--warpfield{RESET} dwi_to_T1w_warp.nii.gz \\
      {YELLOW}--inverse-warpfield{RESET} T1w_to_dwi_warp.nii.gz {YELLOW}--inverse-affine{RESET} T1w_to_dwi_affine.mat \\
      {YELLOW}--synthseg-threads{RESET} 4 {YELLOW}--ants-threads{RESET} 8

    {BLUE}# Generate parcellations separately:{RESET}
    lamar {GREEN}synthseg{RESET} {YELLOW}--i{RESET} subject_t1w.nii.gz {YELLOW}--o{RESET} t1w_parcellation.nii.gz {YELLOW}--parc{RESET}
    lamar {GREEN}synthseg{RESET} {YELLOW}--i{RESET} subject_flair.nii.gz {YELLOW}--o{RESET} flair_parcellation.nii.gz {YELLOW}--parc{RESET}

    {BLUE}# Register using existing parcellations:{RESET}
    lamar {GREEN}register{RESET} {YELLOW}--moving{RESET} subject_flair.nii.gz {YELLOW}--fixed{RESET} subject_t1w.nii.gz \\
      {YELLOW}--output{RESET} registered_flair.nii.gz {YELLOW}--moving-parc{RESET} flair_parcellation.nii.gz \\
      {YELLOW}--fixed-parc{RESET} t1w_parcellation.nii.gz [other arguments...]

    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} LAMAR works with any MRI modality combination
    {MAGENTA}•{RESET} If parcellation files already exist, they will be used directly
    {MAGENTA}•{RESET} All output files need explicit paths to ensure deterministic behavior
    {MAGENTA}•{RESET} The transforms can be reused with the apply-warpfield command
    {MAGENTA}•{RESET} Use dice-compare to evaluate registration quality
    """
    print(help_text)


def main():
    """Main entry point for the LAMAR CLI."""
    parser = argparse.ArgumentParser(
        description="LAMAR: Label Augmented Modality Agnostic Registration"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # WORKFLOW 1: Full registration pipeline
    register_parser = subparsers.add_parser(
        "register", help="Perform full registration pipeline with SynthSeg parcellation"
    )
    register_parser.add_argument(
        "--moving", required=True, help="Input moving image to be registered"
    )
    register_parser.add_argument(
        "--fixed", required=True, help="Reference fixed image (target space)"
    )
    register_parser.add_argument(
        "--output", required=True, help="Output registered image"
    )
    register_parser.add_argument(
        "--moving-parc", required=True, help="Output path for moving image parcellation"
    )
    register_parser.add_argument(
        "--fixed-parc", required=True, help="Output path for fixed image parcellation"
    )
    register_parser.add_argument(
        "--skip-segment-moving", action="store_true", help="Skip segmentation of moving image."
    )
    register_parser.add_argument(
        "--skip-segment-fixed", action="store_true", help="Skip segmentation of fixed image."
    )


    register_parser.add_argument(
        "--registered-parc",
        required=True,
        help="Output path for registered parcellation",
    )
    register_parser.add_argument(
        "--affine", required=True, help="Output path for affine transformation"
    )
    register_parser.add_argument(
        "--warpfield", required=True, help="Output path for warp field"
    )
    register_parser.add_argument(
        "--inverse-warpfield", required=True, help="Output path for inverse warp field"
    )
    register_parser.add_argument(
        "--inverse-affine",
        required=True,
        help="Output path for inverse affine transformation",
    )
    register_parser.add_argument(
        "--registration-method", default="SyNRA", help="Registration method"
    )
    register_parser.add_argument(
        "--synthseg-threads",
        type=int,
        default=1,
        help="Number of threads to use for SynthSeg segmentation (default: 1)",
    )
    register_parser.add_argument(
        "--ants-threads",
        type=int,
        default=1,
        help="Number of threads to use for ANTs registration (default: 1)",
    )
    register_parser.add_argument(
        "--qc-csv", help="Path for quality control Dice score CSV file"
    )

    # WORKFLOW 2: Generate warpfield only
    warpfield_parser = subparsers.add_parser(
        "generate-warpfield", help="Generate registration warpfield without applying it"
    )
    warpfield_parser.add_argument("--moving", required=True, help="Input moving image")
    warpfield_parser.add_argument(
        "--fixed", required=True, help="Reference fixed image"
    )
    warpfield_parser.add_argument(
        "--moving-parc", required=True, help="Output path for moving image parcellation"
    )
    warpfield_parser.add_argument(
        "--fixed-parc", required=True, help="Output path for fixed image parcellation"
    )
    warpfield_parser.add_argument(
        "--registered-parc",
        required=True,
        help="Output path for registered parcellation",
    )
    warpfield_parser.add_argument(
        "--affine", required=True, help="Output path for affine transformation"
    )
    warpfield_parser.add_argument(
        "--warpfield", required=True, help="Output path for warp field"
    )
    warpfield_parser.add_argument(
        "--inverse-warpfield", required=True, help="Output path for inverse warp field"
    )
    warpfield_parser.add_argument(
        "--inverse-affine",
        required=True,
        help="Output path for inverse affine transformation",
    )
    warpfield_parser.add_argument(
        "--registration-method", default="SyNRA", help="Registration method"
    )
    warpfield_parser.add_argument(
        "--synthseg-threads",
        type=int,
        default=1,
        help="Number of threads to use for SynthSeg segmentation (default: 1)",
    )
    warpfield_parser.add_argument(
        "--ants-threads",
        type=int,
        default=1,
        help="Number of threads to use for ANTs registration (default: 1)",
    )
    warpfield_parser.add_argument(
        "--qc-csv", help="Path for quality control Dice score CSV file"
    )

    # WORKFLOW 3: Apply existing warpfield
    apply_parser = subparsers.add_parser(
        "apply-warpfield", help="Apply existing warpfield to an image"
    )
    apply_parser.add_argument(
        "--moving", required=True, help="Input image to transform"
    )
    apply_parser.add_argument("--fixed", required=True, help="Reference space image")
    apply_parser.add_argument("--output", required=True, help="Output registered image")
    apply_parser.add_argument("--warpfield", required=True, help="Path to warp field")
    apply_parser.add_argument(
        "--affine", required=True, help="Path to affine transformation"
    )
    apply_parser.add_argument(
        "--ants-threads",
        type=int,
        default=1,
        help="Number of threads to use for ANTs transformation (default: 1)",
    )

    # DIRECT TOOL ACCESS: SynthSeg
    synthseg_parser = subparsers.add_parser(
        "synthseg", help="Run SynthSeg brain MRI segmentation directly"
    )
    synthseg_parser.add_argument("--i", required=True, help="Input image")
    synthseg_parser.add_argument("--o", required=True, help="Output segmentation")
    synthseg_parser.add_argument(
        "--parc", action="store_true", help="Output parcellation"
    )
    synthseg_parser.add_argument("--cpu", action="store_true", help="Use CPU")
    synthseg_parser.add_argument(
        "--threads", type=int, default=1, help="Number of threads"
    )
    # Add other SynthSeg arguments as needed

    # DIRECT TOOL ACCESS: Coregister
    coregister_parser = subparsers.add_parser(
        "coregister", help="Run coregistration directly"
    )

    # DIRECT TOOL ACCESS: Apply Warp
    apply_warp_parser = subparsers.add_parser(
        "apply-warp", help="Apply transformation to an image directly"
    )

    # Add the dice-compare parser to the subparsers
    dice_compare_parser = subparsers.add_parser(
        "dice-compare",
        help="Calculate Dice similarity coefficient between two parcellation images",
    )
    dice_compare_parser.add_argument(
        "--ref", help="Path to reference parcellation image"
    )
    dice_compare_parser.add_argument(
        "--reg", help="Path to registered parcellation image"
    )
    dice_compare_parser.add_argument("--out", help="Output CSV file path")

    # Parse known args, leaving the rest for the subcommands
    args, unknown_args = parser.parse_known_args()

    # Print comprehensive help if no command provided
    if args.command is None:
        print_cli_help()
        sys.exit(0)
    print(f"Command: {args.command}")
    # Handle command routing
    if args.command == "register":
        lamareg(
            input_image=args.moving,
            reference_image=args.fixed,
            output_image=args.output,
            input_parc=args.moving_parc,
            reference_parc=args.fixed_parc,
            output_parc=args.registered_parc,
            affine_file=args.affine,
            warp_file=args.warpfield,
            inverse_warp_file=args.inverse_warpfield,
            inverse_affine_file=args.inverse_affine,
            registration_method=args.registration_method,
            synthseg_threads=args.synthseg_threads,
            ants_threads=args.ants_threads,
            skip_segment_moving=args.skip_segment_moving,
            skip_segment_fixed=args.skip_segment_fixed,
        )
    elif args.command == "generate-warpfield":
        lamareg(
            input_image=args.moving,
            reference_image=args.fixed,
            output_image=None,  # No output image for generate-warpfield
            input_parc=args.moving_parc,
            reference_parc=args.fixed_parc,
            output_parc=args.registered_parc,
            affine_file=args.affine,
            warp_file=args.warpfield,
            inverse_warp_file=args.inverse_warpfield,
            inverse_affine_file=args.inverse_affine,
            generate_warpfield=True,
            registration_method=args.registration_method,
            synthseg_threads=args.synthseg_threads,
            ants_threads=args.ants_threads,
            skip_segment_moving=args.skip_segment_moving,
            skip_segment_fixed=args.skip_segment_fixed,
        )
    elif args.command == "apply-warpfield":
        lamareg(
            input_image=args.moving,
            reference_image=args.fixed,
            output_image=args.output,
            apply_warpfield=True,
            affine_file=args.affine,
            warp_file=args.warpfield,
            ants_threads=args.ants_threads,
            synthseg_threads=1,  # Not used in this workflow but needed for the function
        )
    elif args.command == "synthseg":
        # Create a clean dictionary with the args provided by the parser
        synthseg_args = {}

        # Add explicit arguments from argparse
        if hasattr(args, "i") and args.i:
            synthseg_args["i"] = args.i
        if hasattr(args, "o") and args.o:
            synthseg_args["o"] = args.o

        # Add flag arguments
        for flag in ["parc", "cpu"]:
            if flag in unknown_args or f"--{flag}" in unknown_args:
                synthseg_args[flag] = True

        # Parse remaining arguments from command line
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i].lstrip("-")
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("-"):
                synthseg_args[arg] = unknown_args[i + 1]
                i += 2
            else:
                # It's a flag
                synthseg_args[arg] = True
                i += 1

        # Set ALL required defaults for SynthSeg
        synthseg_args.setdefault("parc", True)
        synthseg_args.setdefault("cpu", True)
        synthseg_args.setdefault("robust", True)
        synthseg_args.setdefault("v1", False)
        synthseg_args.setdefault("fast", False)
        synthseg_args.setdefault("post", None)
        synthseg_args.setdefault("resample", None)
        synthseg_args.setdefault("ct", None)
        synthseg_args.setdefault("vol", None)
        synthseg_args.setdefault("qc", None)
        synthseg_args.setdefault("device", None)
        synthseg_args.setdefault("crop", None)

        if hasattr(args, "threads") and args.threads:
            synthseg_args["threads"] = str(args.threads)
        else:
            synthseg_args["threads"] = "1"

        try:
            synthseg.main(synthseg_args)
        except Exception as e:
            print(f"SynthSeg error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "coregister":
        # If no additional arguments are provided, print help
        if not unknown_args:
            coregister.print_help()
            sys.exit(0)
        # Forward arguments to coregister
        sys.argv = [sys.argv[0]] + unknown_args
        coregister.main()
    elif args.command == "apply-warp":
        # If no additional arguments are provided, print help
        if not unknown_args:
            apply_warp.print_help()
            sys.exit(0)
        # Forward arguments to apply_warp
        sys.argv = [sys.argv[0]] + unknown_args
        apply_warp.main()
    elif args.command == "dice-compare":
        from lamar.scripts.dice_compare import compare_parcellations_dice, print_help

        print("Dice compare")
        if not hasattr(args, "ref") or not args.ref:
            print_help()
            sys.exit(0)

        compare_parcellations_dice(args.ref, args.reg, args.out)
    elif args.command is None:
        parser.print_help()
        sys.exit(0)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
