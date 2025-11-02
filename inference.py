#!/usr/bin/env python3
"""
SCANEDIT Inference Script
This script performs scene editing using the SCANEDIT model with configurable parameters.
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from PIL import Image

from models.scanedit import SCANEDIT
from models.LLMs.LLM import LLM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run SCANEDIT inference for 3D scene editing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--scene_name',
        type=str,
        required=True,
        help='Name of the scene to edit (e.g., 3e8bba0176, 1ada7a0617)'
    )
    
    parser.add_argument(
        '--text_prompt',
        type=str,
        required=True,
        help='Text prompt describing the desired edit'
    )
    
    # Path arguments
    parser.add_argument(
        '--path_to_ply',
        type=str,
        required=True,
        help='Path to the input .ply file'
    )
    
    parser.add_argument(
        '--masks_classes_path',
        type=str,
        required=True,
        help='Path to the masks and classes .pt file'
    )
    
    # Optional path arguments
    parser.add_argument(
        '--path_to_rgb_data',
        type=str,
        default=None,
        help='Path to RGB image data directory'
    )
    
    parser.add_argument(
        '--path_to_2d_masks',
        type=str,
        default=None,
        help='Path to 2D masks .pt file'
    )
    
    parser.add_argument(
        '--path_to_annotations',
        type=str,
        default=None,
        help='Path to VLM annotations .yaml file'
    )
    
    parser.add_argument(
        '--path_to_save_additional_masks',
        type=str,
        default=None,
        help='Path to save additional background object masks'
    )
    
    parser.add_argument(
        '--img_sample_path',
        type=str,
        default=None,
        help='Path to a sample image for testing'
    )
    
    # Model configuration
    parser.add_argument(
        '--groq_api_key',
        type=str,
        default=None,
        help='GROQ API key for LLM. If not provided, will look for GROQ_API_KEY environment variable'
    )
    
    parser.add_argument(
        '--llm_name',
        type=str,
        default='openai/gpt-oss-120b',
        help='Name of the LLM model to use'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='scannetpp',
        choices=['scannetpp', 'replica'],
        help='Dataset type'
    )
    
    # Flags
    parser.add_argument(
        '--use_preds',
        action='store_true',
        help='Use predictions (adds _mask3d_with_cc extension)'
    )
    
    parser.add_argument(
        '--mesh_type',
        type=str,
        default='mesh',
        choices=['mesh', 'pointcloud'],
        help='Type of 3D representation'
    )
    
    # Output configuration
    parser.add_argument(
        '--folder_name',
        type=str,
        default='inference_output',
        help='Output folder name for results'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Base output directory'
    )
    
    # Additional options
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def validate_paths(args):
    """Validate that required paths exist."""
    required_paths = {
        'path_to_ply': args.path_to_ply,
        'masks_classes_path': args.masks_classes_path,
    }
    
    for name, path in required_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")
    
    # Validate optional paths if provided
    optional_paths = {
        'path_to_rgb_data': args.path_to_rgb_data,
        'path_to_annotations': args.path_to_annotations,
        'img_sample_path': args.img_sample_path,
    }
    
    for name, path in optional_paths.items():
        if path and not os.path.exists(path):
            print(f"Warning: {name} does not exist: {path}")


def load_masks_and_classes(masks_classes_path, verbose=False):
    """Load class names and class-agnostic masks from .pt file."""
    if verbose:
        print(f"Loading masks and classes from: {masks_classes_path}")
    
    class_names, class_agnostic_masks = torch.load(masks_classes_path)
    
    if verbose:
        print(f"Loaded {len(class_names)} classes")
        print(f"Class names: {class_names}")
    
    return class_names, class_agnostic_masks


def initialize_llm(groq_api_key, llm_name, verbose=False):
    """Initialize the LLM model."""
    if not groq_api_key:
        groq_api_key = os.environ.get('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError(
                "GROQ API key not provided. Please provide it via --groq_api_key "
                "or set the GROQ_API_KEY environment variable."
            )
    
    if verbose:
        print(f"Initializing LLM: {llm_name}")
    
    llm = LLM(groq_api_key=groq_api_key, model_name=llm_name)
    return llm


def main():
    """Main inference function."""
    args = parse_args()
    
    if args.verbose:
        print("=" * 80)
        print("SCANEDIT Inference")
        print("=" * 80)
        print(f"Scene: {args.scene_name}")
        print(f"Text Prompt: {args.text_prompt}")
        print(f"Dataset: {args.dataset}")
        print(f"Device: {args.device}")
        print("=" * 80)
    
    # Validate paths
    try:
        validate_paths(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load masks and classes
    try:
        class_names, class_agnostic_masks = load_masks_and_classes(
            args.masks_classes_path, 
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error loading masks and classes: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize LLM
    try:
        llm = initialize_llm(args.groq_api_key, args.llm_name, verbose=args.verbose)
    except Exception as e:
        print(f"Error initializing LLM: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load sample image if provided
    img = None
    if args.img_sample_path:
        try:
            img = Image.open(args.img_sample_path)
            if args.verbose:
                print(f"Loaded sample image: {args.img_sample_path}")
        except Exception as e:
            print(f"Warning: Could not load sample image: {e}")
    
    # Initialize SCANEDIT
    if args.verbose:
        print("Initializing SCANEDIT model...")
    
    try:
        scanedit = SCANEDIT(
            path_to_ply=args.path_to_ply,
            mesh_type=args.mesh_type,
            class_agnostic_masks=class_agnostic_masks,
            class_names=class_names,
            scene_name=args.scene_name,
            text_prompt=args.text_prompt,
            llm=llm,
            dataset=args.dataset,
            path_to_annotations=args.path_to_annotations,
            use_preds=args.use_preds,
            folder_name=args.folder_name
        )
    except Exception as e:
        print(f"Error initializing SCANEDIT: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run editing
    if args.verbose:
        print(f"\nRunning scene editing with prompt: '{args.text_prompt}'")
        print("-" * 80)
    
    try:
        result = scanedit.edit(args.text_prompt)
        
        if args.verbose:
            print("-" * 80)
            print("Editing completed successfully!")
            print(f"Results saved to: {args.output_dir}/{args.folder_name}/{args.scene_name}")
        
        return result
        
    except Exception as e:
        print(f"Error during editing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()