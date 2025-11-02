# SCANEDIT Inference

## 📋 Overview

SCANEDIT is a 3D scene editing framework that uses language models to understand and execute scene manipulation tasks based on text prompts.

## ✅ Project Status

### Completed ✓
- [x] **Inference code released** 
- [x] **Installation scripts** 
- [x] **Documentation**

### TODO 🚧
- [ ] **Inference sample**
- [ ] **Evaluation setup**
- [ ] **Evaluation data**


## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/aminebdj/ScanEdit
cd ScanEdit

# Run the installation script
chmod +x install_environment.sh
./install_environment.sh

# Activate the environment
conda activate scanedit
```

### Set up API Key
```bash
export GROQ_API_KEY="your_api_key_here"
```

## 💻 Usage

### Complete Example

```bash
SCENE_NAME="3e8bba0176"

python inference.py \
    --scene_name "$SCENE_NAME" \
    --text_prompt "Create a seating area for a lecture on the whiteboard" \
    --path_to_ply "/data/scannetpp/plys/validation/${SCENE_NAME}.ply" \
    --masks_classes_path "/data/scannetpp/semantic_3d_masks/${SCENE_NAME}.pt" \
    --path_to_rgb_data "/data/scannetpp/data/${SCENE_NAME}/dslr/undistorted_images" \
    --path_to_2d_masks "/data/scannetpp/undistorted_rast_masks/${SCENE_NAME}.pt" \
    --path_to_annotations "/data/scannetpp/vlm_annotations/${SCENE_NAME}.yaml" \
    --dataset "scannetpp" \
    --llm_name "openai/gpt-oss-120b" \
    --folder_name "results" \
    --output_dir "./outputs" \
    --verbose
```

## 🔧 Arguments Reference

<details>
<summary>Click to expand full argument list</summary>

#### Paths
- `--path_to_rgb_data`: RGB images directory
- `--path_to_2d_masks`: 2D projection masks (.pt)
- `--path_to_annotations`: VLM annotations (.yaml)
- `--path_to_save_additional_masks`: Output path for masks
- `--img_sample_path`: Sample image for testing

#### Model Configuration
- `--groq_api_key`: GROQ API key (default: env var)
- `--llm_name`: LLM model name (default: "openai/gpt-oss-120b")
- `--dataset`: Dataset type (choices: scannetpp, replica)

#### Output Settings
- `--folder_name`: Output subdirectory name
- `--output_dir`: Base output directory
- `--device`: Computation device (cuda/cpu)

#### Flags
- `--use_preds`: Use predicted masks (adds extension)
- `--verbose`: Enable detailed logging

</details>

## 📁 Dataset Structure


```
project_root/
├── data/
│   └── scannetpp/
│       ├── data/
│       │   └── {scene_name}/
│       │       └── dslr/
│       │           └── undistorted_images/
│       │               ├── DSC00001.JPG
│       │               └── ...
│       ├── plys/
│       │   └── validation/
│       │       ├── 3e8bba0176.ply
│       │       └── ...
│       ├── undistorted_rast_masks/
│       │   └── {scene_name}.pt
│       │
│       ├── semantic_3d_masks/
│       │   └── {scene_name}.pt
│       │   
│       └── vlm_annotations/
│           └── {scene_name}.yaml
│
└── outputs/
    └── {folder_name}/
        └── {scene_name}/
            ├── final.ply
            └── ...
```


## 📚 Citation

```bibtex
@inproceedings{el2025scanedit,
  title={ScanEdit: Hierarchically-Guided Functional 3D Scan Editing},
  author={El Amine Boudjoghra, Mohamed and Laptev, Ivan and Dai, Angela},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={27105--27115},
  year={2025}
}
```

