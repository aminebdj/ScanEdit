"""
ScanEdit Main Pipeline

Orchestrates the complete scene editing workflow with support for
iterative editing through multiple text prompts.

Pipeline stages:
1. Scene Handler - Load and parse scene (once)
2. Planner - Determine editing operations (per prompt)
3. Initializer - Initialize object transformations (per prompt)
4. Optimizer - Refine object placements (per prompt)
"""

import os
from pathlib import Path
from models.components.scene_handler import SCENE_HANDLER
from models.agents.planner import Planner
from models.agents.hierarchical_scene_initilizer import Initializer
from models.agents.subgraph_identifier import Subgraph_Identifier
from models.components.optimization.optimizer import SceneOptimizer


class SCANEDIT:
    """
    Main ScanEdit pipeline orchestrator with iterative editing support.
    
    Allows multiple editing operations to be performed sequentially on the
    same scene. Scene handler is initialized once, then multiple text prompts
    can be processed iteratively.
    
    Attributes:
        scene_handler (SCENE_HANDLER): Scene data and object manager
        root_path (str): Base directory for saving results
        edit_history (list): History of all edits applied
        current_edit_number (int): Counter for edit operations
    """
    
    def __init__(self,
                 path_to_ply,
                 data_type,
                 class_agnostic_masks,
                 class_names,
                 scene_name=None,
                 text_prompt=None,
                 vlm=None,
                 llm=None,
                 dataset=None,
                 path_to_annotations=None,
                 use_preds=True,
                 folder_name='ours_llm_backup',
                 load_prev=False,
                 run_optimizer=True,
                 auto_run=True,
                 output_root='./outputs',  # Add root output directory
                 save_video=False,  # Add missing parameter
                 number_of_samples=5):  # Add missing parameter
        """
        Initialize ScanEdit pipeline.
        
        Args:
            path_to_ply (str): Path to scene PLY file
            data_type (str): Type of input data ('mesh' or 'pointcloud')
            class_agnostic_masks: Segmentation masks for objects
            class_names (list): Names of object classes
            scene_name (str, optional): Name identifier for scene
            text_prompt (str, optional): Initial editing instruction. If provided
                                        and auto_run=True, runs immediately
            vlm: Vision-Language Model instance
            dataset: Dataset object if applicable
            path_to_annotations (str, optional): Path to annotation files
            use_preds (bool): Whether to use predictions. Default: True
            folder_name (str): Output folder name. Default: 'ours_llm_backup'
            load_prev (bool): Load previous results if available. Default: False
            run_optimizer (bool): Run optimization after initialization. Default: True
            auto_run (bool): If True and text_prompt provided, run immediately. 
                           If False, use .edit() method. Default: True
            output_root (str): Root directory for all outputs. Default: './outputs'
            save_video (bool): Whether to save video. Default: False
            number_of_samples (int): Number of samples. Default: 5
        """
        # Setup LLM/VLM

        self.llm = llm
        self.vlm = vlm
        # Setup structured output directory
        self.output_root = Path(output_root)
        self.scene_name = scene_name or 'unknown_scene'
        self.scene_output_dir = self.output_root / folder_name / self.scene_name 
        

        self.scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store configuration
        self.config = {
            'path_to_ply': path_to_ply,
            'data_type': data_type,
            'class_agnostic_masks': class_agnostic_masks,
            'class_names': class_names,
            'scene_name': scene_name,
            'vlm': vlm,
            'dataset': dataset,
            'path_to_annotations': path_to_annotations,
            'use_preds': use_preds,
            'folder_name': folder_name,
            'save_video': save_video,
            'number_of_samples': number_of_samples,
            'output_root': str(self.output_root),
            'scene_output_dir': str(self.scene_output_dir),
        }
        
        self.load_prev = load_prev
        self.run_optimizer = run_optimizer
        self.save_video_flag = save_video
        self.number_of_samples = number_of_samples
        
        # Initialize tracking
        self.edit_history = []
        self.current_edit_number = 0
        
        # Step 1: Initialize Scene Handler (only once)
        print("=" * 60)
        print("SCANEDIT INITIALIZATION")
        print("=" * 60)
        print(f"Output Root: {self.output_root}")
        print(f"Scene Output: {self.scene_output_dir}")
        print("=" * 60)
        print("STEP 1: Initializing Scene Handler")
        print("=" * 60)
        
        # Use structured folder path for scene handler
        scene_handler_folder = str(self.scene_output_dir)
        
        self.scene_handler = SCENE_HANDLER(
            path_2_mesh=path_to_ply,  # Corrected parameter name
            datatype=data_type,  # Corrected parameter name (no underscore)
            class_agnostic_masks=class_agnostic_masks,
            class_names=class_names,
            scene_name=scene_name,
            text_prompt=text_prompt,
            vlm=vlm,
            dataset=dataset,
            path_to_annotations=path_to_annotations,
            save_video=save_video,
            use_preds=use_preds,
            number_of_samples=number_of_samples,
            folder_name=scene_handler_folder,  # Use structured path
            output_dir=self.scene_output_dir
        )
        
        # # Set base paths
        # self.base_root_path = str(self.output_paths['edits'])
        # self.root_path = self.base_root_path
        
        # print(f"Base results path: {self.base_root_path}")
        
        # Log initialization
        # self._log_initialization()
        
        print("\nScene handler ready. Use .edit(text_prompt) to apply edits.")
        
        print("\n" + "=" * 60)
        print("SCANEDIT READY")
        print("=" * 60)
    
    def _log_initialization(self):
        """Log initialization details to file."""
        log_file = self.output_paths['logs'] / 'initialization.log'
        with open(log_file, 'w') as f:
            f.write("ScanEdit Initialization Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Scene Name: {self.config['scene_name']}\n")
            f.write(f"Data Type: {self.config['data_type']}\n")
            f.write(f"PLY Path: {self.config['path_to_ply']}\n")
            f.write(f"Use Predictions: {self.config['use_preds']}\n")
            f.write(f"Load Previous: {self.load_prev}\n")
            f.write(f"Run Optimizer: {self.run_optimizer}\n")
            f.write(f"Save Video: {self.save_video_flag}\n")
            f.write(f"Number of Samples: {self.number_of_samples}\n")
            f.write("\nOutput Directories:\n")
            for name, path in self.output_paths.items():
                f.write(f"  {name}: {path}\n")
    
    def get_output_path(self, file_type, filename=None, edit_number=None):
        """
        Get the appropriate output path for a given file type.
        
        Args:
            file_type (str): Type of file ('video', 'mesh', 'log', etc.)
            filename (str, optional): Specific filename
            edit_number (int, optional): Edit iteration number
        
        Returns:
            Path: Full path for the output file
        """
        if edit_number is None:
            edit_number = self.current_edit_number
        
        # Map file types to directories
        type_to_dir = {
            'video': 'videos',
            'mesh': 'meshes',
            'point_cloud': 'point_clouds',
            'visualization': 'visualizations',
            'log': 'logs',
            'checkpoint': 'checkpoints',
            'llm': 'llm_outputs',
            'intermediate': 'intermediate',
            'optimized': 'optimized',
            'planner': 'planner',
            'initializer': 'initializer',
            'subgraph_idenfication': 'subgraph_idenfication',
            'edit': 'edits'
        }
        
        dir_name = type_to_dir.get(file_type, 'intermediate')
        base_dir = self.output_paths[dir_name]
        
        if filename:
            # Add edit number to filename if applicable
            if edit_number > 0 and file_type != 'edit':
                name, ext = os.path.splitext(filename)
                filename = f"{name}_edit{edit_number:03d}{ext}"
            return base_dir / filename
        else:
            return base_dir
    
    def edit(self, text_prompt, load_prev=None, run_optimizer=None):
        """
        Apply an editing operation to the scene.
        
        Can be called multiple times to perform sequential edits on the same scene.
        Each edit is saved to a separate subdirectory.
        
        Args:
            text_prompt (str): Editing instruction (e.g., "Move chair closer to table")
            load_prev (bool, optional): Load previous results. If None, uses instance default
            run_optimizer (bool, optional): Run optimization. If None, uses instance default
        
        Returns:
            dict: Edit result containing:
                - hierarchy_dictionary: Editing plan
                - editing_type: Type of operation
                - root_path: Path to edit results
                - success: Whether edit completed successfully
        """
        # Use instance defaults if not specified
        out_dir = self.scene_output_dir / text_prompt
        self.output_paths = {
            'edits': out_dir / 'edits',
            'videos': out_dir / 'videos',
            'optimized': out_dir / 'optimized',
            'meshes': out_dir / 'meshes',
            'point_clouds': out_dir / 'point_clouds',
            'visualizations': out_dir / 'visualizations',
            'logs': out_dir / 'logs',
            'checkpoints': out_dir / 'checkpoints',
            'llm_outputs': out_dir / 'llm_outputs',
            'intermediate': out_dir / 'intermediate',
            'planner': out_dir / 'planner',
            'subgraph_idenfication': out_dir / 'subgraph_idenfication',
            'initializer': out_dir / 'initializer',
        }
        # Create all directories
        for dir_path in self.output_paths.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        if load_prev is None:
            load_prev = self.load_prev
        if run_optimizer is None:
            run_optimizer = self.run_optimizer
        
        # Increment edit counter
        self.current_edit_number += 1
        edit_num = self.current_edit_number
        
        # Create subdirectory for this edit in structured location
        
        edit_dir = self.output_paths['edits'] / f"edit_{edit_num:03d}"
        edit_dir.mkdir(parents=True, exist_ok=True)

        self.scene_handler.save_in_out_steps = edit_dir / 'opt_steps'
        self.scene_handler.save_in_out_steps.mkdir(parents=True, exist_ok=True)
        self.root_path = str(edit_dir)
        
        print("\n" + "=" * 60)
        print(f"EDIT #{edit_num}: {text_prompt}")
        print("=" * 60)
        print(f"Results path: {self.root_path}")
        
        try:
            # Step 2: Run Subgraph identification
            print("\n" + "-" * 60)
            print("Subgraph Identification")
            print("-" * 60)
            subgraph_idenfication_path = str(self.output_paths['subgraph_idenfication'] / f"edit_{edit_num:03d}")
            os.makedirs(subgraph_idenfication_path, exist_ok=True)
            relevent_objects, relevent_relations = Subgraph_Identifier(self.scene_handler,
                text_prompt,
                subgraph_idenfication_path,
                load_prev=load_prev,
                llm = self.llm            )
            # Step 2: Run Planner
            print("\n" + "-" * 60)
            print("Planning Editing Operations")
            print("-" * 60)
            
            # Use planner output directory
            planner_path = str(self.output_paths['planner'] / f"edit_{edit_num:03d}")
            os.makedirs(planner_path, exist_ok=True)
            
            hierarchy_dictionary, editing_type = Planner(
                self.scene_handler,
                text_prompt,
                planner_path,
                load_prev=load_prev,
                relevent_objects = relevent_objects,
                relevent_relations = relevent_relations,
                llm = self.llm

            )
            
            print(f"Editing type: {editing_type}")
            
            # Handle removal operations
            if editing_type == 'remove':
                print("\n" + "-" * 60)
                print("REMOVAL OPERATION DETECTED")
                print("-" * 60)
                self.scene_handler.remove = hierarchy_dictionary
                print("Objects marked for removal. Skipping initialization and optimization.")
                
                result = {
                    'edit_number': edit_num,
                    'text_prompt': text_prompt,
                    'hierarchy_dictionary': hierarchy_dictionary,
                    'editing_type': editing_type,
                    'root_path': self.root_path,
                    'planner_path': planner_path,
                    'success': True
                }
                self.edit_history.append(result)
                self._save_edit_log(result)
                return result
            
            # Step 3: Initialize Object Transformations
            print("\n" + "-" * 60)
            print("Initializing Object Transformations")
            print("-" * 60)
            
            # Use initializer output directory
            initializer_path = str(self.output_paths['initializer'] / f"edit_{edit_num:03d}")
            os.makedirs(initializer_path, exist_ok=True)
            
            Initializer(
                self.scene_handler,
                hierarchy_dictionary,
                text_prompt,
                initializer_path,
                llm = self.llm

            )
            
            # Step 4: Optimize (optional)
            optimizer_path = None
            if run_optimizer:
                print("\n" + "-" * 60)
                print("Optimizing Object Placements")
                print("-" * 60)
                
                optimizer_path = str(self.output_paths['optimized'] / f"edit_{edit_num:03d}")
                os.makedirs(optimizer_path, exist_ok=True)
                
                optimizer = SceneOptimizer(
                    handler=self.scene_handler,
                )
                optimizer.optimize_scene()
                print("Optimization complete!")
                
                # Save optimized mesh
                if hasattr(self.scene_handler, 'save_scene'):
                    optimized_mesh_path = self.get_output_path('mesh', f'optimized_scene_edit{edit_num:03d}.ply')
                    self.scene_handler.save_scene(str(optimized_mesh_path))
            else:
                print("\n" + "-" * 60)
                print("Skipping Optimization")
                print("-" * 60)
            
            # Save current scene state
            scene_path = self.get_output_path('mesh', f'scene_edit{edit_num:03d}.ply')
            if hasattr(self.scene_handler, 'save_scene'):
                self.scene_handler.save_scene(str(scene_path))
            
            # Save edit result
            result = {
                'edit_number': edit_num,
                'text_prompt': text_prompt,
                'hierarchy_dictionary': hierarchy_dictionary,
                'editing_type': editing_type,
                'root_path': self.root_path,
                'planner_path': planner_path,
                'initializer_path': initializer_path,
                'optimizer_path': optimizer_path,
                'scene_path': str(scene_path),
                'success': True
            }
            # self.edit_history.append(result)
            # self._save_edit_log(result)
            
            print("\n" + "=" * 60)
            print(f"EDIT #{edit_num} COMPLETE")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            print(f"\n❌ ERROR during edit #{edit_num}: {e}")
            result = {
                'edit_number': edit_num,
                'text_prompt': text_prompt,
                'error': str(e),
                'success': False
            }
            self.edit_history.append(result)
            self._save_edit_log(result)
            raise
    
    def _save_edit_log(self, result):
        """Save edit result to log file."""
        log_file = self.output_paths['logs'] / f"edit_{result['edit_number']:03d}.log"
        with open(log_file, 'w') as f:
            f.write(f"Edit #{result['edit_number']} Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Text Prompt: {result.get('text_prompt', 'N/A')}\n")
            f.write(f"Success: {result.get('success', False)}\n")
            if 'editing_type' in result:
                f.write(f"Editing Type: {result['editing_type']}\n")
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            f.write("\nPaths:\n")
            for key in ['root_path', 'planner_path', 'initializer_path', 'optimizer_path', 'scene_path']:
                if key in result:
                    f.write(f"  {key}: {result[key]}\n")
    
    def batch_edit(self, text_prompts, run_optimizer=None):
        """
        Apply multiple edits sequentially.
        
        Args:
            text_prompts (list of str): List of editing instructions
            run_optimizer (bool, optional): Run optimization for each edit
        
        Returns:
            list of dict: Results for each edit
        """
        results = []
        for i, prompt in enumerate(text_prompts, 1):
            print(f"\n{'#' * 60}")
            print(f"BATCH EDIT {i}/{len(text_prompts)}")
            print(f"{'#' * 60}")
            result = self.edit(prompt, run_optimizer=run_optimizer)
            results.append(result)
        
        # Save batch summary
        summary_file = self.output_paths['logs'] / 'batch_summary.log'
        with open(summary_file, 'w') as f:
            f.write("Batch Edit Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total Edits: {len(results)}\n")
            f.write(f"Successful: {sum(1 for r in results if r.get('success', False))}\n")
            f.write("\nEdits:\n")
            for r in results:
                f.write(f"  Edit #{r['edit_number']}: {r.get('text_prompt', 'N/A')} - ")
                f.write(f"{'✓' if r.get('success', False) else '✗'}\n")
        
        return results
    
    def get_scene_handler(self):
        """Get the scene handler instance."""
        return self.scene_handler
    
    def get_results_path(self):
        """Get path to current results directory."""
        return self.root_path
    
    def get_base_path(self):
        """Get path to base results directory."""
        return str(self.scene_output_dir)
    
    def get_edit_history(self):
        """Get history of all edits applied."""
        return self.edit_history
    
    def save_current_state(self, output_path=None):
        """
        Save current scene state.
        
        Args:
            output_path (str, optional): Custom output path
        """
        if output_path is None:
            output_path = self.get_output_path('mesh', f'current_state_edit{self.current_edit_number:03d}.ply')
        
        print(f"\nSaving current state to: {output_path}")
        if hasattr(self.scene_handler, 'save_scene'):
            self.scene_handler.save_scene(str(output_path))
            print("State saved successfully!")
    
    def save_video(self, name="output"):
        """Save video with organized naming."""
        if self.save_video_flag:
            filename = f"{name}_edit{self.current_edit_number:03d}.mp4"
            output_path = self.get_output_path('video', filename)
            print(f"Saving video to: {output_path}")
            # Add actual video saving logic here
            return output_path
    
    def reset_scene(self):
        """Reset scene to initial state (reload from original file)."""
        print("\n" + "=" * 60)
        print("RESETTING SCENE TO INITIAL STATE")
        print("=" * 60)
        
        # Reinitialize scene handler with structured path
        scene_handler_folder = str(self.scene_output_dir)
        
        self.scene_handler = SCENE_HANDLER(
            path_2_mesh=self.config['path_to_ply'],
            datatype=self.config['data_type'],
            class_agnostic_masks=self.config['class_agnostic_masks'],
            class_names=self.config['class_names'],
            scene_name=self.config['scene_name'],
            text_prompt="",
            vlm=self.config['vlm'],
            dataset=self.config['dataset'],
            path_to_annotations=self.config['path_to_annotations'],
            save_video=self.save_video_flag,
            use_preds=self.config['use_preds'],
            number_of_samples=self.number_of_samples,
            folder_name=scene_handler_folder
        )
        
        # Reset tracking
        self.edit_history = []
        self.current_edit_number = 0
        self.root_path = self.base_root_path
        
        print("Scene reset complete!")


# Convenience function for backward compatibility
def create_scanedit(path_to_ply, data_type, class_agnostic_masks, class_names,
                    scene_name=None, text_prompt=None, vlm=None, dataset=None,
                    path_to_annotations=None, use_preds=True, 
                    folder_name='ours_llm_backup', load_prev=False, 
                    run_optimizer=True, output_root='./outputs',
                    save_video=False, number_of_samples=5):
    """
    Create SCANEDIT pipeline (backward compatible interface).
    
    Returns:
        SCANEDIT: Initialized pipeline instance
    """
    return SCANEDIT(
        path_to_ply=path_to_ply,
        data_type=data_type,
        class_agnostic_masks=class_agnostic_masks,
        class_names=class_names,
        scene_name=scene_name,
        text_prompt=text_prompt,
        vlm=vlm,
        dataset=dataset,
        path_to_annotations=path_to_annotations,
        use_preds=use_preds,
        folder_name=folder_name,
        load_prev=load_prev,
        run_optimizer=run_optimizer,
        auto_run=True,
        output_root=output_root,
        save_video=save_video,
        number_of_samples=number_of_samples
    )
