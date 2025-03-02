# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import shutil
import traceback
import importlib.util
import re
import gc

# Suppress the transformers cache migration message
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 

# Patch the diffusers import issue by modifying the file
try:
    # Find the location of the diffusers package
    spec = importlib.util.find_spec('diffusers')
    if spec and spec.submodule_search_locations:
        diffusers_path = spec.submodule_search_locations[0]
        dynamic_modules_file = os.path.join(diffusers_path, 'utils', 'dynamic_modules_utils.py')
        
        if os.path.exists(dynamic_modules_file):
            with open(dynamic_modules_file, 'r') as f:
                content = f.read()
            
            # Check if the file contains the problematic import
            if 'from huggingface_hub import cached_download' in content:
                # Create a backup
                backup_file = dynamic_modules_file + '.backup'
                shutil.copy2(dynamic_modules_file, backup_file)
                
                # Replace the problematic import line
                fixed_content = re.sub(
                    r'from huggingface_hub import cached_download, hf_hub_download, model_info',
                    'from huggingface_hub import hf_hub_download, model_info',
                    content
                )
                
                # Add a workaround
                if 'cached_download = hf_hub_download' not in fixed_content:
                    fixed_content = fixed_content.replace(
                        'from huggingface_hub import hf_hub_download, model_info',
                        'from huggingface_hub import hf_hub_download, model_info\n# Compatibility fix\ncached_download = hf_hub_download'
                    )
                
                # Write the fixed content
                with open(dynamic_modules_file, 'w') as f:
                    f.write(fixed_content)
                
                print(f"Fixed the cached_download import issue in {dynamic_modules_file}")
            else:
                print(f"No problematic import found in {dynamic_modules_file}")
        else:
            print(f"Could not find dynamic_modules_utils.py at {dynamic_modules_file}")
    else:
        print("Could not locate diffusers package")
except Exception as e:
    print(f"Error while trying to fix import: {e}")

import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf
from cog import BasePredictor, Input, Path as CogPath

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to the path
sys.path.append(".")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        logger.info("Setting up environment...")
        
        # Create directories for outputs and temporary files
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("tmp", exist_ok=True)
        
        # Pre-import needed modules
        from controlnet.diffusers_cnet_txt2img import txt2imgControlNet
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load the UV controlnet model during setup
        logger.info("Preloading ControlNet model...")
        # Load SD config template
        sd_config_path = "controlnet/config/UV_gen_template.yaml"
        self.sd_cfg_template = OmegaConf.load(sd_config_path)
        
        # Create and cache the ControlNet model
        self.UV_cnet = txt2imgControlNet(self.sd_cfg_template.txt2img)
        logger.info("ControlNet model loaded")
        
        logger.info("Setup complete!")

    def predict(
        self,
        mesh_path: CogPath = Input(description="Input 3D mesh file (OBJ format)"),
        prompt: str = Input(description="Text prompt for texture generation", default="Realistic textures, detailed materials"),
        negative_prompt: str = Input(description="Negative text prompt", default="low quality, blurry, distorted"),
        seed: int = Input(description="Random seed for generation (use -1 for random)", default=42),
        texture_resolution: str = Input(description="Texture resolution (width,height)", default="1024,1024")
    ) -> CogPath:
        """Generate textures for a 3D mesh using Paint3D"""
        
        try:
            # Create output directory
            output_dir = f"outputs/generation_{seed}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Process the uploaded file
            mesh_path_str = str(mesh_path)
            logger.info(f"Original mesh path: {mesh_path_str}")
            
            # Create a proper mesh path with .obj extension
            proper_mesh_path = os.path.join("tmp", "input_mesh.obj")
            logger.info(f"Copying to: {proper_mesh_path}")
            
            # Copy the file with .obj extension
            shutil.copy2(mesh_path_str, proper_mesh_path)
            
            # Parse texture resolution
            try:
                tex_res = [int(x) for x in texture_resolution.split(",")]
                if len(tex_res) != 2:
                    raise ValueError("Texture resolution must be in format 'width,height'")
                logger.info(f"Using texture resolution: {tex_res}")
            except ValueError as e:
                logger.warning(f"Invalid texture resolution format: {e}, using default 1024x1024")
                tex_res = [1024, 1024]
            
            # Now run the actual pipeline
            try:
                # Import these here to avoid errors during setup
                from paint3d import utils
                from paint3d.models.textured_mesh import TexturedMeshModel
                from paint3d.dataset import init_dataloaders
                
                # Load render config
                render_config_path = "paint3d/config/train_config_paint3d.py"
                pathdir, filename = Path(render_config_path).parent, Path(render_config_path).stem
                sys.path.append(str(pathdir))
                render_cfg = __import__(filename, ).TrainConfig()
                
                # Update render config
                render_cfg.guide.shape_path = proper_mesh_path
                render_cfg.guide.texture_resolution = tex_res
                render_cfg.log.exp_path = output_dir
                
                # Initialize mesh model - this is specific to each input mesh
                logger.info("Initializing mesh model...")
                
                # Memory cleanup before loading the model
                torch.cuda.empty_cache()
                gc.collect()
                
                mesh_model = TexturedMeshModel(cfg=render_cfg, device=self.device).to(self.device)
                dataloaders = init_dataloaders(render_cfg, self.device)
                
                # Generate UV position map
                logger.info("Rendering texture and position map")
                UV_pos = mesh_model.UV_pos_render()
                UV_pos_path = os.path.join(output_dir, "UV_pos.png")
                utils.save_tensor_image(UV_pos.permute(0, 3, 1, 2), UV_pos_path)
                
                # Memory cleanup after UV map generation
                del UV_pos
                torch.cuda.empty_cache()
                gc.collect()
                
                # Create a new config for this specific run
                sd_cfg = OmegaConf.create(self.sd_cfg_template)
                
                # Update the SD config with user inputs
                sd_cfg.txt2img.prompt = prompt
                sd_cfg.txt2img.negative_prompt = negative_prompt
                sd_cfg.txt2img.seed = seed
                
                # Update ControlNet config with UV position map
                sd_cfg.txt2img.controlnet_units[0].condition_image_path = UV_pos_path
                
                # Generate texture using the preloaded ControlNet
                logger.info("Generating texture with ControlNet")
                images = self.UV_cnet.infernece(config=sd_cfg.txt2img)
                
                # Memory cleanup after texture generation
                torch.cuda.empty_cache()
                gc.collect()
                
                # Save the result
                for i, img in enumerate(images):
                    save_path = os.path.join(output_dir, f'UV_gen_res_{i}.png')
                    img.save(save_path)
                
                # Apply the texture and export the mesh
                logger.info("Applying texture to mesh and exporting...")
                mesh_model.initial_texture_path = os.path.join(output_dir, "UV_gen_res_0.png")
                mesh_model.refresh_texture()
                mesh_model.export_mesh(output_dir)
                
                # Memory cleanup after export
                del mesh_model
                del dataloaders
                torch.cuda.empty_cache()
                gc.collect()
                
                # Return the path to the mesh file
                result_path = os.path.join(output_dir, "mesh.obj")
                logger.info(f"Complete! Result saved to {result_path}")
                return CogPath(result_path)
                
            except Exception as e:
                logger.exception(f"Error during texture generation: {e}")
                
                # Create a detailed error report
                error_path = os.path.join(output_dir, "error_details.txt")
                with open(error_path, 'w') as f:
                    f.write(f"Error during texture generation: {str(e)}\n\n")
                    f.write(f"Traceback:\n{traceback.format_exc()}")
                
                # Return the error report
                return CogPath(error_path)
            
        except Exception as e:
            # Create error report for any other exceptions
            error_file = os.path.join("outputs", "error_report.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error: {str(e)}\n\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
            
            logger.exception("Error during processing")
            return CogPath(error_file)