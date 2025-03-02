# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import shutil
import traceback
import torch
import gc
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from cog import BasePredictor, Input, Path as CogPath

# Configure environment variables to avoid HF hub errors
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Setting up environment...")
        
        # Create necessary directories
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("tmp", exist_ok=True)
        
        # Fix the huggingface_hub import issue
        try:
            import diffusers
            dynamic_modules_path = os.path.join(os.path.dirname(diffusers.__file__), 'utils', 'dynamic_modules_utils.py')
            if os.path.exists(dynamic_modules_path):
                with open(dynamic_modules_path, 'r') as f:
                    content = f.read()
                
                # Replace problematic import with fixed version
                if 'from huggingface_hub import cached_download' in content:
                    fixed_content = content.replace(
                        'from huggingface_hub import cached_download, hf_hub_download, model_info',
                        'from huggingface_hub import hf_hub_download, model_info\n# Fix for cached_download\ncached_download = hf_hub_download'
                    )
                    
                    with open(dynamic_modules_path, 'w') as f:
                        f.write(fixed_content)
                    print(f"Fixed huggingface_hub import issue")
        except Exception as e:
            print(f"Warning: Could not fix huggingface_hub import: {e}")
        
        # Add the current directory to the path
        sys.path.append(".")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configuration template
        self.sd_cfg_template = OmegaConf.load("controlnet/config/UV_gen_template.yaml")
        
        # Pre-load the model
        try:
            from controlnet.diffusers_cnet_txt2img import txt2imgControlNet
            print("Loading UV ControlNet model...")
            self.UV_cnet = txt2imgControlNet(self.sd_cfg_template.txt2img)
            print("ControlNet model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not preload model: {e}")
            print(f"Will attempt to load model during prediction")
            self.UV_cnet = None
        
        print("Setup complete")

    def predict(
        self,
        mesh_path: CogPath = Input(description="Input 3D mesh file (OBJ format)"),
        prompt: str = Input(description="Text prompt for texture generation", default="Realistic textures, detailed materials"),
        negative_prompt: str = Input(description="Negative text prompt", default="low quality, blurry, distorted"),
        seed: int = Input(description="Random seed for generation (use -1 for random)", default=42),
        texture_resolution: str = Input(description="Texture resolution (width,height)", default="1024,1024")
    ) -> CogPath:
        """Generate textures for a 3D mesh using Paint3D UV-only approach"""
        
        output_dir = f"outputs/generation_{seed}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Write diagnostic information
        with open(os.path.join(output_dir, "diagnostic.txt"), "w") as f:
            f.write(f"Mesh path: {mesh_path}\n")
            f.write(f"File exists: {os.path.exists(mesh_path)}\n")
            f.write(f"File size: {os.path.getsize(mesh_path)} bytes\n")
            
            # Copy mesh to temp location
            temp_mesh_path = "tmp/input_mesh.obj"
            shutil.copy2(mesh_path, temp_mesh_path)
            
            f.write(f"Copied to: {temp_mesh_path}\n")
            f.write(f"Copy exists: {os.path.exists(temp_mesh_path)}\n")
            f.write(f"Copy size: {os.path.getsize(temp_mesh_path)} bytes\n")
            
            # Add environment info for debugging
            f.write(f"Environment: {os.environ}\n")
            f.write(f"Current directory: {os.getcwd()}\n")
            f.write(f"Directory contents: {os.listdir()}\n")
            f.write(f"tmp directory contents: {os.listdir('tmp')}\n")
        
        try:
            # Parse texture resolution
            tex_res = [int(x) for x in texture_resolution.split(",")]
            if len(tex_res) != 2:
                raise ValueError("Texture resolution must be in format 'width,height'")
            
            # Load Paint3D modules
            from paint3d import utils
            from paint3d.models.textured_mesh import TexturedMeshModel
            from paint3d.dataset import init_dataloaders
            
            # Configure render settings
            render_config_path = "paint3d/config/train_config_paint3d.py"
            pathdir, filename = Path(render_config_path).parent, Path(render_config_path).stem
            sys.path.append(str(pathdir))
            render_cfg = __import__(filename).TrainConfig()
            
            # Update configuration
            render_cfg.guide.shape_path = temp_mesh_path
            render_cfg.guide.texture_resolution = tex_res
            render_cfg.log.exp_path = output_dir
            
            # Seed for reproducibility
            utils.seed_everything(seed)
            
            # Clean up memory before loading models
            torch.cuda.empty_cache()
            gc.collect()
            
            # Initialize mesh model
            mesh_model = TexturedMeshModel(cfg=render_cfg, device=self.device).to(self.device)
            dataloaders = init_dataloaders(render_cfg, self.device)
            
            # Generate UV position map
            UV_pos = mesh_model.UV_pos_render()
            UV_pos_path = os.path.join(output_dir, "UV_pos.png")
            utils.save_tensor_image(UV_pos.permute(0, 3, 1, 2), UV_pos_path)
            
            # Free memory after generating UV map
            del UV_pos
            torch.cuda.empty_cache()
            
            # Create config for this specific run
            sd_cfg = OmegaConf.create(self.sd_cfg_template)
            sd_cfg.txt2img.prompt = prompt
            sd_cfg.txt2img.negative_prompt = negative_prompt
            sd_cfg.txt2img.seed = seed
            sd_cfg.txt2img.controlnet_units[0].condition_image_path = UV_pos_path
            
            # Load model if not already loaded
            if self.UV_cnet is None:
                from controlnet.diffusers_cnet_txt2img import txt2imgControlNet
                self.UV_cnet = txt2imgControlNet(sd_cfg.txt2img)
            
            # Generate texture
            images = self.UV_cnet.infernece(config=sd_cfg.txt2img)
            
            # Save generated textures
            texture_path = os.path.join(output_dir, "UV_gen_res_0.png")
            images[0].save(texture_path)
            
            # Apply texture to mesh and export
            mesh_model.initial_texture_path = texture_path
            mesh_model.refresh_texture()
            mesh_model.export_mesh(output_dir)
            
            # Clean up resources
            del mesh_model
            del dataloaders
            torch.cuda.empty_cache()
            gc.collect()
            
            # Return the path to the textured mesh
            return CogPath(os.path.join(output_dir, "mesh.obj"))
            
        except Exception as e:
            error_path = os.path.join(output_dir, "error_details.txt")
            with open(error_path, 'w') as f:
                f.write(f"Error during texture generation: {str(e)}\n\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
            return CogPath(error_path)