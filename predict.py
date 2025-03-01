from cog import BasePredictor, Input, Path
import os
import sys
import torch
import tempfile
from pathlib import Path as PathLib
from PIL import Image
from omegaconf import OmegaConf

class Predictor(BasePredictor):
    def setup(self):
        """Load the basic configuration"""
        # Don't load models here to avoid memory issues during type signature analysis
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create default config paths
        self.default_sd_config = "controlnet/config/depth_based_inpaint_template.yaml"
        self.default_render_config = "paint3d/config/train_config_paint3d.py"
        
        # Add path for imports
        sys.path.append(".")
    
    def predict(
        self,
        mesh: Path = Input(description="3D mesh file (.obj)"),
        prompt: str = Input(description="Text prompt to guide texture generation", default="Sci-Fi digital painting, colorful, high quality"),
        image_prompt: Path = Input(description="Optional image prompt for IP-Adapter conditioning", default=None),
        pipeline_type: str = Input(
            description="Which pipeline to use", 
            choices=["stage1", "UV_only"],
            default="stage1"
        ),
        seed: int = Input(description="Random seed", default=42),
    ) -> list:
        """Run texture generation on the provided 3D mesh"""
        
        # Import needed modules
        from controlnet.diffusers_cnet_txt2img import txt2imgControlNet
        from paint3d import utils
        from paint3d.models.textured_mesh import TexturedMeshModel
        from paint3d.dataset import init_dataloaders
        
        # Set seed for reproducibility
        utils.seed_everything(seed)
        
        # Create temporary directory for results
        temp_dir = PathLib(tempfile.mkdtemp())
        results = []
        
        try:
            # Load configs
            sd_cfg = OmegaConf.load(self.default_sd_config)
            
            # Import render config module
            import importlib.util
            spec = importlib.util.spec_from_file_location("render_config", self.default_render_config)
            render_config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(render_config_module)
            
            render_cfg = render_config_module.TrainConfig()
            
            # Update configs with user inputs
            render_cfg.guide.shape_path = str(mesh)
            if prompt:
                sd_cfg.txt2img.prompt = prompt
            if image_prompt:
                sd_cfg.txt2img.ip_adapter_image_path = str(image_prompt)
            
            # Set output directory
            outdir = temp_dir / "output"
            outdir.mkdir(exist_ok=True)
            render_cfg.log.exp_path = str(outdir)
            
            if pipeline_type == "UV_only":
                # Simplified UV_only pipeline
                mesh_model = TexturedMeshModel(cfg=render_cfg, device=self.device).to(self.device)
                UV_cnet = txt2imgControlNet(sd_cfg.txt2img)
                
                from pipeline_UV_only import UV_gen
                
                print("Running UV position-based texture generation...")
                uv_images = UV_gen(
                    sd_cfg=sd_cfg,
                    cnet=UV_cnet,
                    mesh_model=mesh_model,
                    outdir=outdir,
                )
                
                # Convert images to PIL and add to results
                for img in uv_images:
                    results.append(img)
                
            else:  # stage1 by default
                # Create models
                dataloaders = init_dataloaders(render_cfg, self.device)
                mesh_model = TexturedMeshModel(cfg=render_cfg, device=self.device).to(self.device)
                depth_cnet = txt2imgControlNet(sd_cfg.txt2img)
                
                from pipeline_paint3d_stage1 import gen_init_view
                
                # Initial view generation
                print("Generating initial views...")
                init_images = gen_init_view(
                    sd_cfg=sd_cfg,
                    cnet=depth_cnet,
                    mesh_model=mesh_model,
                    dataloaders=dataloaders,
                    outdir=outdir,
                    view_ids=render_cfg.render.views_init,
                )
                
                # Convert images to PIL and add to results
                for img in init_images:
                    results.append(img)
            
            return results
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Create a simple error image
            error_img = Image.new('RGB', (512, 512), color=(255, 0, 0))
            return [error_img]