# Prediction interface for Cog
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
import tempfile
import shutil

from cog import BasePredictor, Input, Path as CogPath

# Import Paint3D modules
sys.path.append(".")
from controlnet.diffusers_cnet_txt2img import txt2imgControlNet
from controlnet.diffusers_cnet_inpaint import inpaintControlNet
from controlnet.diffusers_cnet_img2img import img2imgControlNet
from paint3d.dataset import init_dataloaders
from paint3d import utils
from paint3d.models.textured_mesh import TexturedMeshModel
from paint3d.trainer import dr_eval


class Predictor(BasePredictor):
    def setup(self):
        """Load models into memory to make running multiple predictions efficient"""
        # This is just initializing, the actual models will be loaded per-request
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create default config paths
        self.default_sd_config = "controlnet/config/depth_based_inpaint_template.yaml"
        self.default_render_config = "paint3d/config/train_config_paint3d.py"
        self.default_sd_config_stage2 = "controlnet/config/UV_based_inpaint_template.yaml"
        self.default_uv_config = "controlnet/config/UV_gen_template.yaml"
        
        # Verify configs exist
        assert os.path.exists(self.default_sd_config), f"Config not found: {self.default_sd_config}"
        assert os.path.exists(self.default_render_config), f"Config not found: {self.default_render_config}"
        assert os.path.exists(self.default_sd_config_stage2), f"Config not found: {self.default_sd_config_stage2}"
        assert os.path.exists(self.default_uv_config), f"Config not found: {self.default_uv_config}"
    
    def init_process(self, sd_config_path, render_config_path):
        """Initialize configurations"""
        # Parse SD config
        sd_cfg = OmegaConf.load(sd_config_path)
        
        # Import render config module
        import importlib.util
        spec = importlib.util.spec_from_file_location("render_config", render_config_path)
        render_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(render_config_module)
        
        render_cfg = render_config_module.TrainConfig()
        
        return sd_cfg, render_cfg
    
    def update_configs(self, sd_cfg, render_cfg, mesh_path=None, texture_path=None, prompt=None, ip_image_path=None):
        """Update configurations with user inputs"""
        if mesh_path:
            render_cfg.guide.shape_path = str(mesh_path)
        
        if texture_path:
            render_cfg.guide.initial_texture = str(texture_path)
            
        if prompt is not None:
            # Update prompt in all relevant config sections
            if hasattr(sd_cfg, 'txt2img'):
                sd_cfg.txt2img.prompt = prompt
            if hasattr(sd_cfg, 'inpaint'):
                sd_cfg.inpaint.prompt = prompt
            if hasattr(sd_cfg, 'img2img'):
                sd_cfg.img2img.prompt = prompt
                
        if ip_image_path:
            # Update IP adapter image path in all relevant config sections
            if hasattr(sd_cfg, 'txt2img'):
                sd_cfg.txt2img.ip_adapter_image_path = str(ip_image_path)
            if hasattr(sd_cfg, 'inpaint'):
                sd_cfg.inpaint.ip_adapter_image_path = str(ip_image_path)
            if hasattr(sd_cfg, 'img2img'):
                sd_cfg.img2img.ip_adapter_image_path = str(ip_image_path)
        
        return sd_cfg, render_cfg
    
    def predict(
        self,
        mesh: CogPath = Input(description="3D mesh file (.obj)"),
        prompt: str = Input(description="Text prompt to guide texture generation", default=""),
        image_prompt: CogPath = Input(description="Optional image prompt for IP-Adapter conditioning", default=None),
        texture_path: CogPath = Input(description="Optional initial texture image (for stage 2 only)", default=None),
        pipeline_type: str = Input(
            description="Which pipeline to use", 
            choices=["stage1", "stage2", "UV_only", "full"],
            default="full"
        ),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        guidance_scale: float = Input(description="Classifier-free guidance scale", default=7.5, ge=0.0, le=20.0),
        seed: int = Input(description="Random seed", default=42),
    ) -> list:
        """Run texture generation on the provided 3D mesh"""
        
        # Set seed for reproducibility
        utils.seed_everything(seed)
        
        # Create temporary directory for results
        temp_dir = Path(tempfile.mkdtemp())
        results = []
        
        try:
            # Process based on the selected pipeline type
            if pipeline_type in ["stage1", "full"]:
                # Stage 1: Coarse texture generation
                stage1_out_dir = temp_dir / "stage1"
                stage1_out_dir.mkdir(exist_ok=True)
                
                # Load configs
                sd_cfg, render_cfg = self.init_process(self.default_sd_config, self.default_render_config)
                
                # Update configs with user inputs
                sd_cfg, render_cfg = self.update_configs(
                    sd_cfg, render_cfg,
                    mesh_path=mesh,
                    prompt=prompt,
                    ip_image_path=image_prompt
                )
                
                # Update guidance scale
                if hasattr(sd_cfg, 'txt2img'):
                    sd_cfg.txt2img.guidance_scale = guidance_scale
                if hasattr(sd_cfg, 'inpaint'):
                    sd_cfg.inpaint.guidance_scale = guidance_scale
                
                # Update negative prompt
                if hasattr(sd_cfg, 'txt2img'):
                    sd_cfg.txt2img.negative_prompt = negative_prompt
                if hasattr(sd_cfg, 'inpaint'):
                    sd_cfg.inpaint.negative_prompt = negative_prompt
                
                # Create models
                mesh_model = TexturedMeshModel(cfg=render_cfg, device=self.device).to(self.device)
                dataloaders = init_dataloaders(render_cfg, self.device)
                depth_cnet = txt2imgControlNet(sd_cfg.txt2img)
                inpaint_cnet = inpaintControlNet(sd_cfg.inpaint)
                
                # Run the stage1 pipeline
                from pipeline_paint3d_stage1 import gen_init_view, inpaint_viewpoint, forward_texturing
                
                # Initial view generation
                print("Generating initial views...")
                init_images = gen_init_view(
                    sd_cfg=sd_cfg,
                    cnet=depth_cnet,
                    mesh_model=mesh_model,
                    dataloaders=dataloaders,
                    outdir=stage1_out_dir,
                    view_ids=render_cfg.render.views_init,
                )
                
                # Process each generated image
                for i, init_image in enumerate(init_images):
                    view_outdir = stage1_out_dir / f"res-{i}"
                    view_outdir.mkdir(exist_ok=True)
                    
                    # Back-projection init view
                    print("Running back-projection for initial view...")
                    mesh_model.initial_texture_path = None
                    mesh_model.refresh_texture()
                    view_imgs = utils.split_grid_image(img=np.array(init_image), size=(1, 2))
                    forward_texturing(
                        cfg=render_cfg,
                        dataloaders=dataloaders,
                        mesh_model=mesh_model,
                        save_result_dir=view_outdir,
                        device=self.device,
                        view_imgs=view_imgs,
                        view_ids=render_cfg.render.views_init,
                        verbose=False,
                    )
                    
                    # View inpainting
                    print("Running view inpainting...")
                    for view_group in render_cfg.render.views_inpaint:
                        inpainted_images = inpaint_viewpoint(
                            sd_cfg=sd_cfg,
                            cnet=inpaint_cnet,
                            save_result_dir=view_outdir,
                            mesh_model=mesh_model,
                            dataloaders=dataloaders,
                            inpaint_view_ids=[view_group],
                        )
                        
                        # Process inpainted images
                        view_imgs = []
                        for img_t in inpainted_images:
                            view_imgs.extend(utils.split_grid_image(img=np.array(img_t), size=(1, 2)))
                        
                        forward_texturing(
                            cfg=render_cfg,
                            dataloaders=dataloaders,
                            mesh_model=mesh_model,
                            save_result_dir=view_outdir,
                            device=self.device,
                            view_imgs=view_imgs,
                            view_ids=view_group,
                            verbose=False,
                        )
                    
                    # Export textured mesh
                    mesh_model.export_mesh(view_outdir)
                    
                    # Add to results
                    stage1_texture = view_outdir / "albedo.png"
                    texture_image = Image.open(stage1_texture)
                    results.append(texture_image)
                    
                    # Save for potential stage2 use
                    stage1_result_path = stage1_texture
                
            if pipeline_type in ["stage2", "full"]:
                # Stage 2: Refined texture generation
                stage2_out_dir = temp_dir / "stage2"
                stage2_out_dir.mkdir(exist_ok=True)
                
                # Load configs
                sd_cfg, render_cfg = self.init_process(self.default_sd_config_stage2, self.default_render_config)
                
                # Update texture path from stage1 if doing full pipeline
                if pipeline_type == "full" and 'stage1_result_path' in locals():
                    texture_input = stage1_result_path
                else:
                    # Otherwise use user-provided texture
                    if texture_path is None:
                        raise ValueError("For stage2 pipeline, you must provide a texture_path")
                    texture_input = texture_path
                
                # Update configs with user inputs
                sd_cfg, render_cfg = self.update_configs(
                    sd_cfg, render_cfg,
                    mesh_path=mesh,
                    texture_path=texture_input,
                    prompt=prompt,
                    ip_image_path=image_prompt
                )
                
                # Update guidance scale
                if hasattr(sd_cfg, 'inpaint'):
                    sd_cfg.inpaint.guidance_scale = guidance_scale
                if hasattr(sd_cfg, 'img2img'):
                    sd_cfg.img2img.guidance_scale = guidance_scale
                
                # Update negative prompt
                if hasattr(sd_cfg, 'inpaint'):
                    sd_cfg.inpaint.negative_prompt = negative_prompt
                if hasattr(sd_cfg, 'img2img'):
                    sd_cfg.img2img.negative_prompt = negative_prompt
                
                # Create models
                mesh_model = TexturedMeshModel(cfg=render_cfg, device=self.device).to(self.device)
                dataloaders = init_dataloaders(render_cfg, self.device)
                UVInpaint_cnet = inpaintControlNet(sd_cfg.inpaint)
                UVtile_cnet = img2imgControlNet(sd_cfg.img2img)
                
                # Run the stage2 pipeline
                from pipeline_paint3d_stage2 import UV_inpaint, UV_tile
                
                # UV inpaint
                print("Running UV inpainting...")
                UV_inpaint_res = UV_inpaint(
                    sd_cfg=sd_cfg,
                    cnet=UVInpaint_cnet,
                    mesh_model=mesh_model,
                    outdir=stage2_out_dir,
                )
                
                # Process UV inpaint results
                for i, (_, init_img_path) in enumerate(UV_inpaint_res):
                    tile_outdir = stage2_out_dir / f"tile_res_{i}"
                    tile_outdir.mkdir(exist_ok=True)
                    
                    # UV tile
                    print("Running UV tiling...")
                    mesh_model.initial_texture_path = init_img_path
                    mesh_model.refresh_texture()
                    UV_tile_images = UV_tile(
                        sd_cfg=sd_cfg,
                        cnet=UVtile_cnet,
                        mesh_model=mesh_model,
                        outdir=tile_outdir,
                    )
                    
                    # Export final textured mesh
                    final_texture_path = tile_outdir / "UV_tile_res_0.png"
                    mesh_model.initial_texture_path = str(final_texture_path)
                    mesh_model.refresh_texture()
                    mesh_model.export_mesh(tile_outdir)
                    
                    # Add to results
                    tile_image = Image.open(final_texture_path)
                    results.append(tile_image)
            
            if pipeline_type == "UV_only":
                # UV-only pipeline (simpler, directly using UV position controlnet)
                uv_out_dir = temp_dir / "uv_only"
                uv_out_dir.mkdir(exist_ok=True)
                
                # Load configs
                sd_cfg, render_cfg = self.init_process(self.default_uv_config, self.default_render_config)
                
                # Update configs with user inputs
                sd_cfg, render_cfg = self.update_configs(
                    sd_cfg, render_cfg,
                    mesh_path=mesh,
                    prompt=prompt,
                    ip_image_path=image_prompt
                )
                
                # Update guidance scale and negative prompt
                if hasattr(sd_cfg, 'txt2img'):
                    sd_cfg.txt2img.guidance_scale = guidance_scale
                    sd_cfg.txt2img.negative_prompt = negative_prompt
                
                # Create models
                mesh_model = TexturedMeshModel(cfg=render_cfg, device=self.device).to(self.device)
                dataloaders = init_dataloaders(render_cfg, self.device)
                UV_cnet = txt2imgControlNet(sd_cfg.txt2img)
                
                # Run the UV-only pipeline
                from pipeline_UV_only import UV_gen
                
                print("Running UV position-based texture generation...")
                uv_images = UV_gen(
                    sd_cfg=sd_cfg,
                    cnet=UV_cnet,
                    mesh_model=mesh_model,
                    outdir=uv_out_dir,
                )
                
                # Add results
                for i, img in enumerate(uv_images):
                    results.append(img)
                    
                    # Use this texture to generate final mesh
                    result_path = uv_out_dir / f"UV_gen_res_{i}.png"
                    img.save(result_path)
                    
                    mesh_model.initial_texture_path = str(result_path)
                    mesh_model.refresh_texture()
                    mesh_model.export_mesh(uv_out_dir)
            
            return results
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)