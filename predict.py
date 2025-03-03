# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import shutil
import traceback
import time
import torch
import gc
import numpy as np
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from cog import BasePredictor, Input, Path as CogPath

def check_numpy_installation():
    try:
        import numpy
        return f"NumPy version: {numpy.__version__}, path: {numpy.__file__}"
    except ImportError as e:
        return f"Failed to import NumPy: {str(e)}"
    except Exception as e:
        return f"Error checking NumPy: {str(e)}"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Setting up environment...")
        
        # Create necessary directories
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("tmp", exist_ok=True)
        
        # Add the current directory to the path
        sys.path.append(".")
        
        # Check NumPy status
        print(f"NumPy status: {check_numpy_installation()}")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configuration templates
        print("Loading configuration templates...")
        self.depth_sd_cfg = OmegaConf.load("controlnet/config/depth_based_inpaint_template.yaml")
        self.uv_sd_cfg = OmegaConf.load("controlnet/config/UV_based_inpaint_template.yaml")
        
        # Don't preload models - load them dynamically during prediction to save memory
        self.depth_cnet = None
        self.inpaint_cnet = None
        self.uv_inpaint_cnet = None
        self.uv_tile_cnet = None
        
        print("Setup complete")

    def predict(
        self,
        mesh_path: CogPath = Input(description="Input 3D mesh file (OBJ format)"),
        image_prompt: CogPath = Input(description="Reference image for texture generation"),
        negative_prompt: str = Input(description="Negative prompt to guide generation", default="strong light, shadows, blurry, low quality, distortion"),
        seed: int = Input(description="Random seed for generation (use -1 for random)", default=42),
        texture_resolution: str = Input(description="Texture resolution (width,height)", default="1024,1024")
    ) -> list:
        """Generate high-quality textures for 3D meshes guided by reference images"""
        
        # Create unique output directory
        actual_seed = int(time.time()) if seed == -1 else seed
        output_dir = f"outputs/generation_{actual_seed}"
        os.makedirs(output_dir, exist_ok=True)
        
        stage1_dir = os.path.join(output_dir, "stage1")
        stage2_dir = os.path.join(output_dir, "stage2")
        final_dir = os.path.join(stage2_dir, "final")
        os.makedirs(stage1_dir, exist_ok=True)
        
        # Prepare files and paths
        print(f"Processing mesh and reference image...")
        temp_mesh_path = "tmp/input_mesh.obj"
        shutil.copy2(mesh_path, temp_mesh_path)
        
        image_prompt_path = os.path.join(output_dir, "image_prompt.png")
        shutil.copy2(image_prompt, image_prompt_path)
        
        # Write diagnostic information for troubleshooting
        with open(os.path.join(output_dir, "diagnostic.txt"), "w") as f:
            f.write(f"Mesh path: {mesh_path}\n")
            f.write(f"Image prompt: {image_prompt}\n")
            f.write(f"NumPy status: {check_numpy_installation()}\n")
            f.write(f"Environment: {os.environ}\n")
        
        try:
            # Parse texture resolution
            tex_res = [int(x) for x in texture_resolution.split(",")]
            if len(tex_res) != 2:
                raise ValueError("Texture resolution must be in format 'width,height'")
            
            # Import modules only when needed to avoid potential import issues
            print("Loading Paint3D modules...")
            from paint3d import utils
            from paint3d.models.textured_mesh import TexturedMeshModel
            from paint3d.dataset import init_dataloaders
            from paint3d.trainer import dr_eval, forward_texturing
            import torchvision
            import cv2
            
            # Configure render settings
            render_config_path = "paint3d/config/train_config_paint3d.py"
            pathdir, filename = Path(render_config_path).parent, Path(render_config_path).stem
            sys.path.append(str(pathdir))
            render_cfg = __import__(filename).TrainConfig()
            
            # Update configuration
            render_cfg.guide.shape_path = temp_mesh_path
            render_cfg.guide.texture_resolution = tex_res
            render_cfg.log.exp_path = output_dir
            
            # Set seed for reproducibility
            utils.seed_everything(actual_seed)
            
            # Clean memory before loading models
            torch.cuda.empty_cache()
            gc.collect()
            
            # Initialize mesh model and dataloaders
            print("Initializing mesh model...")
            dataloaders = init_dataloaders(render_cfg, self.device)
            mesh_model = TexturedMeshModel(cfg=render_cfg, device=self.device).to(self.device)
            
            # Setup model configs for image-guided generation
            # Use a space as the text prompt as required by the underlying models
            prompt = " "  # Empty space as required by IP-Adapter
            
            # Create config for stage 1
            sd_cfg_stage1 = OmegaConf.create(self.depth_sd_cfg)
            sd_cfg_stage1.txt2img.prompt = prompt
            sd_cfg_stage1.txt2img.negative_prompt = negative_prompt
            sd_cfg_stage1.txt2img.seed = actual_seed
            sd_cfg_stage1.txt2img.ip_adapter_image_path = image_prompt_path
            
            sd_cfg_stage1.inpaint.prompt = prompt
            sd_cfg_stage1.inpaint.negative_prompt = negative_prompt
            sd_cfg_stage1.inpaint.seed = actual_seed
            sd_cfg_stage1.inpaint.ip_adapter_image_path = image_prompt_path
            
            # Create config for stage 2
            sd_cfg_stage2 = OmegaConf.create(self.uv_sd_cfg)
            sd_cfg_stage2.inpaint.prompt = prompt
            sd_cfg_stage2.inpaint.negative_prompt = negative_prompt
            sd_cfg_stage2.inpaint.seed = actual_seed
            sd_cfg_stage2.inpaint.ip_adapter_image_path = image_prompt_path
            
            sd_cfg_stage2.img2img.prompt = prompt
            sd_cfg_stage2.img2img.negative_prompt = negative_prompt
            sd_cfg_stage2.img2img.seed = actual_seed
            sd_cfg_stage2.img2img.ip_adapter_image_path = image_prompt_path
            
            #######################
            # STAGE 1: Coarse Texture Generation
            #######################
            print("STAGE 1: Generating coarse texture from image reference...")
            
            # Load stage 1 models
            if self.depth_cnet is None:
                from controlnet.diffusers_cnet_txt2img import txt2imgControlNet
                self.depth_cnet = txt2imgControlNet(sd_cfg_stage1.txt2img)
            
            # Collect initial depth maps
            init_depth_map = []
            view_angle_info = {i: data for i, data in enumerate(dataloaders['train'])}
            for view_id in render_cfg.render.views_init:
                data = view_angle_info[view_id]
                theta, phi, radius = data['theta'], data['phi'], data['radius']
                outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
                depth_render = outputs['depth']
                init_depth_map.append(depth_render)

            # Fix the tensor dimensions issue
            init_depth_map = torch.cat(init_depth_map, dim=0).repeat(1, 3, 1, 1)
            init_depth_grid = torchvision.utils.make_grid(init_depth_map, nrow=2, padding=0)
            init_depth_map_path = os.path.join(output_dir, "init_depth_render.png")
            
            # Save depth map
            utils.save_tensor_image(init_depth_grid.unsqueeze(0), save_path=init_depth_map_path)
            
            # Post-process depth map
            depth_dilated = utils.dilate_depth_outline(init_depth_map_path, iters=5, dilate_kernel=3)
            depth_dilated_path = os.path.join(output_dir, "init_depth_dilated.png")
            cv2.imwrite(depth_dilated_path, depth_dilated)
            
            # Generate initial view
            sd_cfg_stage1.txt2img.controlnet_units[0].condition_image_path = depth_dilated_path
            init_images = self.depth_cnet.infernece(config=sd_cfg_stage1.txt2img)
            
            # Save the first generated image only
            if len(init_images) > 0:
                init_image_path = os.path.join(output_dir, 'init-img-0.png')
                init_images[0].save(init_image_path)
            
            # Unload the depth_cnet to save memory
            self.depth_cnet = None
            torch.cuda.empty_cache()
            gc.collect()
            
            # Load inpaint model for stage 1
            if self.inpaint_cnet is None:
                from controlnet.diffusers_cnet_inpaint import inpaintControlNet
                self.inpaint_cnet = inpaintControlNet(sd_cfg_stage1.inpaint)
            
            # Process each generated image
            stage1_result = False
            for i, init_image in enumerate(init_images):
                try:
                    # Project initial views onto mesh
                    mesh_model.initial_texture_path = None
                    mesh_model.refresh_texture()
                    view_imgs = utils.split_grid_image(img=np.array(init_image), size=(1, 2))
                    forward_texturing(
                        cfg=render_cfg,
                        dataloaders=dataloaders,
                        mesh_model=mesh_model,
                        save_result_dir=stage1_dir,
                        device=self.device,
                        view_imgs=view_imgs,
                        view_ids=render_cfg.render.views_init,
                        verbose=False,
                    )
                    
                    # Perform inpainting for additional views
                    for view_group in render_cfg.render.views_inpaint:
                        # Project inpaint view
                        inpaint_used_key = ["image", "depth", "uncolored_mask"]
                        for j, one_batch_id in enumerate([view_group]):
                            one_batch_img = []
                            for view_id in one_batch_id:
                                data = view_angle_info[view_id]
                                theta, phi, radius = data['theta'], data['phi'], data['radius']
                                outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
                                view_img_info = [outputs[k] for k in inpaint_used_key]
                                one_batch_img.append(view_img_info)
                            
                            for j, img in enumerate(zip(*one_batch_img)):
                                img = torch.cat(img, dim=3)
                                if img.size(1) == 1:
                                    img = img.repeat(1, 3, 1, 1)
                                t = '_'.join(map(str, one_batch_id))
                                name = inpaint_used_key[j]
                                if name == "uncolored_mask":
                                    img[img>0] = 1
                                save_path = os.path.join(stage1_dir, f"view_{t}_{name}.png")
                                utils.save_tensor_image(img, save_path=save_path)
                        
                        # Inpaint additional views
                        for j, one_batch_id in enumerate([view_group]):
                            t = '_'.join(map(str, one_batch_id))
                            rgb_path = os.path.join(stage1_dir, f"view_{t}_{inpaint_used_key[0]}.png")
                            depth_path = os.path.join(stage1_dir, f"view_{t}_{inpaint_used_key[1]}.png")
                            mask_path = os.path.join(stage1_dir, f"view_{t}_{inpaint_used_key[2]}.png")
                            
                            # Pre-processing inpaint mask: dilate
                            mask = cv2.imread(mask_path)
                            dilate_kernel = 10
                            mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))
                            dilated_mask_path = os.path.join(stage1_dir, f"view_{t}_{inpaint_used_key[2]}_d{dilate_kernel}.png")
                            cv2.imwrite(dilated_mask_path, mask)
                            
                            sd_cfg_stage1.inpaint.image_path = rgb_path
                            sd_cfg_stage1.inpaint.mask_path = dilated_mask_path
                            sd_cfg_stage1.inpaint.controlnet_units[0].condition_image_path = depth_path
                            inpainted_images = self.inpaint_cnet.infernece(config=sd_cfg_stage1.inpaint)
                            
                            # Save only the first inpainted image to save space
                            if len(inpainted_images) > 0:
                                save_path = os.path.join(stage1_dir, f"view_{t}_rgb_inpaint_0.png")
                                inpainted_images[0].save(save_path)
                            
                            # Project inpainted views (using only the first image)
                            if len(inpainted_images) > 0:
                                view_imgs = utils.split_grid_image(img=np.array(inpainted_images[0]), size=(1, 2))
                                forward_texturing(
                                    cfg=render_cfg,
                                    dataloaders=dataloaders,
                                    mesh_model=mesh_model,
                                    save_result_dir=stage1_dir,
                                    device=self.device,
                                    view_imgs=view_imgs,
                                    view_ids=view_group,
                                    verbose=False,
                                )
                    
                    # Save stage 1 results
                    stage1_texture_path = os.path.join(stage1_dir, "albedo.png")
                    mesh_model.initial_texture_path = stage1_texture_path
                    mesh_model.refresh_texture()
                    mesh_model.export_mesh(stage1_dir)
                    stage1_result = True
                    break  # Process only the first image that succeeds
                    
                except Exception as e:
                    print(f"Error processing image {i} in stage 1: {str(e)}")
                    continue
            
            # Unload the inpaint_cnet to save memory
            self.inpaint_cnet = None
            torch.cuda.empty_cache()
            gc.collect()
            
            # If stage 1 failed, return error
            if not stage1_result:
                raise Exception("Failed to generate initial texture in stage 1")
            
            #######################
            # STAGE 2: Refined Texture Generation
            #######################
            print("STAGE 2: Refining texture...")
            os.makedirs(stage2_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Step 1: UV Inpainting
            print("Performing UV Inpainting...")
            
            # Load UV inpaint model
            if self.uv_inpaint_cnet is None:
                from controlnet.diffusers_cnet_inpaint import inpaintControlNet
                self.uv_inpaint_cnet = inpaintControlNet(sd_cfg_stage2.inpaint)
            
            # Export current texture and generate UV position map
            mesh_model.export_mesh(stage2_dir, export_texture_only=True)
            albedo_path = os.path.join(stage2_dir, "albedo.png")
            UV_pos = mesh_model.UV_pos_render()
            UV_pos_path = os.path.join(stage2_dir, "UV_pos.png")
            utils.save_tensor_image(UV_pos.permute(0, 3, 1, 2), UV_pos_path)
            
            # Process mask for inpainting
            mask_dilated = utils.extract_bg_mask(albedo_path, dilate_kernel=15)
            mask_path = os.path.join(stage2_dir, "mask.png")
            cv2.imwrite(mask_path, mask_dilated)
            
            # UV inpainting
            sd_cfg_stage2.inpaint.image_path = albedo_path
            sd_cfg_stage2.inpaint.mask_path = mask_path
            sd_cfg_stage2.inpaint.controlnet_units[0].condition_image_path = UV_pos_path
            
            inpaint_images = self.uv_inpaint_cnet.infernece(config=sd_cfg_stage2.inpaint)
            uv_inpaint_path = os.path.join(stage2_dir, 'UV_inpaint_res_0.png')
            if len(inpaint_images) > 0:
                inpaint_images[0].save(uv_inpaint_path)
            
            # Unload the uv_inpaint_cnet to save memory
            self.uv_inpaint_cnet = None
            torch.cuda.empty_cache()
            gc.collect()
            
            # Step 2: UV Tiling (detail enhancement)
            print("Performing UV Tiling (detail enhancement)...")
            
            # Load UV tile model
            if self.uv_tile_cnet is None:
                from controlnet.diffusers_cnet_img2img import img2imgControlNet
                self.uv_tile_cnet = img2imgControlNet(sd_cfg_stage2.img2img)
            
            mesh_model.initial_texture_path = uv_inpaint_path
            mesh_model.refresh_texture()
            
            # Export current texture and generate UV position map
            mesh_model.export_mesh(final_dir, export_texture_only=True)
            albedo_path = os.path.join(final_dir, "albedo.png")
            UV_pos = mesh_model.UV_pos_render()
            UV_pos_path = os.path.join(final_dir, "UV_pos.png")
            utils.save_tensor_image(UV_pos.permute(0, 3, 1, 2), UV_pos_path)
            
            # UV tile enhancement
            sd_cfg_stage2.img2img.image_path = albedo_path
            sd_cfg_stage2.img2img.controlnet_units[0].condition_image_path = UV_pos_path
            sd_cfg_stage2.img2img.controlnet_units[1].condition_image_path = albedo_path
            
            tile_images = self.uv_tile_cnet.infernece(config=sd_cfg_stage2.img2img)
            final_texture_path = os.path.join(final_dir, 'final_texture.png')
            if len(tile_images) > 0:
                tile_images[0].save(final_texture_path)
            
            # Unload the uv_tile_cnet to save memory
            self.uv_tile_cnet = None
            torch.cuda.empty_cache()
            gc.collect()
            
            # Evaluate and export the final texture
            mesh_model.initial_texture_path = final_texture_path
            mesh_model.refresh_texture()
            dr_eval(
                cfg=render_cfg,
                dataloaders=dataloaders,
                mesh_model=mesh_model,
                save_result_dir=final_dir,
                valset=True,
                verbose=False,
            )
            
            # Export final textured mesh
            mesh_model.export_mesh(final_dir)
            
            # Clean up resources
            mesh_model.empty_texture_cache()
            torch.cuda.empty_cache()
            gc.collect()
            
            # Return the final outputs
            obj_path = os.path.join(final_dir, "mesh.obj")
            mtl_path = os.path.join(final_dir, "mesh.mtl")
            
            print(f"Texture generation complete!")
            
            return [
                CogPath(final_texture_path),
                CogPath(obj_path),
                CogPath(mtl_path)
            ]
                        
        except Exception as e:
            # Check if we have stage 1 results to return as fallback
            stage1_texture_path = os.path.join(stage1_dir, "albedo.png")
            stage1_obj_path = os.path.join(stage1_dir, "mesh.obj")
            stage1_mtl_path = os.path.join(stage1_dir, "mesh.mtl")
            
            if os.path.exists(stage1_texture_path) and os.path.exists(stage1_obj_path) and os.path.exists(stage1_mtl_path):
                print(f"Stage 2 failed but returning Stage 1 results as fallback")
                return [
                    CogPath(stage1_texture_path),
                    CogPath(stage1_obj_path),
                    CogPath(stage1_mtl_path)
                ]
            
            # Log the error and return the error details
            error_path = os.path.join(output_dir, "error_details.txt")
            with open(error_path, 'w') as f:
                f.write(f"Error during texture generation: {str(e)}\n\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
            return CogPath(error_path)