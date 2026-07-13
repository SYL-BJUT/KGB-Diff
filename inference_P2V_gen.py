"""
Baseline IP-Adapter inference for the Palmprint -> Palmvein direction.

Loads a fine-tuned IP-Adapter checkpoint produced by ``train_P2V_gen.py`` and
runs DDIM sampling to translate palmprint images into palmvein images. There
is no knowledge graph, no retrieval, no extra conditioning — this is the
image-only baseline that the KG pipeline in stage 2 is compared against.

Stage-1 release: shipped alongside ``train_P2V_gen.py`` so reviewers can
reproduce the "no-retrieval" baseline outputs for the P2V direction.
"""

import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import argparse
import re


def _make_id(stem):
    """标准化输出文件名：去掉重复前缀和冗余后缀，固定无后缀。
    Polyu 平铺 P2V: PalmPrint_003_2 -> PalmVein_003_2
    Polyu 平铺 V2P: PalmVein_003_2 -> PalmVein_003_2
    Polyu palm_XXXXXX: palm_000107_gen -> palm_000107
    CUMT 递归: 001_001_7 -> 001_7
    其余: 原样返回
    """
    m_polyu = re.match(r"^PalmPrint_(\d+)_(\d+)$", stem, flags=re.IGNORECASE)
    if m_polyu:
        return f"PalmVein_{m_polyu.group(1)}_{m_polyu.group(2)}"
    m_palm = re.match(r"^(palm_\d+)$", stem, flags=re.IGNORECASE)
    if m_palm:
        return m_palm.group(1)
    m = re.match(r"^(.+?)_(.+?)_(\d+)$", stem, flags=re.IGNORECASE)
    if m and m.group(1) == m.group(2):
        return f"{m.group(1)}_{m.group(3)}"
    return stem

# 导入IP-Adapter模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IP-Adapter-main'))
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class BaselineInference:
    """Baseline IP-Adapter inference (Palmprint -> Palmvein, no KG)."""

    def __init__(self, sd_model_path, ip_adapter_path, image_encoder_path, resolution: int = 512, device="cuda"):
        self.device = device
        self.resolution = int(resolution)
        if self.resolution % 8 != 0:
            raise ValueError(f"resolution must be divisible by 8, got: {self.resolution}")
        print(f"Loading models on {device}...")
        
        # 加载SD组件
        self.vae = AutoencoderKL.from_pretrained(sd_model_path, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(sd_model_path, subfolder="unet").to(device)
        
        # 加载CLIP Image Encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device)
        self.clip_image_processor = CLIPImageProcessor()
        
        # 加载Scheduler
        self.scheduler = DDIMScheduler.from_pretrained(sd_model_path, subfolder="scheduler")
        
        # 初始化IP-Adapter
        self._init_ip_adapter(ip_adapter_path)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print("Models loaded successfully!")
    
    def _init_ip_adapter(self, ip_adapter_path):
        """初始化IP-Adapter"""
        print("Initializing IP-Adapter...")
        
        # Image Projection Model
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        ).to(self.device)
        
        # 初始化Attention Processors
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim
                )
                attn_procs[name].load_state_dict(weights)
        
        self.unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
        
        # 加载训练好的权重
        print(f"Loading IP-Adapter weights from {ip_adapter_path}")
        state_dict = torch.load(ip_adapter_path, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        adapter_modules.load_state_dict(state_dict["ip_adapter"])
        
        # 确保adapter_modules在正确的设备上
        adapter_modules.to(self.device)
        
        print("IP-Adapter initialized!")
    
    @torch.no_grad()
    def generate(self, palmprint_image, num_inference_steps=50, guidance_scale=1.0, seed=None):
        """
        生成掌静脉
        
        Args:
            palmprint_image: PIL Image或路径
            num_inference_steps: 推理步数
            guidance_scale: 引导强度（1.0表示无引导）
            seed: 随机种子
        
        Returns:
            生成的掌静脉图像（PIL Image）
        """
        # 加载图像
        if isinstance(palmprint_image, str):
            palmprint_image = Image.open(palmprint_image).convert("RGB")
        
        # 编码掌纹
        clip_image = self.clip_image_processor(images=palmprint_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device)
        
        palmprint_embeds = self.image_encoder(clip_image).image_embeds
        palmprint_tokens = self.image_proj_model(palmprint_embeds)
        
        # 复制掌纹tokens，让两对手都处理掌纹（与训练时一致）
        encoder_hidden_states = torch.cat([palmprint_tokens, palmprint_tokens], dim=1)  # [B, 8, 768]
        
        # 设置随机种子
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # 初始化随机噪声
        latent_hw = self.resolution // 8
        latents = torch.randn(
            (1, self.unet.config.in_channels, latent_hw, latent_hw),
            generator=generator,
            device=self.device
        )
        
        # 设置scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        # 去噪循环
        for t in self.scheduler.timesteps:
            # 预测噪声
            latent_model_input = latents
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states  # 使用双路tokens
            ).sample
            
            # 更新latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 解码到图像空间
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        
        # 转换为PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype("uint8")
        image = Image.fromarray(image)
        
        return image


def main():
    parser = argparse.ArgumentParser(
        description="Baseline IP-Adapter inference for Palmprint -> Palmvein (no KG)."
    )

    # ---- Model paths (must be supplied explicitly) ----
    parser.add_argument(
        "--sd_model_path", type=str, required=True,
        help="Path to a local Stable-Diffusion-1.5 checkpoint "
             "(folder containing {unet, vae, scheduler} subfolders).",
    )
    parser.add_argument(
        "--image_encoder_path", type=str, required=True,
        help="Path to the local CLIP-ViT-L/14 checkpoint (e.g. .../clip-vit-large-patch14).",
    )
    parser.add_argument(
        "--ip_adapter_path", type=str, required=True,
        help="Path to the trained IP-Adapter state_dict produced by train_P2V_gen.py "
             "(e.g. <output_dir>/final_model/ip_adapter.bin).",
    )

    # ---- I/O ----
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory of test palmprint images (input modality).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write generated palmvein images into.",
    )

    # ---- Sampling ----
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Output spatial resolution. Must be divisible by 8 (e.g. 512 or 128).",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="Number of DDIM steps.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed. Each image uses seed + index as its actual seed.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively walk subdirectories of --input_dir. Required for CUMT-style "
             "datasets that organise images by identity.",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # Init inferencer
    inferencer = BaselineInference(
        sd_model_path=args.sd_model_path,
        ip_adapter_path=args.ip_adapter_path,
        image_encoder_path=args.image_encoder_path,
        resolution=args.resolution,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Collect all test images
    if args.recursive:
        test_images = []
        for dp, _, fns in os.walk(args.input_dir):
            for fn in fns:
                if fn.lower().endswith(('.jpg', '.png', '.bmp')):
                    rel = os.path.relpath(os.path.join(dp, fn), args.input_dir)
                    rel = rel.replace(os.sep, "_").rsplit(".", 1)[0]
                    test_images.append((os.path.join(dp, fn), rel))
    else:
        test_images = [
            (os.path.join(args.input_dir, f), os.path.splitext(f)[0])
            for f in sorted(os.listdir(args.input_dir))
            if f.lower().endswith(('.jpg', '.png', '.bmp'))
        ]

    print(f"\nFound {len(test_images)} test images")
    print(f"Generating palmvein images (Palmprint -> Palmvein, baseline) ...\n")

    for i, (input_path, stem) in enumerate(test_images):
        out_stem = _make_id(stem)
        out_name = f"{out_stem}.png"
        output_path = os.path.join(args.output_dir, out_name)

        print(f"[{i+1}/{len(test_images)}] Processing {os.path.basename(input_path)} ...")

        generated_vein = inferencer.generate(
            palmprint_image=input_path,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed + i,
        )

        generated_vein.save(output_path, format="PNG")
        print(f"  -> {output_path}")

    print(f"\nAll done. Generated images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
