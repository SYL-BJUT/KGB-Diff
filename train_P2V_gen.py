"""
Baseline IP-Adapter training for the Palmprint -> Palmvein direction.

This is the *image-only* baseline used in the ablation study (no knowledge graph).
The model learns a 1-to-1 mapping between paired palmprint and palmvein images:
CLIP-ViT-L/14 encodes the palmprint, the IP-Adapter projection maps it to 4 tokens
that are duplicated to 8 (filling both the `to_k/to_v` and `to_k_ip/to_v_ip`
attention slots of every cross-attention block in the SD-1.5 UNet).

Stage-1 release: we ship this script so reviewers can reproduce the
"no-retrieval" baseline numbers from the paper.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection

# 导入IP-Adapter模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IP-Adapter-main'))
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class PalmPrintVeinDataset(torch.utils.data.Dataset):
    """掌纹-掌静脉配对数据集"""
    
    def __init__(self, palmprint_dir, palmvein_dir, size=512, recursive: bool = False):
        super().__init__()

        self.palmprint_dir = palmprint_dir
        self.palmvein_dir = palmvein_dir
        self.size = size

        def normalize_key(filename: str) -> str:
            stem, _ext = os.path.splitext(filename)

            # Tongji600 / Polyu: PalmPrint_0_1 -> 0_1
            m = re.match(r"^Palm(Print|Vein)_(\d+)_(\d+)$", stem, flags=re.IGNORECASE)
            if m:
                return f"{m.group(2)}_{m.group(3)}"
            # palm_xxxxx style
            m = re.match(r"^palm_(\d+)$", stem, flags=re.IGNORECASE)
            if m:
                return m.group(1)
            # CUMT: 001_1 -> 001_1 (zero-pad id to 3 digits for consistent ordering)
            m = re.match(r"^(\d+)_(\d+)$", stem, flags=re.IGNORECASE)
            if m:
                return f"{int(m.group(1)):03d}_{m.group(2)}"
            return stem.lower()

        def collect_files(root: str) -> list[str]:
            if recursive:
                found = []
                for dp, _, fns in os.walk(root):
                    for fn in fns:
                        if fn.lower().endswith((".jpg", ".png")):
                            found.append(os.path.relpath(os.path.join(dp, fn), root))
                return sorted(found)
            return sorted(
                f for f in os.listdir(root) if f.lower().endswith((".jpg", ".png"))
            )

        palmprint_files = collect_files(palmprint_dir)
        palmvein_files = collect_files(palmvein_dir)

        vein_map = {}
        for f in palmvein_files:
            k = normalize_key(f)
            if k not in vein_map:
                vein_map[k] = f

        self.pairs = []
        unmatched_palmprints = []
        for f in palmprint_files:
            k = normalize_key(f)
            pv = vein_map.get(k)
            if pv is not None:
                self.pairs.append((f, pv))
            else:
                unmatched_palmprints.append((f, k))

        print(
            f"Found {len(self.pairs)} paired samples "
            f"(palmprint_files={len(palmprint_files)}, palmvein_files={len(palmvein_files)}, unmatched_palmprints={len(unmatched_palmprints)})"
        )
        if len(unmatched_palmprints) > 0:
            preview = unmatched_palmprints[:10]
            print("Unmatched palmprint examples (filename -> key):")
            for fn, k in preview:
                print(f"  {fn} -> {k}")
        
        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        palmprint_filename, palmvein_filename = self.pairs[idx]
        
        # 加载掌纹（输入）
        palmprint_path = os.path.join(self.palmprint_dir, palmprint_filename)
        palmprint_raw = Image.open(palmprint_path).convert("RGB")
        palmprint = self.transform(palmprint_raw)
        clip_palmprint = self.clip_image_processor(images=palmprint_raw, return_tensors="pt").pixel_values
        
        # 加载掌静脉（目标）
        palmvein_path = os.path.join(self.palmvein_dir, palmvein_filename)
        palmvein_raw = Image.open(palmvein_path).convert("RGB")
        palmvein = self.transform(palmvein_raw)
        
        return {
            "palmprint": palmprint,
            "palmvein": palmvein,
            "clip_palmprint": clip_palmprint,
            "filename": palmprint_filename
        }
    
    def __len__(self):
        return len(self.pairs)


def collate_fn(data):
    """数据整理函数"""
    palmprints = torch.stack([example["palmprint"] for example in data])
    palmveins = torch.stack([example["palmvein"] for example in data])
    clip_palmprints = torch.cat([example["clip_palmprint"] for example in data], dim=0)
    filenames = [example["filename"] for example in data]
    
    return {
        "palmprints": palmprints,
        "palmveins": palmveins,
        "clip_palmprints": clip_palmprints,
        "filenames": filenames
    }


class IPAdapterPalm2Vein(torch.nn.Module):
    """IP-Adapter for Palm2Vein (Baseline)"""
    
    def __init__(self, unet, image_proj_model, adapter_modules):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
    
    def forward(self, noisy_latents, timesteps, palmprint_embeds):
        """
        前向传播（使用掌纹特征作为双路条件）
        Args:
            noisy_latents: 加噪的latent
            timesteps: 时间步
            palmprint_embeds: 掌纹图像的CLIP embedding [B, 768]
        """
        # 将掌纹embedding投影为tokens
        palmprint_tokens = self.image_proj_model(palmprint_embeds)  # [B, 4, 768]
        
        # 复制掌纹tokens，让两对手都处理掌纹
        # 第一对手（to_k/to_v）：处理前4个tokens
        # 第二对手（to_k_ip/to_v_ip）：处理后4个tokens
        encoder_hidden_states = torch.cat([palmprint_tokens, palmprint_tokens], dim=1)  # [B, 8, 768]
        
        # 预测噪声
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        return noise_pred


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline IP-Adapter training for Palmprint -> Palmvein (no KG, no retrieval)."
    )

    # ---- Model paths (must be supplied explicitly) ----
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to a local Stable-Diffusion-1.5 checkpoint "
             "(folder containing {unet, vae, scheduler} subfolders).",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        required=True,
        help="Path to the local CLIP-ViT-L/14 checkpoint (e.g. .../clip-vit-large-patch14).",
    )

    # ---- Data paths ----
    parser.add_argument(
        "--palmprint_dir",
        type=str,
        required=True,
        help="Directory of training palmprint images (input modality).",
    )
    parser.add_argument(
        "--palmvein_dir",
        type=str,
        required=True,
        help="Directory of training palmvein images (ground-truth target).",
    )

    # ---- Training hyperparameters ----
    parser.add_argument("--resolution", type=int, default=128,
                        help="Spatial resolution. Must be divisible by 8 (default: 512).")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan sub-directories (needed for datasets that use sub-folders, e.g. CUMT).",
    )
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of forward passes between optimizer steps. "
                             "Use a higher value if you want a larger effective batch.")

    # ---- Saving ----
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where checkpoints and the final model are written.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save a numbered checkpoint every N global steps.")

    # ---- Misc ----
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Distributed-training rank (set automatically by torchrun).")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 设置accelerator
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 60)
        print("Baseline IP-Adapter  |  Palmprint -> Palmvein  (no KG)")
        print("=" * 60)
        for k, v in vars(args).items():
            print(f"  {k:35s} = {v}")
        print("=" * 60)

    # ============ Load models ============
    if accelerator.is_main_process:
        print("\nLoading SD-1.5 + CLIP-ViT-L/14 ...")

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder_path)

    # Freeze everything except the IP-Adapter modules we will train below.
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # ============ 初始化IP-Adapter ============
    if accelerator.is_main_process:
        print("Initializing IP-Adapter...")
    
    # Image Projection Model
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    
    # 初始化Attention Processors
    attn_procs = {}
    unet_sd = unet.state_dict()
    
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        
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
    
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    # 创建IP-Adapter模型
    ip_adapter = IPAdapterPalm2Vein(unet, image_proj_model, adapter_modules)
    
    # ============ 设置数据类型 ============
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # ============ 优化器 ============
    import itertools
    params_to_opt = itertools.chain(
        ip_adapter.image_proj_model.parameters(),
        ip_adapter.adapter_modules.parameters()
    )
    optimizer = torch.optim.AdamW(
        params_to_opt,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # ============ 数据加载器 ============
    if accelerator.is_main_process:
        print("Preparing dataset...")
    
    # ============ Data ============
    if accelerator.is_main_process:
        print("\nPreparing dataset ...")

    train_dataset = PalmPrintVeinDataset(
        palmprint_dir=args.palmprint_dir,
        palmvein_dir=args.palmvein_dir,
        size=args.resolution,
        recursive=args.recursive,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # ============ Accelerator prepare ============
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(
        ip_adapter, optimizer, train_dataloader
    )

    # ============ Training loop ============
    if accelerator.is_main_process:
        print(f"\nStarting training ...")
        print(f"Total samples : {len(train_dataset)}")
        print(f"Steps/epoch   : {len(train_dataloader)}")
        print(f"Total steps   : {len(train_dataloader) * args.num_train_epochs}\n")
    
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        begin = time.perf_counter()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin

            with accelerator.accumulate(ip_adapter):
                # Encode ground-truth palmvein into latent space (no gradient through VAE).
                with torch.no_grad():
                    latents = vae.encode(
                        batch["palmveins"].to(accelerator.device, dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Add noise -> noisy_latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps,
                    (bsz,), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode palmprint image into CLIP embedding (no gradient).
                with torch.no_grad():
                    palmprint_embeds = image_encoder(
                        batch["clip_palmprints"].to(accelerator.device, dtype=weight_dtype)
                    ).image_embeds

                # Predict noise from the IP-Adapter (uses duplicated CLIP tokens).
                noise_pred = ip_adapter(
                    noisy_latents,
                    timesteps,
                    palmprint_embeds,
                )

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                current_loss = loss.item()
                epoch_loss += current_loss

                if accelerator.is_main_process:
                    print(f"Epoch {epoch+1}/{args.num_train_epochs}, "
                          f"Step {step+1}/{len(train_dataloader)}, "
                          f"Loss: {current_loss:.4f}, "
                          f"Time: {time.perf_counter() - begin:.2f}s")
            
            global_step += 1

            # Periodic checkpoint (every save_steps)
            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    unwrapped_model = accelerator.unwrap_model(ip_adapter)
                    torch.save({
                        "image_proj": unwrapped_model.image_proj_model.state_dict(),
                        "ip_adapter": unwrapped_model.adapter_modules.state_dict(),
                    }, os.path.join(save_path, "ip_adapter.bin"))

                    print(f"\n  -> Saved checkpoint to {save_path}\n")

            begin = time.perf_counter()

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1} finished.  Average loss: {avg_epoch_loss:.4f}")
            print(f"{'='*60}\n")

            save_path = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)

            unwrapped_model = accelerator.unwrap_model(ip_adapter)
            torch.save({
                "image_proj": unwrapped_model.image_proj_model.state_dict(),
                "ip_adapter": unwrapped_model.adapter_modules.state_dict(),
            }, os.path.join(save_path, "ip_adapter.bin"))

            print(f"  -> Saved epoch {epoch+1} to {save_path}\n")

    # ============= Save final model =============
    if accelerator.is_main_process:
        print("Training complete. Saving final model ...")
        final_save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(ip_adapter)
        torch.save({
            "image_proj": unwrapped_model.image_proj_model.state_dict(),
            "ip_adapter": unwrapped_model.adapter_modules.state_dict(),
        }, os.path.join(final_save_path, "ip_adapter.bin"))

        print(f"\nFinal model -> {final_save_path}")
        print("Done.")


if __name__ == "__main__":
    main()
