"""
Baseline IP-Adapter training for the Palmvein -> Palmprint direction (128 resolution).

This is the *image-only* baseline used in the ablation study (no knowledge graph).
It mirrors `train_P2V_gen.py` but trains in the opposite direction:

    input  = palmvein image  (encoded by CLIP-ViT-L/14)
    target = palmprint image (encoded by the VAE)

The 4 CLIP tokens are duplicated to 8 to feed both attention slots
(`to_k/to_v` and `to_k_ip/to_v_ip`) of every cross-attention block in the SD-1.5 UNet.

Stage-1 release: shipped alongside `train_P2V_gen.py` so reviewers can reproduce
the "no-retrieval" baseline for both translation directions.
"""

import os
import sys
import time
import argparse
import re
from pathlib import Path

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


class PalmVeinPrintDataset(torch.utils.data.Dataset):
    """Palmvein-Palmprint paired dataset (128-resolution V2P baseline).

    Supports multiple filename conventions used across palm datasets:
      - Tongji600 / Polyu: ``PalmPrint_0_1.png`` / ``PalmVein_0_1.png``
      - palm_xxxxx style:  ``palm_00012.png``
      - CUMT (identity/sub): ``001_1.png``  (id zero-padded to 3 digits for consistent ordering)

    Pairing is performed by normalizing both sides into a canonical key and
    matching them up. Files whose key has no counterpart on the other side are
    skipped (and reported).
    """

    def __init__(self, palmvein_dir, palmprint_dir, size=128, recursive: bool = False):
        super().__init__()

        self.palmvein_dir = palmvein_dir
        self.palmprint_dir = palmprint_dir
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

        vein_files = collect_files(palmvein_dir)
        print_files = collect_files(palmprint_dir)

        # Build a vein-side index keyed by the canonical id; first occurrence wins
        # (avoids silently picking a different capture if two files normalize to the
        # same key, which can happen with overlapping sub-folders).
        vein_map = {}
        for f in vein_files:
            k = normalize_key(f)
            vein_map.setdefault(k, f)

        self.pairs = []
        unmatched_prints = []
        for f in print_files:
            k = normalize_key(f)
            pv = vein_map.get(k)
            if pv is not None:
                self.pairs.append((f, pv))
            else:
                unmatched_prints.append((f, k))

        print(
            f"Found {len(self.pairs)} paired samples "
            f"(vein_files={len(vein_files)}, print_files={len(print_files)}, "
            f"unmatched_prints={len(unmatched_prints)})"
        )
        if unmatched_prints:
            preview = unmatched_prints[:10]
            print("Unmatched palmprint examples (filename -> key):")
            for fn, k in preview:
                print(f"  {fn} -> {k}")

        # Image transforms (resize + center-crop, then to [-1, 1])
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        palmprint_filename, palmvein_filename = self.pairs[idx]

        # Load palmvein (input modality) -> tensor + CLIP pixels
        palmvein_path = os.path.join(self.palmvein_dir, palmvein_filename)
        palmvein_raw = Image.open(palmvein_path).convert("RGB")
        palmvein = self.transform(palmvein_raw)
        clip_palmvein = self.clip_image_processor(images=palmvein_raw, return_tensors="pt").pixel_values

        # Load palmprint (target modality) -> tensor only
        palmprint_path = os.path.join(self.palmprint_dir, palmprint_filename)
        palmprint_raw = Image.open(palmprint_path).convert("RGB")
        palmprint = self.transform(palmprint_raw)

        return {
            "palmvein": palmvein,
            "palmprint": palmprint,
            "clip_palmvein": clip_palmvein,
            "filename": palmvein_filename,
        }

    def __len__(self):
        return len(self.pairs)


def collate_fn(data):
    """数据整理函数"""
    palmveins = torch.stack([example["palmvein"] for example in data])
    palmprints = torch.stack([example["palmprint"] for example in data])
    clip_palmveins = torch.cat([example["clip_palmvein"] for example in data], dim=0)
    filenames = [example["filename"] for example in data]

    return {
        "palmveins": palmveins,
        "palmprints": palmprints,
        "clip_palmveins": clip_palmveins,
        "filenames": filenames
    }


class IPAdapterVein2Palm(torch.nn.Module):
    """IP-Adapter for Vein2Palm (Baseline)"""

    def __init__(self, unet, image_proj_model, adapter_modules):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

    def forward(self, noisy_latents, timesteps, palmvein_embeds):
        """
        前向传播（使用掌静脉特征作为双路条件）
        Args:
            noisy_latents: 加噪的latent
            timesteps: 时间步
            palmvein_embeds: 掌静脉图像的CLIP embedding [B, 768]
        """
        palmvein_tokens = self.image_proj_model(palmvein_embeds)  # [B, 4, 768]
        encoder_hidden_states = torch.cat([palmvein_tokens, palmvein_tokens], dim=1)  # [B, 8, 768]
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline IP-Adapter training for Palmvein -> Palmprint "
                    "at 128 resolution (no KG, no retrieval)."
    )

    # ---- Model paths (must be supplied explicitly) ----
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str, required=True,
        help="Path to a local Stable-Diffusion-1.5 checkpoint "
             "(folder containing {unet, vae, scheduler} subfolders).",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str, required=True,
        help="Path to the local CLIP-ViT-L/14 checkpoint.",
    )

    # ---- Data paths ----
    parser.add_argument(
        "--palmvein_dir", type=str, required=True,
        help="Directory of training palmvein images (input modality).",
    )
    parser.add_argument(
        "--palmprint_dir", type=str, required=True,
        help="Directory of training palmprint images (ground-truth target).",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively walk subdirectories of --palmvein_dir and --palmprint_dir. "
             "Required for CUMT-style datasets that organise images by identity.",
    )

    # ---- Training hyperparameters ----
    parser.add_argument("--resolution", type=int, default=128,
                        help="Spatial resolution. Must be divisible by 8 (default: 128).")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of forward passes between optimizer steps.")

    # ---- Saving ----
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where checkpoints and the final model are written.")
    parser.add_argument("--save_steps", type=int, default=3000,
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
        print("Baseline IP-Adapter  |  Palmvein -> Palmprint  (no KG, 128)")
        print("=" * 60)
        for k, v in vars(args).items():
            print(f"  {k:35s} = {v}")
        print("=" * 60)

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

    if accelerator.is_main_process:
        print("Initializing IP-Adapter...")

    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

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

    ip_adapter = IPAdapterVein2Palm(unet, image_proj_model, adapter_modules)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

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

    if accelerator.is_main_process:
        print("\nPreparing dataset ...")

    train_dataset = PalmVeinPrintDataset(
        palmvein_dir=args.palmvein_dir,
        palmprint_dir=args.palmprint_dir,
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

    ip_adapter, optimizer, train_dataloader = accelerator.prepare(
        ip_adapter, optimizer, train_dataloader
    )

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
            with accelerator.accumulate(ip_adapter):
                # ---- Forward: encode palmprint (target) into latent space ----
                with torch.no_grad():
                    latents = vae.encode(
                        batch["palmprints"].to(accelerator.device, dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps,
                    (bsz,), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ---- Encode palmvein (input) into CLIP embedding ----
                with torch.no_grad():
                    palmvein_embeds = image_encoder(
                        batch["clip_palmveins"].to(accelerator.device, dtype=weight_dtype)
                    ).image_embeds

                noise_pred = ip_adapter(
                    noisy_latents,
                    timesteps,
                    palmvein_embeds,
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
