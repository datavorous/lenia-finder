"""Optional MobileCLIP2 image and text embeddings."""

import copy
import hashlib
from contextlib import nullcontext

import torch
import torch.nn.functional as F

MODELS = {
    "mobileclip2-s0": (
        "apple/MobileCLIP2-S0",
        "3136ea51c8ed56b9f9abfab04cb816735aaad6cb",
    ),
    "mobileclip2-s2": (
        "apple/MobileCLIP2-S2",
        "72424e7025436db18f15c3eff6ee8c7c15ad4481",
    ),
}


def target_loss(image_embeddings, text_embeddings):
    """ASAL's negative mean cosine over equally grouped prompt frames."""
    samples = image_embeddings.shape[1]
    aligned_text = text_embeddings.repeat(samples // len(text_embeddings), 1)
    return -(image_embeddings * aligned_text[None]).sum(-1).mean(-1)


class Encoder:
    def __init__(self, name, device="cuda", mixed_precision=True):
        import open_clip
        from huggingface_hub import hf_hub_download

        repository, self.model_revision = MODELS[name]
        self.name = name
        self.device = torch.device(device)
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        variant = "MobileCLIP2-S0" if name.endswith("s0") else "MobileCLIP2-S2"
        checkpoint = hf_hub_download(
            repository, f"mobileclip2_{name[-2:]}.pt", revision=self.model_revision
        )
        digest = hashlib.sha256()
        with open(checkpoint, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.checkpoint_sha256 = digest.hexdigest()

        model, _, _ = open_clip.create_model_and_transforms(
            variant, pretrained=checkpoint, image_mean=(0, 0, 0), image_std=(1, 1, 1)
        )
        # MobileCLIP's published inference conversion fuses its blocks.
        model = copy.deepcopy(model.eval())
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        self.model = model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(variant)
        self.mean = torch.tensor((0, 0, 0), device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor((1, 1, 1), device=self.device).view(1, 3, 1, 1)
        self.image_size = 256

    def _autocast(self):
        return (
            torch.autocast("cuda", dtype=torch.float16)
            if self.mixed_precision
            else nullcontext()
        )

    @torch.inference_mode()
    def text(self, prompts):
        tokens = self.tokenizer(prompts).to(self.device)
        with self._autocast():
            features = self.model.encode_text(tokens)
        return F.normalize(features.float(), dim=-1)

    @torch.inference_mode()
    def images(self, images, batch_size=64):
        outputs = []
        for chunk in images.split(batch_size):
            pixels = chunk.to(self.device).permute(0, 3, 1, 2)
            if pixels.shape[-2:] != (self.image_size, self.image_size):
                pixels = F.interpolate(
                    pixels, size=(self.image_size, self.image_size), mode="nearest"
                )
            pixels = (pixels - self.mean) / self.std
            with self._autocast():
                features = self.model.encode_image(pixels)
            outputs.append(F.normalize(features.float(), dim=-1))
        return torch.cat(outputs)
