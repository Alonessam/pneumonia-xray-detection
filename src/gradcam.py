from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from .config import DEFAULT_MODELS_DIR, DEFAULT_OUTPUTS_DIR, IMAGENET_MEAN, IMAGENET_STD
from .models import build_model, get_gradcam_target_layer
from .utils import ensure_dir, get_device, load_checkpoint


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(probs.argmax(dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM aktivasyonları üretilemedi.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_tensor = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam_tensor = F.interpolate(cam_tensor, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_tensor = cam_tensor.squeeze()
        cam_tensor -= cam_tensor.min()
        cam_tensor /= cam_tensor.max().clamp_min(1e-8)
        return cam_tensor.cpu().numpy(), class_idx, probs.detach().cpu().numpy()[0]

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def preprocess_image(image: Image.Image, img_size: int, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    image = image.convert("RGB")
    display_image = image.resize((img_size, img_size))
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, display_image


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    image = image.convert("RGB")
    colored = matplotlib.colormaps["jet"](heatmap)[..., :3]
    colored = Image.fromarray(np.uint8(colored * 255)).resize(image.size)
    return Image.blend(image, colored, alpha=alpha)


def load_model_for_gradcam(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model(
        checkpoint["model_name"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODELS_DIR / "best_resnet50_finetuned.pt")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    device = get_device()
    model, checkpoint = load_model_for_gradcam(args.checkpoint, device)
    input_tensor, display_image = preprocess_image(Image.open(args.image), checkpoint["img_size"], device)

    gradcam = GradCAM(model, get_gradcam_target_layer(model, checkpoint["model_name"]))
    heatmap, class_idx, probs = gradcam(input_tensor, class_idx=args.class_idx)
    gradcam.close()

    overlay = overlay_heatmap(display_image, heatmap)
    out_path = args.out_dir / f"gradcam_{args.image.stem}.png"
    overlay.save(out_path)
    class_names = checkpoint["class_names"]
    print(f"Prediction: {class_names[class_idx]} | probability={probs[class_idx]:.4f}")
    print(f"Grad-CAM saved to: {out_path}")


if __name__ == "__main__":
    main()
