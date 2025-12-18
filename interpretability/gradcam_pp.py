# interpretability/gradcam_pp.py
# Grad-CAM++ for PyTorch models.
# Designed to work cleanly with MagniNet by default on model.swin_proj (4D feature map).

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def _normalize_cam(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # cam: [H,W] or [B,H,W]
    if cam.dim() == 2:
        cam_min, cam_max = cam.min(), cam.max()
        return (cam - cam_min) / (cam_max - cam_min + eps)
    cam_min = cam.amin(dim=(-2, -1), keepdim=True)
    cam_max = cam.amax(dim=(-2, -1), keepdim=True)
    return (cam - cam_min) / (cam_max - cam_min + eps)


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for a given model and target layer.

    Works best when the target layer produces 4D activations [B,C,H,W].
    For MagniNet, a good default is: model.swin_proj

    You can also pass any nn.Module layer as target_layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        device: str | torch.device = "cuda",
    ):
        self.model = model
        self.target_layer = target_layer
        self.device = torch.device(device)

        self._activations = None
        self._grads = None

        self._h_fwd = self.target_layer.register_forward_hook(self._forward_hook)
        self._h_bwd = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # out is activation tensor [B,C,H,W] (expected)
        self._activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] matches activations shape
        self._grads = grad_output[0].detach()

    def remove_hooks(self):
        self._h_fwd.remove()
        self._h_bwd.remove()

    @torch.no_grad()
    def _infer_target_layer_shape(self, x: torch.Tensor) -> tuple[int, int]:
        _ = self.model(x)
        if self._activations is None or self._activations.dim() != 4:
            raise RuntimeError("Target layer did not produce 4D activations. Choose a conv/feature map layer.")
        return int(self._activations.shape[-2]), int(self._activations.shape[-1])

    def generate(
        self,
        x: torch.Tensor,
        mode: str = "binary",
        class_index: int | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Generate Grad-CAM++ heatmap.

        Args:
            x: input tensor [B,3,H,W] in [0,1] range
            mode: "binary" or "multiclass"
            class_index: for multiclass: which class to explain; if None -> predicted class
            normalize: min-max normalize the CAM to [0,1]

        Returns:
            cam: [B,H,W] float tensor
        """
        self.model.eval()
        x = x.to(self.device)

        # forward
        logits_mc, logits_bin = self.model(x)

        if mode.lower() == "binary":
            # explain malignant score (logit)
            score = logits_bin  # [B]
        elif mode.lower() == "multiclass":
            if class_index is None:
                class_index = int(torch.argmax(logits_mc, dim=1)[0].item())
            score = logits_mc[:, class_index]  # [B]
        else:
            raise ValueError("mode must be 'binary' or 'multiclass'.")

        # backward
        self.model.zero_grad(set_to_none=True)
        score.sum().backward(retain_graph=False)

        A = self._activations  # [B,C,H,W]
        dYdA = self._grads     # [B,C,H,W]
        if A is None or dYdA is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Ensure target_layer is correct.")

        # Grad-CAM++ weights
        # alpha = d2 / (2*d2 + sum(A * d3))
        # weights = sum(alpha * relu(d1))
        # cam = relu(sum(weights * A))
        dYdA_2 = dYdA ** 2
        dYdA_3 = dYdA ** 3

        # sum over spatial dims
        sum_A_dYdA_3 = (A * dYdA_3).sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * dYdA_2 + sum_A_dYdA_3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))

        alpha = dYdA_2 / denom  # [B,C,H,W]

        relu_dYdA = F.relu(dYdA)
        weights = (alpha * relu_dYdA).sum(dim=(2, 3), keepdim=True)  # [B,C,1,1]

        cam = (weights * A).sum(dim=1)  # [B,H,W]
        cam = F.relu(cam)

        # upsample CAM to input size
        cam = cam.unsqueeze(1)  # [B,1,H,W]
        cam = F.interpolate(cam, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)  # [B,H,W]

        if normalize:
            cam = _normalize_cam(cam)

        return cam

    @staticmethod
    def overlay_on_image(
        image_rgb: np.ndarray,
        cam_01: np.ndarray,
        alpha: float = 0.35,
    ) -> np.ndarray:
        """
        Overlay CAM on an RGB image (both numpy arrays).

        Args:
            image_rgb: uint8 RGB image [H,W,3]
            cam_01: float CAM [H,W] in [0,1]
            alpha: blending factor

        Returns:
            overlay: uint8 RGB image [H,W,3]
        """
        import cv2

        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        cam_01 = np.clip(cam_01, 0.0, 1.0)

        heatmap = (cam_01 * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = (1 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay
