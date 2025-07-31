import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Use new API to avoid broken hooks
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, image_tensor, landmark_tensor, class_idx=None):
        self.model.eval()  # <-- Ensure eval mode
        self.model.zero_grad()

        # Forward pass
        output = self.model(image_tensor, landmark_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        # Backward pass
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        # Compute weights: global average pooling on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # shape: [B, C, 1, 1]

        # Compute CAM: weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # shape: [B, 1, H, W]
        cam = F.relu(cam)

        # Normalize & resize to 224x224
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
