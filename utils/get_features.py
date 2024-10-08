import torch
import numpy as np
import torch.nn.functional as F

def extract_cams(model, dataloader, target_layer):

    # Storage for feature maps
    features = None

    # Hook function to capture features
    def hook_fn(module, input, output):
        nonlocal features
        features = output

    # Register the hook
    handle = target_layer.register_forward_hook(hook_fn)

    # List to store the CAMs
    cam_features = []

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for data in dataloader:
            # Move inputs to the specified device
            inputs = data.to(device)

            # Forward pass through the model
            outputs = model(inputs)

            # Get the predicted class index
            predicted_class = torch.argmax(outputs, dim=1)

            # Compute gradients of the output w.r.t. the captured features
            grads = torch.autograd.grad(outputs[:, predicted_class], features, grad_outputs=torch.ones_like(outputs))[0]

            # Calculate the weights for each feature map
            weights = torch.mean(grads, dim=(2, 3))

            # Compute the CAM by weighting the feature maps
            cam = torch.sum(weights[:, :, None, None] * features, dim=1)

            # Store the CAM result as a NumPy array
            cam_features.append(cam.cpu().detach().numpy())

            # Clean up to free memory
            torch.cuda.empty_cache()
            del inputs, outputs, predicted_class, grads, weights, cam

    # Remove the hook to clean up
    handle.remove()

    return cam_features
