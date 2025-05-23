import torch
import torch.nn.functional as F

def project_token_embeds(vector_to_project, target_vector, rho):
    # L2 norm projection
    # scaled_vector = vector_to_project * rho / torch.norm(vector_to_project, p=2, dim=1) * torch.norm(target_vector, p=2, dim=1)

    assert len(vector_to_project.shape) == 1, f"vector_to_project should be a 1D tensor but got {vector_to_project.shape}"
    assert vector_to_project.shape[0] == target_vector.shape[0], f"vector_to_project and target_vector should have the same dimension"
    assert type(target_vector) == torch.Tensor, f"target_vector should be a tensor"

    # Compute the L2 norm of the vector
    l2_norm = torch.norm(vector_to_project, p=2, dim=0)
    target_l2_norm = torch.norm(target_vector, p=2, dim=0)
    # Project the vector onto the unit sphere
    unit_vector = vector_to_project / l2_norm
    # Scale the vector by rho
    scaled_vector = rho * unit_vector * target_l2_norm
    return scaled_vector


def slerp(vector_to_project, target_vector, rho):
    """
    Spherical linear interpolation between two vectors.
    
    Args:
        vector_to_project: torch.Tensor (1D), the source vector
        target_vector: torch.Tensor (1D), the destination vector
        rho: float, interpolation factor [0, 1]
        rho = 0 --> vector_to_project is unchanged
        rho = 1 --> vector_to_project is the target_vector
        
    Returns:
        torch.Tensor: Interpolated vector (1D)
    """
    assert vector_to_project.ndim == 1, "vector_to_project must be a 1D tensor"
    assert target_vector.ndim == 1, "target_vector must be a 1D tensor"
    assert vector_to_project.shape == target_vector.shape, "Vectors must have the same shape"
    assert rho >= 0.0 and rho <= 1.0, "rho must be between 0 and 1"
    
    # Normalize input vectors
    v0 = torch.nn.functional.normalize(vector_to_project, dim=0)
    v1 = torch.nn.functional.normalize(target_vector, dim=0)

    # Compute cosine of angle and clamp
    dot = torch.clamp(torch.dot(v0, v1), -1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # If angle is small, fallback to linear interpolation
    if sin_theta < 1e-6:
        return (1 - rho) * vector_to_project + rho * target_vector

    # Compute slerp
    interp = (torch.sin((1 - rho) * theta) / sin_theta) * vector_to_project + \
            (torch.sin(rho * theta) / sin_theta) * target_vector
    return interp



def slerp_batched(vector_to_project: torch.Tensor, target_vector: torch.Tensor, rho: float) -> torch.Tensor:
    """
    Batched Slerp for inputs of shape [batch_size, num_tokens, embedding_dim].

    Args:
        vector_to_project (Tensor): Source tensor of shape [B, N, D]
        target_vector (Tensor): Target tensor of shape [B, N, D]
        rho (float): Interpolation factor (0.0 to 1.0)

    Returns:
        Tensor: Interpolated tensor of shape [B, N, D]
    """
    assert vector_to_project.shape == target_vector.shape, "Input tensors must have the same shape"
    assert vector_to_project.ndim == 3, "Inputs must be 3D: [batch_size, num_tokens, embedding_dim]"

    # Normalize input vectors along embedding dimension
    v0 = F.normalize(vector_to_project, dim=-1)
    v1 = F.normalize(target_vector, dim=-1)

    # Compute cosine similarity between corresponding vectors
    dot = (v0 * v1).sum(dim=-1, keepdim=True)  # shape [B, N, 1]
    dot = torch.clamp(dot, -1.0, 1.0)

    # Compute angle and its sine
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Fallback to lerp for small angles
    small_angle_mask = sin_theta.abs() < 1e-6
    lerp_result = (1 - rho) * vector_to_project + rho * target_vector

    # Apply Slerp formula
    slerp_result = (
        torch.sin((1 - rho) * theta) / sin_theta * vector_to_project +
        torch.sin(rho * theta) / sin_theta * target_vector
    )

    # Blend based on angle size
    output = torch.where(small_angle_mask, lerp_result, slerp_result)
    return output

def adjust_norm_and_slerp(vector_to_project, target_vector, rho, alpha):
    # alpha is the scaling factor to adjust the ratio of the norm of the vector to the target vector
    # alpha = 0 --> norm of the vector is unchanged
    # alpha = 1 --> norm of the vector is the norm of the target vector
    # rho is the rotation factor to adjust the direction of the vector in slerp
    # rho = 0 --> vector_to_project is unchanged
    # rho = 1 --> vector_to_project is the target_vector

    vector_norm = torch.norm(vector_to_project, p=2, dim=0)
    target_norm = torch.norm(target_vector, p=2, dim=0)

    # assert alpha >= 0.0 and alpha <= 1.0, "alpha must be between 0 and 1"
    assert rho >= 0.0 and rho <= 1.0, "rho must be between 0 and 1"
    assert vector_norm.shape == torch.Size([]), f"vector_norm should be a 1D tensor but got {vector_norm.shape}"
    assert target_norm.shape == torch.Size([]), f"target_norm should be a 1D tensor but got {target_norm.shape}"

    adjusted_vector = ((1 - alpha) * vector_norm + alpha * target_norm ) * vector_to_project / vector_norm # alpha = 0 --> norm of the vector is unchanged, alpha = 1 --> norm of the vector is the norm of the target vector
    adjusted_target = ((1 - alpha) * vector_norm + alpha * target_norm ) * target_vector / target_norm # change to the same norm as the vector

    return slerp(adjusted_vector, adjusted_target, rho)

def adjust_norm_and_slerp_3d(vector_to_project, target_vector, rho, alpha):
    # vector_to_project is a 3D tensor, [batch_size, num_tokens, embedding_dim]
    # target_vector is a 3D tensor, [batch_size, num_tokens, embedding_dim]
    # rho is a float - the interpolation factor [0, 1] of the slerp
    # alpha is a float - the scaling factor to adjust the ratio of the norm of the vector to the target vector
    # alpha = 0 --> norm of the vector is unchanged
    # alpha = 1 --> norm of the vector is the norm of the target vector
    # rho is the rotation factor to adjust the direction of the vector in slerp
    # rho = 0 --> vector_to_project is unchanged
    # rho = 1 --> vector_to_project is the target_vector

    
    assert vector_to_project.ndim == 3, f"vector_to_project should be a 3D tensor but got {vector_to_project.shape}"
    assert target_vector.ndim == 3, f"target_vector should be a 3D tensor but got {target_vector.shape}"
    assert vector_to_project.shape[1] == target_vector.shape[1], f"vector_to_project and target_vector should have the same dimension"
    assert vector_to_project.shape[2] == target_vector.shape[2], f"vector_to_project and target_vector should have the same dimension"

    # duplicate the target vector to match the batch size of the vector to project
    # target_vector = torch.repeat_interleave(target_vector, vector_to_project.shape[0], dim=0)

    assert vector_to_project.shape[0] == target_vector.shape[0], f"vector_to_project and target_vector should have the same batch size"
    
    vector_norm = torch.norm(vector_to_project, p=2, dim=2)
    target_norm = torch.norm(target_vector, p=2, dim=2)

    # assert alpha >= 0.0 and alpha <= 1.0, "alpha must be between 0 and 1"
    assert rho >= 0.0 and rho <= 1.0, "rho must be between 0 and 1"
    assert vector_norm.ndim == 2, f"vector_norm should be a 2D tensor but got {vector_norm.shape}"
    assert target_norm.ndim == 2, f"target_norm should be a 2D tensor but got {target_norm.shape}"

    vector_norm = vector_norm.unsqueeze(2) # [batch_size, num_tokens, 1]
    target_norm = target_norm.unsqueeze(2) # [batch_size, num_tokens, 1]

    adjusted_vector = ((1 - alpha) * vector_norm + alpha * target_norm ) * vector_to_project / vector_norm # alpha = 0 --> norm of the vector is unchanged, alpha = 1 --> norm of the vector is the norm of the target vector
    adjusted_target = ((1 - alpha) * vector_norm + alpha * target_norm ) * target_vector / target_norm # change to the same norm as the vector
    
    return slerp_batched(adjusted_vector, adjusted_target, rho)

    

