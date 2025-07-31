import os
import glob
import re

def find_latest_checkpoint(trained_model_dir):
    """
    Find the latest checkpoint in the trained model directory.
    
    Args:
        trained_model_dir (str): Path to the trained model directory
        
    Returns:
        str: Latest checkpoint directory name (e.g., "checkpoint-1000") or None if not found
    """
    if not os.path.exists(trained_model_dir):
        print(f"Model directory not found: {trained_model_dir}")
        return None
    
    # Find all checkpoint directories
    checkpoint_pattern = os.path.join(trained_model_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        print(f"No checkpoints found in {trained_model_dir}")
        return None
    
    # Extract step numbers and find the latest
    latest_step = -1
    latest_checkpoint = None
    
    for checkpoint_dir in checkpoint_dirs:
        # Extract step number from checkpoint directory name
        match = re.search(r'checkpoint-(\d+)', os.path.basename(checkpoint_dir))
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_checkpoint = os.path.basename(checkpoint_dir)
    
    return latest_checkpoint

def list_available_checkpoints(trained_model_dir):
    """
    List all available checkpoints in the trained model directory.
    
    Args:
        trained_model_dir (str): Path to the trained model directory
        
    Returns:
        list: List of checkpoint directory names sorted by step number
    """
    if not os.path.exists(trained_model_dir):
        print(f"Model directory not found: {trained_model_dir}")
        return []
    
    # Find all checkpoint directories
    checkpoint_pattern = os.path.join(trained_model_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        print(f"No checkpoints found in {trained_model_dir}")
        return []
    
    # Extract step numbers and sort
    checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        match = re.search(r'checkpoint-(\d+)', os.path.basename(checkpoint_dir))
        if match:
            step = int(match.group(1))
            checkpoints.append((step, os.path.basename(checkpoint_dir)))
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    
    return [checkpoint[1] for checkpoint in checkpoints]

def validate_checkpoint(trained_model_dir, checkpoint_name):
    """
    Validate that a checkpoint contains the required files.
    
    Args:
        trained_model_dir (str): Path to the trained model directory
        checkpoint_name (str): Checkpoint directory name (e.g., "checkpoint-1000")
        
    Returns:
        dict: Status of required files
    """
    checkpoint_path = os.path.join(trained_model_dir, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        return {"valid": False, "error": f"Checkpoint directory not found: {checkpoint_path}"}
    
    required_files = [
        "transformer_lora.safetensors",
        "text_encoder_one_lora.safetensors"
    ]
    
    file_status = {}
    all_present = True
    
    for file_name in required_files:
        file_path = os.path.join(checkpoint_path, file_name)
        file_status[file_name] = os.path.exists(file_path)
        if not file_status[file_name]:
            all_present = False
    
    return {
        "valid": all_present,
        "checkpoint_path": checkpoint_path,
        "files": file_status
    }

if __name__ == "__main__":
    # Example usage
    trained_model_dir = "./models/subject_model_with_text_encoder"
    
    print("=== Checkpoint Finder ===")
    print(f"Searching in: {trained_model_dir}")
    print()
    
    # List all available checkpoints
    checkpoints = list_available_checkpoints(trained_model_dir)
    if checkpoints:
        print("Available checkpoints:")
        for checkpoint in checkpoints:
            print(f"  - {checkpoint}")
        print()
    
    # Find latest checkpoint
    latest = find_latest_checkpoint(trained_model_dir)
    if latest:
        print(f"Latest checkpoint: {latest}")
        
        # Validate the latest checkpoint
        validation = validate_checkpoint(trained_model_dir, latest)
        print(f"Validation: {'✓ Valid' if validation['valid'] else '✗ Invalid'}")
        
        if validation['valid']:
            print("Required files found:")
            for file_name, exists in validation['files'].items():
                status = "✓" if exists else "✗"
                print(f"  {status} {file_name}")
        else:
            print("Missing files:")
            for file_name, exists in validation['files'].items():
                if not exists:
                    print(f"  ✗ {file_name}")
    else:
        print("No checkpoints found") 