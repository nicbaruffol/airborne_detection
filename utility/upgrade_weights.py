import torch

def upgrade_checkpoint_for_fusion(old_ckpt_path, new_ckpt_path):
    print(f"Loading original weights from: {old_ckpt_path}")
    
    # Added weights_only=False to silence the PyTorch warning
    checkpoint = torch.load(old_ckpt_path, map_location='cpu', weights_only=False)
    
    # The actual model weights are usually stored under 'model_state_dict'
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    layer_modified = False
    
    for layer_name, weight_tensor in state_dict.items():
        # Look for the first convolutional layer (4D tensor: [out_channels, in_channels, H, W])
        if len(weight_tensor.shape) == 4 and weight_tensor.shape[1] == 2:
            print(f"Found input layer: '{layer_name}'")
            print(f"Old shape: {weight_tensor.shape}")
            
            # Duplicate the weights for the new IR channels and divide by 2
            new_weight = torch.cat([weight_tensor / 2.0, weight_tensor / 2.0], dim=1)
            
            state_dict[layer_name] = new_weight
            print(f"New shape: {new_weight.shape}")
            
            layer_modified = True
            break # We only want to modify the very first input layer
            
    if not layer_modified:
        print("Error: Could not find a 2-channel input layer in this checkpoint.")
        return

    # Save the upgraded checkpoint
    checkpoint['model_state_dict'] = state_dict
    torch.save(checkpoint, new_ckpt_path)
    print(f"Success! Saved fused checkpoint to: {new_ckpt_path}")

if __name__ == "__main__":
    # Using absolute paths based on your terminal output
    OLD_WEIGHTS = "/cluster/home/nbaruffol/airborne_detection/output/checkpoints/120_hrnet32_all/0/2220.pt" 
    NEW_WEIGHTS = "/cluster/home/nbaruffol/airborne_detection/output/checkpoints/120_hrnet32_fused_init.pt" 
    
    upgrade_checkpoint_for_fusion(OLD_WEIGHTS, NEW_WEIGHTS)