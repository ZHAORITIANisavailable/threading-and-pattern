import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from RESNETEE import *





def get_flops(model, input_tensor):
    """Helper function to calculate FLOPs using thop."""
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    # FLOPs is approx. 2 * MACs
    return macs * 2, params

if __name__ == '__main__':
    # --- Setup ---
    INPUT_SIZE = (1, 3, 32, 32) # (Batch, Channels, Height, Width) for CIFAR-10
    NUM_CLASSES = 10
    
    # Create a dummy input tensor
    dummy_input = torch.randn(INPUT_SIZE).to('cuda')


    model = ResNet18AutoEarlyExitstemp()
    #model.load_state_dict(torch.load('model_with_exits_new.pth'))
    model.cuda()
    # model_with_exits.train()  # 默认开启训练模式以支持后续 retrain
    model.eval()  # 切换到评估模式

    print(f"--- FLOPs Analysis for ResNet18 with Early Exits ---")
    print(f"Input tensor shape: {dummy_input.shape}\n")
    
    # --- 1. Calculation by Path ---
    print("--- 1. FLOPs per Execution Path ---")

    # Define model segments for each path
    # Common trunk for all paths
    common_trunk_1 = nn.Sequential(
        model.conv1,
        model.bn1,
        nn.ReLU(),
        model.layer1,
        model.layer2
    )
    
    # Path 1: Common Trunk -> Early Exit 1
    exit1_branch = model.early_exit1

    # Path 2 requires running through layer3
    middle_trunk = model.layer3
    exit2_branch = model.early_exit2
    
    # Path 3 requires running through layer4 and final classifier
    final_trunk = nn.Sequential(
        model.layer4,
        model.avgpool,
        nn.Flatten(),
        model.fc
    )

    # Calculate FLOPs for each segment
    # Note: We need intermediate tensors to calculate FLOPs for later segments
    with torch.no_grad():
        out_common_trunk_1 = common_trunk_1(dummy_input)
        out_middle_trunk = middle_trunk(out_common_trunk_1)
        
    flops_common_trunk_1, _ = get_flops(common_trunk_1, dummy_input)
    flops_exit1, _ = get_flops(exit1_branch, out_common_trunk_1)
    
    flops_middle_trunk, _ = get_flops(middle_trunk, out_common_trunk_1)
    flops_exit2, _ = get_flops(exit2_branch, out_middle_trunk)

    flops_final_trunk, _ = get_flops(final_trunk, out_middle_trunk)

    # Calculate total FLOPs for each path
    # path1_total_flops = flops_common_trunk_1 + flops_exit1
    # path2_total_flops = flops_common_trunk_1 + flops_middle_trunk + flops_exit2
    # path3_total_flops = flops_common_trunk_1 + flops_middle_trunk + flops_final_trunk


    path1_total_flops = flops_common_trunk_1 + flops_exit1
    path2_total_flops = flops_common_trunk_1 + flops_exit1 + flops_middle_trunk + flops_exit2
    path3_total_flops = flops_common_trunk_1 + flops_middle_trunk + flops_final_trunk + flops_exit1 + flops_exit2
    print(f"Path 1 (Exit 1): {path1_total_flops / 1e9:.4f} GFLOPs")
    print(f"Path 2 (Exit 2): {path2_total_flops / 1e9:.4f} GFLOPs")
    print(f"Path 3 (Main Exit): {path3_total_flops / 1e9:.4f} GFLOPs\n")


    # --- 2. Calculation by Layer ---
    print("--- 2. FLOPs per Layer/Module (for Main Path) ---")
    
    # To analyze by layer, we can compose the full main path model
    main_path_model = nn.Sequential(
        common_trunk_1,
        middle_trunk,
        final_trunk
    )
    
    # Use thop's verbose mode for a layer-by-layer breakdown
    macs, params = profile(main_path_model, inputs=(dummy_input,), verbose=True)
    total_flops_main_path_verbose = macs * 2
    
    print(f"\nTotal FLOPs for Main Path (from verbose analysis): {total_flops_main_path_verbose / 1e9:.4f} GFLOPs")
    print(f"Total Parameters for Main Path: {params / 1e6:.2f} M")