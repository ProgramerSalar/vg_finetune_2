import torch

# # Replace with the actual path to the file you uploaded
# file_path = "/content/video_generation_from_scratch/clip_video/Algebra Basics： Graphing On The Coordinate Plane - Math Antics [9Uc62CuQjc4]/videos/clip_0000.mp4.pt" 

def checker(file_path):

    try:
        data = torch.load(file_path, map_location="cpu")
        print(f"✅ File loaded successfully.")
        
        # Check if it's a dictionary or a direct tensor
        if isinstance(data, dict):
            print(f"Structure: Dictionary with keys: {data.keys()}")
            # Usually latents are stored under a key like 'latents' or just raw
            # Let's assume the first value is what we need if it's a dict
            tensor_data = list(data.values())[0]
        else:
            tensor_data = data
            
        print(f"Shape: {tensor_data.shape}")
        print(f"Dtype: {tensor_data.dtype}")
        
        # CRITICAL CHECKS
        has_nan = torch.isnan(tensor_data).any().item()
        has_inf = torch.isinf(tensor_data).any().item()
        
        print(f"\n--- SAFETY CHECK ---")
        print(f"Contains NaNs?  : {'❌ YES (BAD)' if has_nan else '✅ NO'}")
        print(f"Contains Infs?  : {'❌ YES (BAD)' if has_inf else '✅ NO'}")
        
        print(f"Min Value: {tensor_data.min().item()}")
        print(f"Max Value: {tensor_data.max().item()}")

        if has_nan or has_inf:
            print("\nCONCLUSION: This file is corrupted. You must delete it and regenerate your data.")
        else:
            print("\nCONCLUSION: The data file looks healthy. The NaN error is likely from the Model/Learning Rate, not the data.")

    except Exception as e:
        print(f"Error loading file: {e}")


if __name__ == "__main__":

    import glob, os
    from pathlib import Path
# /home/manish/Desktop/projects/video_generation_from_scratch/clip_video/Algebra Basics： Graphing On The Coordinate Plane - Math Antics [9Uc62CuQjc4]/videos/clip_0000.mp4.pt
    path = "/home/manish/Desktop/projects/video_generation_from_scratch/clip_video"
    for root, dirs, files in  os.walk(path):
        # path_check = glob.glob()
        for file in files:
            pt_file = Path(root+'/'+file)
            if pt_file.suffix.lower() == ".pt":
                checker(pt_file)

            
            