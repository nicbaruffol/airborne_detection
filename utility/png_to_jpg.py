import os
import glob
from PIL import Image
import concurrent.futures
from tqdm import tqdm  # Gives us a beautiful progress bar!

def convert_to_jpeg(png_path):
    try:
        # Check if the PNG actually exists (in case a previous run deleted it)
        if not os.path.exists(png_path):
            return False
            
        jpg_path = png_path.rsplit('.', 1)[0] + '.jpg'
        
        # Open and save (JPEG doesn't support alpha channels)
        with Image.open(png_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.save(jpg_path, 'JPEG', quality=95)
        
        # Delete original
        os.remove(png_path)
        return True
    except Exception as e:
        # We don't want 1 bad file to crash the whole pool
        return False

if __name__ == '__main__':
    dataset_root = '/cluster/scratch/nbaruffol/raw_videos/rgb_fixed_1_ground_truth' 
    
    print("Finding all PNG images... (This might take 10-20 minutes on a network drive)")
    # Using iglob creates an iterator instead of a massive list, saving RAM!
    search_pattern = os.path.join(dataset_root, 'Images', 'part*', '*', '*.png')
    
    # We still need a list to feed the executor, but we can build it safely
    png_files = list(glob.glob(search_pattern))
    total_files = len(png_files)
    
    print(f"Found {total_files} PNG files. Starting parallel conversion...")
    
    if total_files > 0:
        successful = 0
        # Process images in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # chunksize=100 tells each CPU core to grab 100 files at a time, 
            # drastically reducing inter-process communication overhead!
            results = executor.map(convert_to_jpeg, png_files, chunksize=100)
            
            # Wrap the results in tqdm to get a live progress bar
            for r in tqdm(results, total=total_files, desc="Converting"):
                if r:
                    successful += 1
                    
        print(f"Done! Successfully converted {successful} out of {total_files} images to JPEG.")
    else:
        print("No PNG files found! They might have already been converted.")