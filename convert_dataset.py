import os
import shutil
import pandas as pd

# Define paths
base_dir = "/home/prajyotbhagat/my_projects/my_projects/Football_AI/football-players-detection.v1i.tensorflow"
output_dir = "/home/prajyotbhagat/my_projects/my_projects/Football_AI/yolo_dataset"

splits = ['train', 'valid', 'test']

# Class mapping based on _annotations.csv
class_mapping = {
    'player': 0,
    'referee': 1,
    'goalkeeper': 2,
    'ball': 3
}

# Create output directories
for split in splits:
    out_split = 'val' if split == 'valid' else split
    os.makedirs(os.path.join(output_dir, 'images', out_split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', out_split), exist_ok=True)

def convert_to_yolo(df, img_width, img_height):
    """Converts bounding box coordinates to YOLO format."""
    x_center = (df['xmin'] + df['xmax']) / 2 / img_width
    y_center = (df['ymin'] + df['ymax']) / 2 / img_height
    width = (df['xmax'] - df['xmin']) / img_width
    height = (df['ymax'] - df['ymin']) / img_height
    return pd.DataFrame({
        'class_id': df['class'].map(class_mapping),
        'x_center': x_center,
        'y_center': y_center,
        'width': width,
        'height': height
    })

print("Starting conversion...")

for split in splits:
    split_dir = os.path.join(base_dir, split)
    csv_file = os.path.join(split_dir, '_annotations.csv')
    
    if not os.path.exists(csv_file):
        print(f"Skipping {split}, no _annotations.csv found.")
        continue
        
    df = pd.read_csv(csv_file)
    
    # Process each image
    grouped = df.groupby('filename')
    
    # Use valid dir for yolo (yolo expects 'val' for validation by default convention, but 'valid' is fine if specified in yaml)
    out_split = 'val' if split == 'valid' else split
    
    for filename, group in grouped:
        # Copy image
        src_img = os.path.join(split_dir, filename)
        dst_img = os.path.join(output_dir, 'images', out_split, filename)
        
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f"Warning: Image {src_img} not found!")
            
        # Convert coordinates
        # Assumes all bounding boxes for an image have the same image width and height
        img_width = group['width'].iloc[0]
        img_height = group['height'].iloc[0]
        
        yolo_df = convert_to_yolo(group, img_width, img_height)
        
        # Write to txt
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        dst_txt = os.path.join(output_dir, 'labels', out_split, txt_filename)
        
        yolo_df.to_csv(dst_txt, sep=' ', index=False, header=False)

print("Conversion completed!")

# Generate data.yaml
yaml_content = f"""path: {output_dir}
train: images/train
val: images/val
test: images/test

nc: 4
names: ['player', 'referee', 'goalkeeper', 'ball']
"""

with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

print("data.yaml created!")
