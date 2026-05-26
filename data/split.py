import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

def get_video_id(path, gen_id=None):
    """
    Extracts a unique video identity from a frame filename.
    
    In FF++, real/000_frame_0001.jpg and fake/Deepfakes/000_frame_0001.jpg
    share the same base video ID '000'. Including gen_id prevents merging
    real and fake frames of the same source video during aggregation.
    """
    basename = os.path.basename(path)
    if "_frame" in basename:
        base_vid = basename.split("_frame")[0]
    elif "_" in basename:
        base_vid = basename.rsplit("_", 1)[0]
    else:
        base_vid = os.path.splitext(basename)[0]
    
    if gen_id is not None:
        return f"{gen_id}_{base_vid}"
    # Fallback: use parent directory name to disambiguate
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}_{base_vid}"

def get_source_video_id(path):
    """
    Extracts the TARGET person identity from a frame filename.
    
    FF++ naming conventions:
      Real:           000_frame_0001.jpg       → target = "000"
      Deepfakes:      000_003_frame_0001.jpg   → target = "000" (003 is the source face)
      NeuralTextures: 000_003_frame_0001.jpg   → target = "000"
    
    The first number is ALWAYS the target identity (the person whose face appears).
    We extract only this to group all 6 versions (real + 5 fakes) of the same person.
    """
    basename = os.path.basename(path)
    # Strip the "_frame_NNNN.ext" suffix first for FF++
    if "_frame" in basename:
        base = basename.split("_frame")[0]  # "000" or "000_003"
        # Extract only the FIRST part (target identity)
        # "000" → "000", "000_003" → "000"
        return base.split("_")[0]
        
    # For independent image datasets (no "_frame" suffix), 
    # treat the entire filename (without extension) as the unique identity.
    return os.path.splitext(basename)[0]

def split_by_video_identity(samples, test_size=0.2, random_state=42):
    """
    Splits samples into train and val ensuring that ALL versions of the same
    source video (real + all 5 fakes) stay in the same split.
    
    This prevents identity leakage: the model cannot memorize a person's face
    from the real video in training and then be tested on a manipulated version
    of that same face in validation.
    
    Returns: train_samples, val_samples
    """
    # Group by SOURCE video ID (e.g. '000') — NOT by gen_id-prefixed ID.
    # This groups real/000 + Deepfakes/000 + Face2Face/000 + ... together.
    source_groups = defaultdict(list)
    for sample in samples:
        source_vid = get_source_video_id(sample[0])
        source_groups[source_vid].append(sample)
    
    source_vids = list(source_groups.keys())
    
    # No stratification needed — each source video contains exactly 1 real + 5 fakes,
    # so every split will be perfectly balanced by generator.
    train_source_vids, val_source_vids = train_test_split(
        source_vids, 
        test_size=test_size, 
        random_state=random_state, 
    )
    
    train_samples = []
    for vid in train_source_vids:
        train_samples.extend(source_groups[vid])
        
    val_samples = []
    for vid in val_source_vids:
        val_samples.extend(source_groups[vid])
    
    print(f"  Split by source identity: {len(source_vids)} source videos -> "
          f"{len(train_source_vids)} train, {len(val_source_vids)} val")
    print(f"  Train: {len(train_samples)} frames | Val: {len(val_samples)} frames")
        
    return train_samples, val_samples

def find_cross_split_duplicates(splits_dict):
    """Placeholder to maintain compatibility."""
    return []

def find_cross_split_near_duplicates(splits_dict, max_hamming_distance=1):
    """Placeholder to maintain compatibility."""
    return []
