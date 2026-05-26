import os

# Issue #6: Only accept known image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def load_samples(root_dir):
    samples = []
    gen_map = {}
    gen_counter = 0

    real_dir = os.path.join(root_dir, "real")
    if os.path.exists(real_dir):
        for fname in sorted(os.listdir(real_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue
            full_path = os.path.join(real_dir, fname)
            if not os.path.isfile(full_path):
                continue
            samples.append((full_path, 0, "real"))

    fake_dir = os.path.join(root_dir, "fake")
    if os.path.exists(fake_dir) and os.path.isdir(fake_dir):
        has_subdirs = any(os.path.isdir(os.path.join(fake_dir, d)) for d in os.listdir(fake_dir))
        if not has_subdirs:
            for fname in sorted(os.listdir(fake_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VALID_EXTENSIONS:
                    continue
                full_path = os.path.join(fake_dir, fname)
                if not os.path.isfile(full_path):
                    continue
                samples.append((full_path, 1, "unknown_fake"))
        else:
            for gen_name in sorted(os.listdir(fake_dir)):
                gen_path = os.path.join(fake_dir, gen_name)
                if not os.path.isdir(gen_path):
                    continue
                for fname in sorted(os.listdir(gen_path)):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in VALID_EXTENSIONS:
                        continue
                    full_path = os.path.join(gen_path, fname)
                    if not os.path.isfile(full_path):
                        continue
                    samples.append((full_path, 1, gen_name))
    else:
        for gen_name in sorted(os.listdir(root_dir)):
            if gen_name.lower() == "real":
                continue
            gen_path = os.path.join(root_dir, gen_name)
            if not os.path.isdir(gen_path):
                continue
            for fname in sorted(os.listdir(gen_path)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VALID_EXTENSIONS:
                    continue
                full_path = os.path.join(gen_path, fname)
                if not os.path.isfile(full_path):
                    continue
                samples.append((full_path, 1, gen_name))

    return samples