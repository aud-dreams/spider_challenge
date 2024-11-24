import os
import SimpleITK as stik

def load_mha_files(directory):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mha')])
    arrays = [stik.GetArrayFromImage(stik.ReadImage(f)) for f in files]
    return arrays