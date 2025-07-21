import os
import numpy as np

# def off_to_npz(off_path, npz_path):
#     with open(off_path, 'r') as f:
#         if f.readline().strip() != 'OFF':
#             raise ValueError(f"Not a valid OFF file. -> {off_path}")
#         n_verts, n_faces, _ = map(int, f.readline().strip().split())
#         verts = np.array([list(map(float, f.readline().split())) for _ in range(n_verts)])
#         faces = np.array([list(map(int, f.readline().split()[1:])) for _ in range(n_faces)])  # [1:] skip face size
#     np.savez(npz_path, vertices=verts, faces=faces)

def off_to_npz(off_path, npz_path):
    with open(off_path, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.startswith('OFF'):
            raise ValueError(f"Not a valid OFF file. -> {off_path}")
        
        parts = first_line.split()
        if len(parts) > 1:
            n_verts, n_faces, _ = map(int, parts[1:4])
        else:
            n_verts, n_faces, _ = map(int, f.readline().strip().split())
        
        verts = np.array([list(map(float, f.readline().split())) for _ in range(n_verts)])
        faces = np.array([list(map(int, f.readline().split()[1:])) for _ in range(n_faces)])
    
    np.savez(npz_path, vertices=verts, faces=faces)

import numpy as np

def obj_to_npz(obj_path, npz_path):
    vertices = []
    faces = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Vertex
                vertices.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith('f '):  # Face
                face = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                faces.append(face)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    np.savez_compressed(npz_path, vertices=vertices, faces=faces)


dataset_folder = os.path.join('dataset', 'Manifold40')
preprocessed_ds_folder = os.path.join('dataset', 'Manifold40_preprocessed')


for categorie in os.listdir(dataset_folder):
    print(f'Processing {Categorie}...')
    for train_test in os.listdir(os.path.join(dataset_folder, categorie)):
        for off_file in os.listdir(os.path.join(dataset_folder, categorie, train_test)):
            if off_file.endswith('.obj'):
                off_path = os.path.join(os.path.join(dataset_folder, categorie, train_test, off_file))
                npz_path = os.path.join(os.path.join(preprocessed_ds_folder, categorie, train_test, off_file))[:-4] + '.npz'
                if not os.path.exists(npz_path):
                    os.makedirs(os.path.join(os.path.join(preprocessed_ds_folder, categorie, train_test)), exist_ok=True)
                    obj_to_npz(off_path, npz_path)

