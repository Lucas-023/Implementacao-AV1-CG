from Primitives.Triangle import Triangle
import numpy as np
from collections import defaultdict
import torch
from Primitives.base import *


class Triangle_Soup:
    def __init__(self, triangles, device):
        self.num_tris = len(triangles)
        self.triangles = triangles

    def __str__(self):
        return f"Triangle_Soup com {self.num_tris} triângulos"


def ObjToTri(path, device):
    vertices = []
    faces = []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "v":
                vertices.append(np.array(list(map(float, parts[1:4]))))

            elif parts[0] == "f":
                #converte os índices
                indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]

                #se a face tiver 4 vértices divide em 2 triângulos
                if len(indices) == 3:
                    faces.append(indices)
                elif len(indices) == 4:
                    faces.append([indices[0], indices[1], indices[2]])
                    faces.append([indices[0], indices[2], indices[3]])

    #calculo das normais dos vertices
    vertex_normals = defaultdict(lambda: np.zeros(3))
    for f in faces:
        i1, i2, i3 = f
        v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]

        #vetor normal da face (sem normalizar ainda para depois ponderar pela área)
        n = np.cross(v2 - v1, v3 - v1)

        vertex_normals[i1] += n
        vertex_normals[i2] += n
        vertex_normals[i3] += n

    #normaliza as normais acumuladas
    for k in vertex_normals:
        norm = np.linalg.norm(vertex_normals[k])
        if norm > 1e-9:
            vertex_normals[k] /= norm

    # Criando triangulos
    triangles = []
    for f in faces:
        i1, i2, i3 = f

        v1_real = vertices[i1]
        v2_real = vertices[i2]
        v3_real = vertices[i3]

        tri = Triangle(
            [v1_real, v2_real, v3_real],
            vertex_normals=[vertex_normals[i1], vertex_normals[i2], vertex_normals[i3]],
            device=device,
        )
        triangles.append(tri)

    return triangles
