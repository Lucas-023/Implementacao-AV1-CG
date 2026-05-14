import torch
import numpy as np
from Base.Cam import Camera
from Base.Material import SimpleMaterial
from Primitives.base import HitRecord
from Primitives.Triangle import Triangle
from Base.Lights import PointLight
from ObjTotri import ObjToTri
from ObjTotri import Triangle_Soup
from Primitives.Plane import Plane
from BaseScene import Scene

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Certifique-se de que o ateneav.obj está na mesma pasta
Triangles_atena = Triangle_Soup(
    ObjToTri("OBJ_S/ateneav.obj", device=DEVICE), device=DEVICE
)

class Scene:
    def __init__(
        self, device=DEVICE, material_name=None, object_size=1.0, rotate_cam=0.0
    ):

        self.depth_max = 3
        self.background = [0.15, 0.15, 0.18]
        # Aumentei um pouco a luz ambiente para evitar sombras totalmente negras
        self.ambient_light = [0.25, 0.25, 0.25] 

        self.camera = Camera(
            eye=torch.tensor([6000.0, 2000.0, 500.0], device=DEVICE),
            look_at=torch.tensor([-1500.0, 2000.0, 500.0], device=DEVICE),
            up=torch.tensor([0.0, -1.0, 0.0], device=DEVICE),
            fov=45,
            img_width=800,
            img_height=600,
            device=DEVICE,
        )

        # --- NOVAS LUZES (Esquema de 3 pontos corrigido) ---
        self.lights = [
            # 1. LUZ PRINCIPAL (Key Light): Posicionada à frente (X=3000), acima do rosto (Y=3500)
            PointLight(pos=[3000.0, 3500.0, 500.0], color=[1.0, 0.95, 0.85], intensity=1.0),
            
            # 2. LUZ DE PREENCHIMENTO (Fill Light): De lado para suavizar as sombras do rosto
            PointLight(pos=[2000.0, 2000.0, 3000.0], color=[0.7, 0.8, 1.0], intensity=0.5),
            
            # 3. LUZ DE RECORTE (Rim Light): Fica atrás da estátua (X=-4000) para dar volume e destacar a silhueta contra o fundo
            PointLight(pos=[-4000.0, 4000.0, -2000.0], color=[1.0, 1.0, 1.0], intensity=0.7),
        ]

        # Material simulando mármore
        material_triangulo = SimpleMaterial(
            ambient_coefficient=0.8,
            diffuse_coefficient=0.85,
            diffuse_color=[0.92, 0.92, 0.90],
            specular_coefficient=0.3,
            specular_color=[1.0, 1.0, 1.0],
            specular_shininess=32,
            reflectivity=0.0,
        )

        material_plane = SimpleMaterial(
            ambient_coefficient=0.2,
            diffuse_coefficient=0.9,
            diffuse_color=[0.2, 0.2, 0.22],
            specular_coefficient=0.4,
            specular_color=[0.8, 0.8, 0.8],
            specular_shininess=16,
            reflectivity=0.1,
        )

        plane = Plane(normal=[0.0, 1.0, 0.0], point=[0.0, -100.0, 0.0], device=DEVICE)

        self.objects = []
        for tri in Triangles_atena.triangles:
            tri.material_id = 1
            self.objects.append(tri)
        self.objects.append(plane)
        plane.material_id = 2
        self.materials = [material_triangulo, material_plane]

    def hit(self, ray):
        N = ray.ori.shape[0]
        device = ray.ori.device

        best_t = torch.full((N,), float("inf"), device=device)
        best_mat_id = torch.zeros((N,), dtype=torch.long, device=device)
        best_points = torch.zeros((N, 3), device=device)
        best_normals = torch.zeros((N, 3), device=device)

        for shape in self.objects:
            current_hit = shape.hit(ray)
            hit_mask_bool = current_hit.hit_mask.bool().to(device)
            current_t = current_hit.t.to(device)

            is_closer = hit_mask_bool & (current_t < best_t)

            best_t = torch.where(is_closer, current_t, best_t)
            mat_id_tensor = torch.tensor(
                shape.material_id, device=device, dtype=torch.long
            )
            best_mat_id = torch.where(is_closer, mat_id_tensor, best_mat_id)

            is_closer_xyz = is_closer.unsqueeze(1)
            best_points = torch.where(
                is_closer_xyz, current_hit.point.to(device), best_points
            )
            best_normals = torch.where(
                is_closer_xyz, current_hit.normal.to(device), best_normals
            )

        return HitRecord(best_mat_id, best_t, best_points, best_normals, None)