import torch
from Base.Cam import Camera
from Primitives.base import HitRecord


class Scene:
    def __init__(self, materials, objects, object_size=1.0, rotate_cam=0.0):

        self.background = [0, 0, 0]
        self.ambient_light = [0, 0, 0]

        self.Camera = None

        self.lights = []

        self.objects = objects

        self.materials = materials

    def hit(self, ray):

        N = ray.ori.shape[0]
        device = ray.ori.device

        best_t = torch.full((N,), float("inf"), device=device)
        best_mask = torch.zeros((N,), dtype=torch.long, device=device)
        best_points = torch.zeros((N, 3), device=device)
        best_normals = torch.zeros((N, 3), device=device)
        for _, shape in enumerate(self.objects):
            current_hit = shape.hit(ray)
            hit_mask_bool = current_hit.hit_mask.bool().to(device)
            current_t = current_hit.t.to(device)

            is_closer = hit_mask_bool & (current_t < best_t)

            best_t = torch.where(is_closer, current_t, best_t)
            best_mask = torch.where(
                is_closer, torch.tensor(shape.material_id, device=device), best_mask
            )

            is_closer_xyz = is_closer.unsqueeze(1)
            best_points = torch.where(
                is_closer_xyz, current_hit.point.to(device), best_points
            )
            best_normals = torch.where(
                is_closer_xyz, current_hit.normal.to(device), best_normals
            )

        return HitRecord(best_mask, best_t, best_points, best_normals, None)
