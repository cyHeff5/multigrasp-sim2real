from __future__ import annotations

import pybullet as p


_COLOR_SPHERE        = [0.8, 0.4, 0.1, 1.0]
_COLOR_CUBE          = [0.2, 0.6, 0.8, 1.0]
_COLOR_CYLINDER      = [0.4, 0.8, 0.2, 1.0]
_COLOR_RECT_CYLINDER = [0.8, 0.2, 0.4, 1.0]


class GraspObject:

    def __init__(self, object_id: int, spec: dict, client_id: int = 0) -> None:
        self.object_id  = int(object_id)
        self._cid       = int(client_id)
        self.shape: str      = spec["shape"]
        self.mass_kg: float  = float(spec["mass_kg"])

    @classmethod
    def spawn(
        cls,
        spec:            dict,
        pedestal_height: float,
        spawn_xy:        list[float],
        client_id:       int = 0,
    ) -> "GraspObject":
        # z wird so gesetzt, dass die Unterseite des Objekts bündig auf dem Podest liegt.
        shape = spec["shape"]

        if shape == "sphere":
            radius = spec["size_cm"] / 200.0
            z      = pedestal_height + radius
            col    = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
            vis    = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                          rgbaColor=_COLOR_SPHERE, physicsClientId=client_id)

        elif shape == "cube":
            half = spec["size_cm"] / 200.0
            z    = pedestal_height + half
            col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3, physicsClientId=client_id)
            vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3,
                                        rgbaColor=_COLOR_CUBE, physicsClientId=client_id)

        elif shape == "cylinder":
            radius = spec["thickness_cm"] / 200.0
            half_h = spec["height_cm"]    / 200.0
            z      = pedestal_height + half_h
            col    = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius,
                                             height=spec["height_cm"] / 100.0,
                                             physicsClientId=client_id)
            vis    = p.createVisualShape(p.GEOM_CYLINDER, radius=radius,
                                          length=spec["height_cm"] / 100.0,
                                          rgbaColor=_COLOR_CYLINDER, physicsClientId=client_id)

        elif shape == "rect_cylinder":
            half_t = spec["thickness_cm"] / 200.0
            half_w = spec["width_cm"]     / 200.0
            half_h = spec["height_cm"]    / 200.0
            z      = pedestal_height + half_h
            col    = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_t, half_w, half_h],
                                             physicsClientId=client_id)
            vis    = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_t, half_w, half_h],
                                          rgbaColor=_COLOR_RECT_CYLINDER, physicsClientId=client_id)

        elif shape == "urdf":
            # spawn_z_offset_m wird per AABB in eval_sim.py berechnet.
            z         = pedestal_height + float(spec.get("spawn_z_offset_m", 0.05))
            if "base_orientation_xyzw" in spec:
                obj_quat = [float(v) for v in spec["base_orientation_xyzw"]]
            else:
                obj_quat = p.getQuaternionFromEuler([0.0, 0.0, spec.get("yaw_rad", 0.0)])
            object_id = p.loadURDF(str(spec["urdf_path"]),
                                    basePosition=[spawn_xy[0], spawn_xy[1], z],
                                    baseOrientation=obj_quat,
                                    physicsClientId=client_id)
            p.changeDynamics(object_id, -1,
                              lateralFriction=spec["lateral_friction"],
                              rollingFriction=0.01,
                              spinningFriction=0.01,
                              physicsClientId=client_id)
            return cls(object_id, spec, client_id)  # loadURDF erstellt den Körper direkt, createMultiBody unten wird nicht gebraucht

        else:
            raise ValueError(f"Unknown shape: {shape!r}")

        obj_quat  = p.getQuaternionFromEuler([0.0, 0.0, spec["yaw_rad"]])
        object_id = p.createMultiBody(
            baseMass=spec["mass_kg"],
            baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[spawn_xy[0], spawn_xy[1], z],
            baseOrientation=obj_quat, physicsClientId=client_id,
        )
        p.changeDynamics(object_id, -1, lateralFriction=spec["lateral_friction"],
                          physicsClientId=client_id)
        return cls(object_id, spec, client_id)

    def position(self) -> list[float]:
        pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self._cid)
        return list(pos)

    def height(self) -> float:
        return self.position()[2]

    def orientation_quat(self) -> list[float]:
        _, quat = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self._cid)
        return list(quat)


def apply_pedestal_magnet(
    obj_id:             int,
    pedestal_center_xy: list[float],
    k:                  float,
    client_id:          int = 0,
) -> None:
    # Federkraft in Richtung Podest-Mittelpunkt - simuliert den echten Magneten
    # auf der Plattform, der verhindert dass Kugeln wegrollen. Nur XY, kein Z.
    pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=client_id)
    fx = k * (pedestal_center_xy[0] - pos[0])
    fy = k * (pedestal_center_xy[1] - pos[1])
    p.applyExternalForce(obj_id, -1, [fx, fy, 0.0], pos, p.WORLD_FRAME,
                          physicsClientId=client_id)
