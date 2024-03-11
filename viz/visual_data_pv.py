from collections import deque
from dataclasses import dataclass

import numpy as np
from vispy import scene
from vispy.scene import ViewBox
from vispy.visuals import PlaneVisual
from vispy.visuals.transforms import STTransform, MatrixTransform
from xml.etree import ElementTree as ET

import pyvista as pv


class BeatSaberVisualDataContainer:

    def __init__(self, view: ViewBox):
        self.view = view
        plane = scene.visuals.Plane(
            width=3,
            height=3,
            color=(0.5, 0.5, 0.5, 1),
            parent=view.scene
        )
        self.body_spheres = []
        body_colors = [(1, 0, 0, 0.2), (0, 1, 0, 0.2), (0, 0, 1, 0.2)]
        self.shadow_ellipses = []
        self.saber_lines = scene.visuals.Line(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            parent=view.scene,
        )
        self.shadow_lines = scene.visuals.Line(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            parent=plane,
        )
        self.shadow_lines.set_gl_state(depth_test=False)
        for i in range(3):
            body_sphere = scene.visuals.Sphere(
                color=body_colors[i],
                edge_color=(0, 0, 0, 0),
                parent=view.scene,
                radius=0.1,
            )
            body_sphere.transform = STTransform(scale=(1.0, 1.0, 1.0), translate=(0, 0, 0))
            self.body_spheres.append(body_sphere)

            shadow_ellipse = scene.visuals.Ellipse(
                center=np.array([0, 0, 0]),
                radius=(0.1, 0.1),
                color=(0, 0, 0, 0.5),
                parent=plane,
            )
            self.shadow_ellipses.append(shadow_ellipse)
            shadow_ellipse.set_gl_state(depth_test=False)

        # Render order
        for body_sphere in self.body_spheres:
            body_sphere.order = 0
        plane.order = 1


class VisualDataContainer:

    def __init__(self, view: ViewBox):
        self.view = view
        plane = scene.visuals.Plane(
            width=6,
            height=6,
            color=(0.5, 0.5, 0.5, 1),
            parent=view.scene
        )
        # shadow_image = scene.visuals.Image(
        #     parent=plane,
        # )
        # shadow_canvas = scene.canvas.SceneCanvas(show=False, )
        # shadow_view = scene.widgets.ViewBox()
        self.body_markers = []
        self.body_lines = []
        self.sparse_markers = scene.visuals.Markers(
            pos=np.array([[0, 0, 0]]),
            parent=view.scene,
            size=0.1,
            edge_width=0.01,
            scaling=True,
        )
        # self.axis = scene.visuals.XYZAxis(parent=view.scene)
        self.sparse_axes = []
        for i in range(3):
            axis = scene.visuals.XYZAxis(
                # width=(i + 1) * 2,
                width=1,
                parent=view.scene
            )
            axis.transform = MatrixTransform(matrix=np.eye(4))
            self.sparse_axes.append(axis)

        self.shadow_ellipses = []
        self.shadow_markers = []
        for i in range(3):
            body_markers = scene.visuals.Markers(
                pos=np.array([[0, 0, 0]]),
                parent=view.scene,
                size=0.1,
                edge_width=0.01,
                scaling="visual",
                # spherical=True,
            )
            # body_markers.set_gl_state(depth_test=False)
            self.body_markers.append(body_markers)
            body_lines = scene.visuals.Line(parent=view.scene)
            self.body_lines.append(body_lines)

            # shadow_markers = scene.visuals.Markers(
            #     parent=view.scene,
            #     size=0.1,
            #     edge_width=0.01,
            #     scaling=True,
            # )
            # shadow_markers.set_gl_state(depth_test=False)
            # self.shadow_markers.append(shadow_markers)

        for i in range(3):
            shadow_ellipse = scene.visuals.Ellipse(
                center=np.array([0, 0, 0]),
                radius=(0.1, 0.1),
                color=(0, 0, 0, 0.5),
                parent=plane,
                num_segments=12,
            )
            self.shadow_ellipses.append(shadow_ellipse)
            shadow_ellipse.set_gl_state(depth_test=False)

        # Render order
        # i = 0
        # # for ax in self.sparse_axes:
        # #     ax.order = i
        # # i += 1
        # self.body_markers[2].order = i
        # i += 1
        # self.body_markers[1].order = i
        # i += 1
        # self.body_markers[0].order = i
        # i += 1
        # plane.order = i
        # # i += 1
        # # self.shadow_markers[2].order = i
        # # # i += 1
        # # self.shadow_markers[1].order = i
        # # i += 1
        # # self.shadow_markers[0].order = i
        # # for shadow_ellipse in self.shadow_ellipses:
        # #     shadow_ellipse.order = 1


class XMLVisualDataContainer:

    def __init__(self, xml_path: str):
        tree = ET.parse(xml_path)
        worldbody_tag = tree.find("worldbody")
        self.geoms = []  # sorted in the DFS encounter order
        self.offsets = []

        # A body has multiple geom children optionally.
        body_stack = deque()
        for body_tag in worldbody_tag.findall("body")[::-1]:
            body_stack.append((None, body_tag, np.zeros(3)))

        while body_stack:
            parent, child, global_parent_offset = body_stack.pop()
            local_child_offset = np.array(child.attrib.get("pos", "0 0 0").split(), dtype=float)
            global_child_offset = global_parent_offset + local_child_offset

            for geom in child.findall("geom")[::-1]:
                self.geoms.append(geom)
                geom_pos = np.array(geom.attrib.get("pos", "0 0 0").split(), dtype=float)
                self.offsets.append(local_child_offset)

            for grandchild in child.findall("body")[::-1]:
                body_stack.append((child, grandchild, global_child_offset))

        # self.plane = pv.Cube(center=(0, 0, -0.05), x_length=5, y_length=5, z_length=0.1)
        self.plane = pv.Plane(center=(0, 0, 0), i_size=10, j_size=10)

        # Following the DFS encounter order, create PyVista meshes
        self.meshes = []
        self.axes = []
        self.centers = []
        for geom_tag, offset in zip(self.geoms, self.offsets):
            geom_pos = np.array(geom_tag.attrib.get("pos", "0 0 0").split(), dtype=float)

            if geom_tag.attrib["type"] == "sphere":
                size = float(geom_tag.attrib["size"])
                center = np.zeros(3)
                mesh = pv.Sphere(radius=size, center=geom_pos)
                self.centers.append(center)

            elif geom_tag.attrib["type"] == "box":
                x_length, y_length, z_length = [float(k) * 2 for k in geom_tag.attrib["size"].split()]
                center = np.zeros(3)
                mesh = pv.Cube(center=geom_pos, x_length=x_length, y_length=y_length, z_length=z_length)
                self.centers.append(center)

            elif geom_tag.attrib["type"] == "capsule":
                size = float(geom_tag.attrib["size"])
                fromto = np.array([float(k) for k in geom_tag.attrib["fromto"].split()])
                fro = fromto[:3]
                to = fromto[3:]

                direction = to - fro
                length = np.linalg.norm(direction)
                direction /= length
                center = direction * 0.5 * length
                mesh = pv.Arrow(start=fro, direction=direction, scale=length, shaft_radius=size)

                self.centers.append(center)
            else:
                raise NotImplementedError
            self.meshes.append(mesh)
            ax = pv.Axes()
            self.axes.append(ax)

        self.targets = []
        for _ in range(24):
            sphere = pv.Sphere(radius=0.05)
            self.targets.append(sphere)
