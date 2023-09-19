import numpy as np
np.float = float # for compatibility with old vedo
import vedo as vd

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)

# from vedo 2023.4.6
def intersect_with_plane(mesh, origin=(0, 0, 0), normal=(1, 0, 0)):
    """
    Intersect this Mesh with a plane to return a set of lines.

    Example:
        ```python
        from vedo import *
        sph = Sphere()
        mi = sph.clone().intersect_with_plane().join()
        print(mi.lines())
        show(sph, mi, axes=1).close()
        ```
        ![](https://vedo.embl.es/images/feats/intersect_plane.png)
    """
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    cutter = vtk.vtkPolyDataPlaneCutter()
    cutter.SetInputData(mesh.polydata())
    cutter.SetPlane(plane)
    cutter.InterpolateAttributesOn()
    cutter.ComputeNormalsOff()
    cutter.Update()

    msh = vd.Mesh(cutter.GetOutput(), "k", 1).lighting("off")
    msh.GetProperty().SetLineWidth(3)
    msh.name = "PlaneIntersection"
    return msh

import numpy as np
class Plane:
    def __init__(self, origin, u, v) -> None:
        self.o = origin
        self.u = u / np.linalg.norm(u)
        self.v = v / np.linalg.norm(v)
        assert np.isclose(np.dot(self.u, self.v), 0), f"The plane vectors must be orthonormal to each other (u ⋅ v = {np.dot(self.u, self.v)})"
        self.n = np.cross(self.u, self.v)
        self.M = np.vstack([u, v]).T
    def P3toP2(self, ps):
        return (ps - self.o) @ self.M
    # alternative to get_structures_slice_coords()
    def get_projections(self, actors):
        projected = {}
        for actor in actors:
            mesh = actor._mesh
            intersection = intersect_with_plane(mesh, origin=self.o, normal=self.n) #.c('purple5')
            # intersection = mesh.intersect_with_plane(origin=self.o, normal=self.n) #.c('purple5')
            # slices = [s.triangulate() for s in intersection.join_segments()]
            # vd.show(mesh, intersection, vd.merge(slices), axes=1, viewup='z')
            if not intersection.points().shape[0]:
                # print(actor.name+": no intersection --- BoundingBox"+str(actor._mesh.GetBounds()))
                continue
            pieces = intersection.splitByConnectivity()
            # pieces = intersection.split()
            for piece_n, piece in enumerate(pieces):
                # sort coordinates
                points = piece.join(reset=True).points()
                projected[actor.name + f"_segment_{piece_n}"] = self.P3toP2(points)
            # vd.show(mesh, intersection, vd.merge(slices), axes=1, viewup='z')
        return projected


if __name__ == "__main__":
    import brainrender as br
    atlas = br.Atlas(atlas_name="allen_mouse_25um")
    root = atlas.get_region("root", alpha=1, color=None)
    major_divisions = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH", "HY", "MB", "P", "MY", "CB"]
    regions_meshes = atlas.get_region(*major_divisions, alpha=1, color=None)
    if not isinstance(regions_meshes, list):
        regions_meshes = [regions_meshes]

    for actor in (root, *regions_meshes):
        # from Render._prepare_actor()
        mtx = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        actor._mesh = actor.mesh.clone()
        actor._mesh.applyTransform(mtx)
        # actor._mesh.apply_transform(mtx)
        actor._is_transformed = True

    regions_meshes = [r for r in regions_meshes if r.br_class == "brain region"]

    # CRASHES
    # import bgheatmaps as bgh
    # s = bgh.slicer.Slicer(6000, "frontal", 100, root)
    # s.get_structures_slice_coords(regions_meshes, root)

    # DOESN' CRASH
    p = Plane(root._mesh.centerOfMass(), np.array([0,0,1]), np.array([0,1,0]))
    p.get_projections([*regions_meshes, root])
