import time
import scipy
import numpy as np
import inspect

class Component:
    def __init__(self, vertices, polygon):
        self.vertices = vertices
        self.polygon = polygon

class NaiveDisjoinSet:
    def __init__(self, n_points):
        self.universal = [Component({i}, [(0,0), (0,0)]) for i in range(n_points)]
    def Find(self, a):
        for ele in self.universal:
            if a in ele.vertices:
                return ele
        raise Exception("Element Not Found")
    def Union(self, a, b, merged_hull):
        s1 = self.Find(a)
        s2 = self.Find(b)
        if s1.vertices == s2.vertices:
            raise Exception("Elements belong to same set")
        s3 = s1.vertices.union(s2.vertices)
        nc = Component(s3, merged_hull)
        self.universal.remove(s1)
        self.universal.remove(s2)
        self.universal.append(nc)


def customComputeConvexHull(points):
    points = np.array(points)
    convex_hull = []
    if points.shape[0] == 1:
        convex_hull.append(tuple(points[0]))
        convex_hull.append(tuple(points[0]))
    elif points.shape[0] ==2:
        convex_hull.append(tuple(points[0]))
        convex_hull.append(tuple(points[1]))
        convex_hull.append(tuple(points[0]))
    else:
        
        n_hull= list(scipy.spatial.ConvexHull(points).vertices)
        n_hull.append(n_hull[0])
        convex_hull = [tuple(points[i]) for i in n_hull]
    return convex_hull
        