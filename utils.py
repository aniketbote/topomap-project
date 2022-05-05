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