import numpy as np
from geoutils import find_angle
try:
    from convhull.convhull import customComputeConvexHull
except:
    print("Using Scipy Convex Hull")
    from utils import customComputeConvexHull
from utils import NaiveDisjoinSet
from emst import computeMST
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")

def alignHull(hull, p1, top):
    v= -1
    d2 = np.Inf
    for i, p2 in enumerate(hull):
        d = np.linalg.norm(np.array(p1) - np.array(p2))
        if d < d2:
            v = i
            d2 = d
    
    if top:
        v1 = hull[v]
        v2 = hull[v+1]
    else:
        if v == 0:
            v = -1
        v1 = hull[v]
        v2 = hull[v-1]
        
    trans = {}
    trans['tx'] = -hull[v][0]
    trans['ty'] = -hull[v][1]

    if len(hull) > 2:
        trans = find_angle(v1, v2, trans)
        
    else:
        trans['sin'] = 0
        trans['cos'] = 1

    return trans


def transform(p, t, yoff):
    x = p[0] + t['tx']
    y = p[1] + t['ty']
    xx = x*t['cos'] - y*t['sin']
    yy = x*t['sin'] + y*t['cos']
    yy += yoff
    return (xx, yy)

def transform_component(c, t, yoff):
    for v in c.vertices:
        verts[v] = transform(verts[v], t, yoff)

def mergeComponents(c1_o,c2_o,v1,v2,length):
    c1 = c1_o.vertices
    c2 = c2_o.vertices
    merged = set(list(c1)+list(c2))
    
    if length > 0:
        t1 = alignHull(c1_o.polygon, verts[v1], True)
        transform_component(c1_o, t1, 0)
        t2 = alignHull(c2_o.polygon, verts[v2], False)
        transform_component(c2_o, t2, length)
        points = [list(verts[v]) for v in merged]
        n_hull= customComputeConvexHull(points)
    else:
        raise Exception("Same co-ordinates")
    return list(n_hull)

class TopoMap:
    def __init__(self):
        pass

    def compute_metrics(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            self.total_time = time.time() - start_time
            return result
        return wrapper

    @compute_metrics
    def transform(self, data_df):
        global verts
        nd_points = np.array(data_df)

        nd = NaiveDisjoinSet(len(nd_points))

        verts = {}
        for i in range(len(nd_points)):
            verts[i] = (0,0)

        mst_df = computeMST(nd_points)
            
        for i in range(len(mst_df)):
            p1 = mst_df["src"][i]
            p2 = mst_df["dst"][i]

            c1 = nd.Find(p1)
            c2 = nd.Find(p2)
            
            if c1.vertices == c2.vertices :
                raise Exception("Error")
            hull = mergeComponents(c1,c2,p1,p2,mst_df["dist"][i])
            nd.Union(p1, p2, hull)
            
        t_points = np.array([list(e) for e in verts.values()])
        out_df = pd.DataFrame(t_points, columns=['x','y'])
        self._tpoints = t_points
        return out_df
    
    def plot(self, y = None, save_path = None):
        sns.scatterplot(x = self._tpoints[:,0], y = self._tpoints[:,1], hue = y, style=y)
        if save_path is not None:
            plt.savefig(save_path, dpi = 300)
        plt.show()



    


if __name__ == "__main__":
    from validate import validate_points
    data = pd.read_csv("data\iris.csv")
    x = data.iloc[:,:3]
    y = data.iloc[:,-1]
    T = TopoMap()
    out = T.transform(x)
    # print(out)
    # print(T._tpoints)
    print(T.total_time)
    # T.plot(y, 'img.png')
    x = np.array(x)
    tx = T._tpoints
    print(validate_points(x, tx))
    # out.to_csv("output_topomap.csv", index = False)
            
    



        
        
        
        
        
        