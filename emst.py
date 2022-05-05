from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd

def computeMST(nd_points):
    distance_matrix = pairwise_distances(nd_points)
    mst = minimum_spanning_tree(distance_matrix)
    mst_cords = mst.tocoo()
    mst_df = pd.DataFrame({'src':mst_cords.row, 'dst':mst_cords.col, 'dist':mst_cords.data})
    mst_df = mst_df.sort_values(by='dist').reset_index(drop = True)
    for i in range(len(mst_df)):
        if mst_df['src'][i] > mst_df['dst'][i]:
            mst_df.iloc[0,i], mst_df[1,i] = mst_df[1,i], mst_df[0,i]
    return mst_df

if __name__ == "__main__":
    import numpy as np
    np.random.seed(10)
    POINTS_ARRAY = np.random.randint(1, 15, size = (6,3))
    print(computeMST(POINTS_ARRAY))



