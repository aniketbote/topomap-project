# Parallelized Topomap

Topomap introduces a novel way for projecting multidimensional
data. There are many visual projection techniques designed map the
data. Many measures of similarity and measures of distances are
considered while designing these techniques. TopoMap maps highdimensional data to a visual space while preserving 0-homology
(Betti-0) topological persistence. This is defined by Rips filtration
over a set of points. Topomap uses the Euclidean distance as a
measure to map the data from high dimensional space. However the
compuations involved in Topomap over the exact Euclidean distance
can be time consuming.
Hence our goal is to improve upon the time complexity involved
during the formation of minimum spanning tree in Topomap using
parallelized Euclidean Minimum Spanning Tree. Considering the
work-depth approach, where work can be defined as computation
of instructions and depth as total computation in a single sequence,
parallelizing can be done in W/p+D. This approach is based on
generating well separated pairs using well separated pair decomposition and then computing the minimum spanning tree using Kruskal’s
algorithm and bichromatic closest pairs.

# System Requirements

OS - Linux  
Python - 3.8  

# Usage

`example.py` contains minimal example that uses the implementation

