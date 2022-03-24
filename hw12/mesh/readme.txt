Each folder here represents a mesh containing:

--- nodes.csv ---
A comma sperated list of floats that contains a list of possible endpoints for any triangle in the mesh.

--- elements.csv ---
A comma seperated list of integers i1,i2,i3 that forms a single triangle of the mesh with edges nodes[i1],nodes[i2],nodes[i3].

--- faces.csv ---
A comma sperated list of integers i1,i2,bc where nodes[i1],nodes[i2] is one element of the polygon that forms the outside and bc is a identifier, e.g the index for $\Gamma_{bc}$.
