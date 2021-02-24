from scipy.spatial import SphericalVoronoi,ConvexHull,cKDTree,Delaunay
import numpy as np
import networkx as nx
import warnings
from matplotlib.tri.triangulation import Triangulation
from operator import itemgetter
from scipy.optimize import linear_sum_assignment


def matchKITsystems(chanposA,chanposB):
    """
    returns idxes such that
    chanposA==chanposB[matching]
    returns:
    siteBchannels  -> one to one match
    correspondings -> greedy match not one-to-one
    """
    sensFeature = 'chanpos'

    print('Distance between channels without positional matching:\n',np.sqrt(np.sum((chanposB-chanposA)**2,axis=-1)).sum())
    ch_idxA = np.arange(160)
    ch_idxB = np.arange(160)
    correspondings = np.zeros(160)
    comp_dist = []
    for j in ch_idxA:
        dists=[]
        for i in ch_idxB:
            distance = np.sum((chanposA[j]-chanposB[i])**2)
            dists+=[np.sqrt(distance)]
        min_dist = np.min(dists)
        arg_min  = np.argmin(dists)
        correspondings[j]=arg_min
        comp_dist+=[dists]
    correspondings = np.array(correspondings).astype('int')
    print('Greedy min distance matching is not a one-to-one matching: ')
    print('channel correspondings: (Position denotes the ch_idx in A and value denotes the ch_idx in B)\n',correspondings)
    print('Greedy matching has non unique correspondences: len(np.unique(correspondings))!=160 but:',len(np.unique(correspondings)))
    print('Distance between channels with greedy matching: ', np.linalg.norm(chanposA-chanposB[correspondings]))
    comp_dist = np.array(comp_dist).reshape(160,160)
    
    print('Create one-to-one matching with bipartite node matching (Hungarian Algorithm).')
    siteAchannels, siteBchannels  = linear_sum_assignment(comp_dist)
    print('Distance between channels with positional matching:\n',np.sum(comp_dist[siteAchannels,siteBchannels]))
    return siteBchannels,correspondings

def edge_attr_dist(graph):
    dist_keys = []
    dist_vals = []
    for n1 in np.arange(160):
        for n2 in np.arange(n1+1,160):
            dist_keys+=[(n1,n2)]
            dist_vals+=[np.sqrt(np.sum((graph.nodes[n1]['sensloc']-graph.nodes[n2]['sensloc'])**2))]
    return dict(zip(dist_keys,dist_vals))
  
def get_sensorGraph(triangles):
    G = nx.Graph()
    for path in triangles:
        path = np.append(path, path[0])
        nx.add_path(G, path)
    return G
def set_node_attribute(graph,pts,attr_name,copy=False):
    if copy:
        raise NotImplementedError("copy has not been implemented")
    nx.set_node_attributes(graph,dict(zip(np.arange(len(pts)),pts)),name=attr_name)
    
def set_edge_attribute(graph,edge_attr,attr_name,copy=False):
    if copy:
        raise NotImplementedError("copy has not been implemented")
    nx.set_edge_attributes(graph,edge_attr,name=attr_name)
    
def triangles_spherical(points,radius,center,threshold=1e-6):
    """
    params:
    points: sensor locations projected on a sphere
    copy pasted from:
    https://github.com/scipy/scipy/blob/v1.5.4/scipy/spatial/_spherical_voronoi.py#L37-L345
    """
    if radius is None:
        radius = 1.
        warnings.warn('`radius` is `None`. '
                      'This will raise an error in a future version. '
                      'Please provide a floating point number '
                      '(i.e. `radius=1`).',
                      DeprecationWarning)

    radius = float(radius)
    points = np.array(points).astype(np.double)
    _dim = len(points[0])
    if center is None:
        center = np.zeros(_dim)
    else:
        center = np.array(center, dtype=float)

    # test degenerate input
    _rank = np.linalg.matrix_rank(points - points[0],
                                       tol=threshold * radius)
    if _rank < _dim:
        raise ValueError("Rank of input points must be at least {0}".format(_dim))

    if cKDTree(points).query_pairs(threshold * radius):
        raise ValueError("Duplicate generators present.")

    radii = np.linalg.norm(points - center, axis=1)
    max_discrepancy = np.abs(radii - radius).max()
    if max_discrepancy >= threshold * radius:
        raise ValueError("Radius inconsistent with generators.")

    
    conv = ConvexHull(points)
    # get circumcenters of Convex Hull triangles from facet equations
    # for 3D input circumcenters will have shape: (2N-4, 3)
    vertices = radius * conv.equations[:, :-1] + center
    simplices = conv.simplices
    return simplices

def triangles_xy_surface(pts):
    tri = Triangulation(pts[:,0],pts[:,1])
    return tri.get_masked_triangles()

def triangles_xyz_Delaunay(pts):
    return Delaunay(pts).simplices


def plot_Spherical_Voronoi(R,radius,center):
    sv = SphericalVoronoi(R, radius, center)
    # sort vertices (optional, helpful for plotting)
    sv.sort_vertices_of_regions()
    t_vals = np.linspace(0, 1, 2000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))*radius-center[0]
    y = np.outer(np.sin(u), np.sin(v))*radius-center[1]
    z = np.outer(np.ones(np.size(u)), np.cos(v))*radius+center[2]

    ax.plot_surface(x, y, z, color='y', alpha=0.1)
    ax.scatter(R[:,0],R[:,1],R[:,2])

    # plot Voronoi vertices
    #ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
    #                   c='g')
    # indicate Voronoi regions (as Euclidean polygons)
    for region in sv.regions:
        n = len(region)
        for i in range(n):
            start = sv.vertices[region][i]
            end = sv.vertices[region][(i + 1) % n]
            #result = geometric_slerp(start, end, t_vals) # spherical interpolated edges
            result = np.stack([start,end]) # direct edges
            ax.plot(result[..., 0],
                   result[..., 1],
                   result[..., 2],
                   c='k')
            #ax.plot(start[])
    ax.azim = 10
    ax.elev = 40
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    _ = ax.set_zticks([])
    fig.set_size_inches(4, 4)
    plt.show()
    
def project_to_sphere(pts,radius,center):
    """
    params: 
    something like
    radius = 5
    center = np.array([0,0,-10])

    https://stackoverflow.com/questions/9604132/how-to-project-a-point-on-to-a-sphere
    For the simplest projection (along the line connecting the point to the center of the sphere):
    Write the point in a coordinate system centered at the center of the sphere (x0,y0,z0):

    P = (x',y',z') = (x - x0, y - y0, z - z0)

    Compute the length of this vector:
    |P| = sqrt(x'^2 + y'^2 + z'^2)

    Scale the vector so that it has length equal to the radius of the sphere:
    Q = (radius/|P|)*P

    And change back to your original coordinate system to get the projection:
    R = Q + (x0,y0,z0)
    """ 
    P = pts-center
    Pnorm = np.expand_dims(np.linalg.norm(P,axis=-1),axis=-1)
    Q = (P*radius/Pnorm)
    R = Q + center    
    return R
    
def scatter_on_sphere(R,radius,center):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))*radius-center[0]
    y = np.outer(np.sin(u), np.sin(v))*radius-center[1]
    z = np.outer(np.ones(np.size(u)), np.cos(v))*radius+center[2]

    ax.plot_surface(x, y, z, color='y', alpha=0.1)
    ax.scatter(R[:,0],R[:,1],R[:,2])
    
    

def befriend_sensors(graph,radius):
    """
    The 2D triangulation is sometimes bad. 
    If two nodes don't have an edge, but have a distance < radius -> make an edge
    
    params:
    --------
    radius:
    """
    pass

def get_distance_distribution(graph):
    distances = []
    for node in np.arange(160):
        neighbors = np.array(list(graph.neighbors(node)))
        distances += [from_graph_get_neighbor_distances(graph,node,neighbors)]
    return distances

def _fit_data_format(node_pos,neighbor_positions):
    """
    node_pos, array shape [1,3] or [3]
    neighbor_positions, array w/ shape [N_neighbors,3]
    returns: array w/ shape [N_neighbors,3]
    """
    node_pos = np.array(node_pos)
    neighbor_positions = np.array(neighbor_positions)
    if len(node_pos.shape)==1:
        node_pos = np.expand_dims(node_pos,axis=0)
    assert node_pos.shape[1]==neighbor_positions.shape[1]
    return node_pos,neighbor_positions

def get_distance_to_neighbors(node_pos,neighbor_positions):
    """
    node_pos, array shape [1,3] or [3]
    neighbor_positions, array shape [N_neighbors,3]
    -----
    returns: array w/ shape [N_neighbors]
    """
    node_pos,neighbor_positions = _fit_data_format(node_pos,neighbor_positions)
    return np.sqrt(np.sum(get_displacement_to_neighbors(node_pos,neighbor_positions)**2,axis=-1))

def get_displacement_to_neighbors(node_pos,neighbor_positions):
    """
    returns: array [N_neighbors,3]
    """
    node_pos,neighbor_positions = _fit_data_format(node_pos,neighbor_positions)
    return node_pos-neighbor_positions

def from_graph_get_displacement_to_neighbors(graph,node,neighbors=[]):
    if len(neighbors)==0:
        neighbors  = np.array(list(graph.neighbors(node)))
    node_pos = graph.nodes[node]['sensloc']
    # itemgetter needs unpacked elements 
    neighbor_positions = np.array(itemgetter(*neighbors)(graph.nodes('sensloc')))
    return get_displacement_to_neighbors(node_pos,neighbor_positions)

def from_graph_get_neighbor_distances(graph,node,neighbors=[]):
    if len(neighbors)==0:
        neighbors  = np.array(list(graph.neighbors(node)))
    node_pos = graph.nodes[node]['sensloc']
    # itemgetter needs unpacked elements 
    neighbor_positions = np.array(itemgetter(*neighbors)(graph.nodes('sensloc')))
    distances = get_distance_to_neighbors(node_pos,neighbor_positions)
    return distances

def remove_neighbor(graph,node,neighbors_to_remove,copy=True):
    """
    removes an edge
    """
    if len(neighbors_to_remove)==0:
        return graph
    if copy:
        reduced_graph = graph.copy()
        for neighbor in neighbors_to_remove:
            reduced_graph.remove_edge(node,neighbor)
        return reduced_graph
    else:
        for neighbor in neighbors_to_remove:
            reduced_graph.remove_edge(node,neighbor)
            
def remove_long_distance_neighbor(graph,node,th_dist,copy=True):
    """
    graph: networkx object, with node_attribute 'sensloc'
    param node: int, idx of the node
    param th_dist: threshold, neighbors that are further away get chopped off
    -------
    return: reduced_graph
    """
    neighbors = np.array(list(graph.neighbors(node)))
    distances = from_graph_get_neighbor_distances(graph,node,neighbors)
    
    neighbors_to_remove = neighbors[np.where(distances>th_dist)]

    return remove_neighbor(graph,node,neighbors_to_remove,copy)

def get_edge_similarity(node_pos,neighbor_positions):
    """
    useful for finding approximate colinear neighbors.
    """
    displacements = get_displacement_to_neighbors(node_pos,neighbor_positions)
    n_neighbors = neighbor_positions.shape[0]
    # Quick and dirty, can reduce computation by factor 2.
    similarity = []
    for d1 in displacements:
        for d2 in displacements:
            similarity+=[np.sum(d1*d2)/(np.linalg.norm(d1)*np.linalg.norm(d2))]
    similarity = np.array(similarity).reshape(n_neighbors,n_neighbors)
    return similarity

def from_graph_get_edge_similarity(graph,node,neighbors=[]):
    """
    useful for finding approximate colinear neighbors.
    """
    if len(neighbors)==0:
        neighbors  = np.array(list(graph.neighbors(node)))
    node_pos = graph.nodes[node]['sensloc']
    # itemgetter needs unpacked elements 
    neighbor_positions = np.array(itemgetter(*neighbors)(graph.nodes('sensloc')))
    
    similarity = get_edge_similarity(node_pos,neighbor_positions)
    return similarity

def remove_colinear_neighbor(graph,node,th_max_colinearity = 0.99,copy=True):
    """
    Remove neighbor if they have almost overlapping edges. (Longer edges will be removed.)
    ------
    params:
    graph: networkx object, with node_attribute 'sensloc'
    param node: int, idx of the central node
    param th_max_colinearity: threshold, of two neighbors that have edges with high colinearity
        the neighbor that is further away is removed.
    ------
    details:
    A similarity matrix is computed, where the entries describe how much 
    the direction of node to neighbor_i is similiar to the direction of node to neighbor_j.
    The neighbors which h

    -------
    return: reduced_graph
    """
    neighbors  = np.array(list(graph.neighbors(node)))
    similarity = from_graph_get_edge_similarity(graph,node,neighbors)
    node_pairs_idx = np.stack(np.where(np.tril(similarity,-1)>th_max_colinearity)).T
    node_pairs = neighbors[node_pairs_idx]
    nodes_w_sim_edges = np.unique(node_pairs)
    
    print('n_wsim_ed ',nodes_w_sim_edges)
    new_graph = graph
    while len(nodes_w_sim_edges)>0:
        # sort nodes by distance from central node
        tmp = np.argsort(from_graph_get_neighbor_distances(new_graph,node,nodes_w_sim_edges))[::-1]
        # remove all but the closest neighbor
        neighbors_to_remove = [nodes_w_sim_edges[tmp][:-1][0]]
        print(neighbors_to_remove)
        new_graph = remove_neighbor(new_graph,node,neighbors_to_remove,copy)
    
    
        neighbors  = np.array(list(new_graph.neighbors(node)))
        similarity = from_graph_get_edge_similarity(new_graph,node,neighbors)
        node_pairs_idx = np.stack(np.where(np.tril(similarity,-1)>th_max_colinearity)).T
        node_pairs = neighbors[node_pairs_idx]
        nodes_w_sim_edges = np.unique(node_pairs)
    return new_graph
def reduceGraph(graph,max_dist=90,max_colinearity=.99,num_nodes=160):
    # todo: remove num_nodes and interfere from graph
    reduced_graph = graph
    for node in np.arange(num_nodes):
        print('Current node: ',node)
        reduced_graph = remove_long_distance_neighbor(reduced_graph,node,th_dist = max_dist,copy=True)
        reduced_col = remove_colinear_neighbor(reduced_graph,node,th_max_colinearity = max_colinearity,copy=True)
        print('Before Filtering: ', np.array(list(graph.neighbors(node))))
        print('Long Dist Filtering: ', np.array(list(reduced_graph.neighbors(node))))
        print('Colinearity Filtering: ',np.array(list(reduced_col.neighbors(node)))),
        reduced_graph = reduced_col
    return reduced_graph
    
def get_Degree_of_nodes(graph,nodes):
    """
    get_agreeDegree(GA,agreeing_channel_assignments_A)
    """
    degrees = []
    for k in nodes:
        degrees+=[graph.degree[k]]
    return np.array(degrees)