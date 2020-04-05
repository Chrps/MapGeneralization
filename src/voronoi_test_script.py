import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.components.connected import connected_components
import random
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import scipy.spatial as spt
from matplotlib.path import Path
from matplotlib.pyplot import cm
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from src.sampler import GraphSampler
import time


def generate_graph():
    G = nx.Graph()

    positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0),
             (3, 1), (3, 2), (3, 3)]
    positions = [list(pos) for pos in positions]

    nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    edges = [(0, 4), (0, 1), (1, 5), (1, 2), (2, 6), (2, 3),
             (3, 7), (4, 8), (4, 5), (5, 9), (5, 6), (6, 10),
             (6, 7), (7, 11), (8, 12), (8, 9), (9, 13), (9, 10),
             (10, 14), (10, 11), (11, 15), (12, 13), (13, 14), (14, 15)]


    rand_int_1_x = random.randint(0, 2)
    rand_int_2_x = random.randint(2, 4)
    rand_int_3_x = random.randint(4, 6)
    rand_int_4_x = random.randint(6, 8)

    rand_int_1_y = random.randint(0, 2)
    rand_int_2_y = random.randint(2, 4)
    rand_int_3_y = random.randint(4, 6)
    rand_int_4_y = random.randint(6, 8)

    for node in nodes:
        # Add some randomness to x coordinates
        if positions[node][0] == 0:
            positions[node][0] += rand_int_1_x
        elif positions[node][0] == 1:
            positions[node][0] += rand_int_2_x
        elif positions[node][0] == 2:
            positions[node][0] += rand_int_3_x
        elif positions[node][0] == 3:
            positions[node][0] += rand_int_4_x
        # Add some randomness to y coordinates
        if positions[node][1] == 0:
            positions[node][1] += rand_int_1_y
        elif positions[node][1] == 1:
            positions[node][1] += rand_int_2_y
        elif positions[node][1] == 2:
            positions[node][1] += rand_int_3_y
        elif positions[node][1] == 3:
            positions[node][1] += rand_int_4_y
        G.add_node(node, pos=tuple(positions[node]), label=0)

    for edge in edges:
        G.add_edge(edge[0], edge[1])

    #num_diamonds = random.randint(2, 5)
    num_diamonds = 4
    # Get a random edge
    idxs = random.sample(range(0, (len(edges)-1)), num_diamonds)

    counter = 0
    for idx in idxs:
        G.remove_edge(edges[idx][0], edges[idx][1])

        # Drawing diamond
        node1 = edges[idx][0]
        node2 = edges[idx][1]
        pos1 = positions[node1]
        pos2 = positions[node2]
        center_point = ((pos1[0]+pos2[0])/2, (pos1[1]+pos2[1])/2)

        # Randomize diamond
        x_ran = (random.randint(2, 4))/10
        y_ran = (random.randint(2, 4))/10

        # If same x coordinate draw diamond certain way
        if pos1[0] == pos2[0]:
            new_node_1 = tuple((center_point[0], center_point[1] - y_ran))
            new_node_2 = tuple((center_point[0], center_point[1] + y_ran))
            new_node_3 = tuple((center_point[0] - x_ran, center_point[1]))
            new_node_4 = tuple((center_point[0] + x_ran, center_point[1]))
        else:
            new_node_1 = tuple((center_point[0] - x_ran, center_point[1]))
            new_node_2 = tuple((center_point[0] + x_ran, center_point[1]))
            new_node_3 = tuple((center_point[0], center_point[1] - y_ran))
            new_node_4 = tuple((center_point[0], center_point[1] + y_ran))

        node_name_1 = 16 + counter
        node_name_2 = 17 + counter
        node_name_3 = 18 + counter
        node_name_4 = 19 + counter

        G.add_node(node_name_1, pos=new_node_1, label=1)
        G.add_node(node_name_2, pos=new_node_2, label=1)
        G.add_node(node_name_3, pos=new_node_3, label=1)
        G.add_node(node_name_4, pos=new_node_4, label=1)

        G.add_edge(node1, node_name_1)
        G.add_edge(node_name_1, node_name_3)
        G.add_edge(node_name_1, node_name_4)
        G.add_edge(node_name_4, node_name_2)
        G.add_edge(node_name_3, node_name_2)
        G.add_edge(node_name_2, node2)
        counter += 4

    pos = nx.get_node_attributes(G, 'pos')

    return G, pos

def GetFirstPoint(dataset):
    ''' Returns index of first point, which has the lowest y value '''
    # todo: what if there is more than one point with lowest y?
    imin = np.argmin(dataset[:,1])
    return dataset[imin]

def GetNearestNeighbors(dataset, point, k):
    ''' Returns indices of k nearest neighbors of point in dataset'''
    # todo: experiment with leafsize for performance
    mytree = spt.cKDTree(dataset,leafsize=10)
    distances, indices = mytree.query(point,k)
    # todo: something strange here, we get more indices than points in dataset
    #       so have to do this
    return dataset[indices[:dataset.shape[0]]]

def SortByAngle(kNearestPoints, currentPoint, prevPoint):
    ''' Sorts the k nearest points given by angle '''
    angles = np.zeros(kNearestPoints.shape[0])
    i = 0
    for NearestPoint in kNearestPoints:
        # calculate the angle
        angle = np.arctan2(NearestPoint[1]-currentPoint[1],
                NearestPoint[0]-currentPoint[0]) - \
                np.arctan2(prevPoint[1]-currentPoint[1],
                prevPoint[0]-currentPoint[0])
        angle = np.rad2deg(angle)
        # only positive angles
        angle = np.mod(angle+360,360)
        #print NearestPoint[0], NearestPoint[1], angle
        angles[i] = angle
        i=i+1
    return kNearestPoints[np.argsort(angles)]

def plotPoints(dataset):
    plt.plot(dataset[:,0],dataset[:,1],'o',markersize=10,markerfacecolor='0.75',
            markeredgewidth=1)
    plt.axis('equal')
    plt.axis([min(dataset[:,0])-0.5,max(dataset[:,0])+0.5,min(dataset[:,1])-0.5,
        max(dataset[:,1])+0.5])
    plt.show()

def plotPath(dataset, path):
    plt.plot(dataset[:,0],dataset[:,1],'o',markersize=10,markerfacecolor='0.65',
            markeredgewidth=0)
    path = np.asarray(path)
    plt.plot(path[:,0],path[:,1],'o',markersize=10,markerfacecolor='0.55',
            markeredgewidth=0)
    plt.plot(path[:,0],path[:,1],'-',lw=1.4,color='k')
    plt.axis('equal')
    plt.axis([min(dataset[:,0])-0.5,max(dataset[:,0])+0.5,min(dataset[:,1])-0.5,
        max(dataset[:,1])+0.5])
    plt.axis('off')
    #plt.savefig('./doc/figure_1.png', bbox_inches='tight')
    plt.show()

def removePoint(dataset, point):
    delmask = [np.logical_or(dataset[:,0]!=point[0],dataset[:,1]!=point[1])]
    newdata = dataset[delmask]
    return newdata

def isPointRightOfLine(a,b,c):
    '''
    Check if a point (c) is right of a line (a-b).
    If (c) is on the line, it is not right it.
    '''
    # move to origin
    aTmp = (0,0)
    bTmp = (b[0] - a[0], b[1] - a[1])
    cTmp = (c[0] - a[0], c[1] - a[1])
    return np.cross(bTmp, cTmp) < 0

def isPointOnLine(a,b,c):
    '''
    Check if a point is on a line.
    '''
    # move to origin
    aTmp = (0,0)
    bTmp = (b[0] - a[0], b[1] - a[1])
    cTmp = (c[0] - a[0], c[1] - a[1])
    r = np.cross(bTmp, cTmp)
    return np.abs(r) < 0.0000000001

def lineSegmentTouchesOrCrossesLine(a,b,c,d):
    '''
    Check if line segment (a-b) touches or crosses
    line segment (c-d).
    '''
    return isPointOnLine(a,b,c) or \
           isPointOnLine(a,b,d) or \
          (isPointRightOfLine(a,b,c) ^
           isPointRightOfLine(a,b,d))

def doBoundingBoxesIntersect(a, b, c, d):
    '''
    Check if bounding boxes do intersect. If one bounding box touches
    the other, they do intersect.
    First segment is of points a and b, second of c and d.
    '''
    ll1_x = min(a[0],b[0]); ll2_x = min(c[0],d[0])
    ll1_y = min(a[1],b[1]); ll2_y = min(c[1],d[1])
    ur1_x = max(a[0],b[0]); ur2_x = max(c[0],d[0])
    ur1_y = max(a[1],b[1]); ur2_y = max(c[1],d[1])

    return ll1_x <= ur2_x and \
           ur1_x >= ll2_x and \
           ll1_y <= ur2_y and \
           ur1_y >= ll2_y

def doLinesIntersect(a,b,c,d):
    '''
    Check if line segments (a-b) and (c-d) intersect.
    '''
    return doBoundingBoxesIntersect(a,b,c,d) and \
           lineSegmentTouchesOrCrossesLine(a,b,c,d) and \
           lineSegmentTouchesOrCrossesLine(c,d,a,b)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def concaveHull(dataset, k):
    assert k >= 3, 'k has to be greater or equal to 3.'
    points = dataset
    # todo: remove duplicate points from dataset
    # todo: check if dataset consists of only 3 or less points
    # todo: make sure that enough points for a given k can be found

    firstpoint = GetFirstPoint(points)
    # init hull as list to easily append stuff
    hull = []
    # add first point to hull
    hull.append(firstpoint)
    # and remove it from dataset
    points = removePoint(points,firstpoint)
    currentPoint = firstpoint
    # set prevPoint to a Point righ of currentpoint (angle=0)
    prevPoint = (currentPoint[0]+10, currentPoint[1])
    step = 2

    while ( (not np.array_equal(firstpoint, currentPoint) or (step==2)) and points.size > 0 ):
        if ( step == 5 ): # we're far enough to close too early
            points = np.append(points, [firstpoint], axis=0)
        kNearestPoints = GetNearestNeighbors(points, currentPoint, k)
        cPoints = SortByAngle(kNearestPoints, currentPoint, prevPoint)
        # avoid intersections: select first candidate that does not intersect any
        # polygon edge
        its = True
        i = 0
        while ( (its==True) and (i<cPoints.shape[0]) ):
                i=i+1
                if ( np.array_equal(cPoints[i-1], firstpoint) ):
                    lastPoint = 1
                else:
                    lastPoint = 0
                j = 2
                its = False
                while ( (its==False) and (j<np.shape(hull)[0]-lastPoint) ):
                    its = doLinesIntersect(hull[step-1-1], cPoints[i-1],
                            hull[step-1-j-1],hull[step-j-1])
                    j=j+1
        if ( its==True ):
            return concaveHull(dataset,k+1)
        prevPoint = currentPoint
        currentPoint = cPoints[i-1]
        # add current point to hull
        hull.append(currentPoint)
        points = removePoint(points,currentPoint)
        step = step+1
    # check if all points are inside the hull
    p = Path(hull)
    pContained = p.contains_points(dataset, radius=0.0000000001)
    if (not pContained.all()):
        return concaveHull(dataset, k+1)

    return hull

def point_inside_polygon(x,y,poly):

    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current

def to_graph(l):
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def get_associated_edges(vertices_associated_vertices, vertices_associated_edges, vertices_associated, room_edges):
    for vertex in vertices_associated:
        room_edges.extend(vertices_associated_edges[vertex])
        room_edges = get_associated_edges(vertices_associated_vertices, vertices_associated_edges, vertices_associated_vertices[vertex], room_edges)
    return room_edges


def merge_associated_edges(vertices_associated_vertices, vertices_associated_edges):
    rooms = []
    for vertex_idx, vertices_associated in enumerate(vertices_associated_vertices):
        room_edges = []
        room_edges.extend(vertices_associated_edges[vertex_idx])
        room_edges = get_associated_edges(vertices_associated_vertices, vertices_associated_edges, vertices_associated, room_edges)
        room_edges = list(set(room_edges))
        rooms.append(room_edges)
    return rooms


def main():
    #G, pos = generate_graph()
    G = nx.read_gpickle("../data/graphs/public_toilet2.gpickle")
    print(G.number_of_nodes())
    graph_sampler = GraphSampler()
    G = graph_sampler.down_sampler(G, 2000, 2000)
    G = nx.relabel.convert_node_labels_to_integers(G)
    pos = nx.get_node_attributes(G, 'pos')
    pos_list = list(map(list, list(pos.values())))
    nx.draw(G, pos, node_size = 30, with_labels=True)

    # Construct Voronoi diagram from nodes
    vor = Voronoi(pos_list)
    voronoi_plot_2d(vor)
    #hull = concaveHull(np.array(pos_list), k=3)
    #hull = np.array(list(map(list, list(hull))))
    #plt.plot(hull[:, 0], hull[:, 1])

    # Calculate convex hull around graph
    hull = ConvexHull(pos_list)
    convex_hull_plot_2d(hull)

    # Select Voronoi points inside convex hull
    # TODO: use clockwise points instead of convex hull
    filt_vor_vertices = []
    for vertex in vor.vertices:
        if point_inside_polygon(vertex[0], vertex[1], np.array(pos_list)[hull.vertices]):
            filt_vor_vertices.append(list(vertex))
    plt.scatter(np.array(vor.vertices)[:,0], np.array(vor.vertices)[:,1], c='g', s=50)
    plt.scatter(np.array(filt_vor_vertices)[:, 0], np.array(filt_vor_vertices)[:, 1], c='r', s=50)

    # Determine edges associated with vertices
    edge_list = list(G.edges)
    start_time = time.time()

    vertices_associated_edges = []
    for vertex in tqdm(filt_vor_vertices):
        vertex_associated_edges = []
        for edge_idx, edge in enumerate(edge_list):
            #edge1_pos1 = list(G._node[edge[0]]['pos'])
            #edge1_pos2 = list(G._node[edge[1]]['pos'])
            edge1_pos1 = pos_list[edge[0]] #list(G._node[edge[0]]['pos'])
            edge1_pos2 = pos_list[edge[1]]
            line_of_sight = True
            for comp_edge_idx, comp_edge in enumerate(edge_list):
                if edge_idx != comp_edge_idx:
                    #edge2_pos1 = list(G._node[comp_edge[0]]['pos'])
                    #edge2_pos2 = list(G._node[comp_edge[1]]['pos'])
                    edge2_pos1 = pos_list[comp_edge[0]]  # list(G._node[edge[0]]['pos'])
                    edge2_pos2 = pos_list[comp_edge[1]]
                    mid_pos = [(edge1_pos1[0]+edge1_pos2[0])/2, (edge1_pos1[1]+edge1_pos2[1])/2]
                    #if intersect(edge1_pos1, vertex, edge2_pos1, edge2_pos2) and \
                    #        intersect(edge1_pos2, vertex, edge2_pos1, edge2_pos2):
                    if intersect(vertex, mid_pos, edge2_pos1, edge2_pos2):
                        line_of_sight = False
                        break
            if line_of_sight:
                vertex_associated_edges.append(edge_idx)
        vertices_associated_edges.append(vertex_associated_edges)
    print(time.time() - start_time)

    # Determine vertices associated with vertices
    vertices_associated_vertices = []
    for vertex_idx, vertex in tqdm(enumerate(filt_vor_vertices)):
        vertex_associated_vertices = []
        vertex_associated_vertices.append(vertex_idx)
        for comp_vertex_idx, comp_vertex in enumerate(filt_vor_vertices):
            if comp_vertex_idx != vertex_idx:
                line_of_sight = True
                for edge in edge_list:
                    edge_pos1 = list(G._node[edge[0]]['pos'])
                    edge_pos2 = list(G._node[edge[1]]['pos'])
                    if intersect(vertex, comp_vertex, edge_pos1, edge_pos2):
                        line_of_sight = False
                        break
                if line_of_sight:
                    vertex_associated_vertices.append(comp_vertex_idx)
        vertices_associated_vertices.append(vertex_associated_vertices)

    room_graph = to_graph(vertices_associated_vertices)
    rooms_vertices = list(connected_components(room_graph))
    #print(rooms_vertices)

    #print(vertices_associated_edges)

    print(rooms_vertices)

    rooms_nodes = []
    for room_vertices in tqdm(rooms_vertices):
        row_idx = np.array(list(room_vertices))
        plt.plot(np.array(filt_vor_vertices)[row_idx, 0], np.array(filt_vor_vertices)[row_idx, 1])
        room_nodes = []
        for vertex in list(room_vertices):
            for edge in vertices_associated_edges[vertex]:
                room_nodes.append(edge_list[edge][0])
                room_nodes.append(edge_list[edge][1])
        rooms_nodes.append(list(set(room_nodes)))

    color = iter(cm.rainbow(np.linspace(0, 1, len(rooms_nodes))))
    for room_nodes in tqdm(rooms_nodes):
        room_pos = []
        c=next(color)
        for node in room_nodes:
            pos = G._node[node]['pos']
            room_pos.append(pos)

        # TODO could be a problem
        #ConvexHull
        #center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), room_pos), [len(room_pos)] * 2))
        #room_pos_sorted = np.array(sorted(room_pos, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))

        #plt.fill(room_pos_sorted[:, 0], room_pos_sorted[:, 1], c=c, alpha=0.2)
        #hull = concaveHull(np.array(room_pos),3)
        #plt.fill(np.array(hull)[:, 0], np.array(hull)[:, 1], c=c, alpha=0.2)
        #plt.fill(np.array(hull)[:, 0], np.array(hull)[:, 1], c=c, alpha=0.2)
        # TODO use known edges
        print(room_pos)
        if len(room_pos) > 2:
            hull = ConvexHull(room_pos)
            plt.fill(np.array(room_pos)[hull.vertices,0], np.array(room_pos)[hull.vertices,1], c=c, alpha=0.2)
        #plt.fill(np.array(room_pos)[hull.vertices][0], np.array(room_pos)[hull.vertices][1], c=c, alpha=0.2)
        print(hull)
        #plt.fill

    #plt.fill(np.array(rooms_nodes[0])[0], np.array(rooms_nodes[0])[1])

    print(rooms_nodes)
    '''for room_nodes in rooms_nodes:
        room_pos = []
        for node in room_nodes:
            edge_pos1 = G._node[node[0]]['pos']
            edge_pos2 = G._node[node[1]]['pos']
        hull = ConvexHull(room)'''


    '''rooms = [[]]
    for idx, vertices_associated in enumerate(vertices_associated_vertices):
        print(str(idx) + '/' + str(len(vertices_associated_vertices)))
        print(rooms)
        for room in rooms:
            if vertices_associated[0] in room:
                room = room + vertices_associated
            else:
                rooms.append(vertices_associated)

    for room in rooms:
        room = list(set(room))

    print(vertices_associated_vertices)
    print(len(rooms))'''

    #rooms = merge_associated_edges(vertices_associated_vertices, vertices_associated_edges)
    #print(len(rooms))

    '''for vertex_idx, vertices_associated in enumerate(vertices_associated_vertices):
        room_edges = []
        room_edges.extend(vertices_associated_edges[vertex_idx])
        for vertex in vertices_associated:
            room_edges.extend(vertices_associated_edges[vertex])
            for 
        
        room_edges = list(set(room_edges))'''


    plt.show()
    print("done")

if __name__ == "__main__":
    main()
