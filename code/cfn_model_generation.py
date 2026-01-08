import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
import tqdm
import random


class Node():    
    def __init__(self, label:int, edge_labels: list):
        self.label = label
        self.edges = edge_labels
        self.spins = None
        if len(self.edges) == 1:
            self.isLeaf = True
        else:
            self.isLeaf = False
        
        self.magnetizations = {e:None for e in self.edges}

    def generate_spins(self, numSignals:int, tree):
        self.spins = 2*np.random.binomial(1,1/2, numSignals)-1
        for e in self.edges:
            e = tree.edges[e]
            neighbors = [node for node in e.vertices if node!=self.label]
            neighbor = neighbors[0]
            neighbor = tree.vertices[neighbor]
            p = (1-e.true_parameter)/2
            flip = 1-2*np.random.binomial(1, p, numSignals)
            give_spin = flip*self.spins
            neighbor.give_spin_and_propagate(give_spin, e.label, tree)

    def give_spin_and_propagate(self, spins, edge, tree):
        self.spins = spins        
        if not self.isLeaf:
            for e in self.edges:
                if e!=edge:    
                    e = tree.edges[e]
                    neighbor = e.other_neighbor(self.label)
                    neighbor = tree.vertices[neighbor]
                    p = (1-e.true_parameter)/2
                    flip = 1-2*np.random.binomial(1, p, spins.shape)
                    give_spin = flip*self.spins
                    neighbor.give_spin_and_propagate(give_spin, e.label, tree)

    def magnetize(self, away_from_edge, tree):
        if self.isLeaf:
            self.magnetizations[away_from_edge] = self.spins
        elif len(self.edges) == 3:
            other_edges = [e for e in self.edges if e!=away_from_edge]
            e1,e2 = other_edges[0], other_edges[1]
            e1,e2 = tree.edges[e1], tree.edges[e2]
            v1, v2 = e1.other_neighbor(self.label), e2.other_neighbor(self.label)
            v1, v2 = tree.vertices[v1], tree.vertices[v2]
            theta1,theta2 = e1.estimated_parameter, e2.estimated_parameter

            q = np.vectorize(lambda x,y: (x+y)/(1+x*y))
           
            if v1.magnetizations[e1.label] is None:
                v1.magnetize(e1.label, tree)
            if v2.magnetizations[e2.label] is None:
                v2.magnetize(e2.label, tree)
            
            Z1, Z2 = v1.magnetizations[e1.label], v2.magnetizations[e2.label]

            self.magnetizations[away_from_edge] = q(theta1*Z1,theta2*Z2)    
        else:
            other_edges = [e for e in self.edges if e!=away_from_edge]
            q = np.vectorize(lambda x,y: (x+y)/(1+x*y))
            Z = np.zeros_like(self.spins)

            for e in other_edges:
                edge_in_tree = tree.edges[e]
                vert = edge_in_tree.other_neighbor(self.label)
                vert_in_tree = tree.vertices[vert]
                theta = edge_in_tree.estimated_parameter
                if vert_in_tree.magnetization[e] is None:
                    vert_in_tree.magnetize(e, tree)
                Z = q(Z, theta*vert_in_tree.magnetization[e])
            self.magnetizations[away_from_edge] = Z

    def update_magnetization(self, included_edge, tree):
        """
        Updates the magnetization once an estimated paramter in a descendent subtree is updated.
        If self.label = v and the included_edge is {u,v} then this updates the magnetizations at
        v whenever we delete any of the edges incident to v that are not included_edge.
        """
        if not self.isLeaf:
            if len(self.edges)==3:
                other_edges = [e for e in self.edges if e!=included_edge]

                ## Get other edges

                e1, e2 = other_edges[0], other_edges[1]
                e1, e2 = tree.edges[e1], tree.edges[e2]
                
                
                ## Get updated magnetization
                e0 = tree.edges[included_edge]
                u = e0.other_neighbor(self.label)
                u = tree.vertices[u]
                theta_u = e0.estimated_parameter
                Z_u = u.magnetizations[included_edge]

                ## Get magnetization from other neighbors
                v1, v2 = e1.other_neighbor(self.label), e2.other_neighbor(self.label)
                v1, v2 = tree.vertices[v1], tree.vertices[v2]

                theta1,theta2 = e1.estimated_parameter, e2.estimated_parameter
                if v1.magnetizations[e1.label] is None:
                    v1.magnetize(e1.label, tree)
                if v2.magnetizations[e2.label] is None:
                    v2.magnetize(e2.label, tree)
                Z1, Z2 = v1.magnetizations[e1.label], v2.magnetizations[e2.label]

                q = np.vectorize(lambda x,y: (x+y)/(1+x*y))

                ## Update magnetization for self

                self.magnetizations[e1.label] = q(theta2*Z2, theta_u*Z_u)
                self.magnetizations[e2.label] = q(theta1*Z1, theta_u*Z_u)

                ## Propogate

                v1.update_magnetization(e1.label, tree)
                v2.update_magnetization(e2.label, tree)
            else:
                other_edges = [e for e in self.edges if e!=included_edge]
                ## Get updated magnetization at vertex u
                e0 = tree.edges[included_edge]
                u = e0.other_neighbor(self.label)
                u = tree.vertices[u]
                theta_u = e0.estimated_parameter
                Z_u = u.magnetizations[included_edge]

                q = np.vectorize(lambda x,y: (x+y)/(1+x*y))
                ## Eventual magnetization at self.magentization[---]
                Z = theta_u*Z_u
                
                ## Compute magnetizations away from each other_edges
                other_mags = {e:Z for e in other_edges}
                for e in other_edges:
                    edge_in_tree = tree.edges[e]
                    vert = edge_in_tree.other_neighbor(self.label)
                    vert_in_tree = tree.vertices[vert]
                    theta = edge_in_tree.estimated_paramter
                    if vert_in_tree.magnetizations[e] is None:
                        vert_in_tree.magnetize(e, tree)
                    mag_at_other_neighbor = vert_in_tree.magnetizaions[included_edge]
                    for edge, mag in other_mags.items():
                        if edge != e:
                            other_mags[edge] = q(mag, mag_at_other_neighbor)

                ## Update and propogate
                for edge, mag in other_mags.items():
                    # Update
                    self.magnetizations[edge] = mag
                    # Propogate
                    edge_in_tree = tree.edges[edge]
                    vert = edge_in_tree.other_neighbor(self.label)
                    vert_in_tree = tree.vertices[vert]
                    vert_in_tree.update_magnetizatoin(edge, tree)                    




class Edge():
    def __init__(self, label, vertices: list[tuple], delta:float = 0.05, warmStart:bool = True, C_val:float = .5, c_val:float = 0.25):
        if delta<=0:
            raise Exception("Delta must be positive.")
        true_ll = np.float128(1-2*C_val*delta)
        gap = np.float128(delta*(C_val-c_val)*2)
        if gap<=0:
            raise Exception("Parameter space is empty.")
        self.label = label
        self.vertices = vertices
        self.true_parameter = true_ll+gap*np.float128(np.random.uniform(0,1))
        if warmStart:
            self.estimated_parameter = true_ll+gap*np.float128(np.random.uniform(0,1))
        else:
            self.estimated_parameter = np.float128(0.8)
        self.upperLimit = 1-c_val*delta*2
        
        self.delta = delta
        self.warmstart = warmStart
    def other_neighbor(self, nei):
        if nei not in self.vertices:
            raise Exception('Vertex not found.')
        else:
            v = [n for n in self.vertices if n!= nei]
            return v[0]

    def update_parameter(self, Zu,Zv, update_rule:str = 'max', learning_rate:float = 0.01):
        prod = Zu*Zv
        if update_rule=='max':
            ## Coordinate maximization
            if all(prod>=0):
                self.estimated_parameter = self.upperLimit
            elif all(prod<=0):
                self.estimated_parameter = 1/2
            else:
                f = lambda t: np.mean(prod/(1+t*prod),dtype= np.float128)
                if f(-1)*f(1)<0:
                    a = scipy.optimize.bisect(f,-1,1, maxiter = 100)
                    self.estimated_parameter = np.float128(a)
        elif update_rule == 'gradient':
            ## Gradient ascent
            grad = np.mean(prod, dtype = np.float128)
            self.estimated_parameter += grad*learning_rate
        
        elif update_rule == 'emp_flips':
            ## Gradient ascent
            same = sum(prod>=0)
            diff = sum(prod<0)
            empiricial_pflip = diff/len(prod)
            self.estimated_parameter = 1- 2*empiricial_pflip


class Tree():
    def __init__(self, vertices:dict, edges:dict,root:int, n_samples:int, delta:float = 0.05, 
                 warmStart:bool = True, C_val:float = .5, c_val:float = 0.25):
        if root not in vertices:
            raise Exception("Root is not a vertex.")
        
        self.edges = dict()
        for e,inc_vert in edges.items():
            self.edges[e] = Edge(e, inc_vert,delta, warmStart, C_val, c_val)
            for v in inc_vert:
                if v not in vertices:
                    raise Exception("Incident vertices not included in vertex dict.")

        self.vertices = dict()
        for v, inc_edges in vertices.items():
            self.vertices[v] = Node(v,inc_edges)
            for e in inc_edges:
                if e not in edges:
                    raise Exception("Incident edge not included in edge dict.")
                
        self.root = root

        self.root_vertex = self.vertices[root]
        self.number_update_rounds = 0
        self.generate_spins(n_samples)


    def generate_spins(self, numSamples):
        self.root_vertex.generate_spins(numSamples, self)
    

    def coordinate_update(self, edge_label, update_rule:str = 'max',learning_rate:float = 0.01):
        ## Get Magnetizatoins
        u, v = self.edges[edge_label].vertices
        u, v = self.vertices[u], self.vertices[v]
        if u.magnetizations[edge_label] is None:
            u.magnetize(edge_label, self)
        if v.magnetizations[edge_label] is None:
            v.magnetize(edge_label, self)
        Zu, Zv = u.magnetizations[edge_label], v.magnetizations[edge_label]
        
        
        ### Update theta

        self.edges[edge_label].update_parameter(Zu,Zv, update_rule, learning_rate)

        #### Proogate
        u.update_magnetization(edge_label,self)
        v.update_magnetization(edge_label, self)


    def update_round(self, orders = [1], update_rule:str = 'max', learning_rate:float = 0.01, print_progress:bool=True):
        error = [[] for _ in orders]
        self.number_update_rounds+=1
        if print_progress:
            print(f"Round {self.number_update_rounds} progress:")
            for e in tqdm.tqdm(self.edges.keys()):
                self.coordinate_update(e, update_rule, learning_rate)
                error_list = self.get_gaps(orders)
                for p,e in enumerate(error_list):
                    error[p].append(e)
        else:
            for e in self.edges.keys():
                self.coordinate_update(e, update_rule, learning_rate)
                error_list = self.get_gaps(orders)
                for p,e in enumerate(error_list):
                    error[p].append(e)
        return error
    

    def get_gaps(self, orders = [2]):
        true_params = np.array([e.true_parameter for e in self.edges.values()], dtype = np.float128)
        est_params = np.array([e.estimated_parameter for e in self.edges.values()], dtype = np.float128)
        norms = [np.linalg.norm(true_params-est_params,p) for p in orders]
        return norms

    def get_l1_gap(self):
        true_params = np.array([e.true_parameter for e in self.edges.values()], dtype = np.float128)
        est_params = np.array([e.estimated_parameter for e in self.edges.values()], dtype = np.float128)
        return np.linalg.norm(true_params-est_params,1)
    



    

class MagNode():
    def __init__(self, label , leftNode = None, rightNode = None,
                 delta:float = 0.05, C_val:float = 0.5, c_val:float = 0.25):
        
        lowerLimit = np.float128(1-2*C_val*delta)
        gap = np.float128(2*delta*(C_val - c_val))
        if gap<=0:
            raise Exception('Parameter space is not well-defined.')
        self.node = label
        if leftNode is not None:
            if rightNode is None:
                raise Exception('Only left child is defined.')
            self.leftChild = leftNode
            self.leftEsimate = lowerLimit +  gap * np.float128(np.random.uniform(0,1))
            self.leftParam = lowerLimit +  gap * np.float128(np.random.uniform(0,1))
            self.rightChild = rightNode
            self.rightEsimate = lowerLimit +  gap * np.float128(np.random.uniform(0,1))
            self.rightParam = lowerLimit +  gap * np.float128(np.random.uniform(0,1))
            self.isLeaf = False
        else:
            if rightNode is not None:
                raise Exception('Only right child is defined.')
            self.isLeaf = True
        self.spins = None
        self.magnetizations = None

    def generateSpins(self, numberSamples):
        self.spins = np.ones(shape = (numberSamples,1))
        self.propogate(self.spins)
    
    def propogate(self,signal):
        signal = np.array(signal)
        self.spins = signal
        if not self.isLeaf:
            pL = (1-self.leftParam)/2
            pR = (1-self.rightParam)/2
            lFlip = 1-2*np.random.binomial(1,pL,size = signal.shape)
            rFlip = 1-2*np.random.binomial(1,pR,size = signal.shape)
            self.leftChild.propogate(lFlip*signal)
            self.rightChild.propogate(rFlip*signal)
        else:
            self.magnetizations = signal    
    
    def magnetize(self):
        if self.magnetizations is None:
            self.leftChild.magnetize()
            self.rightChild.magnetize()
            recurse = np.vectorize(lambda x,y:(x+y)/(1+x*y))
            thetaL = self.leftEsimate
            thetaR = self.rightEsimate
            zL = self.leftChild.magnetizations
            zR = self.rightChild.magnetizations
            self.magnetizations = recurse(thetaL*zL, thetaR*zR)
            self.leftChild.magnetizations = None
            self.rightChild.magnetizations = None

    def constructMagnetization(self, numberSamples):
        self.generateSpins(numberSamples = numberSamples)
        self.magnetize()
        return self.magnetizations

class GenMagNode():
    def __init__(self, label:int, children:list, 
                 delta:float = 0.05, C_val:float = 0.5, c_val:float = 0.25):
        lowerLimit = np.float128(1-2*C_val*delta)
        gap = np.float128(2*delta*(C_val - c_val))
        self.lower_limit = lowerLimit
        self.gap = gap
        if gap<=0:
            raise Exception('Parameter space is not well-defined.')
        self.node = label
        self.children = children
        self.trueParameters = [lowerLimit +  gap * np.float128(np.random.uniform(0,1)) for _ in children]
        self.estParameters = [lowerLimit +  gap * np.float128(np.random.uniform(0,1)) for _ in children]
        if children:
            self.isLeaf = False
        else:
            self.isLeaf = True
        self.spins = None
        self.magnetizations = None

    def add_child(self, child):
        self.trueParameters.append(self.lower_limit +  self.gap * np.float128(np.random.uniform(0,1)))
        self.estParameters.append(self.lower_limit +  self.gap * np.float128(np.random.uniform(0,1)))
        self.children.append(child)
        self.isLeaf = False

    def generateSpins(self, numberSamples):
        self.spins = np.ones(shape = (numberSamples,1))
        self.propogate(self.spins)
    
    def propogate(self,signal):
        signal = np.array(signal)
        self.spins = signal
        if not self.isLeaf:
            for true_param,c in zip(self.trueParameters, self.children):
                p = (1-true_param)/2
                flip = 1-2*np.random.binomial(1,p,size = signal.shape)
                c.propogate(flip*signal)
        else:
            self.magnetizations = signal    

    def magnetize(self):
        if self.isLeaf:
            self.magnetizations = self.spins
        elif self.magnetizations is None:
            Z = np.zeros(shape = self.spins.shape)
            recurse = np.vectorize(lambda x,y:(x+y)/(1+x*y))
            for theta,c in zip(self.estParameters, self.children):
                c.magnetize()
                z_child = c.magnetizations
                Z = recurse(Z, z_child*theta)
                if not c.isLeaf:
                    c.magnetizations = None
            self.magnetizations = Z
            
    def constructMagnetization(self, numberSamples):
        self.generateSpins(numberSamples = numberSamples)
        self.magnetize()
        return self.magnetizations
 
### Magnetization Trees

def UniformMagTree(numberLeaves, delta:float = 0.05, C_val:float= 0.5, c_val:float = 0.25):
    """
    Generates a uniform rooted binary tree by the Kingman coalescent.
    """
    leafs = {i:MagNode(i, delta = delta, C_val = C_val, c_val = c_val) for i in range(numberLeaves)}
    nextLabel = numberLeaves
    while len(leafs)>=2:
        keys = list(leafs.keys())
        l,r = random.sample(keys,2)
        leafs[nextLabel] = MagNode(nextLabel, leafs[l], leafs[r], delta = delta, C_val = C_val, c_val = c_val)
        leafs.pop(l)
        leafs.pop(r)
        nextLabel+=1
    node = leafs[nextLabel-1] 
    return node

def FoataFuchs(n:int, delta:float = 0.05, C_val:float = 0.5, c_val:float = 0.25):
    li = [1+int(np.floor(np.random.uniform(0,n))) for _ in range(n-1)]
    repeats = []
    mult_occur = set()
    no_occur = set(range(1,n+1))
    for i,k in enumerate(li):
        if i==0:
            repeats.append(i)
            mult_occur.add(k)
        else:
            if k in mult_occur:
                repeats.append(i)
            else:
                mult_occur.add(k)
    repeats.append(n)    
    no_occur = no_occur.difference(mult_occur)
    leafs = list(no_occur)
    leafs.sort()
    

    nodes = {i:GenMagNode(i,[], delta, C_val, c_val) for i in range(1,n+1)}
    for i, l in enumerate(leafs):
        d,u = repeats[i], repeats[i+1]
        if u<n:
            path = li[d:u]
        else:
            path = li[d:]
        child = l
        for parent in path[::-1]:
            nodes[parent].add_child(nodes[child])
            child = parent
    return nodes[li[0]]


def GeneralizedFoataFuchs(child_counts:list, delta:float = 0.05, C_val:float = 0.5, c_val:float = 0.25):
    li = []
    i = 1
    for k in child_counts:
        li.extend([i for _ in range(k)])
        i+=1
    random.shuffle(li)
    n = sum(child_counts)+1
    repeats = []
    mult_occur = set()
    no_occur = set(range(1,n+1))
    for i,k in enumerate(li):
        if i==0:
            repeats.append(i)
            mult_occur.add(k)
        else:
            if k in mult_occur:
                repeats.append(i)
            else:
                mult_occur.add(k)
    repeats.append(n)    
    no_occur = no_occur.difference(mult_occur)
    leafs = list(no_occur)
    leafs.sort()
    

    nodes = {i:GenMagNode(i,[], delta, C_val, c_val) for i in range(1,n+1)}
    for i, l in enumerate(leafs):
        d,u = repeats[i], repeats[i+1]
        if u<n:
            path = li[d:u]
        else:
            path = li[d:]
        child = l
        for parent in path[::-1]:
            nodes[parent].add_child(nodes[child])
            child = parent
    return nodes[li[0]]


### Tree type trees

def UniformTree(n_leaves, n_samples = 100, delta:float = 0.05,
                 warmStart:bool =  True, C_val:float = 0.5, c_val:float = 0.25):
    """
    Generates a uniform binary tree and returns a Tree() object for coordinate updates.
    """
    vertices = {j+1: [] for j in range(n_leaves)}
    leafs = {j+1 for j in range(n_leaves)}
    edges = {}
    root = 0
    next_label = n_leaves+1
    edge_label = 1
    while len(leafs)>=2:
        keys = list(leafs)
        l,r = random.sample(keys,2)
        vertices[next_label] = [edge_label, edge_label+1]
        vertices[l].append(edge_label)
        vertices[r].append(edge_label+1)

        edges[edge_label] = [next_label, l]
        edges[edge_label+1] = [next_label, r]


        leafs.remove(l)
        leafs.remove(r)
        leafs.add(next_label)
        next_label+=1
        edge_label+=2
    r = list(leafs)[0]
    vertices[root] = [edge_label]
    vertices[r].append(edge_label)
    edges[edge_label] = [root, r]
    tree = Tree(vertices, edges, root, n_samples, delta, warmStart, C_val, c_val)
    return (tree, edges, vertices, root)


def GeneralizedFoataFuchsTree(child_sequence:list, n_samples = 100, delta:float = 0.05,
                 warmStart:bool =  True, C_val:float = 0.5, c_val:float = 0.25):
    if 1 in child_sequence:
        raise TypeError('Tree contains contractable edge. Child sequence contains a 1.')
    
    li = []
    i = 1
    for k in child_sequence:
        li.extend([i for _ in range(k)])
        i+=1
    random.shuffle(li)
    li.append(li[0])


    n = sum(child_sequence)+1

    vertices = {k:[] for k in range(1,n+1)}
    edges = {k:[] for k in range(1,n)} 

    edge_index = 1

    repeats = []
    mult_occur = set()
    no_occur = set(range(1,n+1))
    for i,k in enumerate(li):
        if i==0:
            repeats.append(i)
            mult_occur.add(k)
        else:
            if k in mult_occur:
                repeats.append(i)
            else:
                mult_occur.add(k)
    repeats.append(n)    
    no_occur = no_occur.difference(mult_occur)
    leafs = list(no_occur)
    leafs.sort()

    for i, l in enumerate(leafs):
        d,u = repeats[i], repeats[i+1]
        if u<n:
            path = li[d:u]
        else:
            path = li[d:]
        child = l
        for parent in path[::-1]:
            vertices[child].append(edge_index)
            vertices[parent].append(edge_index)
            edges[edge_index] = [child, parent]
            child = parent
            edge_index+=1
    
    root = li[0]
    tree = Tree(vertices, edges, root, n_samples, delta, warmStart, C_val, c_val)
    return (tree, edges, vertices, root)