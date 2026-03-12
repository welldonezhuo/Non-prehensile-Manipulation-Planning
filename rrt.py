import numpy as np
import time
import samplers
import utils


class Tree(object):
    """
    The tree class for Kinodynamic RRT.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.dim_state = self.pdef.get_state_dimension()
        self.nodes = []
        self.stateVecs = np.empty(shape=(0, self.dim_state))

    def add(self, node):
        """
        Add a new node into the tree.
        """
        self.nodes.append(node)
        self.stateVecs = np.vstack((self.stateVecs, node.state['stateVec']))
        assert len(self.nodes) == self.stateVecs.shape[0]

    def nearest(self, rstateVec):
        """
        Find the node in the tree whose state vector is nearest to "rstateVec".
        """
        dists = self.pdef.distance_func(rstateVec, self.stateVecs)
        nnode_id = np.argmin(dists)
        return self.nodes[nnode_id]

    def size(self):
        """
        Query the size (number of nodes) of the tree. 
        """
        return len(self.nodes)


class Node(object):
    """
    The node class for Tree.
    """

    def __init__(self, state):
        """
        args: state: The state associated with this node.
                     Type: dict, {"stateID": int, 
                                  "stateVec": numpy.ndarray}
        """
        self.state = state
        self.control = None  # the control asscoiated with this node
        self.parent = None  # the parent node of this node

    def get_control(self):
        return self.control

    def get_parent(self):
        return self.parent

    def set_control(self, control):
        self.control = control

    def set_parent(self, pnode):
        self.parent = pnode


class KinodynamicRRT(object):
    """
    The Kinodynamic Rapidly-exploring Random Tree (RRT) motion planner.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance for the problem to be solved.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.tree = Tree(pdef)
        self.state_sampler = samplers.StateSampler(self.pdef)  # state sampler
        self.control_sampler = samplers.ControlSampler(
            self.pdef)  # control sampler

    def solve(self, time_budget):
        """
        The main algorithm of Kinodynamic RRT.
        args:  time_budget: The planning time budget (in seconds).
        returns: is_solved: True or False.
                      plan: The motion plan found by the planner,
                            represented by a sequence of tree nodes.
                            Type: a list of rrt.Node
        """
        ########## TODO ##########

        solved = False
        plan = None

        # Only initialize it if the tree is empty
        if self.tree.size() == 0:
            start = Node(self.pdef.start_state)
            start.set_parent(None)
            start.set_control(None)
            self.tree.add(start)

        start_clock = time.time()
        num_trials = 1

        while time.time() - start_clock < time_budget:
            # Use only random state sampling
            x_samp = self.state_sampler.sample()
            x_samp_vec = x_samp["stateVec"] if isinstance(
                x_samp, dict) else x_samp

            # Find nearest node in the tree
            nearest = self.tree.nearest(x_samp_vec)

            # propagate toward sampled state
            control, candidate_state = self.control_sampler.sample_to(
                nearest, x_samp_vec, num_trials
            )

            # Skip failed propagation
            if control is None or candidate_state is None:
                continue

            # Skip tiny motion
            step = candidate_state["stateVec"] - nearest.state["stateVec"]
            if np.linalg.norm(step) < 1e-6:
                continue

            # Add new node
            child = Node(candidate_state)
            child.set_control(control)
            child.set_parent(nearest)
            self.tree.add(child)

            # Goal check
            if self.pdef.goal.is_satisfied(child.state):
                path_nodes = []
                current = child

                while current is not None:
                    path_nodes.append(current)
                    current = current.get_parent()

                solved = True
                plan = path_nodes[::-1]
                return solved, plan

        ##########################

        return solved, plan
