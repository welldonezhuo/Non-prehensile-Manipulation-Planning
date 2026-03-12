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

        # Reinitialize the tree and insert the start state as the root
        self.tree = Tree(self.pdef)

        start = Node(self.pdef.start_state)
        start.set_parent(None)
        start.set_control(None)
        self.tree.add(start)

        # Start the timer for the planning loop
        start_clock = time.time()
        num_trials = 1
        iteration_count = 0

        # Run RRT expansion until the time budget is exhausted
        while time.time() - start_clock < time_budget:
            iteration_count += 1

            # With some probability, sample around an existing node (local exploration)
            if self.tree.size() > 1 and np.random.rand() < 0.25:
                base_node = self.tree.nodes[np.random.randint(
                    self.tree.size())]
                x_samp_vec = np.array(base_node.state["stateVec"], copy=True)

                # Add small Gaussian noise to explore nearby states
                noise = np.random.normal(0.0, 0.02, size=x_samp_vec.shape)
                x_samp_vec = x_samp_vec + noise
            else:
                # Otherwise sample a completely random state
                x_samp = self.state_sampler.sample()
                x_samp_vec = x_samp["stateVec"] if isinstance(
                    x_samp, dict) else x_samp

            # Find the nearest node in the tree to the sampled state
            nearest = self.tree.nearest(x_samp_vec)

            # Try to propagate from the nearest node toward the sampled state
            control, candidate_state = self.control_sampler.sample_to(
                nearest, x_samp_vec, num_trials
            )

            # Skip if propagation fails
            if control is None or candidate_state is None:
                continue

            # Skip if the new state is almost identical to the previous one
            step = candidate_state["stateVec"] - nearest.state["stateVec"]
            if np.linalg.norm(step) < 1e-6:
                continue

            # Create a new node and add it to the tree
            child = Node(candidate_state)
            child.set_control(control)
            child.set_parent(nearest)
            self.tree.add(child)

            # Check if the new node satisfies the goal condition
            if self.pdef.goal.is_satisfied(child.state):

                # Reconstruct the path by tracing back through the parents
                path_nodes = []
                current = child

                while current is not None:
                    path_nodes.append(current)
                    current = current.parent

                # Reverse the path so it starts from the root
                return True, path_nodes[::-1]

        # Return failure if no solution is found within the time limit
        return False, None

        ##########################

        return solved, plan
