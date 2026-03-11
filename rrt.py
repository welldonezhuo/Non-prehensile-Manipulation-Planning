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

        start_time = time.time()

        # Initialize the RRT tree with the start state
        start_state = self.pdef.get_start_state()
        start_node = Node(start_state)
        self.tree.add(start_node)

        # Main RRT planning loop
        while time.time() - start_time < time_budget:

            # --------------------------------------------------
            # Goal-biased sampling:
            # With a small probability, sample a state near the goal
            # to encourage the planner to push the target object
            # toward the goal region.
            # --------------------------------------------------
            if np.random.rand() < 0.2:
                rstateVec = np.array(start_state["stateVec"], copy=True)
                goal = self.pdef.goal
                rstateVec[7] = goal.x_g + \
                    np.random.uniform(-goal.r_g, goal.r_g)
                rstateVec[8] = goal.y_g + \
                    np.random.uniform(-goal.r_g, goal.r_g)

                # Randomize target object orientation if present
                if len(rstateVec) > 9:
                    rstateVec[9] = np.random.uniform(-np.pi, np.pi)

            # Otherwise sample a random state from the state space
            else:
                rstateVec = self.state_sampler.sample()

            # --------------------------------------------------
            # Select the node to expand from
            # Sometimes expand from the node whose target object
            # is currently closest to the goal to improve efficiency.
            # Otherwise use the nearest node to the sampled state.
            # --------------------------------------------------
            if self.tree.size() > 1 and np.random.rand() < 0.5:
                goal = self.pdef.goal
                best_node = None
                best_dist = np.inf

                for node in self.tree.nodes:
                    s = node.state["stateVec"]
                    d = np.linalg.norm([s[7] - goal.x_g, s[8] - goal.y_g])

                    if d < best_dist:
                        best_dist = d
                        best_node = node

                nearest_node = best_node
            else:
                nearest_node = self.tree.nearest(rstateVec)

            # --------------------------------------------------
            # Sample control inputs from the nearest node
            # The control sampler tries to find a control that
            # moves the system toward the sampled state.
            # --------------------------------------------------
            ctrl, _ = self.control_sampler.sample_to(
                nearest_node,
                rstateVec,
                1
            )

            # If no valid control is found, skip this iteration
            if ctrl is None:
                continue

            # --------------------------------------------------
            # Propagate the control in simulation to obtain the
            # resulting state and check whether execution is valid.
            # --------------------------------------------------
            self.pdef.panda_sim.restore_state(nearest_node.state)
            _, valid = self.pdef.panda_sim.execute(ctrl)
            outcome = self.pdef.panda_sim.save_state()

            # Skip if execution is invalid
            if not valid:
                continue

            # Check state validity (collision, workspace bounds, etc.)
            if not self.pdef.is_state_valid(outcome):
                continue

            # --------------------------------------------------
            # Create a new node with the resulting state and
            # add it to the RRT tree
            # --------------------------------------------------
            new_node = Node(outcome)
            new_node.set_control(ctrl)
            new_node.set_parent(nearest_node)

            self.tree.add(new_node)

            # --------------------------------------------------
            # Check if the goal condition is satisfied
            # If yes, backtrack through parents to form the plan
            # --------------------------------------------------
            if self.pdef.goal.is_satisfied(new_node.state):
                solved = True
                plan = []

                node = new_node
                while node is not None:
                    plan.append(node)
                    node = node.get_parent()

                # Reverse the sequence to obtain start → goal order
                plan.reverse()
                break
        ##########################

        return solved, plan
