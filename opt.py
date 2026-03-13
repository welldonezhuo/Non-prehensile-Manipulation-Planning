import numpy as np
import copy
import sim
import goal
import rrt
import utils


########## TODO ##########

import samplers


def path_cost(plan):
    """
    Compute the total duration of a plan.
    args: plan: A list of rrt.Node
    returns: total_cost: float
    """
    if plan is None:
        return np.inf

    total_cost = 0.0
    for node in plan:
        ctrl = node.get_control()

        # add control time
        if ctrl is not None:
            total_cost += ctrl[3]

    return total_cost


def path_length(plan):
    """
    Compute the number of nodes in a plan.
    args: plan: A list of rrt.Node
    returns: int
    """
    if plan is None:
        return 0

    # count nodes
    return len(plan)


def clone_plan(plan):
    """
    Make a deep copy of a plan and rebuild parent links.
    args: plan: A list of rrt.Node
    returns: new_plan: A list of rrt.Node
    """
    if plan is None:
        return None
    if len(plan) == 0:
        return []

    new_plan = []
    new_parent = None

    for old_node in plan:
        # copy state
        new_state = copy.deepcopy(old_node.state)
        new_node = rrt.Node(new_state)

        old_ctrl = old_node.get_control()
        if old_ctrl is None:
            new_node.set_control(None)
        else:
            # copy control
            new_node.set_control(np.array(old_ctrl, copy=True))

        # rebuild parent link
        new_node.set_parent(new_parent)
        new_plan.append(new_node)
        new_parent = new_node

    return new_plan


def try_shortcut(pdef, from_node, to_node, num_trials=40, dist_threshold=0.15):
    """
    Try to connect from_node directly toward to_node by sampling controls.

    args:
        pdef: problem definition
        from_node: start node of the shortcut
        to_node: target node of the shortcut
        num_trials: number of sampled controls
        dist_threshold: max allowed distance to accept the shortcut

    returns:
        new_node: a new node reached from from_node, or None if failed
    """
    control_sampler = samplers.ControlSampler(pdef)

    # target state
    target_vec = to_node.state["stateVec"]

    best_ctrl, best_state = control_sampler.sample_to(
        from_node, target_vec, num_trials
    )

    # shortcut failed
    if best_ctrl is None or best_state is None:
        return None

    # check distance
    dist = pdef.distance_func(best_state["stateVec"], target_vec)
    dist = float(np.min(dist))

    if dist > dist_threshold:
        return None

    # build new shortcut node
    new_node = rrt.Node(copy.deepcopy(best_state))
    new_node.set_control(np.array(best_ctrl, copy=True))
    new_node.set_parent(from_node)

    return new_node


def shortcut_plan(pdef, plan, num_trials, dist_threshold):
    """
    Shortcut a plan by trying to skip intermediate nodes.

    args:
        pdef: problem definition
        plan: a list of rrt.Node
        num_trials: number of control samples for each shortcut attempt
        dist_threshold: max allowed distance to accept the shortcut

    returns:
        new_plan: a shortened plan
    """
    if plan is None:
        return None
    if len(plan) <= 2:
        return clone_plan(plan)

    # work on a copy
    original = clone_plan(plan)

    # keep start node
    new_plan = [original[0]]
    i = 0

    while i < len(original) - 1:
        from_node = new_plan[-1]
        shortcut_found = False

        # try longer shortcut first
        for j in range(len(original) - 1, i + 1, -1):
            to_node = original[j]
            shortcut_node = try_shortcut(
                pdef,
                from_node,
                to_node,
                num_trials=num_trials,
                dist_threshold=dist_threshold
            )

            if shortcut_node is not None:
                # accept shortcut
                new_plan.append(shortcut_node)
                i = j
                shortcut_found = True
                break

        if not shortcut_found:
            # keep next node
            next_old = original[i + 1]
            keep_node = rrt.Node(copy.deepcopy(next_old.state))

            ctrl = next_old.get_control()
            if ctrl is None:
                keep_node.set_control(None)
            else:
                keep_node.set_control(np.array(ctrl, copy=True))

            # reconnect parent
            keep_node.set_parent(new_plan[-1])
            new_plan.append(keep_node)
            i += 1

    return new_plan


def print_plan_stats(name, plan):
    """
    Print simple statistics of a plan.
    args:
        name: label of the plan
        plan: a list of rrt.Node
    """
    if plan is None:
        print(name)
        print("  plan = None")
        return

    # print plan info
    print(name)
    print("  nodes =", path_length(plan))
    print("  cost  =", path_cost(plan))


# def same_plan(plan1, plan2, tol=1e-6):
#     """
#     Check whether two plans are effectively identical.
#     """
#     if plan1 is None or plan2 is None:
#         return False
#     if len(plan1) != len(plan2):
#         return False

#     for n1, n2 in zip(plan1, plan2):
#         s1 = n1.state["stateVec"]
#         s2 = n2.state["stateVec"]

#         # compare states
#         if np.linalg.norm(s1 - s2) > tol:
#             return False

#         c1 = n1.get_control()
#         c2 = n2.get_control()

#         if c1 is None and c2 is None:
#             continue
#         if (c1 is None) != (c2 is None):
#             return False

#         # compare controls
#         if np.linalg.norm(np.array(c1) - np.array(c2)) > tol:
#             return False

#     return True


def execute_plan_with_color(panda_sim, plan, rgb):
    """
    Execute a plan and draw the end-effector trajectory with a given color.
    args:
        panda_sim: simulator instance
        plan: a list of rrt.Node
        rgb: [r, g, b], e.g. [1, 0, 0] for red
    """
    if plan is None:
        return

    for i in range(1, len(plan)):
        node = plan[i]
        ctrl = node.get_control()
        if ctrl is None:
            continue

        # pose before motion
        prev_pos, _ = panda_sim.get_ee_pose()

        panda_sim.execute(ctrl)

        # pose after motion
        new_pos, _ = panda_sim.get_ee_pose()

        # draw one segment
        panda_sim.bullet_client.addUserDebugLine(
            prev_pos,
            new_pos,
            lineColorRGB=rgb,
            lineWidth=3,
            lifeTime=0
        )

##########################
