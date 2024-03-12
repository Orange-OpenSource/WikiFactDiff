import pydot
import random

# Node attributes
node_attrs = {
    "shape": "box",
    "style": '"rounded,filled"',
    "fontname": "Helvetica",
    "fontsize": 12,
    "color": "#3498db",
    "fillcolor": "#a9c6d9",
    "margin": 0.2,
}

# Edge attributes
edge_attrs = {
    "color": "#95a5a6",
    "weight": 2
}

# Edge attributes
edge_attrs_down = {
    "color": "#95a5a6",
    "weight": 2,  # Straight down
}

edge_attrs_right = {
    "color": "#95a5a6",
    "weight": 1,  # To the right
}

_weight_idx = 0


def get_node(graph, name):
    return [x for x in graph.get_nodes() if name == x.get_label('label')]

def draw(parent_name, child_name, graph, weight_order):
    global _weight_idx
    if len(get_node(graph, parent_name)) == 0:
        node = pydot.Node(str(random.randint(0, 2**32)), label=parent_name, **node_attrs)
        # node.set_label(parent_name)
        graph.add_node(node)
    node = pydot.Node(str(random.randint(0, 2**32)), label=child_name, **node_attrs)
    # node.set_label(child_name)
    graph.add_node(node)
    if weight_order:
        attrs = edge_attrs.copy()
        attrs['weight'] = weight_order[_weight_idx]
        _weight_idx = (_weight_idx + 1) % len(weight_order)
    else:
        attrs = edge_attrs
    edge = pydot.Edge(get_node(graph,parent_name)[0], node, **attrs)
    graph.add_edge(edge)
    
def _visit(node, parent=None, graph=None, weight_order=None):
        if isinstance(node, dict):
            for k,v in node.items():
            # We start with the root node whose parent is None
            # we don't want to graph the None node
                if parent:
                    draw(parent, k, graph, weight_order)
                _visit(v, k, graph, weight_order)
        elif isinstance(node, (tuple, list)):
            for x in node:
                _visit(x, parent, graph, weight_order)

        else:
            draw(parent, node, graph, weight_order)
            # drawing the label using a distinct name
            # draw(k, k+'_'+v, graph)

def visit(node, graph, weight_order):
    _visit(node, None, graph, weight_order)

def tree_plot(tree : dict, transform_node=None, weight_order : list = None):
    global _weight_idx
    graph = pydot.Dot(rankdir="TB", bgcolor="#ffffff", splines=True, nodesep=0.4)
    _weight_idx = 0
    visit(tree, graph, weight_order)
    if transform_node:
        for x in graph.get_nodes():
            transform_node(x)
    return graph

