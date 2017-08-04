import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # The value of the Node
        self.value = None

        # Node(s) from which this nodes receives values
        self.inbound_nodes = inbound_nodes

        # Node(s) to which this node sends values
        self.outbound_nodes = []

        # Link this Node as an outbound for every inbound Node
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)

        # Gradients of this node for each input
        # Keys are input nodes, values are partial derivatives
        self.gradients = {}

    def forward(self):
        """
        Forward propagation

        Compute the output value based on `inbound_nodes` and
        store the result in `value`
        :return:
        """
        raise NotImplementedError

    def backward(self):
        """
        Backward propagation
        :return:
        """
        raise NotImplementedError


class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes
        Node.__init__(self)

    # NOTE: An Input node is the only type of node where a
    # value can be passed as an argument to `forward`
    #
    # All other nodes receive their values from `inbound_nodes`
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        """
        Calculates the forward pass. Performs no calculation.

        Usually, value is not passed in as parameter, but initialized
        in the `feed_dict` argument to `topological_sort()`. If value
        argument supplied, Node `value` is overwritten.
        :param value:
        :return:
        """
        if value is not None:
            # Overwrite `value` if argument passed in
            self.value = value

    def backward(self):
        """
        Calculates the backward pass. Used in Linear, for example.

        Input nodes have no `inbound_nodes` so their derivative is 0.
        However, weights and biases are usually input nodes, so their
        `outbound_nodes` gradients should be summed up.
        :return:
        """
        self.gradients = {self: 0}  # Zero gradient for the node itself
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

# TODO - upgrade Add and Mul, add nodes for other simple arithmetic
class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = sum(n.value for n in self.inbound_nodes)

class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 1
        for n in self.inbound_nodes:
            self.value *= n.value

# class Linear(Node):
#     def __init__(self, inputs, weights, bias):
#         Node.__init__(self, [inputs, weights, bias])
#
#         # NOTE: The weights and bias properties here are not
#         # numbers, but rather references to other nodes.
#         # The weight and bias values are stored within the
#         # respective nodes.
#
#     def forward(self):
#         """
#         Set self.value to the value of the linear function output.
#
#         Your code goes here!
#         """
#         self.value = sum(i*w for i,w in zip(self.inbound_nodes[0].value,
#                                             self.inbound_nodes[1].value)) + \
#                      self.inbound_nodes[2].value

class Linear(Node):
    def __init__(self, X, W, b):
        # Notice the ordering of the input nodes passed to the
        # Node constructor.
        Node.__init__(self, [X, W, b])

    def forward(self):
        """
        Set the value of this node to the linear transform output.
        """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b

    def backward(self):
        """
        Perform the backpropagation for all three inputs
        """
        # Initialize partials for all input nodes
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Sum the corresponding gradients over the outbound nodes
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            # Set partial of node with respect to its inputs (X)
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set partial of node with respect to its (inbound) weights
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set partial of node with respect to its bias
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    """
    You need to fix the `_sigmoid` and `forward` methods.
    """
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used later with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.

        Your code here!
        """
        return 1. / (1. + np.exp(-x))


    def forward(self):
        """
        Set the value of this node to the result of the
        sigmoid function, `_sigmoid`.

        Your code here!
        """
        # This is a dummy value to prevent numpy errors
        # if you test without changing this method.
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Peform the backward propagation.

        The sigmoid derivative f'(x) = s(x)[(1-s(x)]
        :return:
        """
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sig_input = self.value
            self.gradients[self.inbound_nodes[0]] += sig_input * (1-sig_input) * grad_cost


class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        # Call the base class' constructor.
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y-a

        # column vectors: can use len() and np.mean() directly
        self.value = np.mean(np.square(self.diff))

    def backward(self):
        """
        Perform backpropagation.

        This is an output node, so backpropagation starts with it.
        It has no outbound nodes to sum gradients for.
        """
        # Note: self.diff restores the output dimensions for backpropagation
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff  # y is an input
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff  # NN output


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    # Build the graph G
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    # Build the topological sort L
    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


# def forward_pass(output_node, sorted_nodes):
#     """
#     Performs a forward pass through a list of sorted nodes.
#
#     Arguments:
#
#         `output_node`: A node in the graph, should be the output node (have no outgoing edges).
#         `sorted_nodes`: A topologically sorted list of nodes.
#
#     Returns the output Node's value
#     """
#
#     for n in sorted_nodes:
#         n.forward()
#
#     return output_node.value

def forward_and_backward(graph):
    """
    Performs a forward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # Performs SGD
    #
    # Loop over the trainables
    for t in trainables:
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this
        # trainable.
        partial = t.gradients[t]
        t.value -= learning_rate * partial
