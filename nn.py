"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from miniflow import *

t, u, v, w, x, y, z = Input(), Input(), Input(), Input(), Input(), Input(), Input()

f = Add(y, z)
g = Add(f, u, v)
i = Mul(g, w, x)
h = Mul(i, t)

feed_dict = {t: 3, u: 13, v: 7, w: 12, x: 10, y: 5, z: 8}

sorted_nodes = topological_sort(feed_dict)
output = forward_pass(h, sorted_nodes)

# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
print("({} + {} + {} + {}) * {} * {} * {} = {} (according to miniflow)".format(
    feed_dict[y],
    feed_dict[z],
    feed_dict[u],
    feed_dict[v],
    feed_dict[w],
    feed_dict[x],
    feed_dict[t],
    output))
