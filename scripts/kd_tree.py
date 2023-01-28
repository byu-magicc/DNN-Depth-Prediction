import collections
import operator

"""
This algorithm is taken from https://johnlekberg.com/blog/2020-04-17-kd-tree.html and modified
to work with features as inputs (which have the fields "pos_x" and "pos_y") in a 2d space
"""

BT = collections.namedtuple("BT", ["value", "left", "right"])
BT.__doc__= """
A Binary Tree (BT) with a node value, and left- and right-subtrees.
"""


def SED(X, Y):
    """Compute the squared Euclidean distance between X and Y."""
    xy=["pos_x", "pos_y"]
    return sum((X[cor] - Y[cor])**2 for cor in xy)

def kdtree(points):
    """Construct a k-d tree from an iterable of points.

    This algorithm is taken from Wikipedia. For more details,

    > https://en.wikipedia.org/wiki/K-d_tree#Construction
    """
    k = 2
    xy=["pos_x", "pos_y"]

    def build(*, points, depth):
        """Build a k-d tree from a set of points at a given depth."""

        if len(points) == 0:
            return None
        points.sort(key=lambda x:x[xy[depth%k]])
        middle = len(points)//2

        return BT(
            value = points[middle],
            left = build(
                points=points[:middle],
                depth=depth+1
            ),
            right = build(
                points=points[middle+1:],
                depth=depth+1
            )
        )
    return build(points=points, depth=0)

NNRecord = collections.namedtuple("NNRecord", ["point", "distance"])
NNRecord.__doc__ = """
Used to keep track of the current best guess during a nearest neighbor search.
"""

def find_nearest_neighbor(*, tree, point, best_list=[]):
    """Find the nearest neighbor in a k-d tree for a given point.
    """

    k=2
    xy=["pos_x", "pos_y"]

    if best_list is None:
        best_list = []
    elif len(best_list) == 4:
        
    def add_to_best(item):
        nonlocal best_list

        i = len(best_list) - 1
        while i > 0:
            if best_list[i-1].distance < item.distance:
                break
            i -= 1
        best_list.insert(i, item)
        if len(best_list) > 4:
            best_list.pop()
    def search(*, tree, depth):
        """Recursively search through the k-d tree to find the nearest neighbor.
        """
        nonlocal best_list

        if tree is None:
            return
        
        distance = SED(tree.value, point)
        if len(best_list) == 0 or distance < best_list[-1].distance:
            new_item = NNRecord(point=tree.value, distance=distance)
            add_to_best(new_item)
        
        axis = depth % k
        diff = point[xy[axis]] - tree.value[xy[axis]]
        if diff <= 0:
            close, away = tree.left, tree.right
        else:
            close, away = tree.right, tree.left
        search(tree=close, depth=depth+1)
        if diff**2 < best_list.distance:
            search(tree=away, depth=depth+1)
    
    search(tree=tree, depth=0)
    return best_list
