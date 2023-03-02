import collections

BT = collections.namedtuple("BT", ["value", "left", "right"])
BT.__doc__= """
A Binary Tree (BT) with a node value, and left- and right-subtrees.
"""
NNRecord = collections.namedtuple("NNRecord", ["point", "distance"])
NNRecord.__doc__ = """
Used to keep track of the current best guess during a nearest neighbor search.
"""

class KD_Tree:
    def __init__(self, points, indicies=[0,1]):
        self.indicies = indicies
        self.k = len(indicies)
        self.build_tree(points)

    def SED(self, X,Y):
        """Compute the squared Euclidean distance between X and Y."""
        return sum((X[cor] - Y[cor])**2 for cor in self.indicies)
    
    def build_tree(self, points):
        """Construct a k-d tree from an iterable of points.

        This algorithm is taken from Wikipedia. For more details,

        > https://en.wikipedia.org/wiki/K-d_tree#Construction
        """

        def build(*, points, depth):
            """Build a k-d tree from a set of points at a given depth."""

            if len(points) == 0:
                return None
            points.sort(key=lambda x:x[self.indicies[depth%self.k]])
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
        self.tree = build(points=points, depth=0)

    def find_nearest_neighbors(self, point, previous_best=[], num_neighbors=4):
        """Find the nearest neighbor in a k-d tree for a given point.
        """

        if previous_best is None or len(previous_best) == 0:
            best_list = []
        elif len(previous_best) > 0:
            new_list = []
            for item in previous_best:
                distance = self.SED(point, item)
                new_item = NNRecord(item, distance)
                new_list.append(new_item)
            best_list = new_list
            best_list.sort(key=lambda x:x.distance)

        def add_to_best(item):
            nonlocal best_list

            i = len(best_list) - 1
            while i > 0:
                if best_list[i-1].distance <= item.distance:
                    break
                i -= 1
            if i > 0 and best_list[i-1].point[self.indicies[0]] == item.point[self.indicies[0]] and best_list[i-1].point[self.indicies[1]] == item.point[self.indicies[1]]:
                return
            best_list.insert(i, item)
            if len(best_list) > num_neighbors:
                best_list.pop()
        def search(tree, depth):
            """Recursively search through the k-d tree to find the nearest neighbor.
            """
            nonlocal best_list

            if tree is None:
                return
            
            distance = self.SED(tree.value, point)
            if len(best_list) == 0 or len(best_list) < 4 or distance < best_list[-1].distance:
                new_item = NNRecord(point=tree.value, distance=distance)
                add_to_best(new_item)
            
            axis = depth % self.k
            diff = point[self.indicies[axis]] - tree.value[self.indicies[axis]]
            if diff <= 0:
                close, away = tree.left, tree.right
            else:
                close, away = tree.right, tree.left
            search(tree=close, depth=depth+1)
            if diff**2 < best_list[-1].distance:
                search(tree=away, depth=depth+1)
        
        search(self.tree, depth=0)
        new_list = []
        for item in best_list:
            new_list.append(item.point)
        return new_list