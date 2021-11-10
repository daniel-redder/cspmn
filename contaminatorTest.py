import numpy
import random

class contaminator():
    def __init__(self):
        self.rng = numpy.random.default_rng(seed=42)

    def e_contam(self, node):
        # contaminate them
        node.weight = self.rng.dirichlet(alpha=[random.randint(1,100) for x in node.weight])


class child():
    def __init__(self,weights):
        self.weight=weights

    def toString(self):
        return self.weight


inp = child([4,3,2])
cont = contaminator()

print(cont.e_contam(inp))
print(inp.weight)