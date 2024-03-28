class Node:
    def __str__(self):
        return type(self).__name__

    def __eq__(self, other):
        return isinstance(other, Node)

    def __lt__(self, other):
        self_str = str(self)
        other_str = str(other)
        if self_str == "Hole" and other_str == "Hole":
            return self_str < other_str
        elif self_str == "Hole":
            return False
        elif other_str == "Hole":
            return True
        elif "Hole" in self_str and "Hole" in other_str:
            return self_str < other_str
        elif "Hole" in other_str:
            return True
        elif "Hole" in self_str:
            return False
        return self_str < other_str


class Formula:
    pass


class ForAll(Formula):
    def __init__(self, var, subformula):
        self.var = var
        self.subformula = subformula

    def __str__(self):
        return "ForAll {}.({})".format(self.var, str(self.subformula))


class Exists(Formula):
    def __init__(self, var, subformula):
        self.var = var
        self.subformula = subformula

    def __str__(self):
        return "Exists {}.({})".format(self.var, str(self.subformula))


class Subformula(Formula):
    pass


class And(Formula):
    def __init__(self, subformula1, subformula2):
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def __str__(self):
        return "({}) And ({})".format(str(self.subformula1), str(self.subformula2))


class IfThen(Formula):
    def __init__(self, subformula1, subformula2):
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def __str__(self):
        return "({}) -> ({})".format(str(self.subformula1), str(self.subformula2))


class Not(Formula):
    def __init__(self, subformula):
        self.subformula = subformula

    def __str__(self):
        return "Not ({})".format(str(self.subformula))


class Predicate(Formula):
    pass


class Is(Predicate):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def __str__(self):
        return type(self).__name__ + "(" + str(self.var1) + "," + str(self.var2) + ")"


class IsAbove(Predicate):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def __str__(self):
        return type(self).__name__ + "(" + str(self.var1) + "," + str(self.var2) + ")"


class IsLeft(Predicate):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def __str__(self):
        return type(self).__name__ + "(" + str(self.var1) + "," + str(self.var2) + ")"


class IsNextTo(Predicate):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def __str__(self):
        return type(self).__name__ + "(" + str(self.var1) + "," + str(self.var2) + ")"


class IsInside(Predicate):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def __str__(self):
        return type(self).__name__ + "(" + str(self.var1) + "," + str(self.var2) + ")"
