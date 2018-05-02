# -*- coding: utf-8 -*-

from decision_tree_in_action_id3.tree_in_action import createTree
from decision_tree_in_action_id3.tree_plotter import createPlot


with open("lenses.txt") as f:
    lenses = [inst.strip().split("\t") for inst in f.readlines()]
    lensesLabels = ["age", "prescript", "astigmatic", "tearRate"]
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)

    createPlot(lensesTree)

