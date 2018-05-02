# -*- coding: utf-8 -*-

import tkinter as tk


import matplotlib
matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import decision_tree_in_action_cart.regression_compare as tree





def reDraw(tolS, tolN):
    # 清除之前的图像
    reDraw.f.clf()

    reDraw.a = reDraw.f.add_subplot(111)  # 添加子图

    if chkBtnVar.get():  # 检查复选框是否被选中
        if tolN < 2:
            tolN = 2
        myTree = tree.createTree(reDraw.rawDat, tree.modelLeaf, tree.modelErr, (tolS, tolN))
        yHat = tree.createForecast(myTree, reDraw.testDat, tree.modelTreeEval)
    else:
        myTree = tree.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = tree.createForecast(myTree, reDraw.testDat)

    # 真实值描点, 预测值用于绘制拟合曲线
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.a.scatter(reDraw.rawDat[:, 0].A1, reDraw.rawDat[:, 1].A1, s=5)

    reDraw.canvas.show()


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("Enter Integer for tolN")
        # 清空输入, 并用默认值填充
        tolNentry.delete(0, tk.END)
        tolNentry.insert(0, "10")
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("Enter Integer for tolS")
        tolSentry.delete(0, tk.END)
        tolSentry.insert(0, "1.0")

    return tolN, tolS



def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)



root = tk.Tk()


# columnspan, rowspan 指示是否允许组件跨列或跨行
tk.Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)

tk.Label(root, text="tolN").grid(row=1, column=0)

# 文本输入框组件
tolNentry = tk.Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, "10")  # 预填充值

tk.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = tk.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, "1.0")

tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)

# 复选按钮的变量值
chkBtnVar = tk.IntVar()
# 复选按钮组件
chkBtn = tk.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)


reDraw.rawDat = np.mat(tree.loadDataSet("sine.txt"))
reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)

reDraw(1.0, 10)


root.mainloop()
