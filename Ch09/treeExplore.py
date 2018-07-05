from tkinter import *
from numpy import *
from machineLearnInAction.Ch09.regTrees import *
import matplotlib
matplotlib.use('TKAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def redraw(tolS,tolN):
    print('redraw the plot tols :%d, toln:%d',tolS,tolN)
    redraw.f.clf()
    redraw.a = redraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN<2:tolN=2
        mytree = createTree(redraw.rawData,modelLeaf,modelErr,(tolS,tolN))
        # yHat = cre


def drawNewTree():
    print('redraw the drawNewTree')
    pass

root = Tk()

# Label(root,text='Plot Place Holder').grid(row=0,columnspan=3)

redraw.f = Figure(figsize=(5,4),dpi=100)
redraw.canvas = FigureCanvasTkAgg(redraw.f,master=root)
redraw.canvas.show()
redraw.canvas.get_tk_widget().grid(row=0,columnspan=3)

Label(root,text='tolN').grid(row=1,column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10')

Label(root,text='tolS').grid(row=2,column=0)
tolsEntry = Entry(root)
tolsEntry.insert(0,'1.0')
tolsEntry.grid(row=2,column=1)

Button(root,text='redraw',command=drawNewTree).grid(row=1,column=2,rowspan=3)

chkBtnVar = IntVar()
chkBtnVar = Checkbutton(root,text='Model Tree',variable=chkBtnVar)
chkBtnVar.grid(row=3,column=0,columnspan=2)

redraw.rawData = mat(loadDataSet('sine.txt'))
redraw.testData = arange(min(redraw.rawData[:,0]),max(redraw.rawData[:,0]),0.01)
redraw(1.0,10)
root.mainloop()


