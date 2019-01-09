from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import re

import pandas 
from sys import exit 
import statsmodels.formula.api as smf
import numpy as np

def getfrom_user(optionsList):
	#ask user which columns to use for Y and X1, X2

	#ask user what column has the y dependent values
	print("File opened!")

	while True:
		print("Which column has the y DEPENDENT variable?")
		for counter in range(0,len(optionsList)):
			print(f"{optionsList[counter]} ---> Select {counter}")

		responseY = input(f"Enter your selection. Press Q to quit: ")

		if responseY.upper() == "Q":
			exit("Early Exit")

		elif 0 <= int(responseY) <= len(optionsList):
			break

	responseY = int(responseY)


	#ask user what column has x1 values
	while True:
		print("Which column has the x1 INDEPENDENT variable?")
		for counter in range(0,len(optionsList)):
			print(f"{optionsList[counter]} ---> Select {counter}")

		responseX1 = input(f"Enter your selection. Press Q to quit: ")

		if responseX1.upper() == "Q":
			exit("Early Exit")

		elif int(responseX1) == int(responseY):
			print("Sorry, that's already been chosen. Try again")


		elif 0 <= int(responseX1) <= len(optionsList):
			break

	responseX1 = int(responseX1)

	#ask user what column has x2 values
	while True:
		print("Which column has the x2 INDEPENDENT variable?")
		for counter in range(0,len(optionsList)):
			print(f"{optionsList[counter]} ---> Select {counter}")

		responseX2 = input(f"Enter your selection. Press Q to quit: ")

		if responseX2.upper() == "Q":
			exit("Early Exit")

		elif int(responseX2) == int(responseY) or int(responseX2) == int(responseX1):
			print(f"Sorry. Try again. \n \n ")
			

		elif 0 <= int(responseX2) <= len(optionsList):
			break

	responseX2 = int(responseX2)

	return (responseY, responseX1, responseX2)

def open_file():
	#aks user which file to open and then opens the file

	pattern = re.compile(r"[^A-Za-z0-9_\-\\\.]")

	while True:	
		print("Enter the EXCEL filename you want to open.")
		print(r"Filename must not contain these characters: [\/-@#$%&*?`~:;'{}|<>]")
		filename = input(r"The file must be in the same directory. Press Q to exit. ")

		if filename.upper() == "Q":
			exit("Early Exit")

		result = pattern.findall(filename)

		#check if length of result is empty
		if len(result) == 0:
			break

	#test to see if file can be found or exists
	try:
		with open(filename) as file:
	
			pass
	
	except FileNotFoundError as err:
		print(err)
		exit("Early Exit")

	return filename
	
def update_title():
	#generator function
	#updates the scatter chart title

	max = 0

	while max < 3:
		title = "3D Scatter Chart:" + "\n" + scatter.Yname + " = " + str(list(scatter.LRmodel.params)[0])[0:5] + " + " + str(list(scatter.LRmodel.params)[1])[0:5] + scatter.X1name + " + " + str(list(scatter.LRmodel.params)[2])[0:5]+scatter.X2name

		yield title

		max = max + 1

class Data:

	def __init__(self, filename="butler.xlsx"):
		print("Opening File...")
		
		filename = open_file()
		data_file = pandas.read_excel(filename)

		#get list of column names
		optionsList = tuple(data_file.columns.values)

		#of the list of variables, which one should be y dependent and x1, x2 independent?
		responses = getfrom_user(optionsList)

		responseY = optionsList[responses[0]]
		responseX1 = optionsList[responses[1]]
		responseX2 = optionsList[responses[2]]

		self.data = data_file 

		self.Yname = responseY
		self.X1name = responseX1
		self.X2name = responseX2

		self.pointColor = "tab:red"
		self.edgeColor = (0,0,0,1)  #black RGBA format  (r, g, b, a)
		self.selectColor = (0, 1, 0, 1)   #green RGBA format  (r, g, b, a)

		self._fc = None
		self._coll = None
		self._3doffsets = None

		self.LRmodel = None
		self.LRmodel_eqn = None

		self._dataChg = None 
		self.title = None

	def selectColors(self, color = (0, 1, 0, 1)):
		#sets the color of the user selected scatter plot point
		# RGBA format (r, g, b, a)

		self.selectColor = color

		return self.selectColor

	def edgeColors(self, color = (0,0,0,1)):
		#sets the color of the scatter plot points' borders
		# RGBA format (r, g, b, a)

		self.edgeColor = color

		return self.edgeColor

	def pointColors(self, color = "tab:red"):
		#sets the color of the scatter plot points

		self.pointColor = color

		return self.pointColor

	def scatterPlot(self):
		#draws the scatterplot
		
		#create scatter plot
		#colors must be plainted on individually to enable point selection
		coll = ax.scatter( scatter.data[scatter.X1name].values, scatter.data[scatter.X2name].values, scatter.data[scatter.Yname].values, facecolors=[scatter.pointColors()]*len(scatter.data[scatter.X1name].values), edgecolors=[scatter.edgeColors()]*len(scatter.data[scatter.X1name].values),  depthshade=False, picker=4)

		self._fc = coll.get_facecolors()

		self._coll = coll

		#source: https://stackoverflow.com/questions/51716696/extracting-data-from-a-3d-scatter-plot-in-matplotlib
		self._3doffsets = np.array(coll._offsets3d).T

		#Update scatter chart title:
		
		scatter.title = update_title()
		
		



	def lin_R(self):
		#Set up for Linear Regression and create Linear Regression plane
		#Python method: Formula API

		#put  X independent values into Dataframe
		df = pandas.DataFrame(scatter.data,columns=[ scatter.X1name, scatter.X2name ])

		#put in Y dependant value
		df[scatter.Yname] = pandas.Series(scatter.data[scatter.Yname].values) 


		#Generate the model using OLS = Ordinary Least Squares

		formula = str(scatter.Yname)+ " ~ " + str(scatter.X1name) + " + " + str(scatter.X2name)


		model = smf.ols( formula , data=df)

		model_formula = model.fit()

		self.LRmodel = model_formula		

		#Draw Regression Plane

		#create a Meshgrid to draw a regression plane
		#https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy

		#min/max for X1 values
		X1mn = np.min( df[ scatter.X1name ]  )
		X1mx = np.max( df[ scatter.X1name ]  )


		#min/max for X2 values
		X2mn = np.min( df[  scatter.X2name  ]  )
		X2mx = np.max( df[  scatter.X2name  ]  )

		x_mesh, y_mesh = np.meshgrid(np.linspace(X1mn, X1mx, num=100), np.linspace(X2mn, X2mx, num=100))

		#need to calculate z values for the plane using x and y values from the mesh
		#load x values
		onlyX = pandas.DataFrame({scatter.X1name: x_mesh.ravel(), scatter.X2name: y_mesh.ravel()})

		fittedY = model_formula.predict(exog=onlyX)


		y_fitted = fittedY.values.reshape(x_mesh.shape)

		# #edgecolors:
		# #https://www.mathworks.com/help/matlab/ref/surf.html#bvgppvs_sep_shared-EdgeColor
		# #https://stackoverflow.com/questions/21418255/changing-the-line-color-in-plot-surface
		ax.plot_surface( x_mesh, y_mesh, y_fitted, rstride=10, cstride=10, linewidth = 0.5, alpha=0.1, color='None', edgecolors='k')

def get_lbltext(*args):
	#accepts x1, x2, y2 values and puts them into a string for use as scatterpoint label

	text = scatter.X1name + ": " + str(args[0]) + "\n" + scatter.X2name + ": " + str(args[1]) + " \n" + scatter.Yname + ": " + str(args[2])

	return text



def plot_curves(indexes):
    #enables user to select individual points in the scatter plot

    #create new copy of array of index of scatter points with their individual colors

    new_fc = scatter._fc.copy() 

    offsets = scatter._3doffsets

    for i in indexes: # might be more than one point if ambiguous click
        
        #new_fc has index of all points with their respective colors
        #only index i needs to change its color
        new_fc[i,:] = scatter.selectColors()   # color of selected point; green RGBA format  (r, g, b, a)
        scatter._coll._facecolor3d = new_fc

        #no need to change edgecolor; it's set as black earlier
        #new_fc[i,:] = (0, 0, 0, 1)   #black RGBA format  (r, g, b, a)
        #coll._edgecolor3d = new_fc


    #remove previous label if it exists
    try:
    	label = scatter._dataChg[3]
    	label.remove()
    
    except TypeError as err:
        #error happens if there is no previous label
        #in that case, just move on
        pass 

    
    #annotate selected scatter point
    #source: https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
    x1 = offsets[i][0]
    x2 = offsets[i][1]
    y2 = offsets[i][2]

    x3, y3, _ = proj3d.proj_transform(x1,x2,y2, ax.get_proj())

    label = pyplot.annotate(get_lbltext(x1,x2,y2), xy = (x3, y3), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 1), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    
    scatter._dataChg = (x1,x2,y2,label)



    fig.canvas.draw_idle()


def onpick(event):
    #get index of scatter point clicked. returns in array format
    

    ind = event.ind
    plot_curves(list(ind))
	
def set_xyz_labels():
	#set up the x,y,z axis labels

	ax.set_xlabel(scatter.X1name)
	ax.set_ylabel(scatter.X2name)
	ax.set_zlabel(scatter.Yname)

def update_position(e):
    #update position of label after user initiated changes

	try:
		data = scatter._dataChg  #scatter._dataChg = (x1,x2,y2,label)

	  
		x1 = data[0]
		x2 = data[1]
		y2 = data[2]

		label = data[3]

		x3, y3, _ = proj3d.proj_transform(x1,x2,y2, ax.get_proj())
	    
		label.xy = x3,y3
	    
		label.update_positions(fig.canvas.renderer)
	    
		fig.canvas.draw_idle()	
	except TypeError as err:
		#this just means user has not selected a scatter point, thus, no label exists
		pass



scatter = Data()

fig = pyplot.figure()  #creating a figure which keeps track of everything

ax = fig.add_subplot(111, projection='3d')

# Ensure that the next plot doesn't overwrite the first plot
#source: https://stackoverflow.com/questions/36060933/matplotlib-plot-a-plane-and-points-in-3d-simultaneously
ax = pyplot.gca()




#create and populate the scatter plot
scatter.scatterPlot()

#create the linear regression model and the linear regression plane
scatter.lin_R()

#attributes of Statsmodels here:
#http://www.statsmodels.org/devel/_modules/statsmodels/regression/linear_model.html#RegressionResults

#print(scatter.LRmodel.summary())
# print(scatter.LRmodel.pvalues)
#str(list(scatter.LRmodel.params)[0])[0:5]


set_xyz_labels()

fig.suptitle(next(scatter.title))

#make the scatter plot points clickable
fig.canvas.mpl_connect('pick_event', onpick)

#post labels when scatter point is clicked
fig.canvas.mpl_connect('button_release_event', update_position)

pyplot.show()

print("Program Ends")



