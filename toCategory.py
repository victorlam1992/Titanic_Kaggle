def toCategory(X, listOfVar):
	for i in list(range(0,len(listOfVar))):
		varName = listOfVar[i]
		X[varName] = X[varName].astype('category')
	
	cat_columns = X.select_dtypes(['category']).columns
	X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)