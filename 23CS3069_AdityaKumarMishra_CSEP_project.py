import pandas as pd
import numpy as np


from  sklearn import metrics
from  sklearn import tree
import warnings

warnings.filterwarnings('ignore')


data=pd.read_csv('Crop_recommendation.csv')
data.head()
data.tail()
data.size
data.shape
data.columns
data['label'].unique()
data.dtypes
data['label'].value_counts()
features=data[['N','P','K','temperature','humidity','ph','rainfall']]
target=data['label']
labels=data['label']
acc=[]
model=[]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(features,target,test_size=0.2,random_state=2)  #splitting 20% data for testing and 80% for training
features.size

xtrain.size  # to verify  that splitting has actually been done

from sklearn.tree import DecisionTreeClassifier
acc=[]
model=[]
DecisionTree=DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
DecisionTree.fit(xtrain,ytrain)
predicted_values=DecisionTree.predict(xtest)
x=metrics.accuracy_score(ytest,predicted_values)

print("DecisionTree's Accuracy is:",x*100)






#now deploying RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=20,random_state=0)
RF.fit(xtrain,ytrain)
predicted_values=RF.predict(xtest)
x=metrics.accuracy_score(ytest,predicted_values)
acc.append(x)
model.append('RF')
print("RF accuracy is : ",x*100)





#inputting the values from the farmer for N,P,K,Temp etc...
lst=[]  #to store the data input
print("\n\n\n  please input the values of \n  1.Nitrogen content (kg/hectare)\n  2.Phosphorous content (kg/hectare)\n  3.Potassium Content (kg/hectare)\n  4.Average Temperature\n  5.Relative Humidity\n  6.pH and \n  7.annual rainfall (in mm) at your field in the same order as mentioned..")

for i in range(0, 7):
    ele = float(input())
    # adding the element
    lst.append(ele) 


#now I will predict data using the two methods I have made
#first I will use the Decision tree method to predict

prediction=DecisionTree.predict([lst])
print('DEAR FARMER YOU MUST CULTIVATE ',prediction,'   AT YOUR FARM FOR BEST YIELD(as per DecisionTree Method)')

 

# now I will use the RandomForestClassifier method to predict
prediction=RF.predict([lst])
print('DEAR FARMER YOU MUST CULTIVATE ',prediction,'   AT YOUR FARM FOR BEST YIELD (as per RandomforestClassifier Method....MORE PREFERABLE as it is more accurate)')