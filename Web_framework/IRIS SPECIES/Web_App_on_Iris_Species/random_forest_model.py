import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

iris=pd.read_excel("iris.xls")
x=iris.drop("Classification",axis=1)
y=iris.Classification
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf = rf_clf.fit(x_train, y_train)
y_pred_rf = rf_clf.predict(x_test)
pickle.dump(rf_clf,open('model.pkl','wb'))