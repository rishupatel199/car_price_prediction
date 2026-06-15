import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
df=pd.read_csv("car.csv")
df=df.drop(["doornumber","carbody","price"],axis=1)
df=pd.get_dummies(df,columns=["fueltype"],dtype=int)
x=df.drop(["price_inr"],axis=1)
y=df["price_inr"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(x_train, y_train)
pickle.dump(model, open("model.pkl", "wb"))