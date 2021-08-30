import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
experience=[[1.0],[1.5],[2.0],[2.5],[3.0],[3.5],[4.0],[4.5],[5.0],[5.5]]
salary=[  10000, 15000 , 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
plt.scatter(experience,salary,color='black')
plt.xlabel("Experience (years)")
plt.ylabel("Salary")
reg=LinearRegression()
reg.fit(experience,salary)
exp=[[8.5]]
print(reg.predict(exp))
