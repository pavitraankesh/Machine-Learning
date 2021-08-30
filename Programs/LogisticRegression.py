from sklearn.linear_model import LogisticRegression
age = [[42],[16],[5],[67],[18],[10],[34],[9]]
eligible = [1,0,0,1,1,0,1,0]
classifier=LogisticRegression()
classifier.fit(age,eligible)
age1=[[15]]
print(classifier.predict(age1))
if(classifier.predict(age1)):
  print("Eligible to vote.")
else:
  print("Not eligible to vote.")
