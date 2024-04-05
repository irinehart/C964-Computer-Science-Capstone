import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

df = pd.read_csv('study_performance.csv')

df.hist()
pyplot.show()
scatter_matrix(df)
pyplot.show()

mylog_model = linear_model.LogisticRegression()
x = df.values[:,5:8]
y = df.values[:,0]

mylog_model.fit(x,y)

y_pred = mylog_model.predict(x)

print("This program aims to predict if you are a boy or a girl based on your grades.")
print("We are interested in evaluating if our school has improved in supporting boys academically, considering that girls traditionally outperform boys in school.")
print("This program has a prediction accuracy of", metrics.accuracy_score(y, y_pred), "percent.")
print("")

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
pyplot.show()

def validate_grade(prompt):
    while True:
        try:
            grade = float(input(prompt))
            if 0 <= grade <= 100:
                return grade
            else:
                print("Please enter a valid grade between 0 and 100.")
        except ValueError:
            print("Please enter a valid numeric grade.")

for i in range(len(x)):
    if i == 0:
        math_grade = validate_grade("What is your grade in math? ")
    if i == 1:
        reading_grade = validate_grade("What is your grade in reading? ")
    if i == 2:
        writing_grade = validate_grade("What is your grade in writing? ")

y_pred = mylog_model.predict([[math_grade, reading_grade, writing_grade]])
print("The data suggests you are a", y_pred)

valid_responses = ["y", "n", "yes", "no", "NO", "No", "YES", "Yes", "Y", "N"]
while True:
    response = input("Is this accurate (Y/N)? ").lower()
    if response in valid_responses:
        break
    else:
        print("Please enter Y or N.")

print("Thank you for participating have a good day!")