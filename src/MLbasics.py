import pandas as pd
import matplotlib.pylab as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# EXTRACT
df = pd.read_csv("../datasets/billboard.csv")

# TRANSFORM - clear unrelevant columns and remove NA rows
df['week_id'] = pd.to_datetime(df['week_id'], errors='coerce')
df['month'] = df['week_id'].dt.month
df['year'] = df['week_id'].dt.year

# Understand correlation
# print(df.drop(columns=["url", "song", "performer", "song_id", "instance"]).corr())

# TRANSFORM - clear uncorrelated or undesired columns and remove NA rows
clean_df = df.drop(columns=["url", "song", "performer", "song_id", "instance",
                            "month", "week_id", "year"])  # Weak or no correlation
filtered_df = clean_df.dropna()

X = filtered_df.drop(columns=["peak_position"])  # week_position,previous_week_position,weeks_on_chart
print(X.shape)
y = filtered_df["peak_position"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Show data
plt.scatter(X.values[:, 0], X.values[:, 1], c=y.values)
plt.show()

def confusion_matrix_pd_convertor(clf, predictions):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    GaussianNB(),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42)
]

for clf in classifiers:
    clf.fit(X_train.values, y_train.values)
    predictions = clf.predict(X_test.values)
    print(f"{type(clf).__name__} - accuracy_score: {accuracy_score(y_test.values, predictions)}")