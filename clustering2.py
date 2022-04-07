# Import necessary libaries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import plotly.graph_objects as go
import plotly.figure_factory as ff

# Import the dataframe
features_df = pd.read_csv('music_data.csv')

# Extra columns if necessary
# features_df = features_df[['danceability', 'energy', 'speechiness', \
#     'acousticness', 'valence', 'tempo', 'Category']]

# Columns we want to use for the classification
features_df = features_df[['energy', 'acousticness', 'valence', 'Category']]

# Correctly subtract the data for machine learning
X, Y = features_df.drop('Category', axis=1), features_df['Category']

# Run machine learning algorithm
neigh = KNeighborsClassifier(n_neighbors=5)

# Run cross-validation
predictions_neigh = cross_val_predict(neigh, X, Y, cv=5)
k_score_neigh = cross_val_score(neigh, X, Y, cv=5)

# Calculate accuracy
accuracy_score = k_score_neigh.mean()

# Print the accuracy
print("Accuracy: ", accuracy_score)

# Generate confusion matrix
conf_matrix = confusion_matrix(Y, predictions_neigh)

z = conf_matrix
x = ['Predicted: Summer', 'Predicted: Winter']
y = ['Actual: Summer', 'Actual: Winter']

# Plot a confusion matrix
fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z)
fig.show()

