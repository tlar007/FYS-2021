import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv('SpotifyFeatures.csv')

# Problem 1a: Report the number of samples and features.
num_samples = df.shape[0]
num_features = df.shape[1]
#print(f"Number of samples: {num_samples}, Number of features: {num_features}")

# Problem 1b: Filter the dataset for 'Pop' and 'Classical' genres and create labels.
filtered_df = df[df['genre'].isin(['Pop', 'Classical'])].copy()  # .copy() to avoid SettingWithCopyWarning
filtered_df['label'] = (filtered_df['genre'] == 'Pop').astype(int)
pop_count = filtered_df[filtered_df['genre'] == 'Pop'].shape[0]
classical_count = filtered_df[filtered_df['genre'] == 'Classical'].shape[0]
#print(f"Number of 'Pop' samples: {pop_count}, Number of 'Classical' samples: {classical_count}")

# Problem 1c: Select features 'liveness' and 'loudness' and split the data
features = filtered_df[['liveness', 'loudness']].values
labels = filtered_df['label'].values
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, shuffle=True, random_state=42)

data = X_train, X_test, y_train, y_test

""" # Counting, Max and Min value of features 
for i in data:
    print(i[0:10])

X_1_max = 0
X_1_min = 100

for i in data[:2]:
    print(i[:10])
    
    for X in i:
        # X_1
        if X[1] > X_1_max:
            X_1_max = X[1]
        if X[1] < X_1_min:
            X_1_min = X[1]

print(X_1_max, X_1_min)


for i in data[2:]:
    pop_count_array = sum(i)
    classical_count_array = len(i)-sum(i)
    print(f'pop: {pop_count_array}')
    print(f'classical: {classical_count_array}') """


# Problem 1d: Visualize the features with a scatter plot
""" def plot_scatterplot(X_train, y_train):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Pop', alpha=0.5)
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Classical', alpha=0.5)
    plt.xlabel('Liveness')
    plt.ylabel('Loudness')
    plt.legend()
    plt.title('Liveness vs Loudness for Pop and Classical genres')
    plt.show()

plot_scatterplot(X_train, y_train) """



"""
def logistic_regression_sgd(X, y, alpha=0.01, num_iterations=1000):
    
    for _ in range(num_iterations):
    
        # Iterate over each training example
        for i in range(m):
            
            # Compute the gradients
            error = y_hat - y_i
            gradient_w = error * x_i
            gradient_b = error
            
            # Update weights and bias
            w -= alpha * gradient_w
            b -= alpha * gradient_b
    
    return w, b """


# Problem 2a: Implement your own logistic discrimination classifier.
# Logistic Regression Functions
def logistic_regression_model(X, y, epochs=10, learning_rate=0.001, epsillon = 0.00001):
    """
    Found optimal values after plotting for differnet lr
    X: features
    y: labels
    epochs=10
    learning_rate=0.001
    epsillon = 0.00001
    """
    # Initial values for w and b
    number_of_features = X.shape[1]
    w = np.zeros(number_of_features)
    b = 0

    number_of_samples = len(X)
    epoch_list = []
    training_error_list = []

    print(f'learning rate: {learning_rate}')
    
    for epoch in range(epochs):
        ## SHUFFLE???
        training_error_sum = 0
        for i in range(number_of_samples):
            # step-1 (linear combination) z value (z = np.dot(w.T, X[i]) + b):
            # X[i] as an array is in correct shape even w/o transposing
            z = np.dot(w, X[i]) + b

            # step-2 sigmoid_value for y_hat
            y_hat = 1 / (1 + np.exp(-z))

            # step-3 find the deviance from y to get the gradients
            differance = y_hat - y[i] 
            dw = differance * X[i]
            db = differance

            # steo-4  update the weights w and bias b
            w = w - learning_rate * dw
            b = b - learning_rate * db

            # Training error - only for plotting
            # to avoid log(x=0), use log(x+epsilon), epsilon = 10(^-5)
            training_error_i = - (y[i] * np.log(y_hat+epsillon) + (1-y[i]) * np.log(1-y_hat+epsillon))
            training_error_sum = training_error_sum + training_error_i
        
        # Appending training error after an epochs
        #L_avg = L_sum / number_of_samples
        epoch_list.append(epoch)
        training_error_sum = training_error_sum / number_of_samples
        training_error_list.append(training_error_sum)

        #if epoch == 1:
        print(f'Learning-rate: {learning_rate}, Epoch: {epoch}, Training error L: {training_error_sum}')
        if epoch % 100 == 0:
            print(f'Learning-rate: {learning_rate}, Epoch: {epoch}, Training error L: {training_error_sum}')
    
    # returning the trained weights w, trained bias b and the errors for each epoch: training_error_list
    return w, b, training_error_list



# plot the training array
def plot_errors_with_lr(error_nested):
    learning_rates = [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01]
    epochs = 10

    training_error_for_different_learning_rates = []
    weights_for_different_learning_rates  = []
    bias_for_different_learning_rates= []

    for lr in learning_rates:
        w, b, training_error = logistic_regression_model(X_train, y_train, epochs=epochs, learning_rate=lr)

        # append to lists for plotting
        weights_for_different_learning_rates.append(w)
        bias_for_different_learning_rates.append(b)
        training_error_for_different_learning_rates.append(training_error)
    min_value = min(min(inner_list) for inner_list in error_nested)
    
    plt.figure()
    for i in range(len(error_nested)):
        plt.plot(range(0, len(error_nested[i])), error_nested[i], label=f'Learning rate: {learning_rates[i]}')
    plt.axhline(y=min_value, label = f'Min training error: {min_value:.3f}', linestyle = '--')
    plt.xlabel('Epochs')
    plt.ylabel('Training Error (Loss) per sample')
    plt.title('Training Error vs Epochs for Different Learning Rates')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    
#def plot_errors_with_lr(training_error_for_different_learning_rates)

w, b, training_error = logistic_regression_model(X_train, y_train)

# Problem 2b: Testing my models w and b on the test set
def predict(X, w, b):
    z = np.dot(w, X.T) + b
    y_hat = 1 / (1 + np.exp(z))  # Sigmoid function
    y_predict = np.where(y_hat >= 0.5, 0, 1) #Threshold to classify as 0 or 1
    return y_predict

#Function for evaluation
def evaluate(X, y, w, b):
    y_predicted = predict(X, w, b)
    accuracy = accuracy_score(y, y_predicted) # using sklearn
    return accuracy

testing_accuracy = evaluate(X_test, y_test, w, b)
print('The accuracy of our model on our training set is:', testing_accuracy)


# Problem 2c: Visualize the decision boundary
def plot_decision_boundary(X_train, y_train):
    # Generate x values for the decision boundary
    x_values = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    
    # Compute corresponding y values using the decision boundary equation
    y_values = -(w[0] * x_values + b) / w[1]
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Pop', alpha=0.5)
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Classical', alpha=0.5)
    plt.plot(x_values, y_values, 'g--', label='Decision Boundary')
    plt.xlabel('Liveness')
    plt.ylabel('Loudness')
    plt.legend()
    plt.title('Liveness vs Loudness for Pop and Classical genres')
    plt.show()

plot_decision_boundary(X_train, y_train)

# Import necessary metrics for confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Problem 3a: Create Confusion Matrix for the Test Set
def plot_confusion_matrix(y_true, y_pred):

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create the confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Classical', 'Pop'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix for Test Set')
    plt.show()

# Predict using the trained model
y_pred_test = predict(X_test, w, b)

# Plot the confusion matrix
plot_confusion_matrix(y_test, y_pred_test)


# Problem 3b: Accuracy of the model
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Accuracy of the model on the test set: {test_accuracy * 100:.2f}%")

# Additional explanation of confusion matrix
print("Confusion Matrix gives more insight into:")
print("1. True Positives (Pop correctly classified)")
print("2. True Negatives (Classical correctly classified)")
print("3. False Positives (Classical misclassified as Pop)")
print("4. False Negatives (Pop misclassified as Classical)")

