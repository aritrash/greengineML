import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pickle # Import the pickle library

if __name__ == "__main__":
    # --- Data Loading and Preprocessing ---
    try:
        # Load the dataset from the Excel file
        # Please make sure your file is named 'dataset_ads1115adc.xlsx'
        df = pd.read_excel('datasets/dataset_ads1115adc.xlsx')
    except FileNotFoundError:
        print("Error: 'dataset_ads1115adc.xlsx' not found. Please create the file or change the filename.")
        exit()
    
    # Preprocessing the data
    # Normalizing the 'adc_value' for better model performance
    # Note: scikit-learn's LogisticRegression might not strictly need this,
    # but it's good practice for gradient-based models.
    df['adc_value'] = df['adc_value'] / 32768.0

    # Separate features (X) and labels (y)
    # The model expects a 2D array, so we keep 'day' even if it's currently a placeholder
    X = df[['adc_value', 'day']].values
    y = df['lights_on'].values

    # Divide the dataset into 80% training and 20% validation sets.
    # The `random_state` ensures that the random split is consistent.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Dataset split: {len(X_train)} training samples, {len(X_val)} validation samples.")

    # --- Model Training with scikit-learn ---
    print("\n--- Training Model with scikit-learn's LogisticRegression ---")
    
    # Create an instance of the Logistic Regression model.
    # The solver handles the optimization algorithm.
    model = LogisticRegression(solver='liblinear', random_state=42)
    
    # Train the model using the training data
    model.fit(X_train, y_train)
    
    print("Training complete!")

    # --- Save the Trained Model to a File ---
    print("Saving the trained model to 'lightingmodel.pkl'...")
    try:
        # 'wb' stands for 'write binary'
        with open('lightingmodel.pkl', 'wb') as file:
            pickle.dump(model, file)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Failed to save the model: {e}")

    # --- Model Validation and Evaluation ---
    print("\n--- Model Validation ---")
    
    # Make predictions on the validation set
    y_pred_val = model.predict(X_val)

    # Calculate overall accuracy
    accuracy = accuracy_score(y_val, y_pred_val) * 100
    print(f"Overall model accuracy on validation set: {accuracy:.2f}%")
    
    # Display a confusion matrix for detailed performance
    cm = confusion_matrix(y_val, y_pred_val)
    print("\nConfusion Matrix:")
    print(cm)

    print("\n--- Sample Predictions ---")
    # Randomly select 5 data points from the validation set to show results
    sample_indices = np.random.choice(len(X_val), 5, replace=False)
    for i in sample_indices:
        adc_val_original = X_val[i][0] * 32768
        day_val = X_val[i][1]
        actual = y_val[i]
        predicted = y_pred_val[i]
        print(f"Input: adc={int(adc_val_original)}, day={int(day_val)} | Predicted: {predicted}, Actual: {actual} | Accurate: {predicted == actual}")

    # --- Visualizing Results ---
    print("\n--- Visualizing Confusion Matrix ---")
    # Plotting the confusion matrix for better visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Lights Off (0)', 'Lights On (1)'],
                yticklabels=['Lights Off (0)', 'Lights On (1)'])
    plt.title('Confusion Matrix for Validation Set')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()