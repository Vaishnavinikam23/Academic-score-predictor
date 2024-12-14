import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox

# Load dataset
data = pd.read_csv('scores.csv')

# Drop 'CGPA' and keep it as a reference
cgpa_reference = data['CGPA']
data_without_cgpa = data.drop(columns=['CGPA'])

# Feature Selection
X = data_without_cgpa[['Attendance', 'Internals', 'Screen Time', 'Total Credits']]
y = cgpa_reference

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_scaled, y)

# Create GUI for user input and prediction
def predict_cgpa():
    try:
        # Get user input
        attendance = float(attendance_entry.get())
        internals = float(internals_entry.get())
        screen_time = float(screen_time_entry.get())
        total_credits = float(total_credits_entry.get())
        
        # Standardize the input using the scaler
        user_input = [[attendance, internals, screen_time, total_credits]]
        user_input_scaled = scaler.transform(user_input)
        
        # Predict CGPA
        predicted_cgpa = model.predict(user_input_scaled)[0]
        
        # Display the result
        messagebox.showinfo("Predicted CGPA", f"The predicted CGPA is: {predicted_cgpa:.2f}")
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Create the GUI
root = tk.Tk()
root.title("CGPA Prediction")
root.geometry("400x300")
root.configure(bg="#f0f0f0")

# Title Label
title_label = tk.Label(root, text="CGPA Prediction", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

# Input fields for each feature
tk.Label(root, text="Attendance:", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
attendance_entry = tk.Entry(root, font=("Helvetica", 12), width=30)
attendance_entry.pack(pady=5)

tk.Label(root, text="Internals:", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
internals_entry = tk.Entry(root, font=("Helvetica", 12), width=30)
internals_entry.pack(pady=5)

tk.Label(root, text="Screen Time:", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
screen_time_entry = tk.Entry(root, font=("Helvetica", 12), width=30)
screen_time_entry.pack(pady=5)

tk.Label(root, text="Total Credits:", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
total_credits_entry = tk.Entry(root, font=("Helvetica", 12), width=30)
total_credits_entry.pack(pady=5)

# Predict Button
predict_button = tk.Button(root, text="Predict CGPA", command=predict_cgpa, 
                            bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5)
predict_button.pack(pady=20)

# Exit Button
exit_button = tk.Button(root, text="Exit", command=root.quit,
                        bg="#f44336", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5)
exit_button.pack(pady=10)

# Run the GUI
root.mainloop()
