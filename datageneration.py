import pandas as pd
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv("data.csv")
x = data['Attendance']
y = data['Marks']

# Gradient Descent Function
def gradient_decent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    
    for i in range(n):
        x_i = points.iloc[i].Attendance
        y_i = points.iloc[i].Marks
        m_gradient += -(2/n) * (y_i - (m_now * x_i - b_now)) * x_i  # ✅ Corrected formula
        b_gradient += -(2/n) * (y_i - (m_now * x_i - b_now))  # ✅ Corrected formula

    # Update m and b after processing all data points
    m_new = m_now - L * m_gradient
    b_new = b_now - L * b_gradient

    return m_new, b_new

# Training with Gradient Descent
epochs = 300
L = 0.0001  # Learning rate (too high can diverge)

m, b = 0, 0  # Initialize parameters

for i in range(epochs):
    m, b = gradient_decent(m, b, data, L)  # ✅ Use updated m, b

# Plotting the results
plt.scatter(data.Attendance, data.Marks, color="red", label="Data Points")
plt.plot(data.Attendance, m * data.Attendance + b, color="black", label="Best Fit Line")
plt.xlabel("Attendance")
plt.ylabel("Marks")
plt.legend()
plt.show()

# Print final values of m and b
print(f"Final m: {m}, Final b: {b}")
