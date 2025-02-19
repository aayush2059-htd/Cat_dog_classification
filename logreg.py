import pandas as pd
import matplotlib.pyplot as plt
data= pd.read_csv("data.csv")
x= data['Attendance']
y= data['Marks']
def gradient_decent(m_now,b_now,points, L):
    m_gradient=0
    b_gradient=0
    n=len(points)
    
    for i in range(n):
      x= data.iloc[i].Attendance
      y= data.iloc[i].Marks
      m_gradient+= -(2/n)*(y-(m_now*x+b_now))*x
      b_gradient+= -(2/n)*(y-(m_now*x+b_now))
    m= m_now-m_gradient*L
    b= b_now-b_gradient*L 
    return m, b

epoches=300
L=0.0001
m,b =0,0
for i in range(epoches):
   m, b= gradient_decent(m,b,data,L)
print (m, b)
plt.scatter(data.Attendance, data.Marks, color="red")
plt.plot(data.Attendance, [m * x_val + b for x_val in data.Attendance], color="black", label="Best Fit Line")

plt.xlabel("Attendance")
plt.ylabel("Marks")

plt.show()
    
    
       
    