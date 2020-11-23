import math
import networkx as nx
import matplotlib.pyplot as plt
# creating instances 'e'
e_x = [2,5,4,0,1,2,3,5]
e_y = [1,3,0,4,5,3,4,7]

N = 8
Total_No_of_Same_Instance_classifier = 4
speed = 0.5
Incorrect_Instance = 1/N

plt.title('PB-CUP Algorithm - Classifying the Instances')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.grid()

plt.scatter(e_x,e_y,label='x-axis')
plt.scatter(e_y,e_x,label='y-axis')

plt.show()

print ("e_x[0]:", e_x[0], "e_y[0]:", e_y[0])
print ("e_x[2]:", e_x[2], "e_y[2]:", e_y[2])
print ("e_x[6]:", e_x[6], "e_y[6]:", e_y[6])

Cup0 = [e_x[0],e_y[0]]
Cup2 = [e_x[2],e_y[2]]
Cup6 = [e_x[6],e_y[6]]

print ("Cup1 = ",Cup0)
print ("Cup2 = ",Cup2)
print ("Cup3 = ",Cup6)

First_Cup = e_x[0]+e_y[0]
print ("First_Cup = ", First_Cup)

Second_Cup = e_x[2]+e_y[2]
print ("Second_Cup = ", Second_Cup)

Third_Cup = e_x[6]+e_y[6]
print ("Third_Cup = ", Third_Cup)

cup = nx.DiGraph()
cup.add_nodes_from([1,2,3,4,5,6])
#cup.add_nodes_from([4,5])
cup.add_edge(1,2)
cup.add_edge(1,3)
cup.add_edge(2,1)
cup.add_edge(2,3)
cup.add_edge(3,1)
cup.add_edge(3,2)
cup.add_edge(1,4)
cup.add_edge(1,5)
cup.add_edge(2,4)
cup.add_edge(2,5)
cup.add_edge(3,4)
cup.add_edge(3,5)
cup.add_edge(4,6)
cup.add_edge(5,6)


nx.draw(cup, with_labels = True)
plt.draw()
plt.title('Creation of Cups Interlinks')
plt.show()

R = 1/ Total_No_of_Same_Instance_classifier
print("R = ", R)

SV_1 = math.sqrt(First_Cup)
print("SV_1 = ", SV_1)

SV_2 = math.sqrt(Second_Cup)
print("SV_2 = ", SV_2)

SV_3 = math.sqrt(Third_Cup)
print("SV_3 = ", SV_3)


error_weight_SV_1 = 1/(1+math.exp(-SV_1))
print("error_weight_SV_1 = ", error_weight_SV_1)

error_weight_SV_2 = 1/(1+math.exp(-SV_2))
print("error_weight_SV_2 = ", error_weight_SV_2)

error_weight_SV_3 = 1/(1+math.exp(-SV_3))
print("error_weight_SV_3 = ", error_weight_SV_3)

Total_Initial_Error_Weight = (error_weight_SV_1 + error_weight_SV_2 + error_weight_SV_3)
print("Total_Initial_Error_Weight = ", Total_Initial_Error_Weight)


Probability = 1/N
print("Probability = ", Probability)

Distance_of_CUP1_to_CUP1 = abs(e_x[0]-e_x[0]),abs(e_y[0]-e_y[0])
print("Distance_of_CUP1_to_CUP1 = ", Distance_of_CUP1_to_CUP1)
Min_Distance_of_CUP1_to_CUP1 = min(Distance_of_CUP1_to_CUP1)
print("Min_Distance_of_CUP1_to_CUP1 = ", Min_Distance_of_CUP1_to_CUP1)

Distance_of_CUP1_to_CUP2 = abs(e_x[0]-e_x[2]),abs(e_y[0]-e_y[2])
print("Distance_of_CUP1_to_CUP2 = ", Distance_of_CUP1_to_CUP2)
Min_Distance_of_CUP1_to_CUP2 = min(Distance_of_CUP1_to_CUP2)
print("Min_Distance_of_CUP1_to_CUP2 = ", Min_Distance_of_CUP1_to_CUP2)

Distance_of_CUP1_to_CUP3 = abs(e_x[0]-e_x[6]),abs(e_y[0]-e_y[6])
print("Distance_of_CUP1_to_CUP3 = ", Distance_of_CUP1_to_CUP3)
Min_Distance_of_CUP1_to_CUP3 = min(Distance_of_CUP1_to_CUP3)
print("Min_Distance_of_CUP1_to_CUP3 = ", Min_Distance_of_CUP1_to_CUP3)

V1_CUP_1 = SV_1*speed*Min_Distance_of_CUP1_to_CUP1
V1_CUP_2 = SV_2*speed*Min_Distance_of_CUP1_to_CUP1
V1_CUP_3 = SV_3*speed*Min_Distance_of_CUP1_to_CUP1
print("V1_CUP_1 = ",V1_CUP_1, "V1_CUP_2 = ",V1_CUP_2, "V1_CUP_3 = ",V1_CUP_3)

V2_CUP_1 = SV_1*speed*Min_Distance_of_CUP1_to_CUP2
V2_CUP_2 = SV_2*speed*Min_Distance_of_CUP1_to_CUP2
V2_CUP_3 = SV_3*speed*Min_Distance_of_CUP1_to_CUP2
print("V2_CUP_1 = ",V2_CUP_1, "V2_CUP_2 = ",V2_CUP_2, "V2_CUP_3 = ",V2_CUP_3)

V3_CUP_1 = SV_1*speed*Min_Distance_of_CUP1_to_CUP3
V3_CUP_2 = SV_2*speed*Min_Distance_of_CUP1_to_CUP3
V3_CUP_3 = SV_3*speed*Min_Distance_of_CUP1_to_CUP3
print("V3_CUP_1 = ",V3_CUP_1, "V3_CUP_2 = ",V3_CUP_2, "V3_CUP_3 = ",V3_CUP_3)

Max_CUP_1 = max(V1_CUP_1,V1_CUP_2,V1_CUP_3)
Max_CUP_2 = max(V2_CUP_1,V2_CUP_2,V2_CUP_3)
Max_CUP_3 = max(V3_CUP_1,V3_CUP_2,V3_CUP_3)
print("Max_CUP_1 =", Max_CUP_1, "Max_CUP_2 = ",Max_CUP_2, "Max_CUP_3 = ",Max_CUP_3)


Node_1_Dimension_CUP_2 = speed*Max_CUP_1+speed*Max_CUP_2+speed*Max_CUP_3
print("Node_1_Dimension_CUP_2 = ", Node_1_Dimension_CUP_2)

Node_2_Dimension_CUP_2 = speed*Max_CUP_1+speed*Max_CUP_2+speed*Max_CUP_3
print("Node_2_Dimension_CUP_2 = ", Node_2_Dimension_CUP_2)

F1 = (Incorrect_Instance*2) * Node_1_Dimension_CUP_2
print(" F1 = ", F1)
F2 = (Incorrect_Instance*2) * Node_2_Dimension_CUP_2
print(" F2 = ", F2)
Total_Weight_at_end_Iteration_1 = (F1*(1-Incorrect_Instance*2)/Incorrect_Instance*2)+(F2*(1-Incorrect_Instance*2)/Incorrect_Instance*2)
print("Total_Weight_at_end_Iteration_1=", Total_Weight_at_end_Iteration_1)

Normalize_the_Final_CUP_Weight = (Total_Weight_at_end_Iteration_1/R)
print("Normalize_the_Final_CUP_Weight = ", Normalize_the_Final_CUP_Weight)

Final_CUP_Error_Weight = 1/(1+math.exp(-Normalize_the_Final_CUP_Weight))
print("Final_CUP_Error_Weight = ", Final_CUP_Error_Weight)

Final_CUP_Structure = nx.DiGraph()
Final_CUP_Structure.add_nodes_from([1.73,2,2.64,1.32,1.323,31.68])
Final_CUP_Structure.add_edge(1.73,1.32)
Final_CUP_Structure.add_edge(2,1.32)
Final_CUP_Structure.add_edge(2.64,1.32)
Final_CUP_Structure.add_edge(1.73,1.323)
Final_CUP_Structure.add_edge(2,1.323)
Final_CUP_Structure.add_edge(2.64,1.323)
Final_CUP_Structure.add_edge(1.32,31.68)
Final_CUP_Structure.add_edge(1.323,31.68)
nx.draw(Final_CUP_Structure, with_labels = True)
plt.draw()
plt.show()

Final_CUP_Error_Weight_Structure = nx.DiGraph()
Final_CUP_Error_Weight_Structure.add_nodes_from([0.85,0.88,0.93,0.72,0.73,0.99])
Final_CUP_Error_Weight_Structure.add_edge(0.85,0.72)
Final_CUP_Error_Weight_Structure.add_edge(0.88,0.72)
Final_CUP_Error_Weight_Structure.add_edge(0.93,0.72)
Final_CUP_Error_Weight_Structure.add_edge(0.85,0.73)
Final_CUP_Error_Weight_Structure.add_edge(0.88,0.73)
Final_CUP_Error_Weight_Structure.add_edge(0.93,0.73)
Final_CUP_Error_Weight_Structure.add_edge(0.72,0.99)
Final_CUP_Error_Weight_Structure.add_edge(0.73,0.99)
nx.draw(Final_CUP_Error_Weight_Structure, with_labels = True)
plt.draw()
plt.show()


