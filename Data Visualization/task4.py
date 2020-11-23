Import pandas as pd , numpy as np
Import seaborn as sns
Import math
Import matplotlib.pyplot as plt
Import scipy.stats.pearsonr

Volcano_filepath = “E:/Applied informatics/semester-III/Data Visulaization/Homework4/volcano_data_2010.csv" 
Volcano_data = pd.read_csv(car_filepath, index_col="Month")

Vc = Volcano_data
X = Volcano_filepath
     plt.sns.barplot(x=volcano_data.index, y=Volcano_data['NK']) 
Cov_volcano = cor(Vc, Volcano_data)

Data_Volcano = scipy.stats.pearsonr(x, y)
plt.figure(figsize=(14,7)) 
plt.title(“Volcano Correlation analysis 2010 and 2011") 
sns.heatmap(data=Volcano_data, annot=True) 

