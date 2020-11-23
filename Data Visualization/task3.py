from urllib.request import urlopen 
import json 
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response: counties = json.load(response) 
import pandas as pd 
df = pd.read_csv('F:/Applied Informatics/Semester-III/Data Visualization/Laboratory Works/Homeowrk_1/volcano_data_2010.csv', dtype={"fips": str}) 
df.reset_index(inplace=True)

df['year'] = [d.year for d in df.elevation]
df['elevation'] = [d.strftime('%b') for d in df.elevation]
years = df['year'].unique()

np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)


plt.figure(figsize=(16,12), dpi= 80)
for i, y in enumerate(years):
    if i > 0:        
        plt.plot('month', 'value', data=df.loc[df.year==y, :], color=mycolors[i], label=y)
        plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'value'][-1:].values[0], y, fontsize=12, color=mycolors[i])

# Decoration
plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$Elevation', xlabel='$Year$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot of Volcanic eruption elevation Time Series", fontsize=20)
plt.show()
import plotly.express as px 
fig = px.choropleth(df, geojson=counties, locations='fips', color='elevation', color_continuous_scale="Viridis", range_color=(0, 12), scope='world', labels={'elevation':'Elevtion'} ) 
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}) 
fig.show()
