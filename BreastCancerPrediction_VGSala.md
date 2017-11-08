
<h1> <big> Breast cancer diagnosis prediction: a case study for feature selection </big> </h1> 
<p> <big>Breast Cancer Wisconsin (Diagnostic) Data Set - Kaggle </big></p> 
<h3> Vera Giulia Sala - Ubiqum Code Academy </h3>


<ol> 
  <li><a href="#p1"> Goal of the analysis</a></li> 
  <li><a href="#p2">Dataset</a></li> 
  <li><a href="#p3">Preprocessing and explorative analysis of data</a></li> 
  <li><a href="#p4">Feature selection</a></li> 
      <ol>   <li> <a href="#p4_1">Elimination highly collinear features</a> </li> 
             <li> <a href="#p4_2">Univariate feature selection – (ANOVA, d-cohen)</a> </li>
             <li> <a href="#p4_3">Logistic regression with Lasso penalization</a> </li>
             <li> <a href="#p4_4">Random forest, feature importance</a> </li>
             <li> <a href="#p4_5">Recursive feature elimination (RFE)</a> </li></ol>
  <li><a href="#p5">Comparison predictive models with different feature selection</a></li> 
  <li><a href="#p6">Principal Component Analysis</a></li> 
  <li><a href="#p7">Conclusion</a></li> 
</ol>


# <a id="p1">Goal of the analysis</a> 

The goal of the analysis is to predict whether a breast cancer is benign or malignant depending on the characteristics of the cancer cell nuclei exctracted from digitized images of a fine needle aspirate (FNA) of a breast mass.  
The analysis has been focused on feature selection, with the main objective of determining which nuclei features are more relevant for diagnosis predictions.


# <a id="p2">Dataset</a>

**Data set**        
569 instances,  30 features 


**Features**  
The features are computed from a digitized image of a fine needle aspirate (FNA) 
of a breast mass and they describe the characteristics of the cell nuclei.  

- radius (mean of distances from centre to points on the perimeter) 
- texture (standard deviation of gray-scale values) 
- perimeter 
- area 
- smoothness (local variation in radius lengths) 
- compactness (perimeter^2 / area - 1.0) 
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry 
- fractal dimension ("coastline approximation" - 1)

3 values for each feature: mean, standard error and worst (largest) value


# <a id="p3">Preprocessing and explorative analysis of data</a>

We perform an explorative analysis of data, and we standardize all the features.


```python
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *
from biokit.viz import corrplot
from collections import OrderedDict
from sklearn import preprocessing as pp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

```


```python
dati=pd.read_csv("data.csv")

```


```python
dati.shape
```




    (569, 33)




```python
dati.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
pd.value_counts(dati['diagnosis'])
```




    B    357
    M    212
    Name: diagnosis, dtype: int64




```python
fig = plt.figure(1, figsize=(8, 6))
sns.countplot(x = 'diagnosis', data = dati)
plt.show()
fig.savefig("class_distr.png")
```


![png](BreastCancerPrediction_VGSala_filesoutput_13_0.png)



```python
%pylab inline
dati_melted = pd.melt(dati.iloc[:,1:12], id_vars=['diagnosis'])
lab_new = ["Mean Area", "Mean Compactness", "Mean Concave Points", "Mean Concavity","Mean Fractal Dimension","Mean Perimeter","Mean Radius", "Mean Smoothness","Mean Symmetry","Mean Texture"]
aaa=ggplot(dati_melted, aes(x='value', fill='diagnosis')) +\
     geom_histogram(alpha=0.5)+ facet_wrap("variable",scales="free")+\
     labs( x = 'value', y = 'counts')+ scale_fill_manual(values=("#5F9E6F","#5975A4" ))
aaa.save("bm.pdf", width=15, height=15) 

```

    Populating the interactive namespace from numpy and matplotlib
    

    C:\Users\A\Anaconda3\lib\site-packages\IPython\core\magics\pylab.py:161: UserWarning: pylab import has clobbered these variables: ['copy', 'ylim', 'legend', 'colors', 'xlim']
    `%matplotlib` prevents importing * from pylab and numpy
      "\n`%matplotlib` prevents importing * from pylab and numpy"
    


![png](BreastCancerPrediction_VGSala_filesoutput_14_2.png)


**Standardize data**


```python
dati_scaled = pd.DataFrame(pp.scale(dati.iloc[:,2:32]))
```


```python
repl = dict( zip( list(dati_scaled.columns.values), list((dati.columns.values[2:32]))))
dati_scaled= dati_scaled.rename(columns=repl)
dati_scaled["diagnosis"] = dati.diagnosis
```


```python
datiii = dati_scaled.iloc[:,0:10]
datiii = pd.concat([datiii,dati_scaled.diagnosis],axis=1)
```


```python
datax = pd.melt(datiii,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(y="features", x="value", hue="diagnosis", data=datax,split=True, inner="quart")
plt.xticks(rotation=90)
```




    (array([-6., -4., -2.,  0.,  2.,  4.,  6.,  8.]),
     <a list of 8 Text xticklabel objects>)




![png](BreastCancerPrediction_VGSala_filesoutput_19_1.png)


# <a id="p4">Feature selection</a>

The main goal of the analysis is to determine which features are more relevant for diagnosis prediction.   
After removing highly collinear features, we try and compare different feature selection methods.  

## <a id="p4_1">Elimination highly collinear features</a>

Highly collinear features increase the dimensionality of the problem, without adding valuable information. A clear example is given by the variables: "radius", "perimeter", "area" that are related by a clear functional dependance. We eliminate the highly collinear features (where Pearson correlation coefficient is > 0.85), keeping only the one that has the highest correlation with the dependent variable. 


```python
yyy=plt.figure(figsize=(15, 15))
corr2 = dati_scaled.corr()

c2 = corrplot.Corrplot(corr2)
c2.plot(fig=yyy)
```

    C:\Users\A\Anaconda3\lib\site-packages\biokit\viz\linkage.py:41: ClusterWarning: scipy.cluster: The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix
      Y = hierarchy.linkage(D, method=method, metric=metric)
    


![png](BreastCancerPrediction_VGSala_filesoutput_24_1.png)



```python
rrr = dati_scaled.loc[:,["radius_mean","area_mean","perimeter_mean"]]
g = sns.PairGrid(rrr,diag_sharey=False)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter, edgecolor="w", s=40)
plt.rcParams["axes.labelsize"] = 15
plt.rcParams['figure.figsize'] = 10, 10
plt.show()
```


![png](BreastCancerPrediction_VGSala_filesoutput_25_0.png)



```python
elimin = ["radius_worst", "area_mean", "area_worst", "perimeter_mean", "perimeter_worst",  "perimeter_se", "area_se",
          "texture_worst","concave points_mean", "concave points_worst", "concavity_worst", "compactness_worst"]
dati_red = dati_scaled.drop(elimin, axis=1)
```


```python
dati_red.shape
```




    (569, 19)



## <a id="p4_2">Univariate feature selection – (ANOVA, d-cohen)</a>

Univariate feature selection is done by calculating the correlation of each independent variable separately with the dependent variable, and keeping the features with the highest correlation coefficient. To measure univariate correlation we use ANOVA and d-Cohen cefficient as effect size. 

$dCohen = \frac{\bar{X_{1}}-\bar{X_{2}}}{\sqrt{S_{p}^{2}}} $


```python
from sklearn.feature_selection import f_classif
```


```python
X=dati_red[dati_red.columns[dati_red.columns != "diagnosis" ]]
y=dati_red["diagnosis"]
```


```python
anova_corr = pd.DataFrame((f_classif(X, y))[0])
anova_corr= anova_corr.rename(columns={0: "F-value"})
anova_corr["p-value"] =(f_classif(X, y))[1]
anova_corr["variable"] = X.columns
```


```python
malign = X[(y == "M")]
bening =  X[(y == "B")]
m_std = malign.apply(np.std, 0)
b_std = bening.apply(np.std, 0)
m_mean = malign.apply(np.mean, 0)
b_mean = bening.apply(np.mean, 0)
df_m = malign.shape[0]-1
df_b = bening.shape[0]-1
sp_sq = (m_std**2*df_m + b_std**2*df_b)/(df_m+df_b)
d_cohen =  abs(m_mean - b_mean)/np.sqrt(sp_sq)
anova_corr["d_cohen"] = np.array(d_cohen)
```


```python
anova_corr.sort_values("d_cohen",ascending=False).iloc[0:5,:]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F-value</th>
      <th>p-value</th>
      <th>variable</th>
      <th>d_cohen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>646.981021</td>
      <td>8.465941e-96</td>
      <td>radius_mean</td>
      <td>2.209955</td>
    </tr>
    <tr>
      <th>4</th>
      <td>533.793126</td>
      <td>9.966556e-84</td>
      <td>concavity_mean</td>
      <td>2.007319</td>
    </tr>
    <tr>
      <th>3</th>
      <td>313.233079</td>
      <td>3.938263e-56</td>
      <td>compactness_mean</td>
      <td>1.537618</td>
    </tr>
    <tr>
      <th>7</th>
      <td>268.840327</td>
      <td>9.738949e-50</td>
      <td>radius_se</td>
      <td>1.424834</td>
    </tr>
    <tr>
      <th>15</th>
      <td>122.472880</td>
      <td>6.575144e-26</td>
      <td>smoothness_worst</td>
      <td>0.961294</td>
    </tr>
  </tbody>
</table>
</div>



> **The 5 most relevant features are: radius_mean, concavity_mean, compactness_mean, radius_se, smoothness_worst**

## <a id="p4_3">Logistic regression with Lasso penalization</a>

Logistic regression with Lasso penalization fits the training data minimizing the cost function:

$min_{\theta}\frac{1}{2}\Sigma(y_{i}-X_{i}\theta)^{2}+\lambda \vert\theta \vert_{1} $

For a value for the $\lambda$ coefficient high enough, some feature coefficients go to zero. So a feature selection is performed. For all the remaining feature a further selection can be done by considering the magnitude of the coefficients.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn import linear_model
```


```python
log_reg_cv = linear_model.LogisticRegressionCV(Cs=10,cv=5,random_state=123,penalty="l1",solver="liblinear")
regr = log_reg_cv.fit(X.as_matrix(),np.squeeze(np.asarray(y.as_matrix())))
```

lasso regression for 10 values of lambda (cv = 5): best value is 3.59381366e-01


```python
regr.Cs_
```




    array([  1.00000000e-04,   7.74263683e-04,   5.99484250e-03,
             4.64158883e-02,   3.59381366e-01,   2.78255940e+00,
             2.15443469e+01,   1.66810054e+02,   1.29154967e+03,
             1.00000000e+04])




```python
pd.DataFrame(regr.scores_["M"])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.626087</td>
      <td>0.626087</td>
      <td>0.626087</td>
      <td>0.904348</td>
      <td>0.947826</td>
      <td>0.973913</td>
      <td>0.965217</td>
      <td>0.956522</td>
      <td>0.956522</td>
      <td>0.956522</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.626087</td>
      <td>0.626087</td>
      <td>0.626087</td>
      <td>0.956522</td>
      <td>0.956522</td>
      <td>0.939130</td>
      <td>0.947826</td>
      <td>0.947826</td>
      <td>0.939130</td>
      <td>0.939130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.628319</td>
      <td>0.628319</td>
      <td>0.628319</td>
      <td>0.973451</td>
      <td>0.973451</td>
      <td>0.955752</td>
      <td>0.946903</td>
      <td>0.946903</td>
      <td>0.946903</td>
      <td>0.946903</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.628319</td>
      <td>0.628319</td>
      <td>0.628319</td>
      <td>0.973451</td>
      <td>0.982301</td>
      <td>0.973451</td>
      <td>0.973451</td>
      <td>0.973451</td>
      <td>0.973451</td>
      <td>0.973451</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.628319</td>
      <td>0.628319</td>
      <td>0.628319</td>
      <td>0.955752</td>
      <td>0.964602</td>
      <td>0.982301</td>
      <td>0.973451</td>
      <td>0.973451</td>
      <td>0.973451</td>
      <td>0.973451</td>
    </tr>
  </tbody>
</table>
</div>




```python
(np.mean(pd.DataFrame((regr.scores_["M"]))))
```




    0    0.627426
    1    0.627426
    2    0.627426
    3    0.952705
    4    0.964940
    5    0.964910
    6    0.961370
    7    0.959631
    8    0.957891
    9    0.957891
    dtype: float64




```python
dict4 =dict(zip((X.columns ), (np.squeeze(regr.coef_))))
feature_regr = np.transpose(pd.DataFrame([dict4]))
feature_regr.columns = ["Regr_coeff"]
np.abs(feature_regr).sort_values("Regr_coeff",ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Regr_coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>radius_mean</th>
      <td>2.907901</td>
    </tr>
    <tr>
      <th>radius_se</th>
      <td>1.764286</td>
    </tr>
    <tr>
      <th>concavity_mean</th>
      <td>1.613836</td>
    </tr>
    <tr>
      <th>smoothness_worst</th>
      <td>1.075027</td>
    </tr>
    <tr>
      <th>texture_mean</th>
      <td>1.065012</td>
    </tr>
    <tr>
      <th>symmetry_worst</th>
      <td>1.053832</td>
    </tr>
    <tr>
      <th>symmetry_se</th>
      <td>0.573840</td>
    </tr>
    <tr>
      <th>fractal_dimension_se</th>
      <td>0.489557</td>
    </tr>
    <tr>
      <th>compactness_se</th>
      <td>0.170258</td>
    </tr>
    <tr>
      <th>fractal_dimension_mean</th>
      <td>0.018919</td>
    </tr>
    <tr>
      <th>symmetry_mean</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>compactness_mean</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>smoothness_se</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>smoothness_mean</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>fractal_dimension_worst</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>concavity_se</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>concave points_se</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>texture_se</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



> **The 5 most relevant features are: radius_mean, radius_se, concavity_mean, smoothness_worst, texture_mean**

## <a id="p4_4">Random forest, feature importance</a>


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=120)

X1= X.copy()
Y1= y.copy()
rf = RandomForestClassifier()

risRF = pd.DataFrame(np.array(range(18)))
accurac = []
i=0

for train_index, test_index in rs.split(X1):
    i=i+1
    #print("TRAIN:", train_index, "TEST:", test_index)
    XX_train, XX_test = X1.iloc[train_index,:], X1.iloc[test_index,:]
    YY_train, YY_test = Y1.iloc[train_index], Y1.iloc[test_index]
    r = rf.fit(XX_train, (np.array(YY_train)))
    aaa = accuracy_score(YY_test,r.predict(XX_test))
    ddff = pd.DataFrame(np.squeeze(r.feature_importances_))
    ddff.columns = [str(i)]
    accurac.append(aaa)
    risRF = risRF.join(ddff, lsuffix='_caller', rsuffix='_other')

```


```python
mean(accurac)

```




    0.93508771929824552




```python
risRF["feature"] = np.squeeze(X.columns)
risRF["mean_importance"] = risRF.iloc[:,1:6].mean(1)
a1= risRF.sort_values("mean_importance",ascending=False)
a1.index = np.array(range(18))
a1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>feature</th>
      <th>mean_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.299121</td>
      <td>0.158717</td>
      <td>0.149921</td>
      <td>0.195974</td>
      <td>0.265093</td>
      <td>0.263935</td>
      <td>0.354787</td>
      <td>0.240900</td>
      <td>0.184799</td>
      <td>0.203151</td>
      <td>radius_mean</td>
      <td>0.213765</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>0.126854</td>
      <td>0.199519</td>
      <td>0.233911</td>
      <td>0.172427</td>
      <td>0.094541</td>
      <td>0.174456</td>
      <td>0.189749</td>
      <td>0.176237</td>
      <td>0.182051</td>
      <td>0.285759</td>
      <td>concavity_mean</td>
      <td>0.165451</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0.075668</td>
      <td>0.159051</td>
      <td>0.065683</td>
      <td>0.207480</td>
      <td>0.159917</td>
      <td>0.120992</td>
      <td>0.121586</td>
      <td>0.111612</td>
      <td>0.176155</td>
      <td>0.094302</td>
      <td>radius_se</td>
      <td>0.133560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.179794</td>
      <td>0.025907</td>
      <td>0.148052</td>
      <td>0.115003</td>
      <td>0.056247</td>
      <td>0.122365</td>
      <td>0.046974</td>
      <td>0.063604</td>
      <td>0.096276</td>
      <td>0.056556</td>
      <td>compactness_mean</td>
      <td>0.105001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>0.043444</td>
      <td>0.056925</td>
      <td>0.042327</td>
      <td>0.076981</td>
      <td>0.065831</td>
      <td>0.033502</td>
      <td>0.012997</td>
      <td>0.060058</td>
      <td>0.023735</td>
      <td>0.117935</td>
      <td>concavity_se</td>
      <td>0.057102</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0.061947</td>
      <td>0.043185</td>
      <td>0.021897</td>
      <td>0.034213</td>
      <td>0.083319</td>
      <td>0.041207</td>
      <td>0.054982</td>
      <td>0.053626</td>
      <td>0.068020</td>
      <td>0.021268</td>
      <td>texture_mean</td>
      <td>0.048912</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15</td>
      <td>0.021285</td>
      <td>0.043342</td>
      <td>0.085363</td>
      <td>0.015567</td>
      <td>0.057789</td>
      <td>0.037920</td>
      <td>0.029585</td>
      <td>0.048162</td>
      <td>0.027847</td>
      <td>0.030922</td>
      <td>smoothness_worst</td>
      <td>0.044669</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>0.030341</td>
      <td>0.036135</td>
      <td>0.035737</td>
      <td>0.022747</td>
      <td>0.024552</td>
      <td>0.016994</td>
      <td>0.017489</td>
      <td>0.023624</td>
      <td>0.028500</td>
      <td>0.024034</td>
      <td>fractal_dimension_mean</td>
      <td>0.029903</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12</td>
      <td>0.028312</td>
      <td>0.042936</td>
      <td>0.037185</td>
      <td>0.008564</td>
      <td>0.031645</td>
      <td>0.011743</td>
      <td>0.009353</td>
      <td>0.033666</td>
      <td>0.063279</td>
      <td>0.014131</td>
      <td>concave points_se</td>
      <td>0.029729</td>
    </tr>
    <tr>
      <th>9</th>
      <td>16</td>
      <td>0.018233</td>
      <td>0.048203</td>
      <td>0.033910</td>
      <td>0.021376</td>
      <td>0.023016</td>
      <td>0.036413</td>
      <td>0.029482</td>
      <td>0.034742</td>
      <td>0.016849</td>
      <td>0.036821</td>
      <td>symmetry_worst</td>
      <td>0.028948</td>
    </tr>
    <tr>
      <th>10</th>
      <td>17</td>
      <td>0.021359</td>
      <td>0.046830</td>
      <td>0.028784</td>
      <td>0.014955</td>
      <td>0.026487</td>
      <td>0.034699</td>
      <td>0.031957</td>
      <td>0.030154</td>
      <td>0.013877</td>
      <td>0.016182</td>
      <td>fractal_dimension_worst</td>
      <td>0.027683</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9</td>
      <td>0.010582</td>
      <td>0.024876</td>
      <td>0.016840</td>
      <td>0.020819</td>
      <td>0.030160</td>
      <td>0.022542</td>
      <td>0.014107</td>
      <td>0.032033</td>
      <td>0.020054</td>
      <td>0.009443</td>
      <td>smoothness_se</td>
      <td>0.020655</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10</td>
      <td>0.005963</td>
      <td>0.027990</td>
      <td>0.021871</td>
      <td>0.028752</td>
      <td>0.009216</td>
      <td>0.009712</td>
      <td>0.012484</td>
      <td>0.028973</td>
      <td>0.030065</td>
      <td>0.019424</td>
      <td>compactness_se</td>
      <td>0.018758</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>0.029694</td>
      <td>0.030973</td>
      <td>0.012577</td>
      <td>0.007501</td>
      <td>0.010771</td>
      <td>0.011755</td>
      <td>0.020703</td>
      <td>0.020675</td>
      <td>0.017536</td>
      <td>0.012972</td>
      <td>smoothness_mean</td>
      <td>0.018303</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8</td>
      <td>0.008416</td>
      <td>0.019966</td>
      <td>0.011457</td>
      <td>0.013311</td>
      <td>0.021472</td>
      <td>0.008837</td>
      <td>0.005977</td>
      <td>0.007302</td>
      <td>0.008092</td>
      <td>0.009596</td>
      <td>texture_se</td>
      <td>0.014925</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>0.013897</td>
      <td>0.006004</td>
      <td>0.014794</td>
      <td>0.016803</td>
      <td>0.023099</td>
      <td>0.004250</td>
      <td>0.014838</td>
      <td>0.016651</td>
      <td>0.011141</td>
      <td>0.011760</td>
      <td>symmetry_mean</td>
      <td>0.014920</td>
    </tr>
    <tr>
      <th>16</th>
      <td>14</td>
      <td>0.011641</td>
      <td>0.015202</td>
      <td>0.023069</td>
      <td>0.014994</td>
      <td>0.008216</td>
      <td>0.030276</td>
      <td>0.015612</td>
      <td>0.006588</td>
      <td>0.015351</td>
      <td>0.010788</td>
      <td>fractal_dimension_se</td>
      <td>0.014624</td>
    </tr>
    <tr>
      <th>17</th>
      <td>13</td>
      <td>0.013448</td>
      <td>0.014239</td>
      <td>0.016619</td>
      <td>0.012530</td>
      <td>0.008629</td>
      <td>0.018401</td>
      <td>0.017338</td>
      <td>0.011393</td>
      <td>0.016372</td>
      <td>0.024956</td>
      <td>symmetry_se</td>
      <td>0.013093</td>
    </tr>
  </tbody>
</table>
</div>



> **The 5 most relevant features are: radius_mean, concavity_mean, radius_se, texture_mean, compactness_mean**

## <a id="p4_5">Recursive feature elimination (RFE)</a>


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
from sklearn.feature_selection import RFE
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.2,random_state=101)
```


```python
clf_rf_3 = DecisionTreeClassifier()
rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
rfe = rfe.fit(X_train, y_train)
```


```python
X_train.columns[rfe.support_]
```




    Index(['radius_mean', 'texture_mean', 'smoothness_mean', 'concavity_mean',
           'symmetry_worst'],
          dtype='object')



> **The 5 most relevant features are: radius_mean, texture_mean, smoothness_mean, concavity_mean, symmetry_worst**

# <a id="p5">Comparison predictive models with different feature selection</a>

We train a number of classification predictive models, using 3-fold cross - validation on the full dataset, and we compare the accuracy of prediction. We repeat the predictions using three different feature selections decided from our analysis:  
- all features
- 7 features: "radius_mean","radius_se","concavity_mean","smoothness_worst","fractal_dimension_se","texture_mean","symmetry_worst"
- 3 features: "radius_mean","radius_se","concavity_mean"


```python
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
```


```python
reg= [LogisticRegression(), RandomForestClassifier(), DecisionTreeClassifier(), svm.SVC(), KNeighborsClassifier(n_neighbors=5), 
      MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5, 2),random_state=123), SGDClassifier(),
     GradientBoostingClassifier(n_estimators=150), AdaBoostClassifier(n_estimators=150)]
model = ["Logistic Regression","Random Forest", "Decision Tree", "SVM", "k-nn", 
   "Neural Network","SGD","Gradient Boosting", "AdaBoost"]

performance = pd.DataFrame(model)
performance.rename(columns={0: 'Model'}, inplace=True)
performance["Accuracy 18 features"]= np.zeros(9)
performance["Accuracy 7 features"]= np.zeros(9)
performance["Accuracy 3 features"]= np.zeros(9)

s=0
for i in reg:
    acc = cross_val_score(i,X, y,cv=3)
    #print(model[s] + "   " +str(np.mean(acc)) )
    performance.iloc[s,1] = np.mean(acc)
    s=s+1
```


```python
sel_feat = ["radius_mean","radius_se","concavity_mean","smoothness_worst","fractal_dimension_se","texture_mean","symmetry_worst"]   
X_sel= X[sel_feat]


s=0
for i in reg:
    acc = cross_val_score(i,X_sel, y,cv=3)
    #print(model[s] + "   " +str(np.mean(acc)))
    performance.iloc[s,2] = np.mean(acc)
    s=s+1
         

```


```python
sel_feat1 = ["radius_mean","radius_se","concavity_mean"]      
X_sel1= X[sel_feat1]


s=0
for i in reg:
    acc = cross_val_score(i,X_sel1, y,cv=3)
    #print(model[s] + "   " +str(np.mean(acc)))
    performance.iloc[s,3] = np.mean(acc)
    s=s+1
         

```


```python
performance
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy 18 features</th>
      <th>Accuracy 7 features</th>
      <th>Accuracy 3 features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.970129</td>
      <td>0.964847</td>
      <td>0.915678</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>0.931486</td>
      <td>0.949049</td>
      <td>0.906878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Decision Tree</td>
      <td>0.919122</td>
      <td>0.929704</td>
      <td>0.899824</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVM</td>
      <td>0.963102</td>
      <td>0.964857</td>
      <td>0.913924</td>
    </tr>
    <tr>
      <th>4</th>
      <td>k-nn</td>
      <td>0.949021</td>
      <td>0.949039</td>
      <td>0.915669</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Neural Network</td>
      <td>0.952529</td>
      <td>0.949002</td>
      <td>0.913914</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SGD</td>
      <td>0.954321</td>
      <td>0.964884</td>
      <td>0.882159</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Gradient Boosting</td>
      <td>0.956057</td>
      <td>0.949039</td>
      <td>0.910406</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AdaBoost</td>
      <td>0.959593</td>
      <td>0.961357</td>
      <td>0.920932</td>
    </tr>
  </tbody>
</table>
</div>



**Train-test prediction with logistic regression**


```python
clf_log_reg =LogisticRegression()      

log_reg = clf_log_reg.fit(X_train, y_train)
accuracy_score(y_test,log_reg.predict(X_test))
```




    0.94736842105263153



> **The best prediction accuracy is found using logistic regression with all the features. Neverteless we can observe that just 3 features are enough to predict the diagnosis with a quite high accuracy.**

# <a id="p6">Principal Component Analysis (PCA)</a>

We perform principal component analysis, and perform predicitons with three principal components (random forest, accuracy 92% for predictions on test set).


```python
X=dati_scaled[dati_scaled.columns[dati_scaled.columns != "diagnosis" ]]
y=dati_scaled["diagnosis"]
```

**PCA with full components **


```python
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
```




    PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)




```python
range(0,pca.explained_variance_ratio_.size)
```




    range(0, 30)




```python
aaa =plt.figure(1, figsize=(10, 10))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')
plt.show()
```


![png](BreastCancerPrediction_VGSala_filesoutput_78_0.png)


**PCA with 3 components **


```python
pca = PCA(n_components=3)
pca.fit(X)
```




    PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)




```python
X_pca = pca.transform(X)
```


```python
X_pca
```




    array([[  9.19283683,   1.94858307,  -1.12316625],
           [  2.3878018 ,  -3.76817174,  -0.52929277],
           [  5.73389628,  -1.0751738 ,  -0.55174757],
           ..., 
           [  1.25617928,  -1.90229671,   0.56273056],
           [ 10.37479406,   1.6720101 ,  -1.87702929],
           [ -5.4752433 ,  -0.67063679,   1.49044297]])




```python
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_pca, y,
                                                    stratify=y, 
                                                    test_size=0.2,random_state=201)
```


```python
clf_rf = RandomForestClassifier()      

rf_pca = clf_rf.fit(X_train1, y_train1)
```


```python
accuracy_score(y_test1,rf_pca.predict(X_test1))
```




    0.91228070175438591




```python
%pylab inline
pylab.rcParams['figure.figsize'] = (10, 6)
fig = plt.figure(1, figsize=(8, 6))
plt.scatter(X_train1[:, 0], X_train1[:, 1], c=pd.get_dummies(y_train1)["M"], cmap=plt.cm.coolwarm)  
plt.xlabel('Comp 0')
plt.ylabel('Comp 1')
plt.show()

fig = plt.figure(1, figsize=(8, 6))
plt.scatter(X_train1[:, 0], X_train1[:, 2], c=pd.get_dummies(y_train1)["M"], cmap=plt.cm.coolwarm)  
plt.xlabel('Comp 0')
plt.ylabel('Comp 2')
plt.show()

fig = plt.figure(1, figsize=(8, 6))
plt.scatter(X_train1[:, 1], X_train1[:, 2], c=pd.get_dummies(y_train1)["M"], cmap=plt.cm.coolwarm)  
plt.xlabel('Comp 1')
plt.ylabel('Comp 2')

plt.show()
```

    Populating the interactive namespace from numpy and matplotlib
    


![png](BreastCancerPrediction_VGSala_filesoutput_86_1.png)



![png](BreastCancerPrediction_VGSala_filesoutput_86_2.png)



![png](BreastCancerPrediction_VGSala_filesoutput_86_3.png)



```python
%pylab inline
pylab.rcParams['figure.figsize'] = (10, 10)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.set_xlabel('Comp 0')
ax.set_ylabel('Comp 1')
ax.set_zlabel('Comp 2')
ax.scatter(X_train1[:, 0], X_train1[:, 1], X_train1[:, 2], c=pd.get_dummies(y_train1)["M"],cmap=plt.cm.coolwarm)
plt.show()
```

    Populating the interactive namespace from numpy and matplotlib
    


![png](BreastCancerPrediction_VGSala_filesoutput_87_1.png)


# <a id="p7">Conclusions</a>

> **We can predict the diagnosis with an accuracy of 94.7%**

> **We determined the most relevant features for diagnosis prediction:**
- radius_mean (malignant cancer cells are bigger in average)
- radius_se (the variance of radius values is larger)
- concavity_mean (severity of concave portions of the contour)

> **To improve the accuracy of the predictions we would need to collect more data**
