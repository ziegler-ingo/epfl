# EPFL Course Projects

## Housing-Prices
The housing-prices project uses the Ames, Iowa, housing data set by Dean De Cock. It consists of 2930 observations and 82 features. The features are typical characteristics that may impact the sale price of a house, such as floor sizes, paved or unpaved streets, or availability of pools. 

The project is mainly concerned with linear regression in combination with L1 and/or L2 regularization to combat overfitting. The goal is predict the continuously distributed target variable: sale price. Results are measured by the **mean absolute error**, which is denoted in US Dollar. Furthermore, the adjusted R-squared metric

$$\bar{R}^2 = 1-(1-R^2)\frac{n-1}{n-p-1}, \text{with n = number of observations, p = number of features} 
$$

will be tracked. Throughout the project, I will also track a self-defined metric called *error ratio*, which is supposed to shine light on how large the model error is in comparison to the median sale price of a house. The smaller the resulting ratio, the more reliable our model predictions are (or, i.e. the less impactful our error is) when considering the median house price.

In the project, three models are built: 
* The first one is a simple model with only 2 features.
	* Features are chosen by highest correlation with the target variable
    * To little surprise, the highest correlated features are: overall house quality and above ground living area
* The second model consists of 20 features
    * Features are chosen by creating a RandomForestRegressor model and observing the feature importances
* The third model consists of all 79 features
    * Due to linearly dependent columns, some of the columns - after applying preprocessing steps - had to be dropped. The custom algorithm to find linearly dependent columns is included in the code. 

### Results
| Model | \# of Features| \# of Features (OHE)| MAE (\\$)| RSS | $R^2$ | adj. $R^2$| Error Ratio (%)|
|-------|:-------------:|:-------------------:|:--------:|:---:|:-----:|:---------:|:--------------:|
|Simple Model|2|2|23,479|30.130|0.766|0.765| 13.89|
|Intermediate Model|20|24|15,664|17.633|0.863|0.860|9.27|
|Complex Model|79|243|12,538|14.300|0.889|0.866|7.42|

#### Image Classifier
EPFL Extension School collected an image data set from the streets around Lausanne, Switzerland, of various means of transportation, e.g. cars, trucks, bikes, but also road work machinery. 

Using one of Tensorflow Hub's pre-trained models, MobileNet V2, to extract high-level features from our image collection, I built, trained, and evaluated various machine learning models to see the differences in handling pixel data. The following models were implemented:
* 
