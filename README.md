# EPFL Course Projects

## Housing-Prices
The housing-prices project uses the Ames, Iowa, housing data set by Dean De Cock. It consists of 2930 observations and 82 features. The features are typical characteristics that may impact the sale price of a house, such as floor sizes, paved or unpaved streets, or availability of pools. 

The project is mainly concerned with linear regression in combination with L1 and/or L2 regularization to combat overfitting. The goal is predict the continuously distributed target variable: sale price. Results are measured by the **mean absolute error**, which is denoted in US Dollar. Furthermore, the adjusted R-squared metric will be tracked. Throughout the project, I will also track a self-defined metric called *error ratio*, which is supposed to shine light on how large the model error is in comparison to the median sale price of a house. The smaller the resulting ratio, the more reliable our model predictions are (or, i.e. the less impactful our error is) when considering the median house price.

In the project, three models are built: 
* The first one is a simple model with only 2 features.
	* Features are chosen by highest correlation with the target variable
    * To little surprise, the highest correlated features are: overall house quality and above ground living area
* The second model consists of 20 features
    * Features are chosen by creating a RandomForestRegressor model and observing the feature importances
* The third model consists of all 79 features
    * Due to linearly dependent columns, some of the columns - after applying preprocessing steps - had to be dropped. The custom algorithm to find linearly dependent columns is included in the code. 

### Results
| Model | \# of Features| \# of Features (OHE)| MAE ($)| RSS | R2 | adj. R2| Error Ratio (%)|
|-------|:-------------:|:-------------------:|:--------:|:---:|:-----:|:---------:|:--------------:|
|Simple Model|2|2|23,479|30.130|0.766|0.765| 13.89|
|Intermediate Model|20|24|15,664|17.633|0.863|0.860|9.27|
|Complex Model|79|243|12,538|14.300|0.889|0.866|7.42|

## Image Classifier
EPFL Extension School collected an image data set from the streets around Lausanne, Switzerland, of various means of transportation, e.g. cars, trucks, bikes, but also road work machinery. In total, there are only 280 images, so training a model from scratch using the images' raw pixel inputs would not successfully achieve high accuracy ranges (see later in the dense and convolutional neural networks).

Using one of Tensorflow Hub's pre-trained models, MobileNet V2, to extract high-level features from our image collection, I built, trained, and evaluated various machine learning models to see the differences in handling pixel data. The following models were implemented:
* k-NN, as well as a visual search tool using nearest neighbors
* Simple decision trees
* Logistic regression
* Non-linear estimators: random forests and support vector machines
* Dense and convolutional neural networks
	* For the CNN model, I re-built AlexNet, which was introduced by [Krizhevsky, Sutskever, and Hinton](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf).
	* The CNN is trained from scratch, i.e. it does **not** use the extracted high-level features
	
### Results
|Model   |Test Accuracy|
|:------:|:---------------:|
|k-NN|0.98|
|Decision Tree|0.74|
|Logistic Regression|0.96|
|Random Forests|0.96|
|SVM (linear)|0.96|
|SVM (rbf)|0.96|
|1-layer NN|0.96|
|2-layer NN|0.94|
|CNN|0.56|


## Capstone Project: Predicting Student Performance and Improving Student Network Building in Virtual Learning Environments
#### The Problem
Without direct, face-to-face contact to students, online learning institutions have a harder time recognizing possibly failing students because there is generally less support and outreach opportunities for both students and tutors. As a consequence, supplyer of online learning courses would benefit from a system that allows them to classify the probable success of a student based on only the information that is available to them: the information that is tracked at point of registration (i.e. demographic, socio-economic), the students up-to-date performance in the course (e.g. mid-term grades), and the data of how much the students interact with the provided virtual learning environment (e.g. how many sites the student visits, what kind of materials the student uses, how much he clicks around).

The Open University data set combines all three of these aspects and is thus suited to use as assessment for possible future student performance.

#### Main goal
We will build supervised learning models (logistic regression, decision tree and random forest, k-NN) that aim to classify a student profile into one of 4 possible final result categories: pass with distinction, pass, withdrawn, or fail. The coefficients will be trained on this data set, i.e. historical students, which can then be used to predict the success of new, incoming students based on their course choice, demographic data, expected activity, assessment submission dates, type of resources used, etc.

#### Side tasks
Decision trees are naturally based on sequential, ranked decisions, which compelements the nature of our features very well. Students are faced with lots of decisions: for instance, which course to take, in which term, which resources to use, when to submit the exam, what is the expected level or activity. A decision tree model will, next to the prediction accuracy, shine light onto the most important decisions, where those features split, and how they are ranked.

A scree plot will additionally provide an overview of the features that are contributing most to our classification task.

The k-NN algorithm can also be used to find and identify neighboring students, in effect suggesting peers with similar background and in similar situations. This enables the students to build closer connections, raise the student's motivation, and improve not only the students' results and experiences but also the university's alumni network.

#### Course of Analysis
The data set is provided via 7 different subsets that each track some kind of data in its own way. For instance, one data set only tracks the name, length and presentation year of courses, another set only tracks student assessments via student ID, without linking to courses, and again another set only tracks the demographic data of students without linking to assessments, so it is vital to merge all 7 sets together without losing too much data.  

As the target variable is categorical, it is not possible to rely on correlation data to select the best features. Therefore, extensive EDA as well as feature cleaning and preprocessing steps are necessary. 

Continuously distributed features are log-transformed, z-scaled and cleaned, other categorical and ordinal features are visualized and quantified as best as possible to determine whether there are noticeable trends between features and the target variable. 

For the final input matrix, continuous features will be log-transformed, categorical will be one-hot-encoded, and ordinal features will be ordinally encoded to be readable for algorithms. 

The final, preprocessed and encoded shapes for the models are:
* X_tr final shape = (115027, 223)
* X_val final shape = (41424, 223)
* X_te final shape = (41378, 223)

#### General Accuracy Results
|Model   | Accuracy |
|--------|:--------:|
|Baseline|0.58|
|Logistic Regression| 0.65|
|Decision Tree| 0.65|
|Random Forests| 0.63|
|k-NN| 0.58|

For computational reasons, the k-NN model was modified using singular value decomposition (SVD) to a niveau of 80% variance explained. As SVD was also applied as a reference model in a separate decision tree model, it seems that SVD severely hurts the model performance. In that reference model, the accuracy also dropped to the most-frequent baseline of 0.58.

Further results such as confusion matrices as well as precision and recall scores can be observed in every model notebook. As those results show, our model is extremely biased towards passing-students.

#### Possible Explanation for Model Bias
During EDA process, we could see that there is a lot of variation among the features and the target variable for all classes. There are, for instance, students who submitted their exams too late but still have Pass as a final result. In other cases, students with bad scores in one assessment still finished with Distinction overall.

This is a problem due to the way the data is structured. We merged our data to reflect unique observations in order to keep as much depth as possible. A consequence of this structure is that there are multiple contradictory data points among our total pool, which ML algorithms can't differentiate among. As an example, the algorithm doesn't know whether one bad assessment is just one of many the student wrote. So when the observation says that a student with score = 22 and low overall VLE activity finished with Pass, the model trains on this profile that students with such characteristics can finish with Pass. The model especially doesn't know that some observations belong together - it just calculates based on every observation-label pair. Because the true unique student-module Pass group already accounted for the majority of samples, all following sub-results - indepdently of how bad or good they are - also have Pass as their label. Therefore, the Pass group encompasses virtually all possible background, score, and activity combinations. As it is additionally the largest group, the model learns primarily on those examples.

The consequence is that the model develops a strong bias in favor of Pass and we thus receive almost exclusively predictions in favor of Pass. Because this one class has such an overhelming amount of input variability in terms of feature values, there is no pattern left to be recognized for that class as basically every input value can lead to Pass. This results directly in the problem that there are no significant trends and pattern left to be recognized and learned from for the other 3 classes, because again, the Pass class also covers every possibility. The final result is what we could then see in the confusion matrices, namely that Fail, Withdrawn, and Distinction rarely - if at all - get predicted.

