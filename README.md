# sentiment-analysis

# Abstract
In today’s world, the struggle is not about survival. But it is about quality of life. Customer’s sentiments are, more than ever, dictating business decisions. In today’s dynamic business environment, knowing the sentiment of the customers is a necessary, but a painstakingly slow process. The effect of customer sentiments expressed on social media networks, on the price of stocks and cryptocurrencies has often been a theme for the last few years. Consumer spending is also a key factor, in reviving economies from recession, according to many leading economists, and it has also been seen, in many research studies that the consumer spending is directly affected by consumer sentiment. With the recent advances in machine learning technologies and in computing capacities, it has become possible to know the customer sentiment, through predictive modelling. Machine learning technologies, have made the necessary step of knowing the customer sentiment faster, which has the potential to provide competitive edge to businesses. 

# Problem Statement
In this assessment, a dataset, consisting of over two lakh fifty thousand records, was presented with more than ten columns. The dimensionality was a result of errors in data collation. Reviews for various movies were presented in the ‘review’ column and sentiment was expressed as a dichotomous categorical feature in ‘sentiment’ column. Sentiment could either be positive or negative. The task, was to build a sentiment classification model on top of the presented dataset.

# Data Summary
Upon initial assessment, the assumption of encountering lower dimensional data, based upon the nature of the problem, was proven wrong. It was found that, not only the data was present in more than ten dimensions, but many instances were entirely composed on null values. Mislabelling was another common feature amongst various instances. Two types of mislabelling events were identified:
1. Type 1 Mislabelling: ‘sentiment’ column for eleven instances had junk value (‘)’) present in them. The actual sentiment for the instance was present in ‘Unnamed: 03’ feature.
2. Type 2 Mislabelling: review for different instances was spread out into different columns. This, not only inflated the dimensionality of the dataset, but also caused mislabelling in the ‘sentiment’ column, as the review for the instances often spread into the ‘sentiment’ column.

# Data Pipeline
1. Cleaning the Dataset: 0.8% instances (2191/250,041), which were totally composed of null values, were removed after verification. 
2. Treating Mislabelling: In instances with Type 1 Mislabelling, the value of the sentiment was corrected, by shifting the label from ‘Unnamed: 03’ column to the ‘sentiment’ column. Type 2 Mislabelling, was treated by checking different columns of every instance and collating the review back into the ‘review’ column, and by shifting the sentiment label back into the ‘sentiment’ column.
3. Check for Empty String: ‘sentiment’ and ‘review’ columns were checked for empty strings. One such instance, with an empty review string, was discovered, and the instance was removed.
4. Preprocessing and Feature Engineering: Stop words, numeric digits and non-English (junk) words were removed and Lemmatization was performed.

# Exploratory Analysis
1. Check for imbalance: A slight, but not considerable, imbalance was noted in the dataset.

![download](https://user-images.githubusercontent.com/52705077/138031580-c610ff72-b45d-4070-b9be-acdd795fa322.png)

2. Review Length Comparison: Length of different reviews was visualized using a bar plot. It was found that there were some reviews, which had earlier consisted of junk values. Since we had removed junk values, such reviews now had an empty string and also any instance, with a review lesser than 5 words was also removed. 0.71% (1782/2,50,041), such instances were removed. After processing, it was observed that, there were a few reviews which were skewing the mean and the standard deviation. The median number of words in  reviews was found to be 62.

![download (1)](https://user-images.githubusercontent.com/52705077/138031637-8a8b8d7e-f34a-4586-bf57-27e5e0e24bd1.png)

# Train-Test Split and Vectorization
Sklearn.model_selection library was used to split the dataset into train and test sets. Splitting of the dataset is done, to test the generalization of a model. The model learns the rules from the train set and then, its performance is monitored on a dataset, which is previously unseen by the model, i.e. the test set. An 80-20 train-test split was taken.
 
TF-IDF vectorizer was chosen to vectorize different reviews. Machine learning algorithms do not work on strings. Thus, strings need to be converted into vector form, so that they can be processed effectively. TF stands for Term Frequency.  It is the ratio of number of times a term appears in a document and the total number of terms in the document. IDF stands for Inverse Document Frequency. Document Frequency is a ratio of the number of documents containing a certain term to the total number of documents present in the corpus. It is inverted so that, the terms which are present in a lot of documents have a lesser overall impact on the final TF-IDF vector representation as these words are generic in nature and contribute only little to the classification aspects of the corpus. Log of the IDF is taken, in order to normalize it, otherwise, it can possess a wider range and could potentially overshadow the effect of the TF terms. TF-IDF vectorizer was chosen over countvectorizer, as countvectorizer is only a quantitative vectorizer, whereas TF-IDF is a quantitative as well as a qualitative vectorizer. The model was first fitted with training data and latter the test data was transformed using the model.

# Model Training and Evaluation

1. Approach: Although linearity was not expected, the exact patterns in the dataset was unknown. Due to a large quantity of data and potential non-linearities, XGBoost was expected to perform well. In order to evaluate this hypothesis, first the raw performance of different models was evaluated. Latter on, the best performing models were fine-tuned using CV and hyperparameter tuning.
 
2. Model Pipeline: To evaluate the raw performance of different models, model_result()
function was built, to fit Logistic Regression, Gaussian Naive Bayes, Random Forest and XGBoost Models. evaluator() Function was written to generate an evaluation report for different models using a dataframe. Evaluation metrics such as Recall, Precision, F1-score, AUC-ROC Score and Accuracy were used for both test and train cases. 

3. Base-Line Results: Random Forest, without any hyperparameter running, got overfitted on the data. Logistic Regression, Gaussian Naive Bayes Algorithms and XGBoost gave a balanced performance. If we compare XGBoost with Logistic Regression Algorithm, then in the case of XGBoost, we gain 3% points of recall and loose 6% points of precision, in comparison to Logistic Regression Model. Overall F1-score and Accuracy of Logistic Regression Model is higher than the XGBoost model. 
 
So, Logistic Regression model and XGBoost Model were selected for hyperparameter tuning.

![base-line](https://user-images.githubusercontent.com/52705077/138032194-7628c506-152c-4687-9b60-c484622ebb39.PNG)

# Hyperparameter Tuning with Cross Validation
 
1. Tuning Logistic Regression Model
RepeatedStratifiedKFold with RandomizedSearchCV was used to tune hyperparameters of the Logistic Regression Model, such as C parameter and penalty type. Accuracy and F1-score, for this model, consistently remained high.

![lr](https://user-images.githubusercontent.com/52705077/138032291-378ac6fa-19ee-468b-96bd-18bf1d2ff9be.PNG)

2. Tuning XGBoost Model
RandomizedSearchCV was used to tune hyperparameters such as number of estimators, max depth and min sample split. The results were, once again, consistent with the initial base-line findings. 

![xg](https://user-images.githubusercontent.com/52705077/138032333-a0187630-521e-4a41-9d14-a00c8e0d8396.PNG)

# Conclusion
Logistic Regression Model, had a better F1-score and accuracy. Moreover, the XGBoost model, can be seen slightly overfitting. This could be a result of potential linearities present in the dataset and of the fact that XGBoost is a data demanding model. It is a possibility that, the data is simply not expressive enough, for XGBoost model to capture its details and patterns
 
Thus, an optimal Logistic Regression model, with hyperparameter tuning and cross validation, was built for predicting the customer sentiments, with a Recall of 81% and Precision of 82%


