Predictive-Modelling-on-Hit-Songs-Using-Repeated-Chorus
About Project:
We examined this myth by creating a data set of choruses from popular artists and applied supervised Machine Learning (ML) techniques to predict the popularity of their works purely based on the audio features extracted from the chorus.
Data Collection:
1.	We get top artists from year 2006 to 2021 using Bill Board API and Top-100-Songs.
2.	We get 80 artists from year 2006 to 2021.
3.	Then we collect popular songs data of those 80 artists which includes Title of songs, Artist name.
4.	We define Label=1 for all popular songs.
5.	We collect all other songs of those artist from above mentioned years and make Label = 0 for all unpopular songs.
6.	Final dataset has 3 columns named Artist, Title, Label.
7.	We save the data in csv file.
8.	Shape of the data frame is (809,3).
Songs Downloading:
1.	Using Youtube-dl API, we downloaded songs in mp3 format.
2.	First, we load dataset of song names we prepared above.
3.	We make lists of popular and unpopular songs as follows:
4.	We define a function to download popular and unpopular songs in mp3 format.
5.	We downloaded 403 popular songs and 406 unpopular songs.
6.	We add new column in our data frame named “Song_path”, which have path address of downloaded songs in the system.
7.	We save table to csv file.
8.	Shape of the data frame is (809,4).
Chorus Extraction:
1. Using Pychorus library, we extracted the repeated chorus of our downloaded songs.
2.	First, we load dataset of songs.
3.	We check the null values and found we have 2 null values in song_path column.
4.	We dropped the null values as
	df = df.dropna (axis=0)
5.	Shape of data frame is now (807,4)
6.	We define a function to extract 15s choruses in .wav format from songs as required in paper.
7.	After applying function, it was found that chorus of 15 songs are not extracted.
8.	We add new column in our data frame named “extracted_chorus_path”, which have path address of downloaded Chorus in the system.
9.	We dropped the rows with un extracted values. 
10.	Shape of data frame is now (792,5)
11.	We save table to csv file.
Audio Features:
1.	Using Librosa library, we extracted the audio features of our downloaded choruses.
2.	We define a function to get statistics of each dimension data for each audio raw feature.
def statistics(list, feature, columns_name, data):
3.	We define a function to extract audio features with statistics.
def extract_features(audio_path, title):
4.	After applying functions we get 518 audio features.
5.	We create dataframe for new extracted audio features:
newData = pd.DataFrame(data, columns=cols)
6.	Shape of data frame is now (792,519).
7.	We merge 2 data sets as:
final_data = df.join(newData)
8.	Shape of data frame is now (792,523).
9.	We save table to csv file.
EDA:
1. First, we load dataset.
2.	We check first five rows of data.
3.	We checked shape of data frame: (792,523).
4.	We check data types. It has Integer, Object, Float data type.
5.	We check Summary Statistics using data.describe( )
6.	We remove un necessary columns: 
data.drop(["Artist","Title","song_path","extracted_chorus_path"],axis=1,inplace=True)
7.	We check null values and duplicated values.
8.	Shape of the data now: (792,519)
9.	We check data balance of our target variable. We have 50.4 % popular data (label=1) and 49.6 % unpopular data (label=0).
10.	We check correlation using corr_df = data.corr().
11.	We check MI scores and get top corelated features.
12.	We perform PCA for feature selection.
13.	we can observe that to achieve 95% variance, the dimension was reduced to 173 principal components from the actual 519 dimensions.
14.	We make new data frame with best features as
final_data.to_csv('improved_best_features_df.csv'). Shape of the data frame is (809,4).

Model Building:
1.	We have made 4 models:
Gradient Boosting Classifier.
Linear Discriminant Analysis.
ADA Boosting Classifier.
KNN Classifier.
Gradient Boosting Classifier:
1.	We import useful libraries.
2.	We split data in to test and train sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
3.	We fit our model on train data sets.
4.	We check our model on different learning rates.
5.	We're mainly interested in the classifier's accuracy on the validation set, but it looks like a learning rate of 0.75 gives us the best performance on the validation set i.e. 0.98.
6.	Now we can evaluate the classifier by checking its accuracy and creating a confusion matrix. Let's create a new classifier and specify the best learning rate we discovered.
7.	We have got Confusion Matrix
Confusion Matrix:
[[76  0]
 [ 0 83]]
8.	We get the accuracy of 100% on test data set.
Linear Discriminant Analysis:
1.	We import useful libraries.
2.	We split data in to test and train sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
3.	We fit our model on train data sets.
4.	We test out model on test set.
5.	We get the accuracy score of 100% on test data set.
ADA Boost Classifier:
1.	We import useful libraries.
2.	We split data in to test and train sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
3.	We fit our model on train data sets.
4.	We test out model on test set.
5.	We get the accuracy score of 100% on test data set.
KNN Classifier:
1. We import useful libraries.
2.	We split data in to test and train sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
3.	We fit our model on train data sets.
4.	We test out model on test set.
5.	We get the accuracy score of 100% on test data set.
Cross Validation:
1.	We check over fitting and under fitting of models.
2.	We use K-Fold Methods.
3.	For GB Classifier, we get average CV score of 100.
4.	For LDA Classifier, we get average CV score of 27.
5.	For ADA boost Classifier, we get average CV score of 100.
6.	For KNN Classifier, we get average CV score of 32.8.
7.	Only GB Classifier and ADA Boost are normally fitted.
8.	(e.g. in my case it is chorus_extract/song_name.wav))
