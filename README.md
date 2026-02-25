# Sentiment-Analysis-using-Machine-Learning-Model-Comparison-
This project performs sentiment analysis on customer/movie reviews and classifies them into Positive or Negative categories using Natural Language Processing (NLP) and Machine Learning techniques
The main objective is not only to build a classifier, but also to compare multiple machine learning algorithms, evaluate their performance, and visualize how different models behave on text data.
This project demonstrates the full workflow of a data science pipeline:
Data cleaning -> Text preprocessing  -> Feature extraction -> Model training -> Model evaluation -> Visualization & interpretation
Technologies Used : Python, Pandas & NumPy, Scikit-learn, XGBoost, NLTK (text preprocessing), Matplotlib & Seaborn (visualization), TF-IDF Vectorization
1. Data Preprocessing: Removed punctuation and special characters
                       Converted text to lowercase
                       Removed stopwords using NLTK
                       Tokenized text data
2. Feature Extraction: Text data was converted into numerical features using:
3. TF-IDF Vectorizer: This converts words into weighted vectors based on their importance in the dataset.
4. Machine Learning Models Implemented (trained and compared): Logistic Regression
                                                               Support Vector Machine (LinearSVC)
                                                               Naive Bayes (MultinomialNB)
                                                               Random Forest Classifier
                                                               XGBoost Classifier
5. Model Evaluation: Accuracy Score
                     Confusion Matrix
                     Precision & Recall
                     An accuracy comparison bar chart was generated to compare model performance.

6. Visualizations: The project includes multiple visual outputs:
                   1. Accuracy Comparison: Shows performance differences between models.
                   2. Confusion Matrix (Individual):A confusion matrix was generated for each model to analyze prediction quality:
                                                    True Positives, True Negatives, False Positives, False Negatives
                   3. Cluster Visualization: Using PCA dimensionality reduction, TF-IDF features were reduced to 2D and plotted to visualize how each model separates positive and negative reviews.
