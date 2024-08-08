
# KEY DRIVER ANALYSIS ON IMDb MOVIE REVIEWS 



## Introduction
This report outlines the steps taken to preprocess the IMDb movie review dataset, extract keywords using two algorithms (TF-IDF and TextRank), combine these algorithms, and perform sentiment analysis using SVM and VADER. The methodology, findings, and recommendations are summarized below.

## Methodology

### Data Preprocessing
1. **Text Cleaning:** Removed HTML tags, special characters, and converted text to lowercase.
2. **Tokenization:** Split reviews into individual words.
3. **Stopword Removal:** Eliminated common English stopwords using the `NLTK` library.

### Keyword Extraction
1. **TF-IDF (Term Frequency-Inverse Document Frequency):**
   - **Reasoning:** TF-IDF highlights words that are important in a document relative to the entire corpus, effectively identifying significant terms.
   - **Implementation:** Used `TfidfVectorizer` from `sklearn` to calculate TF-IDF scores for words in each review.

2. **TextRank:**
   - **Reasoning:** TextRank is a graph-based ranking algorithm that captures the importance of words based on their relationships with other words in the text, similar to PageRank.
   - **Implementation:** Constructed a word graph using co-occurrence and applied the PageRank algorithm using `networkx`.

3. **Combining TF-IDF and TextRank:**
   - **Reasoning:** Combining the strengths of both algorithms leverages local importance (TF-IDF) and global importance (TextRank), resulting in more robust keyword extraction.
   - **Implementation:** Averaged the normalized scores from both algorithms to obtain final keyword rankings.

### Sentiment Analysis
1. **Support Vector Machine (SVM):**
   - **Reasoning:** SVMs are effective for text classification, offering a good balance between performance and interpretability.
   - **Implementation:** Trained an SVM classifier with a linear kernel using TF-IDF features of the reviews using `scikit-learn`.

2. **VADER (Valence Aware Dictionary and sEntiment Reasoner):**
   - **Reasoning:** VADER is a lexicon and rule-based sentiment analysis tool specifically designed for social media texts, which is suitable for analyzing movie reviews.
   - **Implementation:** Applied VADER to compute sentiment scores for each review using `vaderSentiment`.

## Results
- **Accuracy:** The SVM model achieved an accuracy of 86% on the test set.
- **Sentiment Distribution:** VADER sentiment analysis tool identified the sentiment of the text as negative with a compound score of -0.9929 and the accuracy of the VADER model in classifying sentiments was found to be 69%.
- **Keyword Insights:** Combined TF-IDF and TextRank provided key terms reflecting movie themes and sentiments effectively.
- **Extract Key Factors:** 
-  Phrases or keywords that are most frequently mentioned in customer feedback are,
  
### **Top Keywords:**
1. movie
2. film
3. one
4. like
5. good
6. time
7. character
8. story
9. really
10. see
    
**Analyze Sentiment:** 
- By comparing  lexicon-based methods, for  load sentiment lexicons and compute sentiment scores and Used  machine learning approaches SVM Support vector Machine for spliting  into training and test sets, train the model, and evaluate its performance.
Report for the SVM:

**Accuracy**: 0.86

Confusion Matrix:

 [[4186  775]

 [ 631 4408]]
  
**Classification  Report**:      precision    recall  f1-score   support

    negative       0.87      0.84      0.86      4961
    positive       0.85      0.87      0.86      5039

    accuracy                           0.86     10000

    macro avg       0.86      0.86      0.86     10000

    weighted avg       0.86      0.86      0.86     10000


**Report for VADER:**

Sentiment Analysis using VADER:

Sentiment: Negative

Compound Score: -0.9929

Accuracy of VADER sentiment analysis: 0.69

### **Calculate Importance:**

**Importance Score of the top keyword:**

{'movie': 0.47443794271635353, 'film': 0.5251822398939695, 'one': 0.516301512780386, 'like': 0.48842592592592593, 'good': 0.5178075271533216, 'time': 0.5272808586762076, 'character': 0.5222779580997344, 'story': 0.5620253164556962, 'really': 0.47505781301618766, 'see': 0.5200469719845664}

### **Actionable Insights:**
Based on the customer experience
1. Keyword: good, Importance Score: 0.74
2. Keyword: story, Importance Score: 0.72
3. Keyword: character, Importance Score: 0.71
4. Keyword: like, Importance Score: 0.70
5. Keyword: time, Importance Score: 0.70
6. Keyword: see, Importance Score: 0.70
7. Keyword: film, Importance Score: 0.69
8. Keyword: one, Importance Score: 0.69
9. Keyword: really, Importance Score: 0.68
10. Keyword: movie, Importance Score: 0.68

## Visualization
Used `matplotlib`, `seaborn` for or visualizing results.
![image](https://github.com/YuvashreeMagesh/Key-Driver-analayis/assets/128991477/487b52c7-b83b-4ba6-a03c-50c7fc358a05)

![image](https://github.com/YuvashreeMagesh/Key-Driver-analayis/assets/128991477/5857e9ed-2e51-46e6-b349-ad4279d604fc)

![image](https://github.com/YuvashreeMagesh/Key-Driver-analayis/assets/128991477/d5865db8-c6c5-46b8-a440-88b689e5fb3e)

![image](https://github.com/YuvashreeMagesh/Key-Driver-analayis/assets/128991477/132411c9-91e0-47c7-9a3b-0dd693f94b82)





## Key Findings
- The combined keyword extraction method effectively identified significant terms related to movie themes and sentiments.
- SVM and VADER provided complementary insights into review sentiments, with SVM offering robust classification and VADER highlighting nuanced sentiment expressions.

## Recommendations
1. **Hybrid Keyword Extraction:** Combining TF-IDF and TextRank is recommended for comprehensive keyword extraction in textual datasets.
2. **SVM for Sentiment Classification:** SVM is effective for text sentiment classification and should be considered for similar tasks.
3. **VADER for Detailed Sentiment Analysis:** VADER is useful for capturing nuanced sentiments in text, particularly in social media and review contexts.

### Conclusion

This project analyzed IMDb movie reviews using advanced NLP techniques. I successfully extracted keywords using TF-IDF and TextRank, combining their strengths for comprehensive insights. Sentiment analysis with SVM and VADER provided accurate sentiment classification and nuanced sentiment scoring. Recommendations include leveraging hybrid keyword extraction and SVM for effective sentiment analysis in similar datasets, with potential for further refinement through preprocessing adjustments and model optimization.
