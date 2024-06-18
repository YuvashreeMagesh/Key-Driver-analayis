
# Report on Key Driver Analysis on IMDb Moive Reviews

## Introduction
This report outlines the steps taken to preprocess the IMDb movie review dataset, extract keywords using two algorithms (TF-IDF and TextRank), combine these algorithms, and perform sentiment analysis using SVM and VADER. The methodology, findings, and recommendations are summarized below.

## Methodology

### Data Preprocessing
1. **Text Cleaning:** Removed HTML tags, special characters, and converted text to lowercase.
2. **Tokenization:** Split reviews into individual words.
3. **Stopword Removal:** Eliminated common English stopwords using the NLTK library.

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
   - **Implementation:** Trained an SVM classifier with a linear kernel using TF-IDF features of the reviews.

2. **VADER (Valence Aware Dictionary and sEntiment Reasoner):**
   - **Reasoning:** VADER is a lexicon and rule-based sentiment analysis tool specifically designed for social media texts, which is suitable for analyzing movie reviews.
   - **Implementation:** Applied VADER to compute sentiment scores for each review.

## Results
- **Accuracy:** The SVM model achieved an accuracy of 86% on the test set.
- **Sentiment Distribution:** VADER sentiment analysis tool identified the sentiment of the text as negative with a compound score of -0.9929.
  The accuracy of the VADER model in classifying sentiments was found to be 69%.
- **Keyword Insights:** Combined TF-IDF and TextRank provided key terms reflecting movie themes and sentiments effectively.

## Visualization
1. **Confusion Matrix (SVM):** Showed the true vs. predicted labels, indicating the classifier's performance.
2. **Sentiment Distribution Pie Chart (VADER):** Illustrated the proportion of positive vs. negative reviews.
3. **Top Keywords Bar Chart:** Displayed the most significant keywords extracted from reviews.

## Key Findings
- The combined keyword extraction method effectively identified significant terms related to movie themes and sentiments.
- SVM and VADER provided complementary insights into review sentiments, with SVM offering robust classification and VADER highlighting nuanced sentiment expressions.

## Recommendations
1. **Hybrid Keyword Extraction:** Combining TF-IDF and TextRank is recommended for comprehensive keyword extraction in textual datasets.
2. **SVM for Sentiment Classification:** SVM is effective for text sentiment classification and should be considered for similar tasks.
3. **VADER for Detailed Sentiment Analysis:** VADER is useful for capturing nuanced sentiments in text, particularly in social media and review contexts.

---

By implementing the above methodologies, you can achieve a comprehensive understanding of sentiments expressed in IMDb movie reviews and extract meaningful keywords that represent core themes and opinions.

---convert 
