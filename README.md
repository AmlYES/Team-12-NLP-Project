# Podcast Topic Modeling & NLP Analysis

## 🎯 Project Overview
This project aims for preprocess a **diverse set of 6 Egyptian Spotify podcasts, each belonging to a different category** for a **Topic Modeling** task. The dataset consists of transcribed episodes from various genres, and the goal is to analyze text data, perform exploratory data analysis (EDA), preprocess text, and prepare the data for topic modeling.

## 📂 Dataset
The dataset consists of **6 different podcasts**, each belonging to a unique category:
- **Food** (16 episodes)
- **Relationships** (11 episodes)
- **Self-Help** (25 episodes)
- **Educational** (2 episodes)
- **Comedy** (22 episodes)
- **TV & Film** (43 episodes)

Each episode is stored as a **text file**, containing transcriptions of spoken dialogue.

The **horizontal** sampling approach ensures category richness and variety to support the training of a topic modeling task.

## 🔍 Exploratory Data Analysis (EDA)
Before preprocessing, **EDA** was performed to understand the dataset:
- **Metadata Enhancement:** The timestamps were used to **calculate episode durations**, as they were missing from the original data.
- **Word & Sentence Counts:** Analyzed **word count per episode**, **sentence count**, **unique words**, and **most frequent words**.
- **Visualization:**
  - **Word Clouds**: Generated for each episode to visualize common words.
  - **Histograms & Bar Charts**: Compared statistics across podcasts.
- **Sentiment Analysis**: Attempted using **pretrained Arabic models**, but results were **inaccurate for Egyptian Arabic words**.
- **Named Entity Recognition (NER)**: which faced similar challenges due to the lack of support for Egyptian Arabic words.

## 🛠 Preprocessing Steps
To prepare the data for topic modeling, the following steps were performed:
1. **Text Cleaning**: Removed punctuation, special characters, timestamps, and non-Arabic symbols.
2. **Tokenization**: Splitting text into individual words.
3. **Stopword Removal**:
    -  Compiled stopwords from various sources: common Egyptian Arabic stopwords, standard Arabic stopwords, and frequently occurring words from the dataset.
    -  Stored stopwords in a JSON file for consistency.
    -  Removed all stopwords from the dataset.
4. **Lemmatization**: Attempted but faced challenges due to limited support for Egyptian Arabic in libraries such as NLTK, SpaCy, and Farasa.
5. **Word Segmentation**: Planned as much as supported by available libraries.
6. **N-gram Extraction**: Identified frequent word phrases (like bigrams) to capture common multi-word expressions.

## 💡 Insights Extraction
With the cleaned dataset, we performed additional analysis:
- **Sentence Length Analysis:** Compared sentence lengths across different podcast categories to identify trends.
- **Word Clouds:** Regenerated word clouds post-cleaning to visualize important words.
- **Keyword Extraction:** Extracted key phrases and relevant words.
- **N-gram Analysis:** Identified common phrases and expressions used in different categories.

## Preparing Data for Topic Modeling
The final step involves structuring the data for effective topic modeling:

1. **Creating a Structured Dataset:**
   - Compiled a DataFrame where each row represents an episode.
   - Columns include podcast name, episode name, category, and text content.
2. **TF-IDF Vectorization:**
   - Converted text into numerical form using Term Frequency-Inverse Document Frequency (TF-IDF).
3. **Clustering for Validation:**
   - Applied K-Means clustering to group episodes and validate whether episodes from the same podcast naturally cluster together.
4. **Finalizing the Dataset:**
   - Saved the structured dataset into a CSV file for future modeling tasks.


## 🏆 Challenges & Insights
- **Egyptian Arabic Complexity**: Existing NLP models for **NER & Sentiment Analysis** performed poorly due to the lack of Egyptian Arabic support.
- **Short Episode Transcripts**: Some podcasts (e.g., Educational with only 2 episodes) lacked sufficient text data.

## 🚀 Future Improvements
- **Train a custom sentiment model** for Egyptian Arabic.
- **Improve entity recognition** using fine-tuned models.
- **Implement topic modeling** using LDA (Latent Dirichlet Allocation) or another suitable model.
- **Evaluate model performance** and interpret discovered topics.
- **Improve preprocessing** to better handle Egyptian Arabic linguistic nuances.

---

## 📌 How to Use This Repository
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```
2. Run the notebook to analyze & preprocess the dataset:
   ```bash
   jupyter notebook NLP.ipynb
   ```
3. Train topic models and visualize results!

---

### 📧 Contact
For any questions, feel free to reach out!
aml.mohamed@student.guc.edu.eg

---

## Acknowledgments
This project was inspired by the need for better NLP solutions for Egyptian Arabic. Thanks to various open-source libraries and datasets that made this research possible.


💡 *If you found this project useful, don't forget to ⭐ the repo!*




