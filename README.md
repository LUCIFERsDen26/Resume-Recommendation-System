
# Resume Recommendation System

Resume Recommendation System is designed to automate the process of
comparing job descriptions (JDs) with resumes. It assists in identifying
relevant candidates by analyzing the textual content of resumes and JDs.

For resumes, the script loads PDF files containing resumes, filters out
stopwords, and creates bigrams from the processed text. Similarly, for
job descriptions, it loads a JD PDF file, removes stopwords, and
extracts bigrams from the JD content.

To facilitate text analysis and feature extraction, the script
initializes key components, such as the TfidfVectorizer and
CountVectorizer from the scikit-learn library.

One of the script\'s core functionalities involves comparing the bigrams
extracted from the job description with those from the resumes,
employing a scoring mechanism. A lower score indicates a higher degree
of similarity between the job description and a particular resume,
aiding in the identification of potential matches.

In addition to bigram comparison, the script employs the ResumeParser
library to extract crucial candidate information from resumes, including
names, email addresses, phone numbers, and skills.

## Overview

This script performs the following key functions:

-   ### 1. Building Supporting Functions 

    The script defines several supporting functions, including:

    -   Extracting text from PDFs.
    -   Removing stopwords (common words like \"the,\" \"and,\" etc.)
        from text.
    -   Generating bigrams (pairs of adjacent words) from text.

-   ### 2. Processing Resumes 

    1.  Load PDF files containing resumes.
    2.  Remove stopwords from the text in the resumes.
    3.  Generate bigrams from the filtered resumes.

-   ### 3. Processing Job Descriptions (JD) 

    1.  Load a JD PDF file.
    2.  Remove stopwords from the JD text.
    3.  Generate bigrams from the filtered JD text.

-   ### 4. Text Vectorization 

    The script initializes the TfidfVectorizer and CountVectorizer from
    scikit-learn. These are essential for text analysis and feature
    extraction.

-   ### 5. Comparing JD and Resume Bigrams 

    The script compares the bigrams extracted from the JD with those
    from the resumes. This comparison is done using a scoring mechanism,
    and a lower score indicates higher similarity between the JD and the
    resume in terms of bigrams.

-   ### 6. Using ResumeParser 

    The script leverages the ResumeParser library to extract important
    information such as names, emails, phone numbers, and skills from
    the resumes.

## Dependencies

-   NumPy
-   NLTK (Natural Language Toolkit)
-   Gensim
-   PyPDF2
-   scikit-learn
-   ResumeParser (for resume data extraction)



# importing library

``` python
# Importing the NumPy library as np
import numpy as np

# Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

# Importing the necessary NLTK libraries
from nltk.collocations import *
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

# Creating a set of English stopwords
stop_words = set(stopwords.words('english'))

# Importing a function from the gensim library for removing stopwords
from gensim.parsing.preprocessing import remove_stopwords

# Importing regular expressions library
import re

# Importing the PyPDF2 library for working with PDF files
import PyPDF2 as pdf

# Importing the os library for working with the operating system
import os
```

# Convert PDF to TXT

``` python
def _convPdfText(path):
    # Initialize a counter to keep track of the PDF files processed
    count = 0
    
    # Create an empty list to store the extracted text from PDFs
    resume = []
    
    # Loop through files in the specified directory
    for file in os.listdir(path):
        # Check if the file has a .pdf extension
        if file.endswith('.pdf'):
            # Create a PdfFileReader object to read the PDF file
            pdf_reader = pdf.PdfFileReader(path + file)
            
            # Get the text content from the first page of the PDF
            page = pdf_reader.getPage(0)
            pageText = page.extractText()
            
            # Append the extracted text and its corresponding count to the 'resume' list
            resume.append([count, pageText])
            
            # Increment the counter
            count += 1
    
    # Return the list containing extracted text from PDFs
    return resume
```

# Functions for cleaning text from pdf

``` python
def StopWordsfilter(text):
    # Initialize an empty list to store filtered text
    filtered_sentence_list = []

    # Loop through each element in the 'text' list
    for p in text:
        # Tokenize the text from the second element of each sub-list
        word_tokens = word_tokenize(p[1])

        # Filter out stopwords from the tokenized words
        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        # Append the filtered sentence along with its count to the result list
        filtered_sentence_list.append([p[0], filtered_sentence])

    # Call 'filterWithRe' function to further process the filtered sentences
    return filterWithRe(filtered_sentence_list)
```

``` python
def filterWithRe(filter_Sent):
    # Initialize an empty list to store processed words
    procssed_words = []

    # Loop through each element in 'filter_Sent'
    for frl in filter_Sent:
        sub_list = []

        # Loop through the filtered words in the second element of each sub-list
        for x in range(len(frl[1])):
            if x != "":
                # Apply regular expressions to clean and format the words
                a = re.sub("[^-9A-Za-z ]", "", frl[1][x])
                b = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', " ", a)
                c = re.sub('[!"#$%&\'()*+-./:;<=>?@[\]^_`{|}~]', "", b)
                d = re.sub('and', "", c)
                sub_list.append(d)
            else:
                pass

        # Append the processed words along with their count to the result list
        procssed_words.append([frl[0], sub_list])

    # Call 'gensimfilter' function to further process the cleaned words
    return gensimfilter(procssed_words)
```

``` python
def gensimfilter(procss_list):
    # Initialize an empty list to store the final result
    final_result = []

    # Loop through each element in 'procss_list'
    for pre_proc in procss_list:
        sub_list_2 = []

        # Loop through the processed words in the second element of each sub-list
        for x in range(len(pre_proc[1])):
            # Remove stopwords using gensim's 'remove_stopwords' function
            filtered_sentence = remove_stopwords(pre_proc[1][x])

            # If the filtered sentence is not empty, convert it to lowercase and append to the result list
            if filtered_sentence != "":
                filtered_sentence = filtered_sentence.lower()
                sub_list_2.append(filtered_sentence)

        # Append the final processed words along with their count to the result list
        final_result.append([pre_proc[0], sub_list_2])

    # Return the final processed result
    return final_result
```

# Function for collecting bigram

``` python
def GetBigram(Words):
    # Create a BigramCollocationFinder from the input list of words
    finder = BigramCollocationFinder.from_words(Words)
    
    # Initialize an empty list to store the bigram key phrases
    lst = []
    
    # Initialize a list to store key phrases from the CV
    Key_word_from_CV = []
    
    # Iterate through the sorted n-grams and their frequency counts
    for i in sorted(finder.ngram_fd.items()):
        # Check if the n-gram occurs more than once (frequency > 1)
        if i[1] > 1:
            # Append the n-gram to the 'lst' list
            lst.append(i[0])
        else:
            pass
    
    # Convert the list of tuples (bigrams) into a list of strings (key phrases)
    for j in lst:
        Key_word_from_CV.append(" ".join(j))
    
    # Return the list of key phrases extracted from the CV
    return Key_word_from_CV
```

# Main Code

### get resume pdf from path and read text ,clean text

``` python
resume_path = "/home/lucifer/Downloads/txt resume/"
RawResume = _convPdfText(resume_path)
FinalResultResume = StopWordsfilter(RawResume)
print(FinalResultResume)
```

::: 
    [[0, ['mahendra', 'gawali', 'data', 'scientist', 'data', 'engineer', 'personal', 'info', 'email', 'gawalimahen', 'gmailcom', 'phone', '99', 'skills', 'languagesstrategic', 'planning', 'data', 'science', 'data', 'engineer', 'data', 'analytics', 'research', 'skills', 'python', 'sql', 'artificial', 'intelligence', 'machine', 'learning', 'convolutional', 'neural', 'network', 'object', 'oriented', 'programming', 'snowflake', 'etlmicrosoft', 'azure', 'java', 'english', 'hindi', 'marathiphd', 'experience', 'agile', 'softwa', 'development', 'testing', 'deployment', 'life', 'cycle', 'best', 'practices', 'in', 'depth', 'understan', 'ding', 'r', 'python', 'related', 'technology', 'like', 'microso', 'ft', 'azure', 'ibm', 'watson', 'studio', 'snowflake', 'framework', 'experience', 'optimization', 'techniques', 'expertise', 'enterprise', 'resource', 'planning', 'development', 'retailed', 'manufacturing', 'domain', 'experience', 'field', 'linked', 'business', 'analytics', 'statistics', 'operatio', 'ns', 'research', 'geography', 'applied', 'mathematics', 'science', 'engineering', 'work', 'history', 'education', 'certificates', 'presentsr', 'consultant', 'data', 'science', 'data', 'engineer', 'quellcode', 't', 'echnology', 'pvt', 'ltd', 'kopargaon', 'completed', 'projects', 'traffic', 'sign', 'identification', 'convolutional', 'neural', 'network', 'artificial', 'intelligence', 'based', 'enterprise', 'resource', 'planning', 'software', 'natural', 'language', 'processing', 'real', 'time', 'customer', 'reviews', 'enterprise', 'resource', 'planning', 'software', 'principal', 'component', 'analysis', 'dimensionality', 'reduction', 'dataset', 'amazon', 'book', 'review', 'data', 'analysis', 'heart', 'disease', 'dataset', 'etl', 'medical', 'data', 'snowflake', 'etl', 'medical', 'data', 'azure', 'data', 'factory', 'presentassociate', 'professor', 'sanjivani', 'college', 'engineering', 'kopargaon', 'professor', 'mentor', 'research', 'grants', 'research', 'paper', 'writing', 'publication', 'cloud', 'computing', 'doctor', 'philosophy', 'thadomal', 'shahani', 'engineering', 'college', 'bra', 'w', 'mumbai', 'computer', 'engineering', 'master', 'engineering', 'north', 'maharashtra', 'university', 'jalgaon', 'computer', 'engineering', 'bachelor', 'engineering', 'north', 'maharashtra', 'university', 'jalgaon', 'ibm', 'data', 'science', 'professional', 'data', 'analysis', 'python', 'introduction', 't', 'ensorflow', 'artificial', 'intelligence', 'machine', 'learning', 'deep', 'learning']]]



### collect bigram from cleaned list of text

``` python
ResumeBigram = []
for count,result in FinalResultResume:
    result = GetBigram(result)
    ResumeBigram.append([count,result])
print(ResumeBigram)
```

::: 
    [[0, ['artificial intelligence', 'computer engineering', 'convolutional neural', 'data analysis', 'data engineer', 'data science', 'engineering north', 'enterprise resource', 'etl medical', 'intelligence machine', 'machine learning', 'maharashtra university', 'medical data', 'neural network', 'north maharashtra', 'planning software', 'resource planning', 'science data', 'university jalgaon']]]




### get JD pdf from path and read text ,clean text

``` python
Jd_Path = '/home/lucifer/Desktop/test_dataset/'
RawJd = _convPdfText(Jd_Path)
FinalResultJd = StopWordsfilter(RawJd)
print(FinalResultJd)
```

::: 
    [[0, ['data', 'scientist', 'skills', 'you', 'need', 'master', 'skills', 'required', 'data', 'scientist', 'jobs', 'industries', 'organizations', 'want', 'pursue', 'data', 'scientist', 'career', 'let', 'look', 'musthave', 'data', 'scientist', 'qualifications', 'key', 'skills', 'needed', 'data', 'scientist', 'programming', 'skills', 'knowledge', 'statistical', 'programming', 'languages', 'like', 'r', 'python', 'database', 'query', 'languages', 'like', 'sql', 'hive', 'pig', 'desirable', 'familiarity', 'scala', 'java', 'c', 'added', 'advantage', 'statistics', 'good', 'applied', 'statistical', 'skills', 'including', 'knowledge', 'statistical', 'tests', 'distributions', 'regression', 'maximum', 'likelihood', 'estimators', 'proficiency', 'statistics', 'essential', 'datadriven', 'companies', 'machine', 'learning', 'good', 'knowledge', 'machine', 'learning', 'methods', 'like', 'knearest', 'neighbors', 'naive', 'bayes', 'svm', 'decision', 'forests', 'strong', 'math', 'skills', 'multivariable', 'calculus', 'linear', 'algebra', 'understing', 'fundamentals', 'multivariable', 'calculus', 'linear', 'algebra', 'important', 'form', 'basis', 'lot', 'predictive', 'performance', 'algorithm', 'optimization', 'techniques', 'data', 'wrangling', 'proficiency', 'hling', 'imperfections', 'data', 'important', 'aspect', 'data', 'scientist', 'job', 'description', 'experience', 'data', 'visualization', 'tools', 'like', 'matplotlib', 'ggplot', 'djs', 'tableau', 'help', 'visually', 'encode', 'data', 'excellent', 'communication', 'skills', 'incredibly', 'important', 'findings', 'technical', 'nontechnical', 'audience', 'strong', 'software', 'engineering', 'background', 'hson', 'experience', 'data', 'science', 'tools', 'problemsolving', 'aptitude', 'analytical', 'mind', 'great', 'business', 'sense', 'degree', 'computer', 'science', 'engineering', 'relevant', 'field', 'preferred', 'proven', 'experience', 'data', 'analyst', 'data', 'scientist']]]




### get bigram from cleaned jd text


``` python
JdBigram = []
for count,result in FinalResultJd:
    result = GetBigram(result)
    JdBigram.append([count,result])
print(JdBigram)
```

::: 
    [[0, ['calculus linear', 'data scientist', 'experience data', 'knowledge statistical', 'languages like', 'linear algebra', 'machine learning', 'multivariable calculus']]]


# Compare function to compare jd bigram and resume bigram

``` python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

``` python
def TFIDF(BgList):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(BgList)
    CosSimilarityTfidf = cosine_similarity(tfidf[0],tfidf[1])
    return CosSimilarityTfidf
```

``` python
def CountVectorize(BgList):
    count_vectorizer = CountVectorizer()
    count = count_vectorizer.fit_transform(BgList)
    CosSimilarityCountVect = cosine_similarity(count[0],count[1])
    return CosSimilarityCountVect
```

### get dict of resume and jd in this format

{ jd number: \[ \[ resume number, array(\[\[score\]\]) \] \] } the score
are two differnt types one is TFID and another is Count Vector

``` python
tfidDict = {}
for JdCount in range(len(JdBigram)):
    result = []
    for resumeCount in range(len(ResumeBigram)):
        wordslist=[' '.join(JdBigram[JdCount][1]),' '.join(ResumeBigram[resumeCount][1])]
        result.append([resumeCount,TFIDF(wordslist)])
    tfidDict[JdCount]=result
print(tfidDict)
```

::: {.output .stream .stdout}
    {0: [[0, array([[0.18681731]])]]}

``` python
countVecDict = {}
for JdCount in range(len(JdBigram)):
    result = []
    for resumeCount in range(len(ResumeBigram)):
        wordslist=[' '.join(JdBigram[JdCount][1]),' '.join(ResumeBigram[resumeCount][1])]
        result.append([resumeCount,CountVectorize(wordslist)])
    tfidDict[JdCount]=result
print(tfidDict)
```

:::
    {0: [[0, array([[0.30987534]])]]}



### Resume parser \[skill compare\]


``` python
from resume_parser.resume_parser import ResumeParser
```

### predict resume skills and jd skills

``` python
resume = ResumeParser("/home/lucifer/Downloads/txt resume/MBG_Resume.pdf")
resume_skill = resume.get_extracted_data()['skills']
print(resume_skill)
print(resume.get_extracted_data())
```

::: 
    ['Ibm', 'Operations', 'Tensorflow', 'Programming', 'Writing', 'English', 'Agile', 'Cloud', 'Sql', 'Computer science', 'Python', 'Testing', 'Life cycle', 'Statistics', 'Java', 'Machine learning', 'Etl', 'Engineering', 'System', 'R', 'Mathematics', 'Analytics', 'Research', 'Analysis', 'Email']
    {'name': 'Mahendra Gawali', 'email': 'gawalimahen123@gmail.com', 'mobile_number': '9171211111', 'skills': ['Ibm', 'Operations', 'Tensorflow', 'Programming', 'Writing', 'English', 'Agile', 'Cloud', 'Sql', 'Computer science', 'Python', 'Testing', 'Life cycle', 'Statistics', 'Java', 'Machine learning', 'Etl', 'Engineering', 'System', 'R', 'Mathematics', 'Analytics', 'Research', 'Analysis', 'Email'], 'education': [], 'experience': [' Optimization'], 'competencies': {}, 'measurable_results': {}}


``` python
jd = ResumeParser('/home/lucifer/Desktop/test_dataset/jd.pdf')
jd_skill = jd.get_extracted_data()
print(jd_skill)
```

::: 
    ['Technical', 'Analytical', 'Database', 'C++', 'Ggplot', 'Scala', 'Communication', 'Calculus', 'Statistics', 'Computer science', 'Engineering', 'Java', 'Tableau', 'Sql', 'Programming', 'R', 'Python', 'Matplotlib', 'Math', 'Hive']


### make list of skills for comparing

``` python
resume_skill = ' '.join(resume_skill)
jd_skill = ' '.join(jd_skill)
skillList = [jd_skill,resume_skill]
```

``` python
skill_Tfid = TFIDF(skillList)
count_vec = CountVectorize(skillList)
```

``` python
print('TFID',skill_Tfid,'\n','CountVec',count_vec)
```

::: 
    TFID [[0.22231997]] 
     CountVec [[0.35320863]]

