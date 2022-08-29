"""
File containing the parameters used for the computation of the similarities 
"""
from nltk.corpus import stopwords


ENG_STOPWORDS = set(stopwords.words("english"))
N_GRAM = 3
GAP_TOLERANCE = 5 
PADDING = 20