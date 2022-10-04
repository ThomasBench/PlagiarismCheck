"""
All the helpers functions needed for the jupyter notebook to work.
"""
import enchant 
from typing import List, Tuple, Dict, Set, Callable
from termcolor import colored
from collections import defaultdict
from nltk import word_tokenize
from tqdm import tqdm
from copy import deepcopy
import pandas as pd 
import plotly.graph_objects as go
import plotly.express as px
import plotly
from scipy.interpolate import interp1d
# import pyspark
eng_dict = enchant.Dict("en")
N_gram = Tuple[str]
Article = Tuple[List[Tuple[int,str]],Dict[N_gram,List[int]]]
def correct_token(token: str) -> str : 
    """
    Get a token, and correct it using Enchant dictionarries

    Parameters
    ----------
    token : str
        Token to correct


    Returns
    -------
    str
        the corrected token 
    """
    if eng_dict.check(token):
        return token
    else:
        suggestions = eng_dict.suggest(token)
        if len(suggestions) > 0:
            return suggestions[0]
        return token

def generate_n_gram(text: List[str], n: int) -> List[N_gram]:
    """
    Takes a list of token and generate the n_grams for this list

    Parameters
    ----------
    text : List[str]
        list of tokens 
    n : int
        the n parameter to generate the n-grams

    Returns
    -------
    List[N_gram]
        List of n-grams
    """
    return zip(*[text[i:] for i in range(n)])

def to_ngram(index_tup):
    return tuple([tup[1] for tup in index_tup])


def align_sequences(art_grams_1: Dict[N_gram,List[int]],art_grams_2: Dict[N_gram,List[int]]) -> List[Tuple[N_gram,int,int]]:
    """
    Takes two dictionnaries that contains n_grams and return a list of matching indexes for each corresponding n_gram

    Parameters
    ----------
    art_grams_i, i = 1,2 : Dict[N_gram,List[int]],
        Dictionaries with n_gram as index and list of indexes as values

    Returns
    -------
    List[Tuple[N_gram,int,int]]
        List of Tuple that are organised as such:
            N_gram: matching n_gram between the two sequences
            int: index of the matching n_gram for the first sequence 
            int: index of the matching n_gram for the first sequence
    """
    matching_grams = []
    for gram_1 in art_grams_1:
        for gram_2 in art_grams_2:
            if gram_1 == gram_2:
                art_1_matching_id = art_grams_1[gram_1].pop(0)
                art_2_matching_id = art_grams_2[gram_1].pop(0)
                matching_grams.append((gram_1,art_1_matching_id,art_2_matching_id))
    return matching_grams

def glue_sequence(sequence: List[Tuple[N_gram,int,int]], gap_tolerance:int) -> List[Tuple[Tuple[int,int],Tuple[int,int]]]:
    """
    Glue together a sequence of n_grams with a certain tolerance between two non matching n_grams
    """
    final = []
    temp = [sequence[0][1:]]
    last_seen = sequence[0][1:]
    for _, ind_1, ind_2 in sequence[1:]:
        if 0<= ind_1 - last_seen[0] < gap_tolerance and 0<= ind_2 - last_seen[1] < gap_tolerance :
            temp.append((ind_1,ind_2))
        else:
            final.append((temp[0], temp[-1]))
            temp.clear()
            temp.append((ind_1,ind_2))
        last_seen = (ind_1,ind_2)
    final.append((temp[0], temp[-1]))
    return final

def retrieve_text(tokenized_article: List[Tuple[int,str]], start_index: int, end_index: int) -> str:
    return " ".join([x[1] for x in tokenized_article[start_index:end_index+1]])

def display_match(match_indexes: Tuple[Tuple[int,int],Tuple[int,int]] ,treated_1: List[Tuple[int,str]], treated_2: List[Tuple[int,str]] , padding:int) -> None:
    # print(sequence)
    start_1,end_1 = match_indexes[0][0] , match_indexes[1][0] +2
    start_2,end_2 = match_indexes[0][1] , match_indexes[1][1] +2
    text_1 = [retrieve_text(treated_1,start_1-padding, start_1), colored(retrieve_text(treated_1,start_1,end_1),"red"),retrieve_text(treated_1, end_1, end_1 + padding)]
    text_2 = [retrieve_text(treated_2,start_2-padding, start_2), colored(retrieve_text(treated_2,start_2,end_2),"red"),retrieve_text(treated_2, end_2, end_2 + padding)]
    print(*text_1)
    print(*text_2)


#### High-level functions 
### Model parameters




def treat_article(article_path:str, stopwords: Set[str], n: int) -> Article:
    with open(article_path, mode = "r", encoding = "utf-8") as f:
        data = ''.join(f.readlines())
    full_article = ''.join([c for c in data if c.isalnum() or c == " "])
    tokenized_article = list(enumerate(word_tokenize(full_article)))
    filtered_article = [(index,token.lower()) for index,token in tokenized_article if token not in stopwords ]
    filtered_indexes = [index for index,_ in filtered_article]
    corrected_article = []
    for _,x in tqdm(filtered_article):
        corrected_article.append(correct_token(x))
    corrected_article = list(zip(filtered_indexes, corrected_article))
    n_grams = list(generate_n_gram(corrected_article, n))
    n_gram_dict =defaultdict(list)
    for n_gram in n_grams:
        n_gram_dict[to_ngram(n_gram)].append(n_gram[0][0])
    return tokenized_article, n_gram_dict

def compute_plagiarism_from_articles(art_1: Article, art_2: Article, show = True, gap_tolerance = 5, padding = 20) -> Tuple[Callable[[int],str],float]:

    # First treat the articles 
    treated_1, grams_1 = deepcopy(art_1)
    treated_2, grams_2 = deepcopy(art_2)

    # Align sequence and glue the sequences 
    matching_sequence = align_sequences(grams_1,grams_2)

    # compute the plagiarism score from both articles  --> Harmonic Mean 
    average_length = 2/(1/len(grams_1) + 1/ len(grams_2))
    score = len(matching_sequence)/average_length * 100
    match_viewer = []
    glued_sequence = []

    if len(matching_sequence) != 0:
        # Create the viewer function 
        glued_sequence = glue_sequence(matching_sequence, gap_tolerance)
        match_viewer = Viewer(glued_sequence, treated_1,treated_2, padding)
    if show:
        print("The two articles have a similarity score of {:.2f}, with {} matching sequences. You can use the viewer to visualize the matching sequences".format(score, len(glued_sequence)))
        print("\n--- --- ---\n")
        print(f"The first article share {len(matching_sequence)/len(grams_1)*100}% of its content with the second")
        print(f"The second article share {len(matching_sequence)/len(grams_2)*100}% of its content with the first")
        print("\n--- --- ---\n")

    return match_viewer, score

def to_rgb(hex:str) -> Tuple[int,int,int]:
    x = hex[1:]
    return tuple(int(x[i:i+2], 16) for i in (0, 2, 4))

def compute_many_to_many_sim(articles: List[Article], names: List[str], plot = True, lat_lon = None) -> None:

    N = len(articles)

    results = pd.DataFrame(columns = names,index = names)
    for i in range(N):
        for j in range(i+1,N):
            _, sim = compute_plagiarism_from_articles(articles[i], articles[j], show = False)
            results[names[i]][names[j]] = sim
    
    print(repr(results))

    interpolate = interp1d([results.min().min(), results.max().max()], [0,1])
    if plot == True:
        if lat_lon is None:
            raise TypeError("lat_lon should be specified if you cant to plot")
    
        colorscales = px.colors.sequential.Viridis
        fig = go.Figure()
        values = list(zip(names,lat_lon))
        for i in range(N):
            for j in range(i+1,N):
                first_name, first_lat_lon = values[i]
                second_name, second_lat_lon = values[j]
                color = plotly.colors.find_intermediate_color((0,0,0),(255,255,255),interpolate(results[first_name][second_name]))
                fig.add_trace(
                    go.Scattergeo(
                        mode = "markers+lines",
                        lon = [first_lat_lon[1],second_lat_lon[1]],
                        lat = [first_lat_lon[0],second_lat_lon[0]],
                        line_color = 'rgb'+str(color),
                        name = first_name + " - "+second_name +" sim : "  + "{:.2f}".format(results[first_name][second_name])
                    )
                )


        fig.update_layout(
            title_text = 'Similarity map between articles of encyclopedias',
            showlegend = True,
            geo = dict(
                resolution = 50,
                showland = True,
                landcolor = 'rgb(204, 204, 204)',
                projection_type = "equirectangular",
                lataxis = dict(
                    range = [35, 58],
                    dtick = 10
                ),
                lonaxis = dict(
                    range = [-15, 20],
                    dtick = 20
                ),
            )
        )   
        fig.update_layout(template = "none", width = 800, height = 600, legend_orientation = "h")
        fig.show()


class Viewer:
    def __init__(self,sequence, treated_1, treated_2, padding):
        self.seq = sequence
        self.t_1 = treated_1
        self.t_2 = treated_2
        self.pad = padding
    def __getitem__(self,key: slice):
        if isinstance(key,int):
            display_match(self.seq[key], self.t_1,self.t_2, self.pad)
        else:
            if key.stop > len(self):
                raise IndexError(f"There is only {len(self)} matching sequences, and you tried to access the {key.stop}th")
            for k in range(key.start, key.stop):
                print("\n")
                print(f"--- Printing matching n°{k} ---")
                display_match(self.seq[k], self.t_1,self.t_2, self.pad)
    def __len__(self):
        return len(self.seq)
