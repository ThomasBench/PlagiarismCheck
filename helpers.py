import enchant 
from typing import List


eng_dict = enchant.Dict("en")

def to_ngram(index_tup):
    return tuple([tup[1] for tup in index_tup])

def correct_token(token: str) -> str : 
    if eng_dict.check(token):
        return token
    else:
        suggestions = eng_dict.suggest(token)
        if len(suggestions) > 0:
            return suggestions[0]
        return token

def generate_n_gram(text: List[str], n: int) -> List[List[str]]:
    return zip(*[text[i:] for i in range(n)])




def align_sequences(art_grams_1,art_grams_2):
    matching_grams = []
    for gram_1 in art_grams_1:
        for gram_2 in art_grams_2:
            if gram_1 == gram_2:
                art_1_matching_id = art_grams_1[gram_1].pop(0)
                art_2_matching_id = art_grams_2[gram_1].pop(0)
                matching_grams.append((gram_1,art_1_matching_id,art_2_matching_id))
    return matching_grams

def glue_sequence(sequence, gap_tolerance):
    final = []
    temp = [sequence[0][1:]]
    last_seen = sequence[0][1:]
    for _, ind_1, ind_2 in sequence[1:]:
        # print(n_gram)
        if 0<= ind_1 - last_seen[0] < gap_tolerance and 0<= ind_2 - last_seen[1] < gap_tolerance :
            temp.append((ind_1,ind_2))
        else:
            final.append((temp[0], temp[-1]))
            temp.clear()
            temp.append((ind_1,ind_2))
        last_seen = (ind_1,ind_2)
    final.append((temp[0], temp[-1]))
    return final

def retrieve_text(tokenized_article, start_index, end_index):
    return " ".join([x[1] for x in tokenized_article[start_index:end_index+1]])

def display_match(match_indexes,treated_1, treated_2, padding):
    # print(sequence)
    start_1,end_1 = match_indexes[0][0] , match_indexes[1][0] +2
    start_2,end_2 = match_indexes[0][1] , match_indexes[1][1] +2
    text_1 = [retrieve_text(treated_1,start_1-padding, start_1), colored(retrieve_text(treated_1,start_1,end_1),"red"),retrieve_text(treated_1, end_1, end_1 + padding)]
    text_2 = [retrieve_text(treated_2,start_2-padding, start_2), colored(retrieve_text(treated_2,start_2,end_2),"red"),retrieve_text(treated_2, end_2, end_2 + padding)]
    print(*text_1)
    print(*text_2)