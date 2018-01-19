## Helper functions for conversion between the two libraries: Deuces and PyPoker.
import deuces as d

def pp_to_array(hand):
    return [card[1] + card[0].lower() for card in hand]

def pp_to_deuces(hand):
    return [d.Card.new(card[1] + card[0].lower()) for card in hand]

## Pre-process Preflop Odds
import pandas as pd
preflop_eq = pd.read_csv('preflop_equity.csv')
preflop_eq['Cards'] = [str(x) for x in preflop_eq['Cards']]
preflop_eq.set_index('Cards')

def preflop(cards, n):
    """
    Input:
        cards -- in size order, with either suited or unsuited: "AKo" and "KAo"
        num of players -- 3
    Output: Preflop win probability
    """
    return float(preflop_eq.loc[(preflop_eq['Cards'] == cards[0]) | (preflop_eq['Cards'] == cards[1])][str(n) + ' plyrs'].tolist()[0].split('%')[0])/100

def add(list_of_lists):
    ## Adds a list of lists togther
    result = []
    for i in range(len(list_of_lists[0])):
        result.append(sum([lst[i] for lst in list_of_lists]))
    return result