from pypokerengine.players import BasePokerPlayer
import helper
from deuces import Card
import deuces as d
import numpy as np

suits = ['s', 'h', 'd', 'c']
vals = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

def normalize(narray):
    return [x/sum(x) for x in narray]

class HeuristicPlayer(BasePokerPlayer):
    def __init__(self, def_prob, agg=1):
        """
        Input: Various hyperparameters that govern play
            Aggression: How aggressively the bot bids
            Default Prob: Default probability
            ity of folding, calling, raising or bluffing at
                          the buckets:
                          - RR <= 0.7 (struggling hand)
                          - 0.7 < R <= 0.9 (bad hand)
                          - 0.9 < R <= 1.1 (average hand)
                          - 1.1 < R <= 1.3 (good hand)
                          - 1.3 <= R (excellent hand)
            Example:  [
                [0.6, 0.2, 0.0, 0.2],
                [0.4, 0.4, 0.1, 0.1],
                [0.1, 0.7, 0.2, 0.0],
                [0.0, 0.6, 0.4, 0.0],
                [0.0, 0.3, 0.7, 0.0]
            ]
        """
        self.aggression = agg
        self.default_prob = def_prob
        self.vals = ['s', 'h', 'd', 'c']
        self.suits = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    def mutate(self):
        """
        Mutate and change form!
        """
        self.aggression = self.aggression * (1 + np.random.uniform(-0.25, 0.25))
        self.default_prob = normalize(self.default_prob * (1 + np.random.uniform(-0.25, 0.25, size=(5,4))))

    def win_prob(self, your_hand, river_cards, no_of_other_hands, sim=10000):
        """
        Input Example:
            Your Hand: ['Ah', '7c'] ---> Ace of Hearts and Seven of Clubs
            River Cards: ['Ad', '2d', '6d'] --> Ace of Diamonds, 2 of Diamonds, 6 of Diamonds, and 2 others (always 5)
            No of Others: 2 ---> 2 other players remain
            Simulations: number of Monte Carlo simulations to run, default=10000
        Output: Your win %
            If river_cards are present: Use MC simulations to estimate odds
            If no river_cards are present: Use lookup table to find pre-flop odds and return immediately
        """
        if len(river_cards) == 0: #Pre-flop, compute with lookup table
            if your_hand[0][1] == your_hand[1][1]: #suited
                return helper.preflop([your_hand[0][0] + your_hand[1][0] + 's', your_hand[1][0] + your_hand[0][0] + 's'], no_of_other_hands+1)
            else: #unsuited
                return helper.preflop([your_hand[0][0] + your_hand[1][0] + 'o', your_hand[1][0] + your_hand[0][0] + 'o'], no_of_other_hands+1)
        else:
            possible_cards = [a+b for a in vals for b in suits]
            for card in your_hand:
                possible_cards.remove(card)
            for card in river_cards:
                possible_cards.remove(card)

            wins = 0
            num_cards = 5 - len(river_cards) + 2 * no_of_other_hands
            evaluator = d.Evaluator()

            for i in range(sim): #10,000 MC simulations
                generated_cards = np.random.choice(possible_cards, num_cards, replace=False)
                counter = 0
                my_hand = [d.Card.new(card) for card in your_hand]
                board = [d.Card.new(card) for card in river_cards]
                while len(board) < 5:
                    board.append(d.Card.new(generated_cards[counter]))
                    counter += 1
                hand_strength = evaluator.evaluate(board, my_hand)
                best_strength = 1e10 #some random large number
                no_best_hands = 1 #number of hands that are at best_strength
                for j in range(no_of_other_hands):
                    new_hand = [d.Card.new(generated_cards[counter]), d.Card.new(generated_cards[counter+1])]
                    counter += 2
                    new_strength = evaluator.evaluate(board, new_hand)
                    if new_strength < best_strength:
                        best_strength = new_strength
                        no_best_hands = 1 #reset
                    elif new_strength == best_strength:
                        no_best_hands += 1 #add one more
                if best_strength > hand_strength:
                    wins += 1
                elif best_strength == hand_strength:
                    wins += 1/no_best_hands

            return wins/sim
        
    def declare_action(self, valid_actions, hole_card, round_state):
        your_hand = helper.pp_to_array(hole_card)
        river_cards = helper.pp_to_array(round_state['community_card'])
        player_no = round_state['next_player'] #your position
        
        players_still_in = 0
        for player in round_state['seats']:
            if player['state'] == "participating":
                players_still_in += 1
        
        pot = round_state['pot']['main']['amount']
        if len(round_state['pot']) == 2:
            for sidepot in round_state['pot']['side']:
                pot += sidepot['amount'] #can always assume you are in sidepot, else you have no choices anyways.
            
        min_bet = valid_actions[1]['amount']
        stack = round_state['seats'][player_no]
        min_raise = valid_actions[2]['amount']['min']
        max_raise = valid_actions[2]['amount']['max']
        
        if min_bet == 0: #when we are first to act
            wp = self.win_prob(your_hand, river_cards, players_still_in-1)
            rr = wp * (players_still_in+1)
            prob = self.default_prob[np.argmin(abs(np.array([0.6, 0.8, 1.0, 1.2, 1.4]) - rr))]
            prob[1] = prob[0] + prob[1]
            prob[0] = 0 #we don't ever want to fold when we don't have to!!
            
        else:
            # Use heuristic to decide optimal move.
            wp = self.win_prob(your_hand, river_cards, players_still_in-1)
            pot_odds = min_bet/(pot+min_bet)

            rr = wp/pot_odds # our main heuristic, expected rate of return
            prob = self.default_prob[np.argmin(abs(np.array([0.6, 0.8, 1.0, 1.2, 1.4]) - rr))]
            ## We should adjust this probability with our rr!!!

        move = np.random.choice(['fold', 'call', 'raise', 'bluff'], p=prob)
        if move == "raise":
            raise_amount = pot / 3 * self.aggression
            ## We should adjust this raise with our rr!!!
            raise_amount = int(max(min(raise_amount, max_raise), min_raise))
            if len(valid_actions) == 3: #is possible to raise, i.e. have enough money to raise
                return ("raise", raise_amount)
            else:
                return ("call", min_bet)
        elif move == "bluff":
            raise_amount = pot / 2 * self.aggression
            ## We should adjust this raise with our rr!!!
            raise_amount = int(max(min(raise_amount, max_raise), min_raise))
            if len(valid_actions) == 3: #is possible to raise, i.e. have enough money to raise
                return ("raise", raise_amount)
            else:
                return ("call", min_bet)
        elif move == "call":
            return (move, min_bet)
        else:
            return (move, 0)

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    init_def_prob = [
        [0.6, 0.2, 0.0, 0.2],
        [0.4, 0.4, 0.1, 0.1],
        [0.1, 0.7, 0.2, 0.0],
        [0.0, 0.6, 0.4, 0.0],
        [0.0, 0.3, 0.7, 0.0]
    ]
    return HeuristicPlayer(init_def_prob)



