import math
import json
import numpy as np
import jsonpickle
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        A super-basic algorithm that:
         1) Maintains a small rolling midprice for each traded symbol
         2) Buys if best ask < rolling average, sells if best bid > rolling average
         3) Uses a small (hard-coded) trade size and respects position limits (50 for both products)
        """

        # state.traderData = Trading Data We Keep Track Of For Ourselves
        # state.timestamp = 
        # state.listings = 
        # state.order_depths = all the buy and sell orders per product that other market participants have sent and that the algorithm is able to trade with
        #                       - dict where the keys are the products and the corresponding values are instances of the OrderDepth class
        #                       - OrderDepth class then contains all the buy and sell orders
        # state.own_trades = the trades the algorithm itself has done since the last TradingState came in
        #                       - dictionary of Trade objects with key being a product name
        # state.market_trades =  the trades that other market participants have done since the last TradingState came in
        #                       - dictionary of Trade objects with key being a product name
        # state.position = the long or short position that the player holds in every tradable product
        #                       - property is a dictionary with the product as the key for which the value is a signed integer denoting the position
        # state.observations = 


        # Overview of Trade Objects
            # symbol - 
            # price - 
            # quantity - 
            # buyer - Only Non-Empty If The Trade Was Made By The Algorithm Itself
            # seller - Only Non-Empty If The Trade Was Made By The Algorithm Itself
            # timestamp - 

        # Overview of OrderDepth Objects
            # buy_orders - Dict - keys indicate the price associated with the order, and the corresponding keys indicate the total volume on that price level.
            # sell_orders - Dict - keys indicate the price associated with the order, and the corresponding keys indicate the total volume on that price level.
            
            # For example, if the buy_orders property would look like this for a certain product {9: 5, 10: 4}
            # That would mean that there is a total buy order quantity of 5 at the price level of 9, and a total buy order quantity of 4 at a price level of 10.
            # Players should note that in the sell_orders property, the quantities specified will be negative. E.g., {12: -3, 11: -2}.

        # Overview of Observation Objects
            # - You need to obtain either long or short position earlier.
            # - Conversion request cannot exceed possessed items count.
            # - In case you have 10 items short (-10) you can only request from 1 to 10. Request for 11 or more will be fully ignored.
            # - While conversion happens you will need to cover transportation and import/export tariff.
            # - Conversion request is not mandatory. You can send 0 or None as value.

        # How to Send Orders using Order class
            # - Order(symbol, price, quantity)
            # - symbol - The symbol of the product to trade
            # - price - The price at which the order is to be executed
            # - quantity - The quantity of the product to be traded

        # If there are active orders from counterparties for the same product against which the algorithms’ orders can be matched, the algorithms’ order will be (partially) executed right away.
        # If no immediate or partial execution is possible, the remaining order quantity will be visible for the bots in the market, and it might be that one of them sees it as a good trading opportunity and will trade against it.
        # If none of the bots decides to trade against the remaining order quantity, it is cancelled. Note that after cancellation of the algorithm’s orders but before the next Tradingstate comes in, bots might also trade with each other.

        # Attempt to decode stored data from previous runs:
        try:
            if state.traderData != "" and state.traderData != "SAMPLE":
                rolling_data = jsonpickle.decode(state.traderData)
            else:
                # For first run or empty stored data, init a dictionary with empty lists
                rolling_data = {'order_book_bids': {}, 'order_book_asks': {}, 'past_theos': {}, 'market_trades_data': {}}
        except:
            rolling_data = {'order_book_bids': {}, 'order_book_asks': {}, 'past_theos': {}, 'market_trades_data': {}}


        result = {stock: [] for stock in state.listings}

        # Getting Information From Previous Period
        past_order_book_bids = rolling_data['order_book_bids']
        past_order_book_asks = rolling_data['order_book_asks']
        past_theos = rolling_data['past_theos']
        past_market_trades_data = rolling_data['market_trades_data']

        rolling_theos = past_theos

        # Make sure each symbol has a list:
        for symbol in state.order_depths.keys():
            if symbol not in rolling_data["past_theos"]:
                rolling_data["past_theos"][symbol] = []


        # Building Current Order Book
        order_book_bids = {}
        order_book_asks = {}
        for symbol, order_depth in state.order_depths.items():

            order_book_bids[symbol] = order_depth.buy_orders.keys()
            order_book_asks[symbol] = order_depth.sell_orders.keys()
        
            if len(order_book_bids[symbol]) > 0 and len(order_book_asks[symbol]) > 0:
                best_bid = max(order_book_bids[symbol])
                best_ask = min(order_book_asks[symbol])
                mid_price = (best_bid + best_ask) / 2.0

                rolling_theos[symbol].append(mid_price)
                if len(rolling_theos[symbol]) > 5:
                    rolling_theos[symbol].pop(0)

                rolling_avg = sum(rolling_theos[symbol]) / len(rolling_theos[symbol])
                acceptable_price = rolling_avg

                # Decide small trade sizes if mispriced
                orders = []
                current_position = state.position.get(symbol, 0)
                position_limit = 50  
                max_buyable = position_limit - current_position       # how many we can buy
                max_sellable = position_limit + current_position      # how many we can sell

                # f) If the best ask is cheaper than our fair (rolling) price, buy up to 5
                if best_ask < acceptable_price and max_buyable > 0:
                    buy_qty = min(5, max_buyable, -order_depth.sell_orders[best_ask])
                    if buy_qty > 0:
                        orders.append(Order(symbol, best_ask, buy_qty))

                # g) If the best bid is more expensive than our fair (rolling) price, sell up to 5
                if best_bid > acceptable_price and max_sellable > 0:
                    sell_qty = min(5, max_sellable, order_depth.buy_orders[best_bid])
                    if sell_qty > 0:
                        orders.append(Order(symbol, best_bid, -sell_qty))

                result[symbol] = orders

            else:
                result[symbol] = []

        # Parsing Market Trades
        market_trades_data = {}
        for symbol, trades in state.market_trades.items():

            market_trades_data[symbol] = {}

            total_volume = 0
            total_price = 0
            for trade in trades:
                # symbol, price, quantity, buyer, seller, timestamp
                total_volume += trade.quantity
                total_price += trade.price * trade.quantity

            market_trades_data[symbol]['average_weighted_price'] = total_price / total_volume if total_volume > 0 else np.nan
            market_trades_data[symbol]['total_volume'] = total_volume
            
        # Parsing Own Trades
        own_trades = state.own_trades

        # Parsing Position
        position = state.position 

        # Set Future Trader Data From Current Information
        new_rolling_data = {}
        new_rolling_data['order_book_bids'] = order_book_bids
        new_rolling_data['order_book_asks'] = order_book_asks
        new_rolling_data['past_theos'] = rolling_theos
        new_rolling_data['market_trades_data'] = market_trades_data
        new_trader_data = jsonpickle.encode(new_rolling_data)

        # Request Conversions
        conversions = 0

        return result, conversions, new_trader_data
