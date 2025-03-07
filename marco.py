import math
import json
import numpy as np
import jsonpickle
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order

################################################################################
# STATUS CLASS
################################################################################

class Status:
    """
    Stores or references per-product data like position limits, etc.
    Here, we keep it minimal since the main snippet is in Trader.run().
    """

    position_limit = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50
    }

    @classmethod
    def get_position_limit(cls, product: str) -> int:
        return cls.position_limit.get(product, 50)

################################################################################
# STRATEGY CLASS
################################################################################

class Strategy:
    """
    For demonstration, we define a single function: 'rolling_average_arb'
    that:
      1) Maintains rolling midprices in rolling_theos
      2) Buys if best ask < that rolling average
      3) Sells if best bid > that rolling average
      4) Slices trades into max 5 units
    """

    @staticmethod
    def rolling_average_arb(
        symbol: str,
        order_depth: OrderDepth,
        rolling_theos_for_symbol: List[float],
        position: int
    ) -> List[Order]:
        """
        symbol: "RAINFOREST_RESIN" or "KELP"
        order_depth: the buy & sell orders from TradingState
        rolling_theos_for_symbol: the historical midprices we have so far
        position: current position in this product
        returns: a list of Orders
        """

        orders: List[Order] = []

        # if no bids or asks, do nothing
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return orders

        # best_bid and best_ask
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        # midprice
        mid_price = (best_bid + best_ask) / 2.0

        # append new midprice
        rolling_theos_for_symbol.append(mid_price)
        # keep only last 5
        if len(rolling_theos_for_symbol) > 5:
            rolling_theos_for_symbol.pop(0)

        # average
        rolling_avg = sum(rolling_theos_for_symbol) / len(rolling_theos_for_symbol)

        # position limit is 50 for each product
        pos_limit = Status.get_position_limit(symbol)
        max_buyable = pos_limit - position
        max_sellable = pos_limit + position  # how many we can sell

        # if best_ask < average, buy small chunk
        if best_ask < rolling_avg and max_buyable > 0:
            # we can also limit by the ask volume (which is negative in sell_orders)
            ask_volume = -order_depth.sell_orders[best_ask]  # flip sign
            buy_qty = min(5, max_buyable, ask_volume)
            if buy_qty > 0:
                orders.append(Order(symbol, int(best_ask), buy_qty))

        # if best_bid > average, sell small chunk
        if best_bid > rolling_avg and max_sellable > 0:
            bid_volume = order_depth.buy_orders[best_bid]
            sell_qty = min(5, max_sellable, bid_volume)
            if sell_qty > 0:
                orders.append(Order(symbol, int(best_bid), -sell_qty))

        return orders


################################################################################
# TRADE CLASS
################################################################################

class Trade:
    """
    For each product, we call 'rolling_average_arb' from Strategy.
    This is to demonstrate how you'd orchestrate multiple products.
    """

    @staticmethod
    def rainforest_resin(
        symbol: str,
        order_depth: OrderDepth,
        rolling_theos_for_symbol: List[float],
        position: int
    ) -> List[Order]:
        return Strategy.rolling_average_arb(symbol, order_depth, rolling_theos_for_symbol, position)

    @staticmethod
    def kelp(
        symbol: str,
        order_depth: OrderDepth,
        rolling_theos_for_symbol: List[float],
        position: int
    ) -> List[Order]:
        return Strategy.rolling_average_arb(symbol, order_depth, rolling_theos_for_symbol, position)

################################################################################
# TRADER CLASS
################################################################################

class Trader:
    """
    This class includes the original snippet EXACTLY as requested inside run(...).
    We also embed calls to 'Trade.rainforest_resin' / 'Trade.kelp' inside that snippet,
    to produce final orders.
    """

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        A super-basic algorithm that:
         1) Maintains a small rolling midprice for each traded symbol
         2) Buys if best ask < rolling average, sells if best bid > rolling average
         3) Uses a small (hard-coded) trade size and respects position limits (50 for both products)

        We keep your original snippet lines & comments below, including the big 'try/except.'
        We'll store our rolling midprices in 'rolling_data["past_theos"][symbol]' to match your snippet.
        We'll then call the new Strategy logic to generate orders.
        """

        # Attempt to decode stored data from previous runs:
        try:
            if state.traderData != "" and state.traderData != "SAMPLE":
                rolling_data = jsonpickle.decode(state.traderData)
            else:
                # For first run or empty stored data, init a dictionary with empty lists
                rolling_data = {
                    'order_book_bids': {},
                    'order_book_asks': {},
                    'past_theos': {},
                    'market_trades_data': {}
                }
        except:
            rolling_data = {
                'order_book_bids': {},
                'order_book_asks': {},
                'past_theos': {},
                'market_trades_data': {}
            }

        # We build the 'result' dictionary for orders
        result = {stock: [] for stock in state.listings}

        # Getting Information From Previous Period
        past_order_book_bids = rolling_data['order_book_bids']
        past_order_book_asks = rolling_data['order_book_asks']
        past_theos = rolling_data['past_theos']
        past_market_trades_data = rolling_data['market_trades_data']

        rolling_theos = past_theos  # We'll store midprices in rolling_theos[symbol]

        # Make sure each symbol has a list in rolling_theos
        for symbol in state.order_depths.keys():
            if symbol not in rolling_theos:
                rolling_theos[symbol] = []

        # Building Current Order Book
        order_book_bids = {}
        order_book_asks = {}

        for symbol, order_depth in state.order_depths.items():
            # the user snippet: store keys in order_book_bids/asks
            order_book_bids[symbol] = order_depth.buy_orders.keys()
            order_book_asks[symbol] = order_depth.sell_orders.keys()

            # We'll get the current position from state.position or 0
            current_position = state.position.get(symbol, 0)

            # Use the new 'Trade' logic for RAINFOREST_RESIN or KELP
            if len(order_book_bids[symbol]) > 0 and len(order_book_asks[symbol]) > 0:
                # if the product is RAINFOREST_RESIN, call that method
                if symbol == "RAINFOREST_RESIN":
                    # This calls Strategy.rolling_average_arb internally
                    orders = Trade.rainforest_resin(
                        symbol=symbol,
                        order_depth=order_depth,
                        rolling_theos_for_symbol=rolling_theos[symbol],
                        position=current_position
                    )
                    result[symbol] = orders

                # if the product is KELP, call that method
                elif symbol == "KELP":
                    orders = Trade.kelp(
                        symbol=symbol,
                        order_depth=order_depth,
                        rolling_theos_for_symbol=rolling_theos[symbol],
                        position=current_position
                    )
                    result[symbol] = orders
                else:
                    # If there's some other product, do nothing here
                    result[symbol] = []
            else:
                # If no buy or sell orders, do nothing
                result[symbol] = []

        # Parsing Market Trades
        market_trades_data = {}
        for symbol, trades in state.market_trades.items():
            market_trades_data[symbol] = {}
            total_volume = 0
            total_price = 0
            for trade in trades:
                total_volume += trade.quantity
                total_price += trade.price * trade.quantity
            market_trades_data[symbol]['average_weighted_price'] = (
                total_price / total_volume if total_volume > 0 else np.nan
            )
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
