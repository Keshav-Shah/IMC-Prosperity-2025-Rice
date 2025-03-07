import math
import json
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

        # 1) LOAD / INITIALIZE TRADERDATA
        # We'll keep a JSON-encoded dictionary of rolling midprices in traderData
        # TraderData structure example:
        #    {
        #      "RAINFOREST_RESIN": [10.0, 10.5, 11.2, ...],
        #      "KELP": [99.0, 101.0, 100.0, ...]
        #    }

        # Attempt to decode stored data from previous runs:
        try:
            if state.traderData != "" and state.traderData != "SAMPLE":
                rolling_data = jsonpickle.decode(state.traderData)
            else:
                # For first run or empty stored data, init a dictionary with empty lists
                rolling_data = {}
        except:
            rolling_data = {}

        # Make sure each symbol has a list:
        for symbol in state.order_depths.keys():
            if symbol not in rolling_data:
                rolling_data[symbol] = []

        # 2) BUILD THE RESULT STRUCTURE
        result = {}

        # 3) ITERATE THROUGH EACH PRODUCT/SYMBOL
        for symbol, order_depth in state.order_depths.items():

            # a) Find the best bid and best ask if they exist
            if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                # If we cannot compute a midprice, skip
                result[symbol] = []
                continue

            # TODO: calculate our theoretical using the best bid bet ask and best quantity fraction
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0

            # b) Keep a small rolling window of midprices
            rolling_data[symbol].append(mid_price)
            # Window Size
            window_size = 5
            if len(rolling_data[symbol]) > window_size:
                rolling_data[symbol].pop(0)

            # c) Compute rolling average
            rolling_avg = sum(rolling_data[symbol]) / len(rolling_data[symbol])

            # d) For demonstration, define "acceptable price" as the rolling average
            acceptable_price = rolling_avg

            # e) Decide small trade sizes if mispriced
            orders = []
            current_position = state.position.get(symbol, 0)
            position_limit = 50  # known from tutorial round
            max_buyable = position_limit - current_position       # how many we can buy
            max_sellable = position_limit + current_position      # how many we can sell

            # f) If the best ask is cheaper than our fair (rolling) price, buy up to max position
            max_buy = 50
            if best_ask < acceptable_price and max_buyable > 0:
                buy_qty = min(max_buy, max_buyable, -order_depth.sell_orders[best_ask])
                if buy_qty > 0:
                    orders.append(Order(symbol, best_ask, buy_qty))

            # g) If the best bid is more expensive than our fair (rolling) price, sell up to max sell
            max_sell = 50
            if best_bid > acceptable_price and max_sellable > 0:
                sell_qty = min(max_sell, max_sellable, order_depth.buy_orders[best_bid])
                if sell_qty > 0:
                    orders.append(Order(symbol, best_bid, -sell_qty))

            result[symbol] = orders

        # 4) UPDATE TRADERDATA WITH OUR ROLLING PRICE LISTS
        new_trader_data = jsonpickle.encode(rolling_data)

        # 5) No conversions in this example
        conversions = 0

        return result, conversions, new_trader_data
