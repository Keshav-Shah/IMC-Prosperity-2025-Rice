import math
import json
import numpy as np
import pandas as pd
import jsonpickle
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order

class Logger:
    """
    A lightweight logger that stores logs in an internal buffer, truncates them
    to avoid length overflow, then flushes them to stdout or log at the end of each run.

    What’s implemented:
    1) An internal buffer (self.logs) to gather log statements.
    2) A max_log_length limit to prevent massive log outputs.
    3) A flush() method that prints your final logs, optionally with your trading data.

    What’s NOT implemented compared to competitor's Logger:
    1) Detailed compression of TradingState or orders with custom data structures.
    2) A multi-item JSON structure. (We only have a single JSON dump of relevant data.)
    """

    def __init__(self, max_log_length=3750):
        self.logs = ""
        self.max_log_length = max_log_length

    def print(self, *objects, sep=" ", end="\n"):
        """
        Works like Python's print(), but accumulates logs into self.logs
        rather than sending them directly to stdout.
        """
        message = sep.join(map(str, objects)) + end
        self.logs += message

    def flush(self, state, result, conversions, trader_data):
        """
        Truncates logs if needed, then prints a JSON structure containing:
            - 'timestamp'
            - 'conversions'
            - 'traderData'
            - 'traderLogs' (the truncated logs)
            - 'orders' (not truncated)

        Note: In your own version, you can add or remove fields as needed.
        """
        # Ensure logs do not exceed self.max_log_length
        if len(self.logs) > self.max_log_length:
            truncated_logs = self.logs[: self.max_log_length - 3] + "..."
        else:
            truncated_logs = self.logs

        # Build a simple data structure to print or store
        output_data = {
            "timestamp": state.timestamp,
            "conversions": conversions,
            "traderData": trader_data,
            "traderLogs": truncated_logs,
            "orders": {
                symbol: [
                    (order.symbol, order.price, order.quantity) for order in order_list
                ]
                for symbol, order_list in result.items()
            },
        }

        # Finally print the entire structure as JSON (or you could return it, etc.)
        print(json.dumps(output_data, separators=(",", ":")))

        # Clear logs after flushing
        self.logs = ""


################################################################################
# STATUS CLASS
################################################################################

class Status:
    """
    Stores or references per-product data like position limits, etc.
    """

    position_limit = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50
    }

    har_lags = {
        'RAINFOREST_RESIN': [1, 2, 5],
        'KELP': [1, 2, 5],
    }

    har_betas = {
        'RAINFOREST_RESIN': np.array([-0.73541417, -0.50879543, -0.75997904]),
        'KELP': np.array([-0.58603507, -0.30877435, -0.34055787])
    }
    
    har_signal_return_correlation = {
        'RAINFOREST_RESIN': 0.6094643900796544,
        'KELP': 0.5111646163263679
    }
    
    @classmethod
    def get_position_limit(cls, product: str) -> int:
        return cls.position_limit.get(product, 50)


################################################################################
# STRATEGY CLASS
################################################################################

class Strategy:

    def har_all_features_for_product(series, lags, window_y=1):
        df = pd.DataFrame(series)
        lag_features = np.zeros((df.shape[0], len(lags)))
        for i, lag in enumerate(lags):
            if i == 0:
                lag_features[:, i] = df.rolling(lag).mean().shift(1).values.flatten()
            else:
                prev_lag = lags[i-1]
                lag_features[:, i] = (
                    (df.rolling(lag).sum().shift(1) - df.rolling(prev_lag).sum().shift(1)) / (lag - prev_lag)
                ).values.flatten()
        features_y = np.zeros(df.shape[0])
        for i in range(len(df) - window_y + 1):
            features_y[i] = df.iloc[i:i+window_y].mean().values[0]
        start_idx = max(lags)
        end_idx = len(df) - window_y + 1
        return lag_features[start_idx:end_idx], features_y[start_idx:end_idx]


    ########################################################################
    # (B) EXACT Weighted Midprice Snippet from Notebook (VERBATIM)
    ########################################################################
    @staticmethod
    def weighted_midprice(order_depth: OrderDepth, levels=1, quantity_power=1):
        """
        Computes the weighted mid-price using the order book depth.

        Parameters:
        - order_depth: OrderDepth object containing buy and sell orders.
        - levels: Number of levels to include in the calculation.
        - quantity_power: Power to raise the volume to when weighting prices.

        Returns:
        - Weighted mid-price if valid prices exist, otherwise NaN.
        """

        bid_side = []
        ask_side = []

        sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)[:levels]
        sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0], reverse=False)[:levels]

        for bid_price, bid_volume in sorted_bids:
            bid_volume = bid_volume ** quantity_power
            if np.isfinite(bid_price) and np.isfinite(bid_volume) and bid_volume > 0:
                bid_side.append((bid_price, bid_volume))

        for ask_price, ask_volume in sorted_asks:
            ask_volume = ask_volume ** quantity_power
            if np.isfinite(ask_price) and np.isfinite(ask_volume) and ask_volume > 0:
                ask_side.append((ask_price, ask_volume))

        if not bid_side or not ask_side:
            return np.nan

        total_bid_weight = sum(volume for _, volume in bid_side)
        total_ask_weight = sum(volume for _, volume in ask_side)

        weighted_bid_price = (
            sum(price * volume for price, volume in bid_side) / total_bid_weight
            if total_bid_weight > 0 else np.nan
        )
        weighted_ask_price = (
            sum(price * volume for price, volume in ask_side) / total_ask_weight
            if total_ask_weight > 0 else np.nan
        )

        return ((weighted_bid_price + weighted_ask_price) / 2
                if np.isfinite(weighted_bid_price) and np.isfinite(weighted_ask_price)
                else np.nan)


    ########################################################################
    # (C) New 'volatility_posting' strategy:
    #     1) Compute a 2-level fair price from the top 2 bids/asks
    #     2) Keep rolling log-returns to compute 10-lag std
    #     3) Post orders at fair +/- 1 * std (small trade sizes).
    ########################################################################
    @staticmethod
    def volatility_posting(
        symbol: str,
        order_depth: OrderDepth,
        theo: float,
        past_log_returns: List[float],
        position: int,
        lag_volatility: int = 10
    ) -> List[Order]:
        """
        Given theo, if we have a valid vol, place a buy at fair - std,
        sell at fair + std at size s.t. full-full takes us halfway to position limit.
        """
        orders: List[Order] = []

        if len(past_log_returns) < lag_volatility:
            return orders

        past_log_returns_rev = past_log_returns[::-1]

        past_log_returns_rev = np.array(past_log_returns[::-1])  # Convert to NumPy array
        lagged_vol = np.sqrt(np.mean(past_log_returns_rev[:lag_volatility] ** 2))  # Now it works

        std_theo = lagged_vol * theo

        pos_limit = Status.get_position_limit(symbol)
        max_buyable = pos_limit - position
        max_sellable = pos_limit + position

        if std_theo > 0:
            buy_price = theo - std_theo
            sell_price = theo + std_theo

            if max_buyable > 0:
                buy_qty = max_buyable // 2
                orders.append(Order(symbol, int(buy_price), buy_qty))

            if max_sellable > 0:
                sell_qty = max_sellable // 2
                orders.append(Order(symbol, int(sell_price), -sell_qty))

        return orders

    @staticmethod
    def clear_levels(
        symbol: str,
        order_depth: OrderDepth,
        theo: float,
        position: int,
    ) -> List[Order]:
        """
        Clear all levels that have expectation relative to theo.
        """
        
        orders = []

        pos_limit = Status.get_position_limit(symbol)

        max_buyable = pos_limit - position

        asks_sorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        for ask_price, ask_vol in asks_sorted:

            # sell_orders have negative volumes, so flip sign
            volume_available = -ask_vol
            
            # If the ask price exceeds our theoretical value, stop buying
            if ask_price > theo:
                break

            # How many we can buy here, respecting our remaining limit
            buy_qty = min(max_buyable, volume_available)
            if buy_qty > 0:
                orders.append(Order(symbol, int(ask_price), buy_qty))
                max_buyable -= buy_qty
            break
            # If we hit our position limit, break out
            if max_buyable <= 0:
                break

        max_sellable = pos_limit + position  # Maximum amount we can sell

        bids_sorted = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
        for bid_price, bid_vol in bids_sorted:

            # If the bid price is below our theoretical value, stop selling
            if bid_price < theo:
                break

            volume_available = bid_vol  # Buy orders are positive volumes

            # How many we can sell here, respecting our remaining limit
            sell_qty = min(max_sellable, volume_available)
            if sell_qty > 0:
                orders.append(Order(symbol, int(bid_price), -sell_qty))
                max_sellable -= sell_qty
            break
            # If we hit our position limit, break out
            if max_sellable <= 0:
                break

        return orders

    ########################################################################
    # (C) New 'volatility_posting' strategy:
    #     1) Compute a 2-level fair price from the top 2 bids/asks
    #     2) Keep rolling log-returns to compute 10-lag std
    #     3) Post orders at fair +/- 1 * std (small trade sizes).
    ########################################################################
    @staticmethod
    def forecasting_returns(
        symbol: str,
        order_depth: OrderDepth,
        theo: float,
        past_log_returns: List[float],
        position: int
    ) -> List[Order]:
        """
        1) Given past log-returns, calculate future return with a HAR model.
        3) If we have orders we'd like to trade against, we lift / sell all levels above / below our fair price.

        Hardcoded Coefficients:
            lags, betas, corr_signal_return
        """

        orders = []

        lags = Status.har_lags[symbol]
        betas = Status.har_betas[symbol]
        corr_signal_return = Status.har_signal_return_correlation[symbol]

        if len(past_log_returns) < max(lags):
            return orders, 0

        features = np.zeros(len(betas))

        past_log_returns_rev = past_log_returns[::-1]
        for i, lag in enumerate(lags):
            if i == 0:
                features[i] = np.mean(past_log_returns_rev[:lag])
            else:
                features[i] = np.mean(past_log_returns_rev[lags[i-1]:lag])

        expected_har_return = np.sum(betas * features)

        future_theo = theo * (1 + corr_signal_return * expected_har_return)

        orders += Strategy.clear_levels(symbol, order_depth, future_theo, position)
                
        return orders, expected_har_return

################################################################################
# TRADE CLASS
################################################################################

class Trade:
    """
    Handles execution for each product.
    - Calls multiple strategies internally, combining the results.
    - This allows the Trader class to remain simple and only call Trade.rainforest_resin() and Trade.kelp().
    """

    @staticmethod
    def rainforest_resin(
        symbol: str,
        theo: float,
        order_depth: OrderDepth,
        past_theos: List[float],
        past_log_returns: List[float],
        position: int
    ) -> List[Order]:
        """
        Executes BOTH strategies for RAINFOREST_RESIN and merges orders.
        Comment out either to disable that strategy.
        """

        orders_maker, orders_taker = [], []

        # Strategy 2: Volatility-Based Posting (Fair Price ± 1 Std)
        orders_maker += Strategy.volatility_posting(symbol, order_depth, theo, past_log_returns, position)

        # Strategy 3: Forecasting Taking Strategy
        orders_taker, expected_return = Strategy.forecasting_returns(symbol, order_depth, theo, past_log_returns, position)

        return orders_maker, orders_taker, expected_return

    @staticmethod
    def kelp(
        symbol: str,
        theo: float,
        order_depth: OrderDepth,
        past_theos: List[float],
        past_log_returns: List[float],
        position: int
    ) -> List[Order]:
        """
        Executes BOTH strategies for RAINFOREST_RESIN and merges orders.
        Comment out either to disable that strategy.
        """

        orders_maker, orders_taker = [], []

        # Strategy 2: Volatility-Based Posting (Fair Price ± 1 Std)
        orders_maker = Strategy.volatility_posting(symbol, order_depth, theo, past_log_returns, position)

        # Strategy 3: Forecasting Taking Strategy
        orders_taker, expected_return = Strategy.forecasting_returns(symbol, order_depth, theo, past_log_returns, position)

        return orders_maker, orders_taker, expected_return



################################################################################
# TRADER CLASS
################################################################################

logger = Logger()

class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Calls Trade functions directly, which now handle multiple strategies.
        """

        # Attempt to decode stored data from previous runs:
        try:
            if state.traderData and state.traderData != "SAMPLE":
                rolling_data = jsonpickle.decode(state.traderData)
            else:
                rolling_data = {
                    'order_book_bids': {}, 'order_book_asks': {},
                    'past_theos': {}, 'market_trades_data': {},
                    'past_log_returns': {},'past_trades':{},'past_position':{},
                    'last_maker_buy_orders': {}, 'last_maker_sell_orders': {},
                    'last_taker_buy_orders': {}, 'last_taker_sell_orders': {},
                    'expected_return': {},
                }
        except:
            rolling_data = {
                'order_book_bids': {}, 'order_book_asks': {},
                'past_theos': {}, 'market_trades_data': {},
                'past_log_returns': {},'past_trades':{},'past_position':{},
                'last_maker_buy_orders': {}, 'last_maker_sell_orders': {},
                'last_taker_buy_orders': {}, 'last_taker_sell_orders': {},
                'expected_return': {},
            }

        timestamp = state.timestamp
        logger.print(f"Timestamp: {state.timestamp}")
                
        # Define expected keys in rolling_data
        expected_keys = ['past_theos', 'past_log_returns', 'past_trades', 'past_position',
                        'last_maker_buy_orders', 'last_maker_sell_orders', 'last_taker_buy_orders', 'last_taker_sell_orders',
                        'expected_return']

        # Ensure top-level keys exist
        for key in expected_keys:
            rolling_data.setdefault(key, {})

        # Ensure every symbol has an entry for each key
        for symbol in state.listings:
            for key in expected_keys:
                rolling_data[key].setdefault(symbol, [])

        # We build the 'result' dictionary for orders
        result = {stock: [] for stock in state.listings}

        # Retrieve rolling storage
        past_theos = rolling_data['past_theos']
        past_log_returns = rolling_data['past_log_returns']
        past_trades = rolling_data['past_trades']
        past_position = rolling_data['past_position']
        past_expected_return = rolling_data['expected_return']

        # Building Current Order Book
        order_book_bids = {}
        order_book_asks = {}

        # Track order fills
        fill_pct = {}

        realized_return = {}

        expected_return_next_period = {}

        # Build Order Book
        for symbol, order_depth in state.order_depths.items():

            theo = Strategy.weighted_midprice(order_depth, levels=1, quantity_power=1)

            past_theos[symbol].append(theo)

            if len(past_theos[symbol]) > 0 and past_theos[symbol][-1] > 0:
                past_log_returns[symbol].append(np.log(theo / past_theos[symbol][-1]))
                realized_return[symbol] = np.log(theo / past_theos[symbol][-1])
            else:
                past_log_returns[symbol].append(0.0)  # Append 0 if no previous theo
                realized_return[symbol] = 0.0

            order_book_bids[symbol] = order_depth.buy_orders
            order_book_asks[symbol] = order_depth.sell_orders

            current_position = state.position.get(symbol, 0)

            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                if symbol == "RAINFOREST_RESIN":
                    orders_maker, orders_taker, expected_return = Trade.rainforest_resin(
                        symbol, theo, order_depth, past_theos[symbol],
                        past_log_returns[symbol], current_position
                    )
                    result[symbol] = orders_maker + orders_taker
                elif symbol == "KELP":
                    orders_maker, orders_taker, expected_return = Trade.kelp(
                        symbol, theo, order_depth, past_theos[symbol],
                        past_log_returns[symbol], current_position
                    )
                    result[symbol] = orders_maker + orders_taker
                    
                expected_return_next_period[symbol] = expected_return

                rolling_data['last_maker_buy_orders'][symbol] = [o for o in orders_maker if o.quantity > 0]
                rolling_data['last_maker_sell_orders'][symbol] = [o for o in orders_maker if o.quantity < 0]
                rolling_data['last_taker_buy_orders'][symbol] = [o for o in orders_taker if o.quantity > 0]
                rolling_data['last_taker_sell_orders'][symbol] = [o for o in orders_taker if o.quantity < 0]


        logger.print(f"Expected Return at timestamp {timestamp}: {past_expected_return}")
        logger.print(f"Realized Return at timestamp {timestamp}: {realized_return}")

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
        for symbol in state.listings:
            trades = own_trades.get(symbol, [])
            past_trades[symbol].append(trades)

            # Compute fill percentage
            total_filled = np.sum([abs(t.quantity) for t in trades])
            total_submitted = np.sum([abs(o.quantity) for o in rolling_data['last_maker_buy_orders'][symbol] +
                                    rolling_data['last_maker_sell_orders'][symbol]])

            fill_pct[symbol] = total_filled / total_submitted if total_submitted > 0 else 0

            logger.print(f"[{timestamp}][{symbol}] Fill %: {fill_pct[symbol]:.2%}")

        # Parsing Position
        for symbol in state.listings:
            pos = state.position.get(symbol, 0)
            past_position[symbol].append(pos)

        # Remove oldest values if memory limit is exceeded
        for symbol in state.order_depths.keys():
            max_lag = max(Status.har_lags[symbol]) + 1  # Define the max memory limit

            past_theos[symbol] = past_theos[symbol][-max_lag:]
            past_log_returns[symbol] = past_log_returns[symbol][-max_lag:]
            past_trades[symbol] = past_trades[symbol][-max_lag:]
            past_position[symbol] = past_position[symbol][-max_lag:]


        # Update stored trader data properly
        new_trader_data = jsonpickle.encode({
            'order_book_bids': order_book_bids,
            'order_book_asks': order_book_asks,
            'past_theos': past_theos,
            'market_trades_data': market_trades_data,
            'past_log_returns': past_log_returns,
            'past_trades': past_trades,
            'past_position': past_position,
            'last_maker_buy_orders': rolling_data['last_maker_buy_orders'],
            'last_maker_sell_orders': rolling_data['last_maker_sell_orders'],
            'last_taker_buy_orders': rolling_data['last_taker_buy_orders'],
            'last_taker_sell_orders': rolling_data['last_taker_sell_orders'],
            'expected_return': expected_return_next_period

        })

        # Request Conversions (default is 0)
        conversions = 0

        logger.flush(state, result, conversions, new_trader_data)

        return result, conversions, new_trader_data
