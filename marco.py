import math
import json
import numpy as np
import pandas as pd
import jsonpickle
from typing import Dict, List, Tuple, Any
from datamodel import Listing, ConversionObservation, Observation, Order, OrderDepth, Trade, TradingState, ProsperityEncoder, Symbol

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        flat_state = {
            "timestamp": state.timestamp,
            "trader_data": trader_data,
            "listings": self.compress_listings(state.listings),
            "order_depths": self.compress_order_depths(state.order_depths),
            "own_trades": self.compress_trades(state.own_trades),
            "market_trades": self.compress_trades(state.market_trades),
            "position": state.position,
            "observations": self.compress_observations(state.observations)
        }

        flat_orders = self.compress_orders(orders)
        flat_logs = self.logs

        flat_structure = {
            "state": flat_state,
            "orders": flat_orders,
            "conversions": conversions,
            "logs": flat_logs
        }

        print(json.dumps(flat_structure, separators=(",", ":")))
        self.logs = ""


    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: ConversionObservation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))


logger = Logger()

################################################################################
# STATUS CLASS
################################################################################

class Status:
    """
    Stores or references per-product data like position limits, etc.
    """

    max_lags = {
        "RAINFOREST_RESIN": 10,
        "KELP": 10
    }

    position_limit = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50
    }

    volatility_product_constant = {
        "RAINFOREST_RESIN": 4.427334e-08 ** 0.5,
        "KELP": 1.489797e-07 ** 0.5
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

    arima_parameters = {
        'RAINFOREST_RESIN': (1, 0, 1),
        'KELP': (0, 0, 1),
    }

    arima_betas = {
        'RAINFOREST_RESIN': (np.array([0.042213]), np.array([-0.79403773])),
        'KELP': (np.array([]), np.array([-0.6]))
    }
    
    arima_signal_return_correlation = {
        'RAINFOREST_RESIN': 0.66059118,
        'KELP': 0.5244263
    }

    max_levels = {
        'RAINFOREST_RESIN': 1e9,
        'KELP': 1e9
    }

    volatility_offset_sample = {
        'RAINFOREST_RESIN': [0.2, 0.5, 1, 2],
        'KELP': [0.2, 0.5, 1, 2],
    }

    offset_quote_size_proportion = {
        'RAINFOREST_RESIN': 0.1,
        'KELP': 0.1,
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
    # (C) New 'volatility_posting' strategy:
    #     1) Compute a 2-level fair price from the top 2 bids/asks
    #     2) Keep rolling log-returns to compute 10-lag std
    #     3) Post orders at fair +/- 1 * std (small trade sizes).
    ########################################################################
    @staticmethod
    def volatility_posting(
        symbol: str,
        theo: float,
        position: int,
        offset_sample: List[float] = [0.2, 0.5, 1, 2],
    ) -> Tuple[List[Order], List[float]]:

        orders: List[Order] = []

        volatility = Status.volatility_product_constant[symbol]
        volatility_offsets = Status.volatility_offset_sample[symbol]
        
        std_theo = volatility * theo

        pos_limit = Status.get_position_limit(symbol)

        max_buyable = pos_limit - position
        max_sellable = pos_limit + position

        offset_quote_size_proportion = Status.offset_quote_size_proportion[symbol]

        last_bid_price, last_ask_price = 1e9, 0

        effective_offsets_demanded = []
        for offset in volatility_offsets:

            buy_price = theo - std_theo * offset
            sell_price = theo + std_theo * offset

            buy_price_floor = math.floor(buy_price)
            sell_price_ceil = math.ceil(sell_price)

            if max_buyable > 0 and buy_price_floor < last_bid_price:
                offset_order = (theo - buy_price_floor) / offset
                offset_order_qty = offset_order if offset_order < 1 else offset_order ** 0.5
                buy_qty = min(max_buyable, math.floor(pos_limit * offset_quote_size_proportion * offset_order_qty))
                orders.append(Order(symbol, buy_price_floor, buy_qty))
                effective_offsets_demanded.append(offset_order)
                max_buyable -= buy_qty

            if max_sellable > 0 and sell_price_ceil > last_ask_price:
                offset_order = (sell_price_ceil - theo) / offset
                offset_order_qty = offset_order if offset_order < 1 else offset_order ** 0.5
                sell_qty = min(max_sellable, math.floor(pos_limit * offset_quote_size_proportion * offset_order_qty))
                orders.append(Order(symbol, sell_price_ceil, -sell_qty))
                effective_offsets_demanded.append(offset_order)
                max_sellable -= sell_qty

            last_bid_price, last_ask_price = buy_price_floor, sell_price_ceil

        return orders, effective_offsets_demanded

    @staticmethod
    def clear_levels(
        symbol: str,
        order_depth: OrderDepth,
        theo: float,
        position: int,
        max_levels: int = 2
    ) -> List[Order]:
        """
        Clear all levels that have expectation relative to theo.
        """
        
        orders = []

        pos_limit = Status.get_position_limit(symbol)

        max_buyable = pos_limit - position

        asks_sorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        levels_bought = 0
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
                levels_bought += 1

            if levels_bought >= max_levels:
                break

            # If we hit our position limit, break out
            if max_buyable <= 0:
                break

        max_sellable = pos_limit + position  # Maximum amount we can sell

        bids_sorted = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
        levels_sold = 0
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
                levels_sold += 1

            if levels_sold >= max_levels:
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
        past_residuals: List[float],
        position: int,
        model = 'HAR',
    ) -> List[Order]:
        """
        1) Given past log-returns, calculate future return with a HAR model.
        3) If we have orders we'd like to trade against, we lift / sell all levels above / below our fair price.

        Hardcoded Coefficients:
            lags, betas, corr_signal_return
        """

        orders = []

        if model == 'HAR':
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

            expected_return = np.sum(betas * features)

            future_theo = theo * (1 + corr_signal_return * expected_return)

        else:
            # ARIMA MODEL
            betas_y, betas_residual = Status.arima_betas[symbol]
            p, d, q = Status.arima_parameters[symbol]
            corr_signal_return = Status.arima_signal_return_correlation[symbol]

            if d > 0:
                past_log_returns = np.diff(past_log_returns, d)
                
            if len(past_log_returns) < max(p, q):
                return orders, 0

            features = np.zeros(len(betas_y))

            past_log_returns_rev = past_log_returns[::-1]

            past_residuals_rev = past_residuals[::-1]

            ar_term = np.dot(betas_y, past_log_returns_rev[:len(betas_y)]) if len(betas_y) > 0 else 0.0
            ma_term = np.dot(betas_residual, past_residuals_rev[:len(betas_residual)]) if len(betas_residual) > 0 else 0.0

            # expected_return = corr_signal_return * (np.exp(ar_term + ma_term) - 1)
            expected_return = np.exp(ar_term + ma_term) - 1

            future_theo = theo * (1 + expected_return)
            
        orders += Strategy.clear_levels(symbol, order_depth, future_theo, position, max_levels = Status.max_levels[symbol])
                
        return orders, expected_return

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
        past_residuals: List[float],
        position: int
    ) -> List[Order]:
        """
        Executes BOTH strategies for RAINFOREST_RESIN and merges orders.
        Comment out either to disable that strategy.
        """

        orders_maker, orders_taker = [], []

        # Forecasting Taking Strategy
        orders_taker, expected_return = Strategy.forecasting_returns(symbol, order_depth, theo, past_log_returns, past_residuals, position, model='ARIMA')

        theo = theo * (1 + expected_return)
        # Volatility-Based Posting (Fair Price ± 1 Std)
        orders_maker, effective_offsets_demanded = Strategy.volatility_posting(symbol, theo, position)

        return orders_maker, orders_taker, expected_return, effective_offsets_demanded

    @staticmethod
    def kelp(
        symbol: str,
        theo: float,
        order_depth: OrderDepth,
        past_theos: List[float],
        past_log_returns: List[float],
        past_residuals: List[float],
        position: int
    ) -> List[Order]:
        """
        Executes BOTH strategies for RAINFOREST_RESIN and merges orders.
        Comment out either to disable that strategy.
        """

        orders_maker, orders_taker = [], []

        # Forecasting Taking Strategy
        orders_taker, expected_return = Strategy.forecasting_returns(symbol, order_depth, theo, past_log_returns, past_residuals, position, model='ARIMA')

        theo = theo * (1 + expected_return)

        # Volatility-Based Posting (Fair Price ± 1 Std)
        orders_maker, effective_offsets_demanded = Strategy.volatility_posting(symbol, theo, position)


        return orders_maker, orders_taker, expected_return, effective_offsets_demanded



################################################################################
# TRADER CLASS
################################################################################

class Trader:

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

        if not order_depth.buy_orders and not order_depth.sell_orders:
            return np.nan  # No valid order data

        bid_side = []
        ask_side = []

        sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)[:levels]
        sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0], reverse=False)[:levels]

        # Process Bids
        for bid_price, bid_volume in sorted_bids:
            if bid_volume > 0 and np.isfinite(bid_price) and np.isfinite(bid_volume):
                bid_side.append((bid_price, bid_volume ** quantity_power))

        # Process Asks - Fix the Negative Volume Issue
        for ask_price, ask_volume in sorted_asks:
            if abs(ask_volume) > 0 and np.isfinite(ask_price) and np.isfinite(ask_volume):
                ask_side.append((ask_price, abs(ask_volume) ** quantity_power))  # Fix: Use abs(ask_volume)

        # If either side is empty, return NaN
        if not bid_side or not ask_side:
            theo = np.nan
        else:
            
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

            if np.isfinite(weighted_bid_price) and np.isfinite(weighted_ask_price):
                theo = (weighted_bid_price * total_ask_weight + weighted_ask_price * total_bid_weight) / (total_bid_weight + total_ask_weight)
            else:
                theo = np.nan
                
        return theo

    @staticmethod
    def compress_trader_data(data: dict, max_entries: int = 50) -> dict:

        compressed = {}
        for key, value in data.items():
            if key in ['past_theos', 'past_log_returns', 'past_residuals', 'expected_return']:
                # For these, assume value is a dict mapping symbol to a list.
                compressed[key] = {
                    symbol: (vals[-max_entries:] if isinstance(vals, list) else vals)
                    for symbol, vals in value.items()
                }
            elif key == 'past_trades':
                # Convert each trade to a flat list.
                compressed[key] = {}
                for symbol, trades in value.items():
                    # Ensure trades is a list.
                    
                    compressed[key][symbol] = []
                    for trade in trades:
                        compressed[key][symbol].append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])


            elif key in ['market_trades_data']:
                # These are assumed to be already small dictionaries.
                compressed[key] = value
            elif key in ['last_maker_orders', 'last_taker_orders']:
                # Convert each order (or order tuple) to a flat list [symbol, price, quantity].
                compressed[key] = {
                    symbol: [
                        [order.symbol, order.price, order.quantity] if isinstance(order, Order)
                        else (list(order) if isinstance(order, tuple) else order)
                        for order in orders
                    ]
                    for symbol, orders in value.items()
                }
            else:
                compressed[key] = value
        return compressed


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
                    'past_theos': {}, 'market_trades_data': {},
                    'past_log_returns': {}, 'past_residuals': {},
                    'last_maker_orders': {}, 'last_taker_orders': {},
                    'expected_return': {},
                }
        except:
            rolling_data = {
                'past_theos': {}, 'market_trades_data': {},
                'past_log_returns': {}, 'past_residuals': {},
                'last_maker_orders': {}, 'last_taker_orders': {},
                'expected_return': {}, 
            }
    
        # Define expected keys in rolling_data
        expected_keys = ['past_theos', 'past_log_returns', 'past_residuals',
                        'last_maker_orders', 'last_taker_orders',
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
        past_residuals = rolling_data['past_residuals']
        past_expected_return = rolling_data['expected_return']

        realized_return = {}

        expected_return_next_period = {}

        # Build Order Book
        for symbol, order_depth in state.order_depths.items():

            theo = Trader.weighted_midprice(order_depth)

            if len(past_theos[symbol]) > 0:
                past_log_returns[symbol].append(np.log(theo / past_theos[symbol][-1]))
                realized_return[symbol] = np.log(theo / past_theos[symbol][-1])
                past_residuals[symbol].append(realized_return[symbol] - past_expected_return[symbol])
            else:
                past_log_returns[symbol].append(0.0) 
                realized_return[symbol] = 0.0
                past_residuals[symbol].append(0.0)

            past_theos[symbol].append(theo)

            current_position = state.position.get(symbol, 0)

            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                if symbol == "RAINFOREST_RESIN":
                    orders_maker, orders_taker, expected_return, effective_offsets_demanded = Trade.rainforest_resin(
                        symbol, theo, order_depth, past_theos[symbol],
                        past_log_returns[symbol], past_residuals[symbol], current_position
                    )
                    result[symbol] = orders_maker + orders_taker
                elif symbol == "KELP":
                    orders_maker, orders_taker, expected_return, effective_offsets_demanded = Trade.kelp(
                        symbol, theo, order_depth, past_theos[symbol],
                        past_log_returns[symbol], past_residuals[symbol], current_position
                    )
                    result[symbol] = orders_maker + orders_taker

                rolling_data['last_maker_orders'][symbol] = []
                for idx_order, order in enumerate(orders_maker):
                    order_info_tup = (order.symbol, order.price, order.quantity, effective_offsets_demanded[idx_order])
                    rolling_data['last_maker_orders'][symbol].append(order_info_tup)

                expected_return_next_period[symbol] = expected_return

                rolling_data['last_taker_orders'][symbol] = orders_taker
                
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




        # Remove oldest values if memory limit is exceeded
        for symbol in state.order_depths.keys():
            max_lag = Status.max_lags[symbol]

            past_theos[symbol] = past_theos[symbol][-max_lag:]
            past_log_returns[symbol] = past_log_returns[symbol][-max_lag:]
            past_residuals[symbol] = past_residuals[symbol][-max_lag:]

        new_trader_data = json.dumps(
            Trader.compress_trader_data({
                'market_trades_data': market_trades_data,
                'past_theos': past_theos,
                'past_log_returns': past_log_returns,
                'past_residuals': past_residuals,
                'last_maker_orders': rolling_data['last_maker_orders'],
                'last_taker_orders': rolling_data['last_taker_orders'],
                'expected_return': expected_return_next_period,
            }), separators=(",", ":")
        )

        # Request Conversions (default is 0)
        conversions = 0

        logger.flush(state, result, conversions, new_trader_data)

        return result, conversions, new_trader_data
