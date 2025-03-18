import math
import json
import numpy as np
import pandas as pd
import jsonpickle
from typing import Dict, List, Tuple, Any
from statistics import NormalDist
from datamodel import Listing, ConversionObservation, Observation, Order, OrderDepth, Trade, TradingState, ProsperityEncoder, Symbol

normalDist = NormalDist(0,1)

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


class Status:

    def __init__(self, max_lags: int,
                 position_limit: int,
                 volatility_product_constant: float,
                 har_intercept: float,
                 har_lags: List[int],
                 har_betas: List[float],
                 har_signal_return_correlation: float,
                 forecast_har_intercept: float,
                 forecast_har_lags: List[int],
                 forecast_har_betas: List[float],
                 forecast_har_signal_return_correlation: float,
                 arima_intercept: float,
                 arima_parameters: Dict[str, Tuple[int, int, int]],
                 arima_betas: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 arima_signal_return_correlation: Dict[str, float],
                 max_levels: Dict[str, int],
                 volatility_offset_sample: Dict[str, List[float]],
                 offset_quote_size_proportion: Dict[str, float],
                 order_book_volume_weights: Dict[str, np.ndarray],
                 market_own_volume_weights: Dict[str, np.ndarray],
                 minimum_edge_taker: float,
                 forecast_model_type: str,
         ):

        self.max_lags = max_lags
        self.position_limit = position_limit
        self.volatility_product_constant = volatility_product_constant
        self.har_intercept = har_intercept
        self.har_lags = har_lags
        self.har_betas = har_betas
        self.har_signal_return_correlation = har_signal_return_correlation
        self.forecast_har_intercept = forecast_har_intercept
        self.forecast_har_lags = forecast_har_lags
        self.forecast_har_betas = forecast_har_betas
        self.forecast_har_signal_return_correlation = forecast_har_signal_return_correlation
        self.arima_intercept = arima_intercept
        self.arima_parameters = arima_parameters
        self.arima_betas = arima_betas
        self.arima_signal_return_correlation = arima_signal_return_correlation
        self.max_levels = max_levels
        self.volatility_offset_sample = volatility_offset_sample
        self.offset_quote_size_proportion = offset_quote_size_proportion
        self.order_book_volume_weights = order_book_volume_weights
        self.market_own_volume_weights = market_own_volume_weights
        self.minimum_edge_taker = minimum_edge_taker
        self.forecast_model_type = forecast_model_type
class Strategy:

    @staticmethod
    def posting_orders(
        status_object: Status,
        symbol: str,
        orderbook_theo: float,
        signal_theo: float,
        forecast_theo: float,
        volatility: float,
        position: int,
    ) -> Tuple[List[Order], List[float]]:

        if np.isnan(orderbook_theo) or np.isnan(signal_theo) or np.isnan(forecast_theo):
            return [], []

        orders: List[Order] = []

        volatility_offsets = status_object.volatility_offset_sample
        
        std_theo = volatility * orderbook_theo

        pos_limit = status_object.position_limit

        max_buyable = pos_limit - position
        max_sellable = pos_limit + position

        offset_quote_size_proportion = status_object.offset_quote_size_proportion

        bid_price, ask_price = 1e9, 0

        effective_offsets_demanded = []
        for offset in volatility_offsets:

            buy_price = theo - std_theo * offset
            sell_price = theo + std_theo * offset

            buy_price_floor = math.floor(buy_price)
            sell_price_ceil = math.ceil(sell_price)

            if max_buyable > 0 and buy_price_floor < bid_price:
                offset_order = (theo - buy_price_floor) / std_theo
                offset_order_qty = offset_order if offset_order < 1 else offset_order ** 0.5
                buy_qty = min(max_buyable, math.floor(pos_limit * offset_quote_size_proportion * offset_order_qty))
                orders.append(Order(symbol, buy_price_floor, buy_qty))
                effective_offsets_demanded.append(offset_order)
                max_buyable -= buy_qty

            if max_sellable > 0 and sell_price_ceil > ask_price:
                offset_order = (sell_price_ceil - theo) / std_theo
                offset_order_qty = offset_order if offset_order < 1 else offset_order ** 0.5
                sell_qty = min(max_sellable, math.floor(pos_limit * offset_quote_size_proportion * offset_order_qty))
                orders.append(Order(symbol, sell_price_ceil, -sell_qty))
                effective_offsets_demanded.append(offset_order)
                max_sellable -= sell_qty

            bid_price, ask_price = buy_price_floor, sell_price_ceil

        return orders, effective_offsets_demanded

    @staticmethod
    def clear_levels(
        status_object: Status,
        symbol: str,
        order_depth: OrderDepth,
        theo: float,
        position: int,
        max_levels: int
    ) -> List[Order]:
        
        orders = []

        pos_limit = status_object.position_limit

        max_buyable = pos_limit - position

        asks_sorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        levels_bought = 0
        for ask_price, ask_vol in asks_sorted:

            # sell_orders have negative volumes, so flip sign
            volume_available = -ask_vol
            
            # If the ask price exceeds our theoretical value, stop buying
            if ask_price > theo:
                break

            edge = theo - ask_price

            if edge < status_object.minimum_edge_taker:
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

            volume_available = bid_vol  # Buy orders are positive volumes

            # If the bid price is below our theoretical value, stop selling
            if bid_price < theo:
                break

            edge = bid_price - theo

            if edge < status_object.minimum_edge_taker:
                break

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

    @staticmethod
    def signal_taking(
        status_object: Status,
        symbol: str,
        order_depth: OrderDepth,
        orderbook_theo: float,
        signal_theo: float,
        forecast_theo: float,
        position: int,
        max_levels: int
    ) -> List[Order]:
        
        orders = []

        # FIX THIS

        # orders += Strategy.clear_levels(status_object, symbol, order_depth, theo, position, max_levels = min(max_levels, status_object.max_levels))
                
        return orders

################################################################################
# TRADE CLASS
################################################################################

class Trade:

    @staticmethod
    def execute_trades(
        status_object: Status,
        symbol: str,
        orderbook_theo: float,
        signal_theo: float,
        forecast_theo: float,
        volatility: float,
        order_depth: OrderDepth,
        position: int
    ) -> List[Order]:
        
        orders_maker, orders_taker, effective_offsets_demanded = [], [], []

        # Forecasting Taking Strategy
        signal_orders = Strategy.signal_taking(status_object, symbol, order_depth, orderbook_theo, signal_theo, forecast_theo, position, 1e9)
        orders_taker += signal_orders

        # Volatility-Based Posting (Fair Price ± 1 Std) 
        quoting_orders, effective_offsets_demanded = Strategy.posting_orders(status_object, symbol, orderbook_theo, signal_theo, forecast_theo, volatility, position)
        orders_maker += quoting_orders

        return orders_maker, orders_taker, effective_offsets_demanded

class ParseState:

    @staticmethod
    def parse_state(state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:

        try:
            if state.traderData and state.traderData != "SAMPLE":
                rolling_data = jsonpickle.decode(state.traderData)
            else:
                rolling_data = {
                    'orderbook_theos': {}, 'signal_theos': {}, 'market_trades_data': {}, 'own_trades_data': {},
                    'returns': {}, 'residuals': {},
                    'maker_orders': {}, 'taker_orders': {},
                    'expected_return': {},
                }
        except:
            rolling_data = {
                'orderbook_theos': {}, 'signal_theos': {}, 'market_trades_data': {}, 'own_trades_data': {},
                'returns': {}, 'residuals': {},
                'maker_orders': {}, 'taker_orders': {},
                'expected_return': {}, 
            }
    
        # Define expected keys in rolling_data
        expected_keys = ['orderbook_theos', 'signal_theos',
                         'returns', 'residuals',
                        'maker_orders', 'taker_orders']

        # Ensure top-level keys exist
        for key in expected_keys:
            rolling_data.setdefault(key, {})

        # Ensure every symbol has an entry for each key
        for symbol in state.listings:
            for key in expected_keys:
                rolling_data[key].setdefault(symbol, [])

        return rolling_data

class Calculation:

    @staticmethod
    def aggregate_trade_data(state: TradingState) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:

        market_trades_data = {symbol: {'average_weighted_price': 0, 'total_volume': 0} for symbol in state.listings}
        for symbol, trades in state.market_trades.items():
            total_volume = 0
            total_price = 0
            for trade in trades:
                total_volume += trade.quantity
                total_price += trade.price * trade.quantity
            market_trades_data[symbol]['average_weighted_price'] = (
                total_price / total_volume if total_volume > 0 else np.nan
            )
            market_trades_data[symbol]['total_volume'] = total_volume

        own_trades_data = {symbol: {'average_weighted_price': 0, 'total_volume': 0} for symbol in state.listings}
        for symbol, trades in state.own_trades.items():
            own_trades_data[symbol] = {}
            total_volume = 0
            total_price = 0
            for trade in trades:
                total_volume += trade.quantity
                total_price += trade.price * trade.quantity
            own_trades_data[symbol]['average_weighted_price'] = (
                total_price / total_volume if total_volume > 0 else np.nan
            )
            own_trades_data[symbol]['total_volume'] = total_volume

        return market_trades_data, own_trades_data
    
    @staticmethod
    def orderbook_theo_function(
        **kwargs
    ):

        order_depth = kwargs['order_depth']
        status_object = kwargs['status_object']
        market_price = kwargs['market_price']
        market_volume = kwargs['market_volume']
        own_price = kwargs['own_price']
        own_volume = kwargs['own_volume']

        if not order_depth.buy_orders and not order_depth.sell_orders:
            return np.nan

        order_book_volume_weights = status_object.order_book_volume_weights
        market_weight, own_weight = status_object.market_own_volume_weights

        market_weighted_volume = market_volume * market_weight
        own_weighted_volume = own_volume * own_weight

        levels = len(order_book_volume_weights)

        sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)[:levels]
        sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0], reverse=False)[:levels]

        total_bid_weight, total_ask_weight, total_bid_volume, total_ask_volume, weighted_bid_price, weighted_ask_price = 0, 0, 0, 0, 0, 0

        for idx_tup, bid_tup in enumerate(sorted_bids):
            bid_price, bid_volume = bid_tup
            if bid_volume > 0 and np.isfinite(bid_price) and np.isfinite(bid_volume):
                total_bid_weight += bid_volume * order_book_volume_weights[idx_tup]
                total_bid_volume += bid_volume
                weighted_bid_price += bid_price * bid_volume * order_book_volume_weights[idx_tup]

        for idx_tup, ask_tup in enumerate(sorted_asks):
            ask_price, ask_volume = ask_tup
            ask_volume *= -1
            if ask_volume > 0 and np.isfinite(ask_price) and np.isfinite(ask_volume):
                total_ask_weight += ask_volume * order_book_volume_weights[idx_tup]
                total_ask_volume += ask_volume
                weighted_ask_price += ask_price * ask_volume * order_book_volume_weights[idx_tup]

        weighted_bid_price /= total_bid_weight
        weighted_ask_price /= total_ask_weight

        if np.isfinite(weighted_bid_price) and np.isfinite(weighted_ask_price):
            theo = (weighted_bid_price * total_ask_volume + weighted_ask_price * total_bid_volume + market_price * market_weighted_volume + own_price * own_weighted_volume) / (total_bid_volume + total_ask_volume + market_weighted_volume + own_weighted_volume)
        else:
            theo = np.nan
                
        return theo
    
    @staticmethod
    def constant_vol(
        **kwargs
    ):
        
        return kwargs['status_object'].volatility_product_constant

    @staticmethod
    def linear_regression_vol(
        **kwargs
    ):

        returns = kwargs['returns']
        status_object = kwargs['status_object']

        return 1


    @staticmethod
    def forecast_returns(
        status_object: Status,
        theo: float,
        returns: Dict[str, List[float]],
        residuals: Dict[str, List[float]],
    ) -> Tuple[float, float, float]:

        expected_return, future_theo, future_trade_theo = 0.0, 0.0, 0.0
        
        if status_object.forecast_model_type == 'HAR':

            intercept = status_object.har_intercept
            lags = status_object.har_lags
            betas = status_object.har_betas
            corr_signal_return = status_object.har_signal_return_correlation

            if len(returns) < max(lags):
                return 0, theo, theo


            features = np.zeros(len(betas))

            returns_rev = returns[::-1]
            for i, lag in enumerate(lags):
                if i == 0:
                    features[i] = np.mean(returns_rev[:lag])
                else:
                    features[i] = np.mean(returns_rev[lags[i-1]:lag])

            expected_return = np.sum(betas * features) + intercept
            future_theo = theo * (1 + expected_return)
            future_trade_theo = theo * (1 + corr_signal_return * expected_return)

        if status_object.forecast_model_type == 'ARIMA':

            intercept = status_object.arima_intercept
            betas_y, betas_residual = status_object.arima_betas
            p, d, q = status_object.arima_parameters
            corr_signal_return = status_object.arima_signal_return_correlation

            if d > 0:
                returns = np.diff(returns, d)
                
            if len(returns) < max(p, q):
                return 0, theo, theo

            features = np.zeros(len(betas_y))

            returns_rev = returns[::-1]
            residuals_rev = residuals[::-1]

            ar_term = np.dot(betas_y, returns_rev[:len(betas_y)]) if len(betas_y) > 0 else 0.0
            ma_term = np.dot(betas_residual, residuals_rev[:len(betas_residual)]) if len(betas_residual) > 0 else 0.0

            expected_return = ar_term + ma_term + intercept
            future_theo = theo * (1 + expected_return)
            future_trade_theo = theo * (1 + corr_signal_return * expected_return)

        return float(expected_return), float(future_theo), float(future_trade_theo)
            
    # @staticmethod
    # def price_option(
    #     underlying_price: float,
    #     strike_price: float,
    #     time_to_expiration: float,
    #     volatility: float,
    # ) -> float:
        
    #     d1 = (math.log(underlying_price / strike_price)
    #             + 0.5 * volatility ** 2 * time_to_expiration
    #         ) / (volatility * math.sqrt(time_to_expiration))

    #     d2 = d1 - volatility * math.sqrt(time_to_expiration)

    #     return (underlying_price * normalDist.cdf(d1)
    #             - strike_price * normalDist.cdf(d2))

    # @staticmethod
    # def backsolve_implied_volatility(
    #     underlying_price: float,
    #     strike_price: float,
    #     time_to_expiration: float,
    #     option_price: float,
    #     max_iter: int = 100,
    #     tol: float = 1e-6
    # ) -> float:
        
    #     if option_price < 0:
    #         return 0
        
    #     vol_lower = 1e-6
    #     vol_upper = 5.0

    #     for _ in range(max_iter):
    #         vol_mid = 0.5 * (vol_lower + vol_upper)
    #         # Compute the price for vol_mid
    #         priced = Calculation.price_option(underlying_price, strike_price, time_to_expiration, vol_mid)

    #         # Compare to the desired price
    #         diff = priced - option_price
    #         if abs(diff) < tol:
    #             return vol_mid

    #         # Adjust bounds
    #         if diff > 0:
    #             vol_upper = vol_mid
    #         else:
    #             vol_lower = vol_mid

    #     return 0.5 * (vol_lower + vol_upper)

        
        
class Trader:

    def __init__(self):

        status_params = {
            "RAINFOREST_RESIN": {
                "max_lags": 3,
                "position_limit": 50,
                "volatility_product_constant": 4.427334e-08 ** 0.5,
                "har_intercept": 0,
                "har_lags": [1],
                "har_betas": np.array([-0.500568135273929]),
                "har_signal_return_correlation": 0.689941382932617,
                "arima_intercept": 0,
                "arima_parameters": (0, 0, 3),
                "arima_betas": (np.array([]), np.array([-0.8729, -0.0108, 0.0683])),
                "arima_signal_return_correlation": 0.686221539,
                "max_levels": 1e9,
                "volatility_offset_sample": [0.2, 0.5, 1, 2],
                "offset_quote_size_proportion": 0.5,
                "order_book_volume_weights": np.array([1, 0.85641534, 0.61227611]),
                "market_own_volume_weights": np.array([0.58079727, 0.0450033]),
                "minimum_edge_taker": 0,
                "forecast_model_type": "ARIMA"
            },
            "KELP": {
                "max_lags": 3,
                "position_limit": 50,
                "volatility_product_constant": 1.489797e-07 ** 0.5,
                "har_intercept": 0,
                "har_lags": [1],
                "har_betas": np.array([-0.2594471579859045]),
                "har_signal_return_correlation": 0.2905178447520445,
                "arima_intercept": 0,
                "arima_parameters": (1, 0, 0),
                "arima_betas": (np.array([-0.2614]), np.array([])),
                "arima_signal_return_correlation": 0.261342687,
                "max_levels": 1e9,
                "volatility_offset_sample": [0.4, 1, 2],
                "offset_quote_size_proportion": 0.4,
                "order_book_volume_weights": np.array([1, 0.8725848, 0.71203151]),
                "market_own_volume_weights": np.array([0.41013304, 0.14665224]),
                "minimum_edge_taker": 0,
                "forecast_model_type": "ARIMA"
            }
        }

        self.status_dict = {key: Status(**params) for key, params in status_params.items()}


        self.theo_map = {
            'RAINFOREST_RESIN': Calculation.orderbook_theo_function,
            'KELP': Calculation.orderbook_theo_function
        }

        self.vol_map = {
            'RAINFOREST_RESIN': Calculation.constant_vol,
            'KELP': Calculation.constant_vol
        }

    @staticmethod
    def compress_trader_data(data: dict, max_entries: int = 50) -> dict:

        compressed = {}
        for key, value in data.items():
            if key in ['orderbook_theos', 'signal_theos', 'returns', 'residuals', 'expected_return']:
                compressed[key] = {
                    symbol: (vals[-max_entries:] if isinstance(vals, list) else vals)
                    for symbol, vals in value.items()
                }
            elif key in ['market_trades_data', 'own_trades_data']:
                compressed[key] = value
            elif key in ['taker_orders']:
                compressed[key] = {
                    symbol: [
                        [order.symbol, order.price, order.quantity] if isinstance(order, Order)
                        else (list(order) if isinstance(order, tuple) else order)
                        for order in orders
                    ]
                    for symbol, orders in value.items()
                }
            elif key in ['maker_orders']:
                compressed[key] = {
                    symbol: [
                        list(order)
                        for order in orders
                    ]
                    for symbol, orders in value.items()
                }
            else:
                compressed[key] = value
        return compressed

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:

        rolling_data = ParseState.parse_state(state)

        result = {symbol: [] for symbol in state.listings}

        # Retrieve rolling storage
        orderbook_theos = rolling_data['orderbook_theos']
        signal_theos = rolling_data['signal_theos']
        returns = rolling_data['returns']
        residuals = rolling_data['residuals']
        past_expected_return = rolling_data['expected_return']

        expected_return_next_period = {}

        market_trades_data, own_trades_data = Calculation.aggregate_trade_data(state)

        # Build Order Book
        for symbol, order_depth in state.order_depths.items():

            kwargs_vol = {
                'status_object': self.status_dict[symbol],
                'returns': returns[symbol],
                # 'ewma_variance': ewma_variance[symbol]
            }

            vol_function = self.vol_map[symbol]

            vol_theo = vol_function(**kwargs_vol)

            kwargs_theo = {
                'status_object': self.status_dict[symbol],
                'order_depth': order_depth,
                'market_trades_data': market_trades_data,
                'market_price': market_trades_data[symbol]['average_weighted_price'],
                'market_volume': market_trades_data[symbol]['total_volume'],
                'own_price': own_trades_data[symbol]['average_weighted_price'],
                'own_volume': own_trades_data[symbol]['total_volume'],
                'vol': vol_theo,
            }

            theo_function = self.theo_map[symbol]

            orderbook_theo = theo_function(**kwargs_theo)

            current_position = state.position.get(symbol, 0)

            if len(orderbook_theos[symbol]) > 0:
                
                realized_return = (orderbook_theo / orderbook_theos[symbol][-1]) - 1
                realized_return = round(realized_return, 8)
                returns[symbol].append(realized_return)
                residuals[symbol].append(realized_return - past_expected_return[symbol])
                
            else:
                returns[symbol].append(0.0) 
                residuals[symbol].append(0.0)

            expected_return, market_making_theo, taking_theo = Calculation.forecast_returns(self.status_dict[symbol], orderbook_theo, returns[symbol], residuals[symbol])

            expected_return_next_period[symbol] = expected_return

            orderbook_theos[symbol].append(round(orderbook_theo, 5))
            signal_theos[symbol].append(round(market_making_theo, 5))

            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:

                orders_maker, orders_taker, effective_offsets_demanded = Trade.execute_trades(
                    self.status_dict[symbol], symbol, market_making_theo, taking_theo, vol_theo, order_depth, 
                    current_position
                )

                result[symbol] = orders_maker + orders_taker
                rolling_data['maker_orders'][symbol] = []

                for idx_order, order in enumerate(orders_maker):
                    order_info_tup = (order.symbol, order.price, order.quantity, round(effective_offsets_demanded[idx_order], 3))
                    rolling_data['maker_orders'][symbol].append(order_info_tup)

                rolling_data['taker_orders'][symbol] = orders_taker

        # Remove oldest values if memory limit is exceeded
        for symbol in state.order_depths.keys():
            max_lag = self.status_dict[symbol].max_lags

            orderbook_theos[symbol] = orderbook_theos[symbol][-max_lag:]
            signal_theos[symbol] = signal_theos[symbol][-max_lag:]
            returns[symbol] = returns[symbol][-max_lag:]
            residuals[symbol] = residuals[symbol][-max_lag:]

        new_trader_data = json.dumps(
            Trader.compress_trader_data({
                'market_trades_data': market_trades_data,
                'own_trades_data': own_trades_data, 
                'orderbook_theos': orderbook_theos,
                'signal_theos': signal_theos,
                'returns': returns,
                'residuals': residuals,
                'maker_orders': rolling_data['maker_orders'],
                'taker_orders': rolling_data['taker_orders'],
                'expected_return': expected_return_next_period,
            }), separators=(",", ":")
        )

        conversions = 0

        logger.flush(state, result, conversions, new_trader_data)

        return result, conversions, new_trader_data
