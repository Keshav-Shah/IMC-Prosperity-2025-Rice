import math
import json
import numpy as np
import pandas as pd
import jsonpickle
from typing import Dict, List, Tuple, Any, Protocol, Callable, Optional
from abc import ABC, abstractmethod
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
    def __init__(self, **kwargs):
        required_keys = [
            "max_lags", "position_limit", "volatility_product_constant",
            "signal_intercept", "signal_lags", "signal_betas",
            "signal_signal_return_correlation", "forecast_intercept",
            "forecast_lags", "forecast_betas", "forecast_signal_return_correlation",
            "volatility_offset_sample", "offset_quote_size_proportion",
            "order_book_volume_weights", "market_own_volume_weights",
            "minimum_edge_taker"
        ]

        for key in required_keys:
            if key not in kwargs:
                raise KeyError(f"Missing required key in Status: {key}")

        self.__dict__.update(kwargs)


# Base Product Strategy class (ABC = Abstract Base Class)
class ProductStrategy(ABC):
    """Abstract base class for product trading strategies"""
    
    @abstractmethod
    def get_volatility(self, **kwargs) -> float:
        """Calculate volatility for this product"""
        pass
    
    @abstractmethod
    def get_theoretical_price(self, **kwargs) -> float:
        """Calculate theoretical price based on inputs"""
        pass


class StandardProductStrategy(ProductStrategy):
    """Standard implementation of product strategy"""
    
    def __init__(self, status: Status, theo_func: Callable, vol_func: Callable):
        self.status = status
        self.theo_func = theo_func
        self.vol_func = vol_func
    
    def get_volatility(self, **kwargs) -> float:
        kwargs['status_object'] = self.status
        return self.vol_func(**kwargs)
    
    def get_theoretical_price(self, **kwargs) -> float:
        kwargs['status_object'] = self.status
        return self.theo_func(**kwargs)
    

class OptionProductStrategy(ProductStrategy):
    """Standard implementation of product strategy"""
    
    def __init__(self, status: Status, theo_func: Callable, vol_func: Callable):
        self.status = status
        self.theo_func = theo_func
        self.vol_func = vol_func
        self.iv_func = vol_func
    
    def get_volatility(self, **kwargs) -> float:
        kwargs['status_object'] = self.status
        return self.vol_func(**kwargs)
    
    def get_theoretical_price(self, **kwargs) -> float:
        kwargs['status_object'] = self.status
        return self.theo_func(**kwargs)
    
    def get_implied_volatility(self, **kwargs) -> float:
        kwargs['status_object'] = self.status
        return self.iv_func(**kwargs)

class ETFProductStrategy(ProductStrategy):
    """Standard implementation of product strategy"""
    
    def __init__(self, status: Status, theo_func: Callable, vol_func: Callable):
        self.status = status
        self.theo_func = theo_func
        self.vol_func = vol_func

    def get_volatility(self, **kwargs) -> float:
        kwargs['status_object'] = self.status
        return self.vol_func(**kwargs)
    
    def get_theoretical_price(self, **kwargs) -> float:
        kwargs['status_object'] = self.status
        return self.theo_func(**kwargs)



class StrategyRegistry:
    """Registry for product strategies"""
    
    def __init__(self):
        self.strategies = {}
    
    def register(self, symbol: str, strategy: ProductStrategy):
        self.strategies[symbol] = strategy
    
    def get(self, symbol: str) -> Optional[ProductStrategy]:
        return self.strategies.get(symbol)


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

        buy_orders, sell_orders = [], []

        volatility_offsets = status_object.volatility_offset_sample
        
        std_theo = volatility * forecast_theo

        pos_limit = status_object.position_limit

        max_buyable = pos_limit - position
        max_sellable = pos_limit + position

        offset_quote_size_proportion = status_object.offset_quote_size_proportion

        effective_offsets_demanded = []

        for idx_offset, offset in enumerate(volatility_offsets):

            if signal_theo < forecast_theo:
                buy_price = signal_theo - 0.5 * std_theo * offset
            else:
                buy_price = forecast_theo - std_theo * offset
            buy_price_floor = math.floor(buy_price)

            if max_buyable > 0:
                offset_order = (forecast_theo - buy_price_floor) / std_theo
                offset_order_qty = offset_order if offset_order < 1 else offset_order ** 0.5
                buy_qty = min(max_buyable, math.floor(pos_limit * offset_quote_size_proportion * offset_order_qty))
                if idx_offset != 0 and buy_orders and buy_orders[-1].price == buy_price_floor:
                    last_order = buy_orders.pop()
                    buy_orders.append(Order(symbol, buy_price_floor, last_order.quantity + buy_qty))
                    
                else:
                    buy_orders.append(Order(symbol, buy_price_floor, buy_qty))
                    effective_offsets_demanded.append(offset_order)

                max_buyable -= buy_qty


        for idx_offset, offset in enumerate(volatility_offsets):
   
            if signal_theo > forecast_theo:
                sell_price = signal_theo + 0.5 * std_theo * offset
            else:
                sell_price = forecast_theo + std_theo * offset

            sell_price_ceil = math.ceil(sell_price)

            if max_sellable > 0:
                offset_order = (sell_price_ceil - forecast_theo) / std_theo
                offset_order_qty = offset_order if offset_order < 1 else offset_order ** 0.5
                sell_qty = min(max_sellable, math.floor(pos_limit * offset_quote_size_proportion * offset_order_qty))
                if idx_offset != 0 and sell_orders and sell_orders[-1].price == sell_price_ceil:
                    last_order = sell_orders.pop()
                    sell_orders.append(Order(symbol, sell_price_ceil, last_order.quantity-sell_qty))
                else:
                    sell_orders.append(Order(symbol, sell_price_ceil, -sell_qty))
                    effective_offsets_demanded.append(offset_order)

                max_sellable -= sell_qty


        return buy_orders + sell_orders, effective_offsets_demanded

    @staticmethod
    def clear_levels(
        status_object: Status,
        symbol: str,
        order_depth: OrderDepth,
        forecast_theo: float,
        position: int,
    ) -> List[Order]:
        

        orders = []

        pos_limit = status_object.position_limit

        max_buyable = pos_limit - position

        asks_sorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        
        for ask_price, ask_vol in asks_sorted:

            # sell_orders have negative volumes, so flip sign
            volume_available = -ask_vol
            
            # If the ask price exceeds our theoretical value, stop buying
            if ask_price > forecast_theo:
                break

            edge = forecast_theo - ask_price

            if edge < status_object.minimum_edge_taker:
                break

            if volume_available >= status_object.adverse_taking_size:
                break

            # How many we can buy here, respecting our remaining limit
            buy_qty = min(max_buyable, volume_available)
            if buy_qty > 0:
                orders.append(Order(symbol, int(ask_price), buy_qty))
                max_buyable -= buy_qty

            # If we hit our position limit, break out
            if max_buyable <= 0:
                break

        max_sellable = pos_limit + position  # Maximum amount we can sell

        bids_sorted = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)

        for bid_price, bid_vol in bids_sorted:

            volume_available = bid_vol  # Buy orders are positive volumes

            # If the bid price is below our theoretical value, stop selling
            if bid_price < forecast_theo:
                break

            edge = bid_price - forecast_theo

            if edge < status_object.minimum_edge_taker:
                break

            if volume_available >= status_object.adverse_taking_size:
                break

            # How many we can sell here, respecting our remaining limit
            sell_qty = min(max_sellable, volume_available)
            if sell_qty > 0:
                orders.append(Order(symbol, int(bid_price), -sell_qty))
                max_sellable -= sell_qty

            # If we hit our position limit, break out
            if max_sellable <= 0:
                break

        return orders

    @staticmethod
    def signal_taking(
        status_object: Status,
        symbol: str,
        order_depth: OrderDepth,
        forecast_theo: float,
        position: int,
    ) -> List[Order]:
        
        orders = []
        orders += Strategy.clear_levels(status_object, symbol, order_depth, forecast_theo, position)
                
        return orders


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
    ) -> Tuple[List[Order], List[Order], List[float]]:
        
        orders_maker, orders_taker, effective_offsets_demanded = [], [], []

        # Forecasting Taking Strategy
        signal_orders = Strategy.signal_taking(status_object, symbol, order_depth, forecast_theo, position)
        orders_taker += signal_orders

        # Volatility-Based Posting (Fair Price ± 1 Std) 
        quoting_orders, effective_offsets_demanded = Strategy.posting_orders(status_object, symbol, orderbook_theo, signal_theo, forecast_theo, volatility, position)
        orders_maker += quoting_orders

        return orders_maker, orders_taker, effective_offsets_demanded


class ParseState:
    @staticmethod
    def parse_state(state: TradingState) -> Dict:
        try:
            if state.traderData and state.traderData != "SAMPLE":
                rolling_data = jsonpickle.decode(state.traderData)
            else:
                rolling_data = {
                    'orderbook_theos': {}, 'signal_theos': {}, 'forecast_theos': {}, 'market_trades_data': {}, 'own_trades_data': {},
                    'returns': {}, 'residuals': {},
                    'maker_orders': {}, 'taker_orders': {},
                    'expected_return': {},
                }
        except:
            rolling_data = {
                'orderbook_theos': {}, 'signal_theos': {}, 'forecast_theos': {}, 'market_trades_data': {}, 'own_trades_data': {},
                'returns': {}, 'residuals': {},
                'maker_orders': {}, 'taker_orders': {},
                'expected_return': {}, 
            }
    
        # Define expected keys in rolling_data
        expected_keys = ['orderbook_theos', 'signal_theos', 'forecast_theos',
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
    def orderbook_theo_function_cleaned_volume(**kwargs):

        order_depth = kwargs['order_depth']  
        status_object = kwargs['status_object']
        
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return np.nan

        sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0], reverse=False)

        total_bid_volume, total_ask_volume, weighted_bid_price, weighted_ask_price = 0, 0, 0, 0

        for bid_tup in sorted_bids:
            bid_price, bid_volume = bid_tup
            if bid_volume > status_object.minimum_quote_volume and np.isfinite(bid_price) and np.isfinite(bid_volume):
                total_bid_volume += bid_volume
                weighted_bid_price += bid_price * bid_volume

        for ask_tup in sorted_asks:
            ask_price, ask_volume = ask_tup
            ask_volume *= -1
            if ask_volume > status_object.minimum_quote_volume and np.isfinite(ask_price) and np.isfinite(ask_volume):
                total_ask_volume += ask_volume
                weighted_ask_price += ask_price * ask_volume

        weighted_bid_price = weighted_bid_price / total_bid_volume if total_bid_volume > 0 else np.nan
        weighted_ask_price = weighted_ask_price / total_ask_volume if total_ask_volume > 0 else np.nan

        if np.isfinite(weighted_bid_price) and np.isfinite(weighted_ask_price):
            theo = (weighted_bid_price * total_ask_volume + weighted_ask_price * total_bid_volume) / (total_bid_volume + total_ask_volume)
        else:
            theo = np.nan
                
        return theo
    
    @staticmethod
    def orderbook_theo_function_weights(**kwargs):

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

        weighted_bid_price = weighted_bid_price / total_bid_weight if total_bid_weight > 0 else np.nan
        weighted_ask_price = weighted_ask_price / total_ask_weight if total_ask_weight > 0 else np.nan

        if np.isfinite(weighted_bid_price) and np.isfinite(weighted_ask_price):
            theo = (weighted_bid_price * total_ask_volume + weighted_ask_price * total_bid_volume + market_price * market_weighted_volume + own_price * own_weighted_volume) / (total_bid_volume + total_ask_volume + market_weighted_volume + own_weighted_volume)
        else:
            theo = np.nan
                
        return theo
    
    @staticmethod
    def constant_vol(**kwargs):
        return kwargs['status_object'].volatility_product_constant

    @staticmethod
    def linear_regression_vol(**kwargs):
        returns = kwargs['returns']
        status_object = kwargs['status_object']
        return 1

    @staticmethod
    def forecast_returns(
        status_object: Status,
        orderbook_theo: float,
        returns: List[float]
    ) -> Tuple[float, float, float, float]:
        signal_return, forecast_return, signal_theo, forecast_theo = 0.0, 0.0, 0.0, 0.0
        
        signal_intercept = status_object.signal_intercept
        signal_lags = status_object.signal_lags
        signal_betas = status_object.signal_betas
        signal_signal_return_correlation = status_object.signal_signal_return_correlation

        forecast_intercept = status_object.forecast_intercept
        forecast_lags = status_object.forecast_lags
        forecast_betas = status_object.forecast_betas
        forecast_signal_return_correlation = status_object.forecast_signal_return_correlation

        if len(returns) < max(max(signal_lags, default=0), max(forecast_lags, default=0)):
            return 0, 0, orderbook_theo, orderbook_theo

        signal_features = np.zeros(len(signal_betas))
        forecast_features = np.zeros(len(forecast_betas))

        returns_rev = returns[::-1]

        for i, lag in enumerate(signal_lags):
            if i == 0:
                signal_features[i] = np.mean(returns_rev[:lag])
            else:
                signal_features[i] = np.mean(returns_rev[signal_lags[i - 1]:lag])

        for i, lag in enumerate(forecast_lags):
            if i == 0:
                forecast_features[i] = np.mean(returns_rev[:lag])
            else:
                forecast_features[i] = np.mean(returns_rev[forecast_lags[i - 1]:lag])

        signal_return = np.sum(signal_betas * signal_features) + signal_intercept
        forecast_return = np.sum(forecast_betas * forecast_features) + forecast_intercept

        signal_theo = orderbook_theo * (1 + signal_return)
        forecast_theo = orderbook_theo * (1 + forecast_return)

        return float(signal_return), float(forecast_return), float(signal_theo), float(forecast_theo)


class Trader:
    def __init__(self):
        # Create status objects for each product
        self.strategy_registry = self._initialize_strategies()
    
    def _initialize_strategies(self) -> StrategyRegistry:
        registry = StrategyRegistry()
        
        # Define product parameters
        status_params = {
            "RAINFOREST_RESIN": {
                "max_lags": 3,
                "position_limit": 50,
                "volatility_product_constant": 4.427334e-08 ** 0.5,
                "signal_intercept": 0,
                "signal_lags": [],
                "signal_betas": np.array([]),
                "signal_signal_return_correlation": 1,
                "forecast_intercept": 0,
                "forecast_lags": [],
                "forecast_betas": np.array([]),
                "forecast_signal_return_correlation": 1,
                "volatility_offset_sample": [1, 2, 4],
                "offset_quote_size_proportion": 0.5,
                "order_book_volume_weights": np.array([1, 0.85641534, 0.61227611]),
                "market_own_volume_weights": np.array([0.58079727, 0.0450033]),
                "minimum_edge_taker": 0,
                "adverse_taking_size": 0,
                "minimum_quote_volume": 12,
            },
            "KELP": {
                "max_lags": 3,
                "position_limit": 50,
                "volatility_product_constant": 1.489797e-07 ** 0.5,
                "signal_intercept": 0,
                "signal_lags": [1],
                "signal_betas": np.array([-0.21478681523602602]),
                "signal_signal_return_correlation": 0.23813325233172267,
                "forecast_intercept": 0,
                "forecast_lags": [1],
                "forecast_betas": np.array([-0.036193151737277295]),
                "forecast_signal_return_correlation": 0.1007053882011983,
                "volatility_offset_sample": [1, 2, 4],
                "offset_quote_size_proportion": 0.5,
                "order_book_volume_weights": np.array([1, 0.8725848, 0.71203151]),
                "market_own_volume_weights": np.array([0.41013304, 0.14665224]),
                "minimum_edge_taker": 0,
                "adverse_taking_size": 0,
                "minimum_quote_volume": 10,
            }
        }

        # Create status objects and register strategies
        for symbol in ['RAINFOREST_RESIN', 'KELP']:
            status = Status(**status_params[symbol])
            strategy = StandardProductStrategy(
                status=status,
                theo_func=Calculation.orderbook_theo_function_cleaned_volume,
                vol_func=Calculation.constant_vol
            )
            registry.register(symbol, strategy)

        for symbol in ['COCONUT']:
            status = Status(**status_params[symbol])
            strategy = OptionProductStrategy(
                status=status,
                theo_func=Calculation.orderbook_theo_function_cleaned_volume,
                vol_func=Calculation.constant_vol,
                iv_func = Calculation.implied_vol,
            )
            registry.register(symbol, strategy)

        for symbol in ['GIFT_BASKET']:
            status = Status(**status_params[symbol])
            strategy = ETFProductStrategy(
                status=status,
                theo_func=Calculation.orderbook_theo_function_cleaned_volume,
                vol_func=Calculation.constant_vol,
            )
            registry.register(symbol, strategy)

        return registry

    @staticmethod
    def compress_trader_data(data: dict, max_entries: int = 50) -> dict:
        compressed = {}
        for key, value in data.items():
            if key in ['orderbook_theos', 'signal_theos', 'forecast_theos', 'returns', 'residuals', 'expected_return']:
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
        """Main trading method that processes the current state and returns orders"""
        # Parse state to get rolling data
        rolling_data = ParseState.parse_state(state)

        # Initialize result structure
        result = {symbol: [] for symbol in state.listings}

        # Retrieve rolling storage
        orderbook_theos = rolling_data['orderbook_theos']
        signal_theos = rolling_data['signal_theos']
        forecast_theos = rolling_data['forecast_theos']
        returns = rolling_data['returns']
        residuals = rolling_data['residuals']
        past_expected_return = rolling_data.get('expected_return', {})

        expected_return_next_period = {}

        # Get market and own trade data
        market_trades_data, own_trades_data = Calculation.aggregate_trade_data(state)

        # Process each product
        for symbol, order_depth in state.order_depths.items():
            # Get product strategy
            strategy = self.strategy_registry.get(symbol)
            if not strategy:
                continue

            status = strategy.status

            # Calculate volatility
            kwargs_vol = {
                'returns': returns[symbol],
            }
            vol_theo = strategy.get_volatility(**kwargs_vol)
            
            # Calculate theoretical price
            kwargs_theo = {
                'order_depth': order_depth,
                'status_object': status,
                'market_price': market_trades_data[symbol]['average_weighted_price'],
                'market_volume': market_trades_data[symbol]['total_volume'],
                'own_price': own_trades_data[symbol]['average_weighted_price'],
                'own_volume': own_trades_data[symbol]['total_volume'],
                'vol': vol_theo,
            }
            orderbook_theo = strategy.get_theoretical_price(**kwargs_theo)
            
            # Get current position
            current_position = state.position.get(symbol, 0)
            
            signal_return, forecast_return, signal_theo, forecast_theo = Calculation.forecast_returns(
                status, orderbook_theo, returns[symbol]
            )
            
            # Process returns and update historical data
            if len(orderbook_theos[symbol]) > 0:
                realized_return = (orderbook_theo / orderbook_theos[symbol][-1]) - 1
                realized_return = round(realized_return, 8)
                
                symbol_expected_return = past_expected_return.get(symbol, 0)
                
                returns[symbol].append(realized_return)
                residuals[symbol].append(realized_return - symbol_expected_return)
            else:
                returns[symbol].append(0.0)
                residuals[symbol].append(0.0)
            
            # Store expected return for next period
            expected_return_next_period[symbol] = signal_return
            
            # Update theoretical prices
            orderbook_theos[symbol].append(round(orderbook_theo, 5))
            signal_theos[symbol].append(round(signal_theo, 5))
            forecast_theos[symbol].append(round(forecast_theo, 5))
            
            # Execute trading strategy if we have enough data
            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:

                if symbol == 'KELP':
                    orders_maker, orders_taker, effective_offsets_demanded = Trade.execute_trades(
                        strategy.status, symbol, orderbook_theo, signal_theo, forecast_theo, vol_theo, 
                        order_depth, current_position
                    )
                else:
                    orders_maker, orders_taker, effective_offsets_demanded = [], [], []


                # Combine orders and store in result
                result[symbol] = orders_maker + orders_taker
                
                # Store order information for later reference
                rolling_data['maker_orders'][symbol] = []
                for idx_order, order in enumerate(orders_maker):
                    if idx_order < len(effective_offsets_demanded):
                        offset = effective_offsets_demanded[idx_order]
                    else:
                        offset = 0.0
                    order_info_tup = (order.symbol, order.price, order.quantity, round(offset, 3))
                    rolling_data['maker_orders'][symbol].append(order_info_tup)
                
                rolling_data['taker_orders'][symbol] = orders_taker
            
            # Trim history to conserve memory
            max_lag = strategy.status.max_lags
            orderbook_theos[symbol] = orderbook_theos[symbol][-max_lag:]
            signal_theos[symbol] = signal_theos[symbol][-max_lag:]
            forecast_theos[symbol] = forecast_theos[symbol][-max_lag:]
            returns[symbol] = returns[symbol][-max_lag:]
            residuals[symbol] = residuals[symbol][-max_lag:]
        
        # Serialize trader data
        new_trader_data = json.dumps(
            self.compress_trader_data({
                'market_trades_data': market_trades_data,
                'own_trades_data': own_trades_data, 
                'orderbook_theos': orderbook_theos,
                'signal_theos': signal_theos,
                'forecast_theos': forecast_theos,
                'returns': returns,
                'residuals': residuals,
                'maker_orders': rolling_data['maker_orders'],
                'taker_orders': rolling_data['taker_orders'],
                'expected_return': expected_return_next_period,
            }), separators=(",", ":")
        )

        # No conversions in this implementation
        conversions = 0
        
        # Log the final state and return results
        logger.flush(state, result, conversions, new_trader_data)
        
        return result, conversions, new_trader_data