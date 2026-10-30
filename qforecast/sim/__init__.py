"""Paper trading with the trained model: historical backtest + live simulation."""

from qforecast.sim.broker import PaperBroker, SimConfig, Trade

__all__ = ["PaperBroker", "SimConfig", "Trade"]
