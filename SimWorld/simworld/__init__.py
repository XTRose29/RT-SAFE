"""SimWorld public API. Modified for RT-SAFE: lazy optional imports."""
from importlib import import_module
_EXPORTS = {
 "CityGenerator": "citygen.city.city_generator", "CityFunctionCall": "citygen.function_call.city_function_call",
 "BaseLLM": "llm.base_llm", "AssetsRetrieverPlacer": "assets_rp.AssetsRP", "Config": "config",
 "Logger": "utils.logger", "TrafficController": "traffic.controller.traffic_controller",
 "PedestrianManager": "traffic.manager.pedestrian_manager", "VehicleManager": "traffic.manager.vehicle_manager",
 "IntersectionManager": "traffic.manager.intersection_manager", "Map": "map.map", "Node": "map.map", "Edge": "map.map",
 "Communicator": "communicator.communicator", "UnrealCV": "communicator.unrealcv", "BaseAgent": "agent.base_agent",
}
__all__ = list(_EXPORTS)
def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"simworld.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
