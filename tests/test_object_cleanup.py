import numpy as np

from base.rt_communicator import RTCommunicator
from simworld.communicator.communicator import Communicator


class _FakeUnrealCV:
    def __init__(self, objects):
        self._objects = objects
        self.destroyed = []
        self.garbage_cleaned = False
        self.physics_updates = []
        self.mobility_updates = []

    def get_objects(self):
        return self._objects

    def destroy(self, object_name):
        self.destroyed.append(object_name)

    def clean_garbage(self):
        self.garbage_cleaned = True

    def set_physics(self, object_name, enabled):
        self.physics_updates.append((object_name, enabled))

    def set_movable(self, object_name, enabled):
        self.mobility_updates.append((object_name, enabled))


def test_clear_agents_accepts_bytes_and_preserves_actor_name_case():
    unrealcv = _FakeUnrealCV(
        np.asarray(
            [b'RT_SIGNAL_VEHICLE_0', b'rt_pedestrian_2', b'GEN_BP_Tree_4'],
            dtype='S32',
        )
    )

    RTCommunicator(unrealcv).clear_agents()

    assert unrealcv.destroyed == [
        'RT_SIGNAL_VEHICLE_0',
        'rt_pedestrian_2',
    ]
    assert unrealcv.garbage_cleaned is True


def test_clear_env_accepts_bytes_and_respects_keep_roads():
    unrealcv = _FakeUnrealCV(
        np.asarray(
            [b'GEN_BP_Car_1', b'GEN_Road_2', b'RT_SIGNAL_VEHICLE_0'],
            dtype='S32',
        )
    )

    Communicator(unrealcv).clear_env(keep_roads=True)

    assert unrealcv.destroyed == ['GEN_BP_Car_1']
    assert unrealcv.garbage_cleaned is True


def test_clear_env_removes_generated_roads_when_requested():
    unrealcv = _FakeUnrealCV(
        np.asarray([b'GEN_BP_Car_1', b'GEN_Road_2'], dtype='S32')
    )

    Communicator(unrealcv).clear_env(keep_roads=False)

    assert unrealcv.destroyed == ['GEN_BP_Car_1', 'GEN_Road_2']


def test_freeze_generated_decorative_vehicles_leaves_runtime_traffic_dynamic():
    decorative_a = (
        b'BP_vehicle06_Car_GEN_VARIABLE_BP_vehicle06_Car_C_CAT_123'
    )
    decorative_b = (
        b'BP_vehicle02_Car_GEN_VARIABLE_BP_vehicle02_Car_C_CAT_456'
    )
    unrealcv = _FakeUnrealCV(
        np.asarray(
            [
                decorative_a,
                b'RT_SIGNAL_VEHICLE_0',
                b'GEN_BP_Tree_4',
                decorative_b,
            ],
            dtype='S80',
        )
    )

    frozen = Communicator(unrealcv).freeze_generated_decorative_vehicles()

    assert frozen == [decorative_b.decode(), decorative_a.decode()]
    assert unrealcv.physics_updates == [
        (decorative_b.decode(), False),
        (decorative_a.decode(), False),
    ]
    assert unrealcv.mobility_updates == [
        (decorative_b.decode(), False),
        (decorative_a.decode(), False),
    ]
    assert all(
        actor_name != 'RT_SIGNAL_VEHICLE_0'
        for actor_name, _ in unrealcv.physics_updates
    )
