import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

TFRECORD_PATH = (
    "uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000"
)

def read_one_scenario(path):
    dataset = tf.data.TFRecordDataset(path, compression_type="")

    for raw_record in dataset.take(1):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw_record.numpy())
        return scenario

if __name__ == "__main__":
    scenario = read_one_scenario(TFRECORD_PATH)

    print(scenario)

    # print("=== Scenario summary ===")
    # print(f"Scenario ID: {scenario.scenario_id}")
    # print(f"Number of tracks (agents): {len(scenario.tracks)}")
    # print(f"Number of map features: {len(scenario.map_features)}")
    # print(f"Number of dynamic map states: {len(scenario.dynamic_map_states)}")

    # # Time info
    # print(f"Number of timestamps: {scenario.timestamps_sec.__len__()}")
    # print(f"First timestamp: {scenario.timestamps_sec[0]:.2f}s")
    # print(f"Last timestamp:  {scenario.timestamps_sec[-1]:.2f}s")

    # # Inspect one agent
    # agent = scenario.tracks[0]
    # print("\n=== First agent ===")
    # print(f"Agent ID: {agent.id}")
    # print(f"Object type: {scenario_pb2.Track.ObjectType.Name(agent.object_type)}")
    # print(f"Number of states: {len(agent.states)}")

    # state = agent.states[0]
    # print("\nFirst state:")
    # print(f"  x={state.center_x:.2f}, y={state.center_y:.2f}")
    # print(f"  vx={state.velocity_x:.2f}, vy={state.velocity_y:.2f}")
    # print(f"  heading={state.heading:.2f}")
