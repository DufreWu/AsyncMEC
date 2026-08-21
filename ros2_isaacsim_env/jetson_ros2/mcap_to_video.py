import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

BAG_PATH = "asyncmec_20260818_172436_0.mcap"
TOPIC_NAME = "/yolo/annotated"
OUTPUT_VIDEO = "yolo_output.mp4"
FPS = 30.0

def decode_frame(msg, msgtype):
    """Convert sensor_msgs/Image or CompressedImage into a BGR frame."""
    if msgtype.endswith("/CompressedImage"):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("OpenCV could not decode a compressed frame")
        return frame

    encoding = msg.encoding.lower()
    channels_by_encoding = {
        "mono8": 1,
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"Unsupported raw image encoding: {msg.encoding}")

    channels = channels_by_encoding[encoding]
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    pixels = rows[:, : msg.width * channels]
    frame = pixels.reshape(msg.height, msg.width, channels) if channels > 1 else pixels

    conversions = {
        "rgb8": cv2.COLOR_RGB2BGR,
        "bgra8": cv2.COLOR_BGRA2BGR,
        "rgba8": cv2.COLOR_RGBA2BGR,
        "mono8": cv2.COLOR_GRAY2BGR,
    }
    return cv2.cvtColor(frame, conversions[encoding]) if encoding in conversions else frame


def main():
    parser = argparse.ArgumentParser(description="Convert a ROS 2 MCAP image topic to MP4.")
    parser.add_argument("bag", nargs="?", default=BAG_PATH, help="MCAP file path")
    parser.add_argument("--topic", default=TOPIC_NAME, help="Image topic")
    parser.add_argument("--output", default=OUTPUT_VIDEO, help="Output MP4 path")
    parser.add_argument("--fps", type=float, default=FPS, help="Output frame rate")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.is_file():
        print(f"Error: MCAP file not found: {bag_path}", file=sys.stderr)
        return 1

    writer = None
    count = 0
    typestore = get_typestore(Stores.ROS2_JAZZY)

    try:
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections if x.topic == args.topic]
            if not connections:
                topics = "\n".join(f"  {x.topic} ({x.msgtype})" for x in reader.connections)
                print(f"Error: Topic '{args.topic}' not found. Available topics:\n{topics}", file=sys.stderr)
                return 1

            unsupported = [x.msgtype for x in connections if not x.msgtype.endswith(("/Image", "/CompressedImage"))]
            if unsupported:
                print(f"Error: '{args.topic}' is not an image topic: {unsupported[0]}", file=sys.stderr)
                return 1

            print(f"Converting {args.topic} from {bag_path} ...")
            for connection, _timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                frame = decode_frame(msg, connection.msgtype)

                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer: {args.output}")

                writer.write(frame)
                count += 1
                if count % 1000 == 0:
                    print(f"  Processed {count} frames")
    finally:
        if writer is not None:
            writer.release()

    if count == 0:
        print("Error: No frames were found.", file=sys.stderr)
        return 1

    print(f"Done! Saved {count} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
