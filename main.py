

video_path = "/Users/jakeliebow/milli/tests/test_data/chunks/test_chunk_010.mp4"
def main():

    frames = extract_object_boxes_and_tag_objects_yolo(video_path)

    inferenced_frames = process_yolo_boxes_to_get_inferenced_detections(
        frames, video_path=video_path
    )
    root = Node(None, 0)
    print("START")
    start = time.perf_counter()

    build_tree( inferenced_frames)

    elapsed = time.perf_counter() - start

    minutes, seconds = divmod(elapsed, 60)
    print(f"Elapsed: {int(minutes)}m {seconds:.2f}s")