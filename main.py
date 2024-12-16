import multiprocessing as mp
import argparse
from streamer import streamer
from detector import detector
from renderer import renderer

def main(video_path: str):
    """
    Main function to orchestrate the video analytics pipeline.

    Args:
        video_path (str): Path to the video file to be processed.
    """
    # Create queues for inter-process communication
    detector_queue = mp.Queue()
    renderer_queue = mp.Queue()

    # Start processes
    streamer_process = mp.Process(target=streamer, args=(video_path, detector_queue))
    detector_process = mp.Process(target=detector, args=(detector_queue, renderer_queue))
    renderer_process = mp.Process(target=renderer, args=(renderer_queue,))

    streamer_process.start()
    detector_process.start()
    renderer_process.start()

    # Wait for processes to finish
    streamer_process.join()
    detector_process.join()
    renderer_process.join()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Video Analytics Pipeline")
    parser.add_argument(
        "-v", "--video", 
        type=str, 
        required=True, 
        help="Path to the video file to be processed."
    )
    args = parser.parse_args()

    # Start the pipeline with the provided video path
    main(args.video)
