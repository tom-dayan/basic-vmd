import multiprocessing as mp
import argparse
from streamer import streamer
from detector import detector
from renderer import renderer

"""
Orchestrates the video analytics pipeline.

Starts and manages the Streamer, Detector, and Renderer processes.
Accepts command-line arguments to specify video input and optional blurring.
"""

def main(video_path: str, enable_blur: bool):
    """
    Sets up and starts the pipeline processes.

    Args:
        video_path: Path to the video file.
        enable_blur: Enables or disables blurring of detected regions.
    """
    # Create queues for inter-process communication
    detector_queue = mp.Queue()
    renderer_queue = mp.Queue()

    # Create a stop signal using multiprocessing.Event
    stop_signal = mp.Event()

    # Start processes
    streamer_process = mp.Process(target=streamer, args=(video_path, detector_queue, stop_signal))
    detector_process = mp.Process(target=detector, args=(detector_queue, renderer_queue, stop_signal))
    renderer_process = mp.Process(target=renderer, args=(renderer_queue, stop_signal, enable_blur))

    streamer_process.start()
    detector_process.start()
    renderer_process.start()

    # Wait for the renderer to stop with a timeout
    renderer_process.join()

    # Set stop signal for other processes
    stop_signal.set()

    # Wait for other processes to terminate with timeouts
    streamer_process.join(timeout=2)
    detector_process.join(timeout=2)

    # Force terminate if processes are still running
    if streamer_process.is_alive():
        streamer_process.terminate()
    if detector_process.is_alive():
        detector_process.terminate()

    # Clean up queues
    detector_queue.close()
    renderer_queue.close()
    detector_queue.join_thread()
    renderer_queue.join_thread()

if __name__ == "__main__":
    """
    Command-line usage (see README):
        python main.py --video <path_to_video> [--blur]
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Video Analytics Pipeline")
    parser.add_argument(
        "-v", "--video",
        type=str,
        required=True,
        help="Path to the video file to be processed."
    )
    parser.add_argument(
        "-b", "--blur",
        action="store_true",
        help="Enable blurring of detected regions."
    )
    args = parser.parse_args()

    # Start the pipeline with the provided arguments
    main(args.video, args.blur)
