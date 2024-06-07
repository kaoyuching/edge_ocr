import cv2
from typing import Generator


def capture_video(video_id: int, inference_rate: float, display: bool = True) -> Generator:
    """
    inference_rate: number of frames to do inference per second
    """
    vid = cv2.VideoCapture(video_id)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print('FPS: ', fps)

    if inference_rate > fps:
        inference_rate = fps
    k = int(fps / inference_rate) # FPS?
    i = 0
    while True:
        i += 1
        # Capture the video frame by frame
        ret = vid.grab()
        if (i % k) != 0 or not ret:
            continue
        _, frame = vid.retrieve()

        # Display the resulting frame
        if display:
            cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        yield frame

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
