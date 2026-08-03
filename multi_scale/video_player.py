import cv2


class VideoPlayer:

    def __init__(
        self,
        video_path,
        reference_speed=1.0
    ):

        self.video_path = video_path
        self.reference_speed = reference_speed

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    def _restart(self):
        self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot reopen {self.video_path}")

    def get_frame(
        self,
        speed,
        control_period
    ):
        """
        Return one frame according to robot speed.
        """

        frames_to_skip = max(
            1,
            int(
                speed /
                self.reference_speed *
                self.video_fps *
                control_period
            )
        )

        frame = None

        for _ in range(frames_to_skip):

            ret, frame = self.cap.read()

            if not ret:
                self._restart()
                ret, frame = self.cap.read()

                if not ret:
                    raise RuntimeError(
                        f"Cannot read {self.video_path}"
                    )

        return frame

    def release(self):

        self.cap.release()