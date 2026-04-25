import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory
import logging_mp

logger_mp = logging_mp.get_logger(__name__)


class ImageClient:
    """
    Image client for THREE cameras: HEAD + WRIST1 + WRIST2
    
    Server concatenates: | HEAD (640px) | WRIST1 (640px) | WRIST2 (640px) | = 1920px total
    """
    def __init__(
        self,
        head_img_shape=None,
        head_img_shm_name=None,
        wrist1_img_shape=None,
        wrist1_img_shm_name=None,
        wrist2_img_shape=None,
        wrist2_img_shm_name=None,
        image_show=False,
        server_address="192.168.123.164",
        port=5555,
        Unit_Test=False,
    ):
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port
        self.temp_head = None
        self.temp_wrist1 = None
        self.temp_wrist2 = None

        self.head_img_shape = head_img_shape
        self.wrist1_img_shape = wrist1_img_shape
        self.wrist2_img_shape = wrist2_img_shape

        # ---------- SHARED MEMORY FOR HEAD CAMERA ----------
        self.head_enable_shm = False
        if head_img_shape and head_img_shm_name:
            self.head_image_shm = shared_memory.SharedMemory(name=head_img_shm_name)
            self.head_img_array = np.ndarray(
                head_img_shape, dtype=np.uint8, buffer=self.head_image_shm.buf
            )
            self.head_enable_shm = True

        # ---------- SHARED MEMORY FOR WRIST1 CAMERA ----------
        self.wrist1_enable_shm = False
        if wrist1_img_shape and wrist1_img_shm_name:
            self.wrist1_image_shm = shared_memory.SharedMemory(name=wrist1_img_shm_name)
            self.wrist1_img_array = np.ndarray(
                wrist1_img_shape, dtype=np.uint8, buffer=self.wrist1_image_shm.buf
            )
            self.wrist1_enable_shm = True

        # ---------- SHARED MEMORY FOR WRIST2 CAMERA ----------
        self.wrist2_enable_shm = False
        if wrist2_img_shape and wrist2_img_shm_name:
            self.wrist2_image_shm = shared_memory.SharedMemory(name=wrist2_img_shm_name)
            self.wrist2_img_array = np.ndarray(
                wrist2_img_shape, dtype=np.uint8, buffer=self.wrist2_image_shm.buf
            )
            self.wrist2_enable_shm = True

        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

    # ------------------------------------------------------------
    def resize_with_letterbox(self, img, target_h, target_w):
        """Resize image with letterbox to maintain aspect ratio"""
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        y0 = (target_h - nh) // 2
        x0 = (target_w - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    # ------------------------------------------------------------
    def split_three_cameras(self, img):
        """
        Splits the concatenated image from server into HEAD + WRIST1 + WRIST2.
        
        Server sends: | HEAD (640px) | WRIST1 (640px) | WRIST2 (640px) | = 1920px total
        
        For your setup:
        - HEAD: 640px (single RealSense camera)
        - WRIST1: 640px (USB webcam /dev/video0)
        - WRIST2: 640px (USB webcam /dev/video2)
        - Total width: 1920px
        """
        H, W, _ = img.shape

        # Your server concatenates: head (640px) + wrist1 (640px) + wrist2 (640px) = 1920px
        if W == 1920:  # 640 + 640 + 640
            head = img[:, :640]          # First 640px is HEAD camera
            wrist1 = img[:, 640:1280]    # Middle 640px is WRIST1 camera
            wrist2 = img[:, 1280:]       # Last 640px is WRIST2 camera
            return head, wrist1, wrist2
        
        # Fallback: try to split in thirds if exact size unknown
        elif W % 3 == 0:
            third = W // 3
            return img[:, :third], img[:, third:2*third], img[:, 2*third:]
        
        # Two camera fallback (for compatibility)
        elif W == 1280:
            head = img[:, :640]
            wrist1 = img[:, 640:]
            return head, wrist1, None
        
        # Single camera fallback
        else:
            return img, None, None

    # ------------------------------------------------------------
    def receive_process(self):
        """Main receiving loop"""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://{self._server_address}:{self._port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")

        logger_mp.info("ImageClient started - THREE CAMERA MODE")

        try:
            while self.running:
                message = sock.recv()
                np_img = np.frombuffer(message, np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                # -------- SPLIT THREE CAMERAS --------
                head_img, wrist1_img, wrist2_img = self.split_three_cameras(img)

                wrist1_img = cv2.cvtColor(wrist1_img, cv2.COLOR_BGR2RGB)
                wrist2_img = cv2.cvtColor(wrist2_img, cv2.COLOR_BGR2RGB)
                head_img = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB) 

                #print("Head Image Shape  1 :", head_img.shape)
                #print("Wrist1 Image Shape 1:", wrist1_img.shape)
                #   print("Wrist2 Image Shape 1:", wrist2_img.shape)

                # -------- HEAD CAMERA --------
                if self.head_enable_shm and head_img is not None:
                    head = self.resize_with_letterbox(
                        head_img,
                        self.head_img_shape[0],
                        self.head_img_shape[1],
                    )
                    # Convert BGR to RGB for policy
                    #head = cv2.cvtColor(head, cv2.COLOR_BGR2RGB)
                    self.temp_head = head
                    #print("Head Image Shape  2 :", head_img.shape)
                    #print("Wrist1 Image Shape 2:", wrist1_img.shape)
                    #print("Wrist2 Image Shape 2:", wrist2_img.shape)
                    np.copyto(self.head_img_array, head)

                # -------- WRIST1 CAMERA --------
                if self.wrist1_enable_shm and wrist1_img is not None:
                    wrist1 = self.resize_with_letterbox(
                        wrist1_img,
                        self.wrist1_img_shape[0],
                        self.wrist1_img_shape[1],
                    )
                    # Convert BGR to RGB for policy
                   # wrist1 = cv2.cvtColor(wrist1, cv2.COLOR_BGR2RGB)
                    self.temp_wrist1 = wrist1
                    np.copyto(self.wrist1_img_array, wrist1)

                # -------- WRIST2 CAMERA --------
                if self.wrist2_enable_shm and wrist2_img is not None:
                    wrist2 = self.resize_with_letterbox(
                        wrist2_img,
                        self.wrist2_img_shape[0],
                        self.wrist2_img_shape[1],
                    )
                    # Convert BGR to RGB for policy
                    #wrist2 = cv2.cvtColor(wrist2, cv2.COLOR_BGR2RGB)
                    self.temp_wrist2 = wrist2
                    np.copyto(self.wrist2_img_array, wrist2)

                # -------- DEBUG VIEW --------
                if self._image_show:
                    # Show all three cameras in a grid
                    if head_img is not None and wrist1_img is not None and wrist2_img is not None:
                        # Create horizontal concatenation for display
                        display = cv2.hconcat([
                            cv2.resize(head_img, (320, 240)),
                            cv2.resize(wrist1_img, (320, 240)),
                            cv2.resize(wrist2_img, (320, 240))
                        ])
                        cv2.putText(display, "HEAD", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display, "WRIST1", (330, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display, "WRIST2", (650, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Three Camera View", display)
                    
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False

        except Exception as e:
            logger_mp.error(f"Error in receive_process: {e}")
        finally:
            sock.close()
            ctx.term()
            cv2.destroyAllWindows()
            logger_mp.info("ImageClient closed")

    def _init_performance_metrics(self):
        """Initialize performance metrics for testing"""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_window = deque(maxlen=30)

    def close(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'head_image_shm'):
            try:
                self.head_image_shm.close()
            except:
                pass
        if hasattr(self, 'wrist1_image_shm'):
            try:
                self.wrist1_image_shm.close()
            except:
                pass
        if hasattr(self, 'wrist2_image_shm'):
            try:
                self.wrist2_image_shm.close()
            except:
                pass


# ------------------------------------------------------------------
if __name__ == "__main__":
    """
    Standalone test mode - displays all three camera feeds
    """
    print("="*60)
    print("THREE CAMERA IMAGE CLIENT - TEST MODE")
    print("="*60)
    print("\nStarting client to receive:")
    print("  - Camera 0: HEAD (RealSense)")
    print("  - Camera 1: WRIST1 (USB Camera)")
    print("  - Camera 2: WRIST2 (USB Camera)")
    print("\nPress 'q' to quit\n")
    
    client = ImageClient(
        image_show=True,
        server_address="192.168.123.164",
        port=5555,
    )
    
    try:
        client.receive_process()
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        client.close()
        print("✅ Client closed")