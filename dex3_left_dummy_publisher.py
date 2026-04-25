"""
Publish dummy commands to the left Dex3 hand.
- Uses the same DDS topics and message types as the regular Dex3 controller.
- Holds a fixed target for all seven joints (default zeros). Override targets with CLI args.
"""
import argparse
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

DEX3_NUM_MOTORS = 7
LEFT_TOPIC = "rt/dex3/left/state_dummy"


class RISMode:
    """Encode RIS mode byte for Dex3 motors."""

    def __init__(self, joint_id: int, status: int = 0x01, timeout: int = 0):
        self.joint_id = joint_id & 0x0F
        self.status = status & 0x07
        self.timeout = timeout & 0x01

    def to_uint8(self) -> int:
        mode = self.joint_id
        mode |= (self.status << 4)
        mode |= (self.timeout << 7)
        return mode


class Dex3LeftDummyPublisher:
    def __init__(self, freq_hz: float, targets):
        self.freq_hz = freq_hz
        self.dt = 1.0 / self.freq_hz
        self.targets = targets

        self.publisher = ChannelPublisher(LEFT_TOPIC, HandCmd_)
        self.publisher.Init()

        self.msg = self._build_base_msg()

    def _build_base_msg(self):
        msg = unitree_hg_msg_dds__HandCmd_()
        kp = 1.5
        kd = 0.2
        dq = 0.0
        tau = 0.0

        for joint_id in range(DEX3_NUM_MOTORS):
            msg.motor_cmd[joint_id].mode = RISMode(joint_id).to_uint8()
            msg.motor_cmd[joint_id].kp = kp
            msg.motor_cmd[joint_id].kd = kd
            msg.motor_cmd[joint_id].dq = dq
            msg.motor_cmd[joint_id].tau = tau
            msg.motor_cmd[joint_id].q = 0.0
        return msg

    def spin(self):
        print(
            f"Publishing fixed Dex3-left commands to {LEFT_TOPIC} "
            f"at {self.freq_hz:.1f} Hz, targets={self.targets}"
        )
        while True:
            for joint_id in range(DEX3_NUM_MOTORS):
                self.msg.motor_cmd[joint_id].q = self.targets[joint_id]
            self.publisher.Write(self.msg)
            time.sleep(self.dt)


def main():
    parser = argparse.ArgumentParser(description="Publish dummy commands to Dex3 left hand.")
    parser.add_argument("--freq", type=float, default=50.0, help="Publish rate in Hz (default: 50)")
    parser.add_argument(
        "--pos",
        type=float,
        nargs="*",
        help=(
            "Target joint positions in radians. "
            "Pass 1 value to broadcast to all 7 joints, or 7 values to set each joint. "
            "Default: all zeros."
        ),
    )
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID (default: 0)")
    args = parser.parse_args()

    if not args.pos:
        targets = [0.0] * DEX3_NUM_MOTORS
    elif len(args.pos) == 1:
        targets = [args.pos[0]] * DEX3_NUM_MOTORS
    elif len(args.pos) == DEX3_NUM_MOTORS:
        targets = args.pos
    else:
        raise ValueError(f"--pos expects 1 or {DEX3_NUM_MOTORS} values; got {len(args.pos)}")

    ChannelFactoryInitialize(args.domain_id)
    node = Dex3LeftDummyPublisher(freq_hz=args.freq, targets=targets)
    node.spin()


if __name__ == "__main__":
    main()
