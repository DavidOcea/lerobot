import time
import math
from typing import List, Dict, Any, Tuple, Optional
import datetime
# 导入更新后的 eu_motor_py 绑定
# eu_motor_py 是机器人机上的 CAN/EYou 硬件 SDK，开发/CI 机器上没有。
# 用 try/except 让本模块（及其中的纯逻辑）在无 SDK 的环境也能 import，
# 便于单元测试；真连硬件时才需要 eu_motor_py 非空（见各方法内的运行时调用）。
try:
    import eu_motor_py
except ImportError:  # pragma: no cover - SDK only present on the robot machine
    eu_motor_py = None
from .hardware_interface import HardwareInterface
from .cst_soft_spring import CstSoftSpringController, CompliantDriftTrip
from lerobot.utils.monitor_utils import monitor_performance

# 第二批 PHU/RHU 谐波电机（joint_6/7, node 16/17/26/27）的减速比：脉冲/输出圈。
# 0x6064 返回输出轴编码器计数，完整一圈 = 52953088 = 101 × 2^19（0x6092 feed constant = 524288）。
# 该批电机 TPDO 被动反馈失效，read() 回退到裸 SDO 读 0x6064 时用它换算角度。
# 诊断实测（0420_1.log）：+5° → Δraw=735384 → 735384/5×360=52947648，与厂商标注 52953088 一致。
_PULSES_PER_REV_SECOND_BATCH = 52953088

class EyouMotorHardware(HardwareInterface):
    """
    一个模仿 supre_robot_control::EyouSystemInterface 的 Python 类。
    
    该类采用混合设计模式：
    1. 它作为有状态对象，在内部维护 hw_states_* 和 hw_commands_* 变量。
    2. read() 和 write() 方法同时提供清晰的参数和返回值，以方便控制循环。
    """

    _POSITION_FEEDBACK_SYNC_TYPE = 1
    # With the default 20 ms SYNC schedule, type 3 reports status/error every 60 ms.
    _ERROR_FEEDBACK_SYNC_TYPE = 3
    _SYNC_FEEDBACK_EVENT_TIMER_MS = 0


    def __init__(self):
        """构造函数。初始化内部状态存储。"""
        self.can_manager_: Optional[eu_motor_py.CanNetworkManager] = None
        self.feedback_manager_: Optional[eu_motor_py.MotorFeedbackManager] = None
        self.motor_nodes_: List[eu_motor_py.EuMotorNode] = []
        self.joint_names_: List[str] = []
        
        # --- 恢复内部状态和指令存储 ---
        self.hw_states_positions_: List[float] = []
        self.hw_states_velocities_: List[float] = []
        self.hw_states_torques_: List[float] = []

        self.hw_commands_positions_: List[float] = []
        self.hw_start_enabled_: List[bool] = []
        
        self._config: Dict[str, Any] = {}
        self._can_device_index: Optional[int] = None
        self._sync_period_ms: int = 20
        self._last_log_time = time.monotonic()
        self._max_write_duration_us = 0.0

    def init(self, config: Dict[str, Any]) -> bool:
        """
        初始化硬件接口。
        ``sync_period_ms`` controls the SDK-managed CANopen SYNC period and
        defaults to 20 ms. SYNC starts only after PDO configuration succeeds
        during :meth:`activate`.
        """
        print("Initializing EyouMotorHardware...")
        self._config = config
        
        try:
            # 读取顶层参数
            can_device_index = int(self._config["can_device_index"])
            baud_rate_str = self._config["can_baud_rate"]
            sync_period_ms = int(self._config.get("sync_period_ms", 20))
            if sync_period_ms <= 0:
                raise ValueError("sync_period_ms must be greater than zero")
            self._can_device_index = can_device_index
            self._sync_period_ms = sync_period_ms
            
            baud_rate_map = {
                "1M": eu_motor_py.Baudrate.BPS_1M,
                "500K": eu_motor_py.Baudrate.BPS_500K,
                "250K": eu_motor_py.Baudrate.BPS_250K,
            }
            can_baud_rate = baud_rate_map[baud_rate_str]

            print(f"CAN Device Index: {can_device_index}, Baud Rate: {baud_rate_str}")
            
            # 验证并初始化内部存储
            if "joints" not in self._config or not self._config["joints"]:
                raise KeyError("config must contain a non-empty 'joints' list")
            num_joints = len(self._config["joints"])
            self.hw_states_positions_ = [0.0] * num_joints
            self.hw_states_velocities_ = [0.0] * num_joints
            self.hw_states_torques_ = [0.0] * num_joints

            self.hw_commands_positions_ = [0.0] * num_joints
            self.hw_start_enabled_ = [True] * num_joints # 默认全部启用

            # 初始化CAN总线
            self.can_manager_ = eu_motor_py.CanNetworkManager()
            self.can_manager_.init_device(eu_motor_py.DeviceType.Canable, can_device_index, can_baud_rate)
            print("CAN device initialized successfully.")

            # 初始化电机节点
            self.motor_nodes_ = []
            self.joint_names_ = []
            
            for i, joint_info in enumerate(self._config["joints"]):
                joint_name = joint_info.get("name")
                if not joint_name:
                    raise ValueError(f"Joint at index {i} is missing a 'name'.")

                parameters = joint_info.get("parameters", {})
                if "node_id" not in parameters:
                    raise KeyError(f"'node_id' is missing in 'parameters' for joint '{joint_name}'.")
                node_id = int(parameters["node_id"])
                
                self.joint_names_.append(joint_name)
                
                print(f"Initializing motor for joint '{joint_name}' with Node ID {node_id}")
                motor = eu_motor_py.EuMotorNode(can_device_index, node_id)
                self.motor_nodes_.append(motor)

                # --- 核心修改在这里 ---
                # 从 parameters 字典中读取 start_enabled
                # .get(key, default_value) 使得这个参数是可选的
                # 如果 'start_enabled' 不存在，默认值为 True
                start_enabled = parameters.get("start_enabled", True)
                
                # YAML解析器可能会将 'false' 读为布尔值 False，但为了安全，我们还是处理字符串
                if str(start_enabled).lower() == 'false':
                    self.hw_start_enabled_[i] = False
                    print(f"  -> Joint '{joint_name}' is configured to be DISABLED on start.")
                else:
                    self.hw_start_enabled_[i] = True
                    # 默认启用时可以不打印，保持日志简洁
                    # print(f"  -> Joint '{joint_name}' is configured to be ENABLED on start (default).")


        except (KeyError, ValueError, RuntimeError, TypeError) as e:
            import traceback
            print(f"Error during EyouMotorHardware initialization: {e}")
            traceback.print_exc()
            return False

        print("Initialization successful.")
        return True

    def activate(self) -> bool:
        """
        模仿 on_activate。激活硬件并更新内部状态为初始值。
        
        :return: 如果成功则返回 True。
        """
        print("Activating EyouMotorHardware...")
        
        try:
            # 1. 读取初始状态并更新内部成员变量
            for i, motor in enumerate(self.motor_nodes_):
                pos = motor.get_position()
                vel = motor.get_velocity()
                torque = motor.get_torque()

                self.hw_states_positions_[i] = pos
                self.hw_states_velocities_[i] = vel
                self.hw_states_torques_[i] = float(torque)

                self.hw_commands_positions_[i] = pos # 防止启动时跳动
                print(f"Initial state for {self.joint_names_[i]}: Pos={pos:.2f}, Vel={vel:.2f}")

            # 2. 配置并使能电机 (逻辑与之前版本相同)
            for i, motor in enumerate(self.motor_nodes_):
                joint_name = self.joint_names_[i]
                if self.hw_start_enabled_[i]:
                    print(f"Enabling motor for joint {joint_name}...")
                    if not motor.clear_fault():
                        print(f"Error: Failed to clear fault for {joint_name}")
                        return False
                    if not motor.configure_csp_mode(0, True):
                        print(f"Error: Failed to configure CSP mode for {joint_name}")
                        return False
                    if not motor.start_auto_feedback(
                        0, self._POSITION_FEEDBACK_SYNC_TYPE, self._SYNC_FEEDBACK_EVENT_TIMER_MS
                    ):
                        print(f"Error: Failed to configure position feedback for {joint_name}")
                        return False
                    if not motor.start_error_feedback_tpdo(
                        1, self._ERROR_FEEDBACK_SYNC_TYPE, self._SYNC_FEEDBACK_EVENT_TIMER_MS
                    ):
                        print(f"Error: Failed to configure error feedback for {joint_name}")
                        return False
                else:
                    print(f"Skipping activation for joint {joint_name} as it is disabled.")
                    motor.disable()
                    if not motor.clear_fault():
                        print(f"Error: Failed to clear fault for disabled joint {joint_name}")
                        return False
                    if not motor.start_auto_feedback(
                        0, self._POSITION_FEEDBACK_SYNC_TYPE, self._SYNC_FEEDBACK_EVENT_TIMER_MS
                    ):
                        print(f"Error: Failed to configure position feedback for disabled joint {joint_name}")
                        return False
                    if not motor.start_error_feedback_tpdo(
                        1, self._ERROR_FEEDBACK_SYNC_TYPE, self._SYNC_FEEDBACK_EVENT_TIMER_MS
                    ):
                        print(f"Error: Failed to configure error feedback for disabled joint {joint_name}")
                        return False

            
            self.feedback_manager_ = eu_motor_py.MotorFeedbackManager.get_instance()
            self.feedback_manager_.register_callback()
            print("Global feedback callback registered.")
            if not self.can_manager_.start_sync(self._can_device_index, self._sync_period_ms):
                print("Error: Failed to start CANopen SYNC scheduler.")
                return False
            print(f"CANopen SYNC scheduler started ({self._sync_period_ms} ms).")


        except RuntimeError as e:
            print(f"Error during activation: {e}")
            return False

        print("Activation successful.")
        return True
    def read(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        更新内部状态并返回一份新的状态拷贝。
        
        :return: (new_positions, new_velocities) 元组。
        """
        for i, motor in enumerate(self.motor_nodes_):
            feedback = motor.get_latest_feedback()

            if feedback.last_update_time > datetime.timedelta(0):
                self.hw_states_positions_[i] = feedback.position_deg
                self.hw_states_velocities_[i] = feedback.velocity_dps
                self.hw_states_torques_[i] = float(feedback.torque_milli)/1000.0
            else:
                # TPDO 被动反馈失效（第二批 PHU/RHU 谐波电机 joint_6/7 不上报），
                # 回退到裸 SDO 读 0x6064（harmonic_readDirectory 通用路径）。
                # 不用 get_position()：它内部先吃 TPDO 缓存、超时才走厂商 harmonic_getActualPos，
                # 后者对第二批电机返回过冻结旧值；read_s32(0x6064) 是实测确认新鲜的通路。
                # velocity 不读：下游 supre_robot_hardware_manager.read() 把 velocities 写死为 0。
                raw = motor.read_s32(0x6064, 0)
                self.hw_states_positions_[i] = raw / _PULSES_PER_REV_SECOND_BATCH * 360.0
                # get_torque() 返回 per-mille（0.1% 额定扭矩），须 /1000 对齐 if 分支的
                # torque_milli/1000.0；漏除会把 4~31 毫放大成 4~31 Nm，触发碰撞检测误报。
                self.hw_states_torques_[i] = float(motor.get_torque()) / 1000.0

        # 返回内部状态的拷贝，防止外部代码意外修改
        return list(zip(self.hw_states_positions_, self.hw_states_torques_))
    def busy_wait(self, wait_time_s):
        end_time = time.perf_counter() + wait_time_s
        while time.perf_counter() < end_time:
            pass    
    def write(self, commands_positions: List[float]):
        """
        用传入的指令更新内部指令，然后发送到硬件。
        
        :param commands_positions: 要发送的目标位置列表。
        """
        start_time = time.perf_counter()

        # 1. 使用传入的参数更新内部指令变量
        self.hw_commands_positions_ = commands_positions

        any_motor_enabled = False
        # 2. 从内部指令变量读取数据并发送
        for i, motor in enumerate(self.motor_nodes_):
            if self.hw_start_enabled_[i]:
                result = motor.send_csp_target_position(self.hw_commands_positions_[i],0, False)
                if result != 0:
                    print(f"Error: Failed to send command to joint {self.joint_names_[i]}")
                if i!=0 and i%6==0:
                    self.busy_wait(0.001)
                any_motor_enabled = True

        #if any_motor_enabled:
        #    for i, motor in enumerate(self.motor_nodes_):
        #        if self.hw_start_enabled_[i]:
        #            motor.send_sync()
        #            break
        
        # 3. 性能日志
        end_time = time.perf_counter()
        current_duration_us = (end_time - start_time) * 1_000_000
        
        if current_duration_us > self._max_write_duration_us:
            self._max_write_duration_us = current_duration_us
            
        now = time.monotonic()
        if (now - self._last_log_time) >= 1.0:
            print(f"Max write() duration in last second: {self._max_write_duration_us:.0f} us")
            self._max_write_duration_us = 0.0
            self._last_log_time = now

    def _wait_stationary(
        self,
        motor,
        settle_window_s: float = 0.2,
        threshold_deg: float = 0.5,
        timeout_s: float = 2.0,
    ) -> float | None:
        """等待关节静止（速度≈0），返回静止位置；超时返回 ``None``。

        CST 下编码器仍在跑、位置仍可读（0x6064），所以用"连续两个采样窗内
        Δθ < ``threshold_deg``"判静止，避免单次巧合误判。这是切 CSP 的前提——
        手册要求"使能**静止**状态下"才能切换运行模式，带速度切进去就是冲击。

        返回的是最后一次读到的静止位置（供回锁后 hold 用），不是调用方预先读的
        final_pos：零扭矩后关节还会再挪一点，hold 目标应是"停住的位置"。

        返回 ``None`` = ``timeout_s`` 内未静止（被推/重载塌下/欠阻尼振荡），
        调用方必须 ``disable()`` 掉电抱闸兜底，绝不继续切 CSP。
        """
        deadline = time.perf_counter() + timeout_s
        try:
            prev = motor.get_position()
        except Exception:
            # 连位置都读不到就无从判静止，按"不安全"处理。
            return None
        stable_streak = 0
        while time.perf_counter() < deadline:
            time.sleep(settle_window_s)
            try:
                cur = motor.get_position()
            except Exception:
                continue
            if abs(cur - prev) < threshold_deg:
                stable_streak += 1
            else:
                stable_streak = 0
            prev = cur
            if stable_streak >= 2:
                return cur
        return None

    def _relatch_csp(self, motor) -> bool:
        """软瘫(CST) → CSP 回锁，带"等静止 + 绝不残留非零扭矩"的双重安全兜底。

        真机翻车教训（0420.log）：旧实现**没先清零扭矩**就切 CSP，此时关节还
        挂着 float 循环最后一步的非零 torque（"非静止"），固件按手册"使能静止
        状态下切换运行模式"拒绝切换（0x6061 停 10），非零扭矩残留在 CST →
        开环空转跑飞。关键不在切法，而在**切之前关节必须静止**。

        真机已验证（0420_1/0420_2.log）：先零扭矩再 ``configure_csp_mode``，
        0x6061 稳定回 8，位置流 hold-ok（max drift +0.00°）。本实现在此之上再
        加一层"等静止"：零扭矩只停掉"驱动"，关节可能还在惯性/被推/塌下地运动，
        必须 Δθ 连续两窗 < 阈值才敢切 CSP。

        顺序（每层都失败才到下一层，任何一层失败都以抱闸收尾）：
        1. **先清零扭矩**：仍在 CST 时下发 tau=0，立刻停掉驱动。这是"绝不跑飞"
           的最低保底——后续任何一步失败，关节都不会再空转。
        2. **等静止**：读位置滑窗判 Δθ，超时未静止 → ``disable()`` 抱闸。
        3. **切 CSP**：用 ``configure_csp_mode``（会顺带把 RPDO 重映射到
           0x607A），读 0x6061 确认切到 8；切不到就不发位置（避免写进扭矩通道）。
        4. **预载目标再使能**：先按"停住的位置"下发 0x607A 目标，再确认真使能。
           第二电机(node 16/17)上 configure_csp_mode 内部 enableStateMachine 没走到
           Operation Enabled，裸发控制字 0x06→0x07→0x0F 补使能（0420.log 验证有效）；
           预载在前使能在后，使能瞬间才不会去追 0x607A 里的旧目标值。
        5. 没真使能（statusword bit2=0）或切不回 CSP → ``disable()`` 掉电抱闸。

        返回 True=已回 CSP 且真使能；False=未回 CSP 或未使能，但已 disable 抱闸
        （同样安全）。
        """
        # 1) 立刻停驱动：CST 下零扭矩。这是"不会跑飞"的最低保底。
        try:
            motor.send_cst_target_torque(0, 0, False)
        except Exception:
            pass

        # 2) 等静止。零扭矩只停驱动，关节可能还在动；不静止就不切 CSP。
        settled = self._wait_stationary(motor)
        if settled is None:
            # 超时未静止（被推/塌下/振荡）：掉电抱闸，绝不切 CSP。
            try:
                motor.disable()
            except Exception:
                pass
            return False

        # 3) 尝试切 CSP。用 configure_csp_mode（而非裸写 0x6060=8）：它会把
        #    RPDO 重映射到 0x607A，下面 send_csp_target_position 才走位置通道。
        try:
            motor.configure_csp_mode(0, False)
        except Exception:
            pass

        # 确认切到 CSP 模式（0x6061==8）。切不到就不发位置——避免位置写进仍映射
        # 扭矩通道的 RPDO（0420 空转根因）。
        mode_ok = False
        for _ in range(10):  # 最多 ~0.5s，容纳异步模式切换到位
            time.sleep(0.05)
            try:
                if motor.read_u8(0x6061, 0) == 8:
                    mode_ok = True
                    break
            except Exception:
                pass
        if not mode_ok:
            try:
                motor.disable()
            except Exception:
                pass
            return False

        # 4) 先把目标位置预载为"停住的位置"（0x607A=settled），再使能。这样使能
        #    瞬间驱动器不会去追 0x607A 里的旧目标值（否则可能跳一段）。手册
        #    §4.6.1 也要求切位置模式前把 0x6064 写入 0x607A。
        try:
            motor.send_csp_target_position(settled, 0, False)
        except Exception:
            pass

        # 5) 第二电机(node 16/17)：configure_csp_mode 内部 enableStateMachine 没走到
        #    Operation Enabled（0420_3.log 卡 0x0331），裸控制字有效（0420.log）。
        #    第一电机已使能则跳过。目标已预载，使能不会追旧目标。
        try:
            already_enabled = bool(motor.read_u16(0x6041, 0) & 0x04)
        except Exception:
            already_enabled = False
        if not already_enabled:
            self._enable_operation(motor)

        # 6) 确认真使能：statusword bit2 operation_enabled=1。
        latched = False
        for _ in range(10):  # 最多 ~0.5s，容纳异步使能到位
            time.sleep(0.05)
            try:
                sw = motor.read_u16(0x6041, 0)
            except Exception:
                continue
            if sw & 0x04:
                latched = True
                break
        if latched:
            return True

        # 7) 没真使能：掉电抱闸，绝不留非零扭矩指令在 CST。
        try:
            motor.disable()
        except Exception:
            pass
        return False

    def _enable_operation(self, motor, poll_s: float = 0.05, max_attempts: int = 20) -> bool:
        """裸发 CiA 402 控制字 0x6040 = 0x06→0x07→0x0F 使能，逐步确认状态机到位。

        第二电机(node 16/17)上 configure_csp_mode 只切模式+重映射 RPDO，内部
        enableStateMachine 没走到 Operation Enabled（statusword 卡 0x0331）；但裸
        控制字序列有效（0420.log：0x06→0x07→0x0F 最终 0x1337 op_enabled=1）。

        关键：每条控制字之间必须**确认上一条已生效**再发下一条。CiA 402 状态机是
        条件转移——必须先 Ready to Switch On 才能 0x07 切 Switched On，必须先
        Switched On 才能 0x0F 使能。第二电机 SDO 慢，盲目连发 0x07/0x0F 会被状态机
        拒绝，最终仍卡 0x0331（0420.log/0420_5.log 2026-11-16 复现）。第一电机已
        使能时调用方应跳过本方法。目标位置需先预载（见 _relatch_csp）。
        """
        def _confirm(cond) -> bool:
            for _ in range(max_attempts):
                time.sleep(poll_s)
                try:
                    sw = motor.read_u16(0x6041, 0)
                except Exception:
                    continue
                if cond(sw):
                    return True
            return False

        try:
            motor.write_u16(0x6040, 0, 0x06)  # Shutdown -> Ready to Switch On
        except Exception:
            return False
        if not _confirm(lambda sw: not (sw & 0x04)):
            return False
        try:
            motor.write_u16(0x6040, 0, 0x07)  # Switch On -> Switched On
        except Exception:
            return False
        if not _confirm(lambda sw: sw & 0x02):
            return False
        try:
            motor.write_u16(0x6040, 0, 0x0F)  # Enable Operation
        except Exception:
            return False
        return _confirm(lambda sw: sw & 0x04)

    def compliant_hold(
        self,
        node_index: int,
        kp: float = 20.0,
        kd: float = 5.0,
        vel_alpha: float = 0.3,
        spring_limit: float = 300.0,
        damp_limit: float = 300.0,
        total_limit: float = 500.0,
        max_drift_deg: float = 15.0,
        seconds: float = 10.0,
        loop_ms: int = 20,
        stop_drift_deg: float | None = None,
        hold_on_stop: bool = False,
    ) -> float:
        """将单个关节切到 CST 转矩模式并进入软弹簧保持（漂浮态）。

        即探针脚本 ``eyou_probe_cst.py --soft-hold`` 的核心循环，作为接触停止后
        "降刚度"的基元：``tau = Kp*(theta0 - theta) - Kd*omega``（主机侧算好经 CST
        下发）。关节在 theta0 附近漂浮、可被手推随动，直到：
        - 漂移达 ``stop_drift_deg``（正常"到位"，返回），或
        - 超 ``max_drift_deg``（安全跳闸，抛 :class:`CompliantDriftTrip`），或
        - 到 ``seconds`` 超时。

        安全前提（务必遵守）：只用轻载关节——右臂 joint_5=node 15 / joint_6=node 16，
        左臂 25/26。**绝不用腰(node 1/2)**：腰承载整机重量，软瘫会直接倒地。

        Args:
            node_index: ``self.motor_nodes_`` 中的下标（不是 CAN node id）。
            kp/kd: 弹簧/阻尼增益（千分比/度、千分比/(度/秒)）。
            vel_alpha: 速度低通 alpha(0,1]，滤噪防自激。
            spring_limit/damp_limit/total_limit: 千分比钳位。
            max_drift_deg: 漂移安全阈值（度），超限抛 :class:`CompliantDriftTrip`。
            seconds: 最长保持时长；loop_ms: 循环周期（毫秒）。
            stop_drift_deg: 正常停止阈值（度）。|drift| 达此值即返回（应 <
                max_drift_deg）；None 表示只靠超时/安全跳闸结束。
            hold_on_stop: True 时结束不 disable，而是切回 CSP 并按最终位置刚性保持
                （"抓住盒子"）；False（默认，探针）结束即 disable。

        Returns:
            结束时刻相对 theta0 的漂移（度）。
        Raises:
            CompliantDriftTrip: 漂移超 ``max_drift_deg``。
            RuntimeError: clear_fault 或 configure_cst_mode 失败。
        """
        motor = self.motor_nodes_[node_index]
        joint_name = self.joint_names_[node_index]

        # 关键顺序：必须在仍处于 CSP 位置保持时读 theta0。CST 下零扭矩=软瘫，
        # 若切到 CST 之后再读，读到的是已下垂的位置，不是原始参考。
        theta0 = motor.get_position()

        if not motor.clear_fault():
            raise RuntimeError(f"clear_fault failed for {joint_name}")
        # 参数与已验证的探针脚本一致（见 eyou_probe_cst.py do_soft_hold）。
        if not motor.configure_cst_mode(20, 0, False):
            raise RuntimeError(f"configure_cst_mode failed for {joint_name}")

        dt = loop_ms / 1000.0
        controller = CstSoftSpringController(
            kp=kp,
            kd=kd,
            theta0=theta0,
            dt=dt,
            vel_alpha=vel_alpha,
            spring_limit=spring_limit,
            damp_limit=damp_limit,
            total_limit=total_limit,
        )

        t_end = time.perf_counter() + seconds
        final_drift = 0.0
        try:
            while time.perf_counter() < t_end:
                step = controller.step(motor.get_position())
                motor.send_cst_target_torque(step.tau, 0, False)
                final_drift = step.drift
                # Safety trip FIRST: a runaway drift past max_drift_deg must
                # raise even when stop_drift_deg is also set (a drift that
                # overshoots max also satisfies >= stop, so a stop-first order
                # would silently treat the runaway as 'conformed' and never trip).
                if abs(step.drift) > max_drift_deg:
                    raise CompliantDriftTrip(
                        f"{joint_name} drifted {step.drift:+.2f} deg "
                        f"(limit {max_drift_deg} deg)"
                    )
                # Normal stop: the box has conformed the joint to the target drift.
                if stop_drift_deg is not None and abs(step.drift) >= stop_drift_deg:
                    break
                time.sleep(dt)
        finally:
            if hold_on_stop:
                # 结束不软瘫：切回 CSP 并按"静止后停住的位置"刚性保持，抓住物体。
                # _relatch_csp 内部会零扭矩→等静止→切 CSP；等不到静止或切不动会
                # 自动 disable 抱闸（见 _relatch_csp），同样安全锁住。
                if not self._relatch_csp(motor):
                    print(f"[warn] {joint_name} 回锁 CSP 失败，已 disable 掉电抱闸（安全锁定，非 CSP）")
            else:
                motor.disable()
        return final_drift

    def zero_g_float(
        self,
        node_indices,
        kd: float = 5.0,
        vel_alpha: float = 0.3,
        backup_kp: float = 5.0,
        max_torque: float = 500.0,
        max_drift_deg: float = 30.0,
        seconds: float = 60.0,
        loop_ms: int = 20,
    ) -> list:
        """多关节零重力漂浮（直接示教 hand-guiding），无重力模型。

        无模型重力补偿的关键：仍处于 CSP 位置保持时读一次每个关节的实际扭矩
        ``tau_hold``（千分比）——它就是该位姿下"重力+摩擦"的平衡力矩。把它作为
        前馈 ``tau_ff`` 下发后，关节"停哪停哪"：松手停在原地（前馈托住），轻推
        随动（只有阻尼 + 小弹簧轻微抵抗）。扭矩优先读 TPDO 反馈
        ``get_latest_feedback().torque_milli``（生产 ``read()`` 的主路径），裸 SDO
        ``get_torque()``(0x6077) 作兜底。**必须在关节仍处于 CSP 位置保持（已下发
        目标位置）时读**：未下发保持命令时读到的保持扭矩是 0——不是反馈坏，是关节
        没在对抗重力。

        ::

            tau = tau_hold + backup_kp*(theta0 - theta) - kd*omega

        ``backup_kp`` 只是安全兜底小弹簧（前馈不准时兜住下垂），不是主承力。
        因此它远小于刚性位置环，能实现"推得动、松手不坠"。

        多关节在同一个 20ms 循环里逐个 step+下发，所以整条臂的重关节可同时漂，
        单人即可拖动整臂摆位。

        安全前提（务必遵守）：只在轻载关节上示范——右臂 joint_5/6(node 15/16)、
        左臂 25/26。**绝不用腰(node 1/2)**；承重的肩/肘大行程关节要先把
        ``max_torque``/``max_drift_deg`` 收紧并有人监护再试。``max_torque`` 是
        扭矩钳位兜底（防止 tau_hold 读错时整臂下坠/上甩）。

        Args:
            node_indices: ``self.motor_nodes_`` 中的下标列表（不是 CAN node id）。
            kd: 阻尼增益（千分比/(度/秒)），低值=轻快，高值=粘滞。
            vel_alpha: 速度低通 alpha(0,1]，滤噪防自激（见 CstSoftSpringController）。
            backup_kp: 兜底弹簧增益（千分比/度），只用于前馈不准时兜住漂移。
            max_torque: 扭矩总钳位（千分比）。
            max_drift_deg: 漂移安全阈值（度），超限抛 :class:`CompliantDriftTrip`。
            seconds: 最长漂浮时长；loop_ms: 循环周期（毫秒）。

        Returns:
            各关节结束时刻相对 theta0 的漂移（度），与 node_indices 同序。

        Raises:
            CompliantDriftTrip: 任一关节漂移超 ``max_drift_deg``。
            RuntimeError: clear_fault 或 configure_cst_mode 失败。

        说明: Ctrl-C 在循环内被捕获并正常返回（finally 仍会锁住全部关节），
        以便调用方（如 record_joint_positions.py 的 REPL）继续运行而不退出。
        """
        motors = [self.motor_nodes_[i] for i in node_indices]
        names = [self.joint_names_[i] for i in node_indices]

        # 关键顺序：必须在仍处于 CSP 位置保持时读 theta0 和实际扭矩（重力+摩擦）。
        # CST 下 0x6077≈命令值（驱动照办），读不到重力；切过去再读就读空了。
        # 扭矩优先走 TPDO 反馈(get_latest_feedback().torque_milli，生产 read() 主路径)，
        # get_torque()(裸 SDO 0x6077) 兜底。前提：此刻关节必须仍处于 CSP 位置保持
        # (已下发目标位置)，否则读到的保持扭矩是 0（关节没在对抗重力）。
        theta0s = [m.get_position() for m in motors]
        tau_holds = []
        for m in motors:
            fb = m.get_latest_feedback()
            if fb is not None and fb.last_update_time > datetime.timedelta(0):
                tau_holds.append(float(fb.torque_milli))
            else:
                tau_holds.append(float(m.get_torque()))

        for j, m in enumerate(motors):
            if not m.clear_fault():
                raise RuntimeError(f"clear_fault failed for {names[j]}")
        for j, m in enumerate(motors):
            if not m.configure_cst_mode(20, 0, False):
                raise RuntimeError(f"configure_cst_mode failed for {names[j]}")

        dt = loop_ms / 1000.0
        controllers = [
            CstSoftSpringController(
                kp=backup_kp,
                kd=kd,
                theta0=theta0s[j],
                dt=dt,
                vel_alpha=vel_alpha,
                tau_ff=tau_holds[j],
                spring_limit=max_torque,
                damp_limit=max_torque,
                total_limit=max_torque,
            )
            for j in range(len(motors))
        ]

        t_end = time.perf_counter() + seconds
        final_drifts = [0.0] * len(motors)
        try:
            while time.perf_counter() < t_end:
                for j, m in enumerate(motors):
                    step = controllers[j].step(m.get_position())
                    m.send_cst_target_torque(step.tau, 0, False)
                    final_drifts[j] = step.drift
                    if abs(step.drift) > max_drift_deg:
                        raise CompliantDriftTrip(
                            f"{names[j]} drifted {step.drift:+.2f} deg "
                            f"(limit {max_drift_deg} deg)"
                        )
                time.sleep(dt)
        except KeyboardInterrupt:
            # 用户 Ctrl-C 结束示教：吞掉，正常返回，让 REPL 继续。
            pass
        finally:
            # 结束切回 CSP 并按"静止后停住的位置"刚性保持（锁住），防止"没人扶就掉"。
            # _relatch_csp 内部会零扭矩→等静止→切 CSP；等不到静止或切不动会自动
            # disable 抱闸（见 _relatch_csp），同样安全锁住。
            for j, m in enumerate(motors):
                if not self._relatch_csp(m):
                    print(f"[warn] {names[j]} 回锁 CSP 失败，已 disable 掉电抱闸（安全锁定，非 CSP）")
        return final_drifts

    def deactivate(self):
        """停用硬件。"""
        print("Deactivating EyouMotorHardware...")
        try:
            if self.can_manager_ is not None and self._can_device_index is not None:
                self.can_manager_.stop_sync(self._can_device_index)
            for motor in self.motor_nodes_:
                motor.disable()
        except RuntimeError as e:
            print(f"Error during deactivation: {e}")
        print("Deactivation successful.")

    def get_joint_count(self) -> int:
        """返回硬件中的电机数量。"""
        return len(self.motor_nodes_)

# --- 主程序：演示如何使用混合模式的硬件接口 ---
if __name__ == "__main__":
    # ==================== 配置修改开始 ====================
    # 根据 EyouMotorHardware.init 的更新，修改了此处的配置结构。
    # 主要变化：
    # 1. 'node_id' 现在位于 'parameters' 字典内部。
    # 2. 'parameters' 中新增了可选的 'start_enabled' 参数。
    # 3. 移除了不再使用的 'can_device_type'。
    robot_config = {
        "can_device_index": 1,
        "can_baud_rate": "1M",
        "joints": [
            {
                "name": "right_arm_joint_2",
                "parameters": {
                    "node_id": 12,
                    # "start_enabled" 在此处被省略，将使用默认值 True
                }
            },
            {
                "name": "right_arm_joint_4",
                "parameters": {
                    "node_id": 14,
                    "start_enabled": True  # 明确设置为 True (与默认行为相同)
                }
            },
            # 添加第三个关节，以演示 "start_enabled": False 的效果
            # 这个关节将被初始化，但不会在 activate() 中被使能
            {
                "name": "right_arm_joint_6",
                "parameters": {
                    "node_id": 16,
                    "start_enabled": False # 设置为在启动时不使能
                }
            }
        ]
    }
    # ==================== 配置修改结束 ====================

    robot = EyouMotorHardware()

    if not robot.init(robot_config):
        print("Failed to initialize robot hardware. Exiting.")
        exit(1)

    try:
        if not robot.activate():
            print("Failed to activate robot hardware. Exiting.")
            exit(1)
        
        initial_positions = list(robot.hw_commands_positions_)
        print(f"\n--- Starting Control Loop (Press Ctrl+C to exit) ---")
        print(f"Initial positions: {[f'{p:.2f}' for p in initial_positions]}")
        
        target_positions = list(initial_positions)
        
        control_frequency = 100
        control_period = 1.0 / control_frequency
        start_loop_time = time.time()
        
        while True:
            loop_start = time.perf_counter()
            
            # a. 读取硬件状态 (更新内部状态并返回拷贝)
            current_positions = robot.read()
            
            # 定期打印状态
            if int(loop_start * 10) % 10 == 0:
                pos_str = ", ".join([f"{p:7.2f}" for p in current_positions])
                # 标记出被禁用的电机
                enabled_status = [" " if robot.hw_start_enabled_[i] else "D" for i in range(len(robot.hw_start_enabled_))]
                status_str = "".join(enabled_status)
                print(f"Time: {time.time() - start_loop_time:5.2f}s | Pos: [{pos_str}] | Enabled: [{status_str}]")


            # b. Controller Logic: 计算新的目标位置
            elapsed_time = time.time() - start_loop_time
            
            amplitude = 30.0 # 幅度增大以便观察
            frequency = 0.2
            
            for i in range(len(target_positions)):
                # 只对已使能的电机应用运动指令
                if robot.hw_start_enabled_[i]:
                    phase = i * (math.pi / 2)
                    
                    # 1. 创建一个在 [0, 1] 范围内振荡的归一化值
                    normalized_oscillation = (math.sin(2 * math.pi * frequency * elapsed_time + phase) + 1) / 2
                    
                    # 2. 计算始终为正的偏移量
                    offset = amplitude * normalized_oscillation
                    
                    # 3. 将偏移量加到初始位置上
                    target_positions[i] = initial_positions[i] + offset
                else:
                    # 对于禁用的电机，保持其目标位置不变
                    target_positions[i] = initial_positions[i]

            # c. 写入硬件
            robot.write(target_positions)
            
            # d. 维持控制频率
            loop_end = time.perf_counter()
            sleep_time = control_period - (loop_end - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Shutting down.")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        robot.deactivate()