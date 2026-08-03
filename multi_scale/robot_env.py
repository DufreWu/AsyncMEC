import time
import os
import sys
import random
import numpy as np
from collections import deque
import onnxruntime as ort
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from jetson_orin_nx import Jetson_Orin_NX

class BatteryMonitor:
    def __init__(
        self,
        cfg,
        rated_capacity_ah=2.0,
        initial_soh = 1.0,
        initial_soc = 1.0,
        nominal_voltage=3.7,
        internal_resistance=0.05,
        ambient_temp=25.0
    ):
        self.cfg = cfg
        self.rated_capacity_ah = rated_capacity_ah
        self.initial_soc = initial_soc
        self.initial_soh = initial_soh
        
        # Physical battery state
        self.capacity_ah = rated_capacity_ah * self.initial_soh
        self.remaining_ah = self.capacity_ah * initial_soc

        self.feature_period = cfg["battery"]["feature_period"]
        self.elapsed = 0.0
        self.health_feature = None
        self.delta_capacity_prediction = None

        self.nominal_voltage = nominal_voltage
        self.temperature = ambient_temp
        self.ambient_temp = ambient_temp  

        # ECM Parameters
        self.ocv = nominal_voltage
        self.R0 = internal_resistance

        self.R1 = 0.015
        self.C1 = 1800

        self.R2 = 0.008
        self.C2 = 6000

        self.v_rc1 = 0.0
        self.v_rc2 = 0.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

        self.state = {
            "voltage": nominal_voltage,
            "current": 0.0,
            "temperature": ambient_temp,
            "soc": initial_soc,
            "soh": self.estimated_soh
        }

        # ------------------------------------
        # Rolling battery telemetry buffer
        # Each item: [voltage, current, temperature]
        # ------------------------------------
        self.window_size = cfg["battery"]["window_size"]
        self.window_buffer = deque(maxlen=self.window_size)

        for _ in range(self.window_size):
            self.window_buffer.append([
                nominal_voltage,
                0.0,
                ambient_temp
            ])

        self.encoder = ort.InferenceSession(
            cfg["battery"]["encoder_path"],
            providers=["CPUExecutionProvider"]
        )

        self.dcap = ort.InferenceSession(
            cfg["battery"]["dcap_path"],
            providers=["CPUExecutionProvider"]
        )

        # encoder scaler
        with open(cfg["battery"]["encoder_x0_scaler"], "rb") as f:
            self.encoder_scaler_v = pickle.load(f)
        with open(cfg["battery"]["encoder_x1_scaler"], "rb") as f:
            self.encoder_scaler_i = pickle.load(f)
        with open(cfg["battery"]["encoder_x2_scaler"], "rb") as f:
            self.encoder_scaler_t = pickle.load(f)
        with open(cfg["battery"]["encoder_y_scaler"], "rb") as f:
            self.encoder_scaler_y = pickle.load(f)

        # dcap scaler
        with open(cfg["battery"]["dcap_x0_scaler"], "rb") as f:
            self.dcap_scaler_v = pickle.load(f)
        with open(cfg["battery"]["dcap_x1_scaler"], "rb") as f:
            self.dcap_scaler_i = pickle.load(f)
        with open(cfg["battery"]["dcap_x2_scaler"], "rb") as f:
            self.dcap_scaler_t = pickle.load(f)
        with open(cfg["battery"]["dcap_y_scaler"], "rb") as f:
            self.dcap_scaler_y = pickle.load(f)
        
        # self._update_health()
   
    @property
    def estimated_soh(self):
        if self.delta_capacity_prediction is None:
            return self.initial_soh
        
        return float(self.capacity_ah / self.rated_capacity_ah)
    
    @property
    def resistance(self):
        return self.R0 * (1 + 2 * (1 - self.estimated_soh))
    
    def reset(self, initial_soc=1.0):
        self.initial_soc = initial_soc
        self.capacity_ah = self.rated_capacity_ah * self.estimated_soh
        self.remaining_ah = self.capacity_ah * initial_soc
        self.temperature = self.ambient_temp

        self.elapsed = 0.0
        self.v_rc1 = 0.0
        self.v_rc2 = 0.0

        self.state = {
            "voltage": self.nominal_voltage,
            "current": 0.0,
            "temperature": self.temperature,
            "soc": self.initial_soc,
            "soh": self.estimated_soh,
        }

        self.window_buffer.clear()
        for _ in range(self.window_size):
            self.window_buffer.append(
                [self.nominal_voltage, 0.0, self.ambient_temp]
            )

        self.health_feature = None
        self.delta_capacity_prediction = None
        # self._update_health()

    def update(self, power, dt):
        """
        Update battery state.

        Args:
            power (float): Battery output power (W)
            dt (float): Time step (s)
        """
        voltage = max(self.state["voltage"], 2.8)
        current = power / voltage
        energy_used = current * dt / 3600

        self.remaining_ah = max(0.0, self.remaining_ah - energy_used)
        soc = np.clip(self.remaining_ah / self.capacity_ah, 0.0, 1.0)

        # ----------------------------------------------------
        # Single-cell OCV model (3.0V ~ 4.2V)
        # ----------------------------------------------------
        ocv = (3.0 + 1.18*soc + 0.05*np.tanh(8*(soc-0.5)))

        a1 = np.exp(-dt / (self.R1 * self.C1))
        a2 = np.exp(-dt / (self.R2 * self.C2))

        self.v_rc1 = a1 * self.v_rc1 + self.R1 * (1 - a1) * current
        self.v_rc2 = a2 * self.v_rc2 + self.R2 * (1 - a2) * current

        # Terminal voltage
        voltage = ocv - current * self.R0 - self.v_rc1 - self.v_rc2
        voltage = float(np.clip(voltage, 2.8, 4.2))

        # Thermal dynamics integration
        total_r = self.resistance + self.R1 + self.R2
        heat_gen = current ** 2 * total_r
        cooling = 0.02 * (self.temperature - self.ambient_temp)

        # Thermal capacity factor (approx. 0.005 C/J heat scaling)
        self.temperature += (heat_gen - cooling) * dt * 0.005

        self.state = {
            "voltage": voltage,
            "current": current,
            "temperature": self.temperature,
            "soc": soc,
            "soh": self.estimated_soh,
        }

        self.window_buffer.append([voltage, current, self.temperature])

        self.elapsed += dt
        if self.elapsed >= self.feature_period:
            self._update_health()
            self.elapsed = 0.0

    def read(self):
        return {
            **self.state,
            "estimated_soh": self.estimated_soh,
            "capacity_prediction": self.capacity_ah,
        }
    
    # ========================================================
    # Get AE input window
    # ========================================================
    def get_window(self):
        battery_window = np.array(
            list(self.window_buffer),
            dtype=np.float32
        ).reshape(-1, 3)

        # ----------------------------------------------------
        # Batch dimension
        # ----------------------------------------------------
        battery_window = np.expand_dims(
            battery_window,
            axis=0
        )

        return battery_window   
    
    def _get_scaled_window(self):
        window = self.get_window().copy()

        # Scale each channel using its respective encoder scaler
        v_seq = window[0, :, 0].reshape(1, -1)
        i_seq = window[0, :, 1].reshape(1, -1)
        t_seq = window[0, :, 2].reshape(1, -1)

        window[0, :, 0] = self.encoder_scaler_v.transform(v_seq).ravel()
        window[0, :, 1] = self.encoder_scaler_i.transform(i_seq).ravel()
        window[0, :, 2] = self.encoder_scaler_t.transform(t_seq).ravel()

        return window.astype(np.float32)

    def get_health_feature(self):
        if self.health_feature is None:
            return np.zeros(self.cfg["battery"]["encoder_dim"], dtype=np.float32)

        return self.health_feature

    def _update_health(self):
        scaled_window = self._get_scaled_window()

        # Step 1: Run Encoder to extract Latent Health Features
        enc_input_name = self.encoder.get_inputs()[0].name
        self.health_feature = self.encoder.run(
            None, 
            {enc_input_name: scaled_window}
        )[0].squeeze(0)

        # Step 2: Run DCAP prediction
        dcap_input_name = self.dcap.get_inputs()[0].name
        raw_pred = self.dcap.run(
            None, 
            {dcap_input_name: scaled_window}
        )[0].squeeze(0)

        # Step 3: Inverse-transform using dcap_scaler_y
        unscaled_capacity = self.dcap_scaler_y.inverse_transform(
            raw_pred.reshape(1, -1)
        ).item()

        # Step 4: Clip output within physical limits [0, rated_capacity_ah]
        self.delta_capacity_prediction = unscaled_capacity * (1e-6)

        if self.delta_capacity_prediction is not None:
            self.capacity_ah = self.capacity_ah - abs(self.delta_capacity_prediction)
            # print(f"capacity: {self.capacity_ah}")

class Motor:
    def __init__(self, cfg):
        self.max_speed = cfg["motor"]["v_max"]
        self.min_speed = cfg["motor"]["v_min"]
        self.speed = 0.5
    
    def set_speed(self, v) -> None:
        self.speed = max(self.min_speed, min(self.max_speed, v))
    
    def get_speed(self):
        return self.speed

class RobotEnv:
    def __init__(self, cfg, is_sim=False):
        # Initialize robot state
        self.board = Jetson_Orin_NX(cfg) 
        self.battery = BatteryMonitor(cfg)
        self.motor = Motor(cfg)

        # Initial actions
        self.fc_list = np.full(8, self.board.cpu_freqs[0])
        self.fg =   self.board.gpu_freqs[0]      
        self.acceleration = 0.5

        # max 
        self.max_mech_power = cfg["controller"]["max_mech_power"]
        self.max_comp_power = cfg["controller"]["max_comp_power"]

        # Others
        self.is_sim = is_sim

    def set_speed(self, speed):
        """Sets the target velocity for the robot."""
        self.motor.set_speed(speed)

    def get_speed(self):
        """Simulates real-time speed with basic physics (acceleration)."""
        return self.motor.get_speed()

    def get_mechanical_power(self):
        """Calculates instantaneous mechanical power based on current speed."""
        v = self.get_speed()
        power = 1.14005 * v + 0.10893 * v ** 3 + 1.1625 

        # noise = random.gauss(0, 0.01 * power)

        return power
    
    def set_board_cpu_dvfs(self, cpu, fc)->None:
        self.fc_list[cpu] = fc

        if self.is_sim == False:
            self.board.set_cpu_freq(cpu, self.fc_list[cpu])
    
    def get_board_cpu_freqs(self):
        if self.is_sim:
            return self.fc_list
        else:
            return self.board.read_cpu_freqs()
    
    def set_board_gpu_dvfs(self, fg)->None:
        self.fg = fg
        if self.is_sim == False:
            self.board.set_gpu_freq(self.fg)
    
    def get_board_gpu_freq(self):
        if self.is_sim:
            return self.fg
        else:
            return self.board.read_gpu_freq()
    
    def get_computational_power(self):
        if self.is_sim:
            P_static = 4.5
            alpha = 2.75
            beta = 13.5
            # ---- CPU power (per-core accumulation) ----
            P_cpu = 0.0
            for fc in self.fc_list:
                P_cpu += alpha * (fc / self.board.cpu_freqs[-1])

            # ---- GPU power ----
            P_gpu = beta * (self.fg / self.board.gpu_freqs[-1])

            power = P_static + P_cpu + P_gpu
            noise = random.gauss(0, 0.01 * power)

            return max(0, power + noise)

        else:
            return self.board.read_power()

    def get_robot_power(self):
        """Combines Computing (Board) and Mechanical (Motors) power."""
        p_comp = self.get_computational_power()
        p_mech = self.get_mechanical_power()

        return (p_comp + p_mech)
    
    def get_robot_battery_soc(self):
        """Updates battery state based on current power draw and returns SOC."""
        return self.battery.read()["soc"]
    
    def update_battery(self, dt) -> None:
        power = self.get_robot_power()
        self.battery.update(power, dt)
    
    def step(self, action, dt=1.0):
        # --------------------------------
        # Apply actions
        # --------------------------------
        self.set_speed(action["speed"])
        for i in range(8):
            self.set_board_cpu_dvfs(i, action["cpu_freq"])
        self.set_board_gpu_dvfs(action["gpu_freq"])

        # --------------------------------
        # Update battery
        # --------------------------------
        self.update_battery(dt)
        battery_state = self.battery.read()

        # --------------------------------
        # Power
        # --------------------------------
        p_comp = self.get_computational_power()
        p_mech = self.get_mechanical_power()

        # --------------------------------
        # State
        # --------------------------------
        state = {
            "p_comp": p_comp,
            "p_mech": p_mech,
            "speed": self.get_speed(),
            "cpu_freq": self.board.read_cpu_freq(0),
            "gpu_freq": self.board.read_gpu_freq(),
            "battery": battery_state
        }

        return state