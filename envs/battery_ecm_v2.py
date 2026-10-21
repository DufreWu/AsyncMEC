"""Battery-aware simulation: SOC/SOH/temperature-dependent 2RC ECM.

All five ECM elements (R0, R1, R2, C1, C2) have independent, explicit
functions of SOC, SOH, and cell/pack temperature. OCV can be supplied as
SOH-indexed measured curves or an explicitly uncalibrated aging perturbation.
No measured impedance, HPPC, OCV or aging calibration is claimed: default
coefficient values are illustrative.
Only discharge (nonnegative constant terminal power) is supported.

At each electrical substep, the RC time constants are recomputed as
    tau1 = R1(SOC, SOH, T) * C1(SOC, SOH, T)
    tau2 = R2(SOC, SOH, T) * C2(SOC, SOH, T)

Reference state for ECM component values: SOC=0.5, SOH=1, T=25 C.

Default pack: Blue Robotics Lithium-ion Battery (4S6P, 18 Ah, 14.8 V,
266.4 Wh nominal energy, 12 V minimum safe voltage).
https://bluerobotics.com/store/comm-control-power/powersupplies-batteries/battery-li-4s-18ah-r3/
Pack ratings do not calibrate the assumed OCV, resistance, RC or aging curves.

Run demo: python battery_ecm_soc_soh_ocv.py --capacity-ah 18 --nominal-voltage 14.8
Tests:    python battery_ecm_soc_soh_ocv.py --self-test
Reference for ECM structure only (NOT default numerical calibration):
https://www.mathworks.com/help/simscape-battery/ref/batteryequivalentcircuit.html
"""
import argparse
import math
from dataclasses import dataclass


class BatteryLimitError(RuntimeError):
    """Requested step not deliverable. State is unchanged; terminate episode."""


@dataclass(frozen=True)
class ECMParameters:
    r0: float
    r1: float
    r2: float
    c1: float
    c2: float
    tau1: float
    tau2: float


class Battery2RCECM:
    # Pack OCV from an illustrative 4S lithium-ion cell curve (SOC, cell V).
    _CELL_OCV = (
        (0.00, 3.00), (0.05, 3.30), (0.10, 3.50),
        (0.20, 3.60), (0.40, 3.70), (0.60, 3.80),
        (0.80, 3.95), (0.90, 4.08), (1.00, 4.20),
    )

    def __init__(self, capacity_ah=18.0, nominal_voltage=14.8,
                 soc_init=1.0, soh_init=1.0, temp_c=25.0,
                 cycles_to_80=1000.0, ocv_age_delta_v_at_soh80=-0.8):
        if capacity_ah <= 0 or nominal_voltage <= 0 or cycles_to_80 <= 0:
            raise ValueError('Capacity, nominal voltage and cycle life must be positive')
        if not 0.02 <= soc_init <= 1 or not 0.5 <= soh_init <= 1:
            raise ValueError('SOC must be in [0.02, 1] and SOH in [0.5, 1]')
        if not -10 <= temp_c <= 50:
            raise ValueError('Temperature must be in [-10, 50] C')
        values = (capacity_ah, nominal_voltage, soc_init, soh_init, temp_c,
                  cycles_to_80, ocv_age_delta_v_at_soh80)
        if not all(math.isfinite(x) for x in values):
            raise ValueError('All inputs must be finite')

        self.capacity_ah = float(capacity_ah)
        self.nominal_voltage = float(nominal_voltage)
        self.soc = float(soc_init)
        self.soh = float(soh_init)
        self.temp_c = float(temp_c)
        self.cycles_to_80 = float(cycles_to_80)
        self.ocv_age_delta_v_at_soh80 = float(ocv_age_delta_v_at_soh80)
        self.cutoff_voltage = self.nominal_voltage * (3.0 / 3.7)

        # Reference components correspond to SOC=0.5, SOH=1, T=25 C.
        # Rough 5 Ah -> pack scaling; replace with pack HPPC fit when available.
        scale = (self.nominal_voltage / 14.8) * (5.0 / self.capacity_ah)
        self.r0_ref, self.r1_ref, self.r2_ref = 0.035 * scale, 0.010 * scale, 0.005 * scale
        self.c1_ref = 7.5 / self.r1_ref
        self.c2_ref = 30.0 / self.r2_ref

        self.v1 = self.v2 = 0.0
        self.elapsed_s = 0.0
        self.load_energy_j = 0.0
        self.source_energy_j = 0.0
        self.last_info = {}

    def ocv(self, soc=None, soh=None, temp_c=None):
        """Pack OCV(SOC, SOH, T), with fixed synthetic aging correction.

        Temperature term is set to zero until measured OCV(T) data exist.
        SOH does not directly scale voltage or replace capacity fade.
        """
        s = self.soc if soc is None else float(soc)
        h = self.soh if soh is None else float(soh)
        t = self.temp_c if temp_c is None else float(temp_c)
        if not (math.isfinite(s) and 0 <= s <= 1 and math.isfinite(h) and 0.5 <= h <= 1
                and math.isfinite(t) and -10 <= t <= 50):
            raise ValueError('Invalid SOC, SOH, or temperature')
        for (sa, va), (sb, vb) in zip(self._CELL_OCV, self._CELL_OCV[1:]):
            if s <= sb:
                cell_v = va + (vb - va) * (s - sa) / (sb - sa)
                break
        else:
            cell_v = self._CELL_OCV[-1][1]

        fresh_v = cell_v * self.nominal_voltage / 3.7

        age_factor = (1.0 - h) / 0.2
        soc_shape = 4.0 * s * (1.0 - s)

        age_shift = age_factor * (
            -0.2 * s - 0.7 * soc_shape
        )

        return fresh_v + age_shift * self.nominal_voltage / 14.8

    def parameters(self, soc=None, soh=None, temp_c=None):
        """R0, R1, R2, C1, C2 all depend independently on SOC, SOH, T."""
        s = self.soc if soc is None else float(soc)
        h = self.soh if soh is None else float(soh)
        t = self.temp_c if temp_c is None else float(temp_c)
        if not (math.isfinite(s) and 0 <= s <= 1 and math.isfinite(h) and 0.5 <= h <= 1
                and math.isfinite(t) and -10 <= t <= 50):
            raise ValueError('Invalid SOC, SOH, or temperature')

        def resistance(ref, soc_k, soh_k, temp_k):
            return (ref * (1 + soc_k * (1 - s)**2) / (1 + soc_k * 0.25)
                    * (1 + 3 * soh_k * (1 - h)) * math.exp(temp_k * (25 - t)))

        def capacitance(ref, soc_k, soh_k, temp_k):
            return (ref * (1 + soc_k * (s - 0.5))
                    * (1 - soh_k * (1 - h)) * math.exp(temp_k * (t - 25)))

        r0 = resistance(self.r0_ref, 0.9, 0.70, 0.018)
        r1 = resistance(self.r1_ref, 1.4, 1.00, 0.022)
        r2 = resistance(self.r2_ref, 1.8, 1.30, 0.027)
        c1 = capacitance(self.c1_ref, 0.35, 0.60, 0.012)
        c2 = capacitance(self.c2_ref, 0.50, 0.90, 0.009)

        return ECMParameters(r0, r1, r2, c1, c2, r1 * c1, r2 * c2)

    def _electrical(self, power, soc, soh, v1, v2, temp):
        p = self.parameters(soc, soh, temp)
        ocv = self.ocv(soc, soh, temp)
        v_eff = ocv - v1 - v2
        disc = v_eff**2 - 4 * p.r0 * power
        if v_eff <= 0 or disc < 0:
            raise BatteryLimitError('Demand exceeds ECM deliverable power')
        current = 2 * power / (v_eff + math.sqrt(disc))
        voltage = v_eff - current * p.r0
        if voltage < self.cutoff_voltage:
            raise BatteryLimitError('Terminal voltage below cutoff')
        return current, voltage, ocv, p

    def estimate_energy_drawn(self, power_w, duration_s=1.0):
        """Instantaneous fixed-state estimate, without advancing battery state."""
        if not math.isfinite(power_w) or power_w < 0 or not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('Power must be >= 0; duration must be > 0')
        i, v, ocv, _ = self._electrical(power_w, self.soc, self.soh,
                                      self.v1, self.v2, self.temp_c)
        load_j = power_w * duration_s
        source_j = ocv * i * duration_s
        return dict(current_a=i, voltage_v=v, load_energy_j=load_j,
                    source_energy_j=source_j, energy_loss_j=source_j - load_j)

    def energy_loss(self, power_w, duration_s=1.0):
        """Estimate battery internal energy loss in joules (state unchanged)."""
        return self.estimate_energy_drawn(power_w, duration_s)['energy_loss_j']

    def step(self, power_w, dt=1.0, temp_c=None):
        """Apply constant terminal load (W) for dt seconds; return (SOC, SOH).

        Always updates SOH, including calendar-free discharge aging.
        On a battery limit violation, no state is committed.
        """
        t = self.temp_c if temp_c is None else float(temp_c)
        if not (math.isfinite(power_w) and power_w >= 0 and math.isfinite(dt) and dt > 0
                and math.isfinite(t) and -10 <= t <= 50):
            raise ValueError('Invalid power, timestep, or temperature')
        n = math.ceil(dt / 0.1)
        if n > 100000:
            raise ValueError('Timestep too long; split it into smaller calls')
        ds = dt / n
        soc, soh, v1, v2 = self.soc, self.soh, self.v1, self.v2
        source_j = 0.0
        for _ in range(n):
            i, v, ocv, p = self._electrical(power_w, soc, soh, v1, v2, t)
            da = i * ds / 3600.0
            soc -= da / (self.capacity_ah * soh)
            if soc < 0.02:
                raise BatteryLimitError('SOC reserve reached')
            v1 = v1 * math.exp(-ds / p.tau1) + i * p.r1 * (-math.expm1(-ds / p.tau1))
            v2 = v2 * math.exp(-ds / p.tau2) + i * p.r2 * (-math.expm1(-ds / p.tau2))
            c_rate = i / (self.capacity_ah * soh)
            stress = 2 ** ((t - 25) / 10) * (1 + 0.2 * c_rate**2)
            soh -= (0.2 / self.cycles_to_80) * (da / self.capacity_ah) * stress
            if soh < 0.5:
                raise BatteryLimitError('SOH below modeled range')
            source_j += ocv * i * ds

        i, v, ocv, p = self._electrical(power_w, soc, soh, v1, v2, t)
        self.soc, self.soh, self.v1, self.v2, self.temp_c = soc, soh, v1, v2, t
        self.elapsed_s += dt
        self.load_energy_j += power_w * dt
        self.source_energy_j += source_j
        self.last_info = dict(
            soc=soc, soh=soh, voltage_v=v, current_a=i, ocv_v=ocv,
            r0_ohm=p.r0, r1_ohm=p.r1, r2_ohm=p.r2, c1_f=p.c1, c2_f=p.c2,
            load_energy_j=power_w * dt, source_energy_j=source_j,
            energy_loss_j=source_j - power_w * dt,
            cumulative_load_energy_j=self.load_energy_j,
            cumulative_source_energy_j=self.source_energy_j,
        )
        return soc, soh


def self_test():
    import copy
    def new(**kw): return Battery2RCECM(18, 14.8,**kw)
    for power in [0,5,25,42]:
        b = new()
        for _ in range(100):
            old_soc,old_soh = b.soc,b.soh
            s,h = b.step(power)
            assert s<=old_soc and h<=old_soh
            assert math.isclose(b.last_info['current_a']*b.last_info['voltage_v'],power,abs_tol=1e-10)
        assert math.isclose(b.load_energy_j,power*100)
        e = b.estimate_energy_drawn(power, duration_s=10)
        assert math.isclose(e['load_energy_j'], power*10, rel_tol=1e-12, abs_tol=1e-12)
        assert e['source_energy_j'] >= e['load_energy_j']
        loss = b.energy_loss(power, duration_s=10)
        assert math.isclose(loss['energy_loss_j'], e['energy_loss_j'], rel_tol=1e-12, abs_tol=1e-12)
        assert loss['energy_loss_j'] >= 0.0
    b = new(aging=False)
    b.step(25)
    assert b.soh==1
    before = copy.deepcopy(b.__dict__)
    try: b.step(1e6)
    except BatteryLimitError: pass
    else: raise AssertionError('overload accepted')
    assert b.__dict__==before
    # Late-substep reserve failure must roll back the entire requested step.
    b = new(soc_init=.0201,cutoff_voltage=1)
    before = copy.deepcopy(b.__dict__)
    try: b.step(25,dt=10)
    except BatteryLimitError: pass
    else: raise AssertionError('reserve crossed')
    assert b.__dict__==before
    fresh,old = new(aging=False),new(soh_init=.8,aging=False)
    for _ in range(600): fresh.step(30); old.step(30)
    assert old.soc<fresh.soc and old.parameters().r0>fresh.parameters().r0
    assert old.source_energy_j>fresh.source_energy_j
    b = new()
    b.step(25)
    prior = b.v1
    b.step(0)
    assert 0<b.v1<prior
    assert b.parameters(soc=.05).r0 > b.parameters(soc=.8).r0
    # OCV must depend on SOC, SOH in synthetic mode, and on T when configured.
    b = new()
    assert b.ocv(soc=.2,soh=1) < b.ocv(soc=.8,soh=1)
    assert b.ocv(soc=.5,soh=.8) < b.ocv(soc=.5,soh=1)
    assert math.isclose(b.ocv(soc=0,soh=.8),b.ocv(soc=0,soh=1))
    assert math.isclose(b.ocv(soc=1,soh=.8),b.ocv(soc=1,soh=1))
    # Measured lookups, including interpolation of SOH and SOC, and clamping.
    base_curve = [(0,12),(1,16)]
    aged_curve = [(0,12),(1,15)]
    measured = new(ocv_tables_by_soh={1.0:base_curve, 0.8:aged_curve},aging=False)
    assert math.isclose(measured.ocv(soc=.5,soh=.9), 13.75)
    assert math.isclose(measured.ocv(soc=.5,soh=.6), 13.5)
    assert math.isclose(measured.ocv(soc=.5,soh=1), 14)
    assert math.isclose(new(ocv_temp_coeff_v_per_c=.005).ocv(soc=.5,soh=1,temp_c=35)
                        -new().ocv(soc=.5,soh=1,temp_c=25),.05,abs_tol=1e-12)
    # Custom reference table defaults to no invented SOH offset.
    assert new(ocv_table=[(0,15),(1,15)]).ocv(soc=.5,soh=.8)==15
    for power in (0, 5, 25):
        sim = new(soh_init=.8)
        sim.step(power)
        assert math.isclose(sim.last_info['ocv_v'], sim.ocv(),abs_tol=1e-9)
    # R0/R1/R2/C1/C2 must EACH depend on all three state variables.
    for name in ('r0', 'r1', 'r2', 'c1', 'c2'):
        base = getattr(b.parameters(soc=.5,soh=1,temp_c=25), name)
        assert getattr(b.parameters(soc=.2,soh=1,temp_c=25), name) != base, (name, 'SOC')
        assert getattr(b.parameters(soc=.5,soh=.8,temp_c=25), name) != base, (name, 'SOH')
        assert getattr(b.parameters(soc=.5,soh=1,temp_c=5), name) != base, (name, 'T')
    for s0 in (0.05,0.3,0.8):
        for h0 in (0.6,0.85,1.0):
            for t0 in (-10,5,25,50):
                p = b.parameters(soc=s0,soh=h0,temp_c=t0)
                assert all(getattr(p,x)>0 for x in ('r0','r1','r2','c1','c2','tau1','tau2'))
                assert math.isclose(p.tau1,p.r1*p.c1,rel_tol=1e-12)
                assert math.isclose(p.tau2,p.r2*p.c2,rel_tol=1e-12)
    p0 = b.parameters(soc=.5,soh=1,temp_c=25)
    assert math.isclose(p0.r0,b.r0_ref*b.resistance_scale)
    assert math.isclose(p0.r1,b.r1_ref*b.resistance_scale)
    assert math.isclose(p0.r2,b.r2_ref*b.resistance_scale)
    assert math.isclose(p0.c1,b.c1_ref)
    assert math.isclose(p0.c2,b.c2_ref)
    assert math.isclose(p0.tau1,b.tau1_ref)
    assert math.isclose(p0.tau2,b.tau2_ref)
    # Integration convergence across varying loads.
    a,c = new(max_substep_s=.1),new(max_substep_s=.01)
    for power in [5,42,10,30]*20: a.step(power); c.step(power)
    assert abs(a.soc-c.soc)<1e-5 and abs(a.v1-c.v1)<1e-4
    # Analytic coulomb counting with a constant supplied OCV table.
    b = new(aging=False,ocv_table=[(0,15),(1,15)])
    start = b.soc
    b.step(25)
    assert math.isclose(start-b.soc,b.discharge_ah/5,abs_tol=1e-12)
    for power in [-1,float('nan'),float('inf')]:
        try: b.step(power)
        except ValueError: pass
        else: raise AssertionError('invalid power accepted')
    print('PASS: SOC/SOH/T OCV, measured OCV table interpolation, 5 independent state-dependent ECM parameters, positive RC elements, power balance, SOC/SOH monotonicity, rollback, relaxation, coulomb counting, integration convergence and validation')


def main():
    parser = argparse.ArgumentParser(description='Uncalibrated 2RC battery simulation')
    parser.add_argument('--capacity-ah',type=float,default=18.0)
    parser.add_argument('--nominal-voltage',type=float,default=14.8)
    parser.add_argument('--soh-init',type=float,default=0.85)
    parser.add_argument('--power-w',type=float,default=30)
    parser.add_argument('--power-demand-w',nargs='*',type=float,default=None,
                        help='Optional list of constant power demands in W to estimate energy for. '
                             'Example: --power-demand-w 10 25 50')
    parser.add_argument('--duration-s',type=float,default=60,help='Duration used for energy estimates when listing power demands')
    parser.add_argument('--steps',type=int,default=60)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--show-parameters',action='store_true',
                        help='Print illustrative ECM parameters at selected SOC/SOH/T combinations')
    parser.add_argument('--show-ocv',action='store_true',
                        help='Print illustrative OCV(SOC,SOH) curves at 25 C')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.steps<0: parser.error('steps must be nonnegative')
    b = Battery2RCECM(args.capacity_ah,args.nominal_voltage,soh_init=args.soh_init)
    if args.show_ocv:
        print('# Synthetic OCV(SOC, SOH), NOT pack measurements')
        print('soc,ocv_soh_1_v,ocv_soh_0.8_v,ocv_soh_0.6_v')
        for soc0 in (0.05,0.2,0.4,0.6,0.8,0.95):
            print(f'{soc0:g},{b.ocv(soc0,1):.6f},{b.ocv(soc0,.8):.6f},{b.ocv(soc0,.6):.6f}')
        return
    if args.show_parameters:
        print('# Illustrative, NOT measured or fitted ECM parameters')
        print('soc,soh,temp_c,r0_ohm,r1_ohm,r2_ohm,c1_f,c2_f,tau1_s,tau2_s')
        for soc0,soh0,t0 in ((.8,1.,25.),(.2,1.,25.),(.8,.8,25.),(.8,1.,5.),(.2,.8,5.)):
            pp = b.parameters(soc=soc0,soh=soh0,temp_c=t0)
            print(f'{soc0:g},{soh0:g},{t0:g},{pp.r0:.6g},{pp.r1:.6g},{pp.r2:.6g},'
                  f'{pp.c1:.6g},{pp.c2:.6g},{pp.tau1:.6g},{pp.tau2:.6g}')
        return
    if args.power_demand_w is not None:
        print('# Uncalibrated energy estimate for power demand(s); no state update is applied')
        print('power_w,duration_s,load_energy_j,source_energy_j,load_energy_wh,source_energy_wh')
        total_source_j = 0.0
        for demand in args.power_demand_w:
            e = b.estimate_energy_drawn(demand, duration_s=args.duration_s)
            total_source_j += e['source_energy_j']
            print(f'{demand:g},{args.duration_s:g},{e["load_energy_j"]:.6f},{e["source_energy_j"]:.6f},'
                  f'{e["load_energy_wh"]:.6f},{e["source_energy_wh"]:.6f}')
        print(f'# total_source_energy_wh,{total_source_j/3600.0:.6f}')
        return
    print('# Uncalibrated simulation; default curves/aging are assumptions')
    print('time_s,soc,soh,current_a,voltage_v')
    for _ in range(args.steps):
        try: soc,soh = b.step(args.power_w)
        except BatteryLimitError as exc:
            print(f'# Episode terminated; failed step rolled back: {exc}')
            break
        print(f'{b.elapsed_s:g},{soc:.9f},{soh:.12f},'
              f'{b.last_info["current_a"]:.6f},{b.last_info["voltage_v"]:.6f}')


if __name__=='__main__':
    main()
