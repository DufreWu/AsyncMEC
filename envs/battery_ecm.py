"""Battery-aware simulation: state-dependent 2RC ECM, standard library only.

All five ECM elements (R0, R1, R2, C1, C2) have independent, explicit
functions of SOC, SOH, and cell/pack temperature. No measured impedance,
HPPC, OCV or aging calibration is claimed: coefficient values are illustrative.
Only discharge (nonnegative constant terminal power) is supported.

At each electrical substep, the RC time constants are recomputed as
    tau1 = R1(SOC, SOH, T) * C1(SOC, SOH, T)
    tau2 = R2(SOC, SOH, T) * C2(SOC, SOH, T)

Reference state for ECM component values: SOC=0.5, SOH=1, T=25 C.

Default pack: Blue Robotics Lithium-ion Battery (4S6P, 18 Ah, 14.8 V,
266.4 Wh nominal energy, 12 V minimum safe voltage).
https://bluerobotics.com/store/comm-control-power/powersupplies-batteries/battery-li-4s-18ah-r3/
Pack ratings do not calibrate the assumed OCV, resistance, RC or aging curves.

Run demo: python envs/battery_ecm.py --capacity-ah 18 --nominal-voltage 14.8
Tests:    python envs/battery_ecm.py --self-test
Reference for ECM structure only (NOT default numerical calibration):
https://www.mathworks.com/help/simscape-battery/ref/batteryequivalentcircuit.html
"""
import argparse
import math
from dataclasses import dataclass


def finite(name, value, lower=None, upper=None):
    v = float(value)
    if not math.isfinite(v) or (lower is not None and v < lower) or (upper is not None and v > upper):
        raise ValueError(f"{name}: invalid value {value}; expected finite in [{lower}, {upper}]")
    return v


def positive(name, value):
    v = finite(name, value)
    if v <= 0:
        raise ValueError(f"{name} must be positive")
    return v


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
    def __init__(self, capacity_ah=18.0, nominal_voltage=14.8, soc_init=1.0,
                 soh_init=1.0, temp_c=25.0, *, aging=True,
                 cycles_to_80=1000.0, resistance_growth=3.0,
                 rate_stress=0.2, r0_ref=0.035, r1_ref=0.010,
                 r2_ref=0.005, tau1_ref=7.5, tau2_ref=30.0,
                 resistance_scale=None, max_substep_s=0.1,
                 soc_min=0.02, cutoff_voltage=None, ocv_table=None):
        self.capacity_ah = positive('capacity_ah', capacity_ah)
        self.nominal_voltage = positive('nominal_voltage', nominal_voltage)
        self.soc_min = finite('soc_min', soc_min, 0, 0.99)
        self.soc = finite('soc_init', soc_init, self.soc_min, 1)
        self.soh = finite('soh_init', soh_init, 0.5, 1)
        self.temp_c = finite('temp_c', temp_c, -10, 50)
        self.aging = bool(aging)
        self.cycles_to_80 = positive('cycles_to_80', cycles_to_80)
        self.resistance_growth = finite('resistance_growth', resistance_growth, 0)
        self.rate_stress = finite('rate_stress', rate_stress, 0)
        self.r0_ref = positive('r0_ref', r0_ref)
        self.r1_ref = positive('r1_ref', r1_ref)
        self.r2_ref = positive('r2_ref', r2_ref)
        self.tau1_ref = positive('tau1_ref', tau1_ref)
        self.tau2_ref = positive('tau2_ref', tau2_ref)
        # Resistance references retain their assumed 5 Ah / 14.8 V baseline.
        self.resistance_scale = positive('resistance_scale',
            (self.nominal_voltage/14.8)*(5/self.capacity_ah)
            if resistance_scale is None else resistance_scale)
        self.max_substep_s = positive('max_substep_s', max_substep_s)
        self.cutoff_voltage = positive('cutoff_voltage',
            self.nominal_voltage*3.0/3.7 if cutoff_voltage is None else cutoff_voltage)
        # Optional custom table contains (SOC, PACK OCV in V), not cell voltages.
        if ocv_table is None:
            cell_table = [(0,3.0),(.05,3.3),(.1,3.5),(.2,3.6),(.4,3.7),
                          (.6,3.8),(.8,3.95),(.9,4.08),(1,4.2)]
            ocv_table = [(s,v*self.nominal_voltage/3.7) for s,v in cell_table]
        self.ocv_table = tuple((finite('table SOC',s,0,1),positive('table OCV',v))
                               for s,v in ocv_table)
        if (len(self.ocv_table)<2 or self.ocv_table[0][0]!=0 or self.ocv_table[-1][0]!=1
            or any(b[0]<=a[0] or b[1]<a[1] for a,b in zip(self.ocv_table,self.ocv_table[1:]))):
            raise ValueError('OCV table must span SOC 0..1 with increasing SOC and nondecreasing voltage')
        self.v1 = self.v2 = self.elapsed_s = self.discharge_ah = 0.0
        self.load_energy_j = self.source_energy_j = 0.0
        self.last_info = {}

    def ocv(self, soc=None):
        s = self.soc if soc is None else finite('soc', soc, 0, 1)
        for (a,va),(b,vb) in zip(self.ocv_table,self.ocv_table[1:]):
            if s <= b:
                return va+(vb-va)*(s-a)/(b-a)
        return self.ocv_table[-1][1]

    def parameters(self, soc=None, soh=None, temp_c=None):
        s = self.soc if soc is None else finite('soc',soc,0,1)
        h = self.soh if soh is None else finite('soh',soh,.5,1)
        t = self.temp_c if temp_c is None else finite('temp_c',temp_c,-10,50)
        low = max(0.0, (0.2-s)/0.2)
        scale = (self.resistance_scale * (1+1.5*low**2)
                 * math.exp(.025*(25-t)) * (1+self.resistance_growth*(1-h)))
        r0, r1, r2 = self.r0_ref*scale, self.r1_ref*scale, self.r2_ref*scale
        tau_scale = (1+0.5*low)*math.exp(.01*(25-t))
        tau1, tau2 = self.tau1_ref*tau_scale, self.tau2_ref*tau_scale
        return ECMParameters(r0,r1,r2,tau1/r1,tau2/r2,tau1,tau2)

    def _electrical(self, power, soc, soh, v1, v2, temp):
        p = self.parameters(soc,soh,temp)
        voc = self.ocv(soc)
        net = voc-v1-v2
        disc = net*net-4*p.r0*power
        if net <= 0 or disc < 0:
            raise BatteryLimitError('Requested power exceeds ECM deliverable power')
        current = 2*power/(net+math.sqrt(disc))
        voltage = net-current*p.r0
        if voltage < self.cutoff_voltage:
            raise BatteryLimitError('Terminal voltage below cutoff')
        return current, voltage, voc, p

    def estimate_energy_drawn(self, soc_or_power, power_w=None, duration_s=1.0, temp_c=None):
        """Estimate battery source and load energy for a constant power demand over time.

        Supported call patterns:
          - estimate_energy_drawn(power_w, duration_s=...)  # uses current SOC
          - estimate_energy_drawn(soc, power_w, duration_s=...)  # uses explicit SOC
        """
        if power_w is None:
            soc = self.soc
            power = finite('power_w', soc_or_power, 0)
        else:
            soc = finite('soc', soc_or_power, self.soc_min, 1)
            power = finite('power_w', power_w, 0)
        dt = positive('duration_s', duration_s)
        temp = self.temp_c if temp_c is None else finite('temp_c',temp_c,-10,50)
        current, voltage, voc, _ = self._electrical(power, soc, self.soh, self.v1, self.v2, temp)
        load_energy_j = power*dt
        source_energy_j = max(voc*current*dt, 0.0)
        energy_loss_j = source_energy_j - load_energy_j
        return dict(
            soc=soc,
            power_w=power,
            duration_s=dt,
            current_a=current,
            voltage_v=voltage,
            ocv_v=voc,
            load_energy_j=load_energy_j,
            source_energy_j=source_energy_j,
            energy_loss_j=energy_loss_j,
            load_energy_wh=load_energy_j/3600.0,
            source_energy_wh=source_energy_j/3600.0,
            energy_loss_wh=energy_loss_j/3600.0,
        )

    def energy_loss(self, soc_or_power, power_w=None, duration_s=1.0, temp_c=None):
        """Return the internal battery energy loss for a given SOC and power demand.

        Supported call patterns:
          - energy_loss(power_w, duration_s=...)  # uses current SOC
          - energy_loss(soc, power_w, duration_s=...)  # explicit SOC input
        """
        estimate = self.estimate_energy_drawn(soc_or_power, power_w, duration_s=duration_s, temp_c=temp_c)
        return dict(
            soc=estimate['soc'],
            power_w=estimate['power_w'],
            duration_s=estimate['duration_s'],
            load_energy_j=estimate['load_energy_j'],
            source_energy_j=estimate['source_energy_j'],
            energy_loss_j=estimate['energy_loss_j'],
            load_energy_wh=estimate['load_energy_wh'],
            source_energy_wh=estimate['source_energy_wh'],
            energy_loss_wh=estimate['energy_loss_wh'],
        )

    def step(self, power_w, dt=1.0, temp_c=None):
        """Return (SOC, SOH); constant terminal load over dt. All-or-nothing update."""
        power = finite('power_w', power_w, 0)
        dt = positive('dt', dt)
        temp = self.temp_c if temp_c is None else finite('temp_c',temp_c,-10,50)
        count = math.ceil(dt/self.max_substep_s)
        if count > 100000:
            raise ValueError('dt too large: split request into smaller steps')
        delta = dt/count
        soc, soh, v1, v2 = self.soc, self.soh, self.v1, self.v2
        ah = source_j = 0.0
        min_voltage = float('inf')
        for _ in range(count):
            current, voltage, voc, p = self._electrical(power,soc,soh,v1,v2,temp)
            min_voltage = min(min_voltage,voltage)
            da = current*delta/3600
            soc -= da/(self.capacity_ah*soh)
            if soc < self.soc_min:
                raise BatteryLimitError('SOC reserve reached')
            v1 = v1*math.exp(-delta/p.tau1)+current*p.r1*(-math.expm1(-delta/p.tau1))
            v2 = v2*math.exp(-delta/p.tau2)+current*p.r2*(-math.expm1(-delta/p.tau2))
            if self.aging:
                c_rate = current/(self.capacity_ah*soh)
                stress = 2**((temp-25)/10)*(1+self.rate_stress*c_rate*c_rate)
                soh -= (.2/self.cycles_to_80)*(da/self.capacity_ah)*stress
                if soh < .5:
                    raise BatteryLimitError('SOH below supported model domain (0.5)')
            ah += da
            source_j += voc*current*delta
        current, voltage, voc, p = self._electrical(power,soc,soh,v1,v2,temp)
        min_voltage = min(min_voltage,voltage)
        self.soc, self.soh, self.v1, self.v2 = soc,soh,v1,v2
        self.temp_c = temp
        self.elapsed_s += dt
        self.discharge_ah += ah
        self.load_energy_j += power*dt
        self.source_energy_j += source_j
        self.last_info = dict(soc=soc,soh=soh,current_a=current,
            mean_current_a=ah*3600/dt, voltage_v=voltage,min_voltage_v=min_voltage,
            ocv_v=voc, effective_capacity_ah=self.capacity_ah*soh,
            load_power_w=power, load_energy_j=power*dt,
            source_energy_j=source_j, elapsed_s=self.elapsed_s,
            cumulative_load_energy_j=self.load_energy_j,
            cumulative_source_energy_j=self.source_energy_j,
            discharge_equivalent_cycles=self.discharge_ah/self.capacity_ah,
            r0_ohm=p.r0,r1_ohm=p.r1,r2_ohm=p.r2,c1_f=p.c1,c2_f=p.c2,
            model_status='uncalibrated_simulation')
        return soc,soh


def self_test():
    import copy
    def new(**kw): return Battery2RCECM(5,14.8,**kw)
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
    assert b.parameters(soc=.05).r0>b.parameters(soc=.8).r0
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
    print('PASS: power balance, SOC/SOH monotonicity, aging toggle, rollback, RC relaxation, aging comparison, coulomb counting, numerical convergence and input validation')


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
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.steps<0: parser.error('steps must be nonnegative')
    b = Battery2RCECM(args.capacity_ah,args.nominal_voltage,soh_init=args.soh_init)
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
