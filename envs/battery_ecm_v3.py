"""Battery-aware simulation: SOC/SOH/temperature-dependent 2RC ECM.

All five ECM elements (R0, R1, R2, C1, C2) have independent, explicit
functions of SOC, SOH, and cell/pack temperature. OCV can be supplied as
SOH-indexed measured curves or an explicitly uncalibrated aging perturbation.
No measured impedance, HPPC, OCV or aging calibration is claimed: default
coefficient values are illustrative.
Only discharge (nonnegative constant terminal power) is supported.

Energy bookkeeping separates resistor heat, frozen-parameter RC energy change,
and signed numerical/parameter residual. These are diagnostic channels, not a
thermal model. The optional current limit must be set from real pack/BMS data.
Default parameters and synthetic OCV are illustrative, not pack calibration.

At each electrical substep, the RC time constants are recomputed as
    tau1 = R1(SOC, SOH, T) * C1(SOC, SOH, T)
    tau2 = R2(SOC, SOH, T) * C2(SOC, SOH, T)

Reference state for ECM component values: SOC=0.5, SOH=1, T=25 C.

Default pack: Blue Robotics Lithium-ion Battery (4S6P, 18 Ah, 14.8 V,
266.4 Wh nominal energy, 12 V minimum safe voltage).
https://bluerobotics.com/store/comm-control-power/powersupplies-batteries/battery-li-4s-18ah-r3/
Pack ratings do not calibrate the assumed OCV, resistance, RC or aging curves.

Run demo: python battery_ecm_v3.py --capacity-ah 18 --nominal-voltage 14.8
Tests:    python battery_ecm_v3.py --self-test
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
                 cycles_to_80=1000.0, ocv_age_delta_v_at_soh80=-0.8,
                 *, aging_enabled=True, max_discharge_current_a=None,
                 cutoff_voltage=None, max_substep_s=0.1,
                 ocv_table=None, ocv_tables_by_soh=None,
                 ocv_temp_coeff_v_per_c=0.0):
        """Construct an illustrative pack-level discharge-only 2RC ECM.

        Optional OCV tables contain (SOC, pack_voltage_V) points, including
        exact SOC endpoints 0 and 1. The SOH-indexed form maps health to tables;
        values between curves are interpolated and outside are clamped. Tables
        take precedence over the synthetic OCV aging correction.

        Aging is disabled for a fixed-health mission with aging_enabled=False.
        None for the current limit means no explicit BMS/current protection;
        do not interpret it as a real battery safety guarantee.
        """
        numeric = (capacity_ah, nominal_voltage, soc_init, soh_init, temp_c,
                   cycles_to_80, ocv_age_delta_v_at_soh80, max_substep_s,
                   ocv_temp_coeff_v_per_c)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in numeric):
            raise ValueError('All numeric inputs must be finite')
        if capacity_ah <= 0 or nominal_voltage <= 0 or cycles_to_80 <= 0:
            raise ValueError('Capacity, nominal voltage and cycle life must be positive')
        if max_substep_s <= 0:
            raise ValueError('max_substep_s must be positive')
        if not 0.02 <= soc_init <= 1 or not 0.5 <= soh_init <= 1:
            raise ValueError('SOC must be in [0.02, 1] and SOH in [0.5, 1]')
        if not -10 <= temp_c <= 50:
            raise ValueError('Temperature must be in [-10, 50] C')
        if max_discharge_current_a is not None:
            if not math.isfinite(max_discharge_current_a) or max_discharge_current_a <= 0:
                raise ValueError('max_discharge_current_a must be positive or None')
        if cutoff_voltage is not None:
            if not math.isfinite(cutoff_voltage) or cutoff_voltage <= 0:
                raise ValueError('cutoff_voltage must be positive or None')
        if not isinstance(aging_enabled, bool):
            raise ValueError('aging_enabled must be a bool')
        if ocv_table is not None and ocv_tables_by_soh is not None:
            raise ValueError('Choose ocv_table OR ocv_tables_by_soh')

        self.capacity_ah = float(capacity_ah)
        self.nominal_voltage = float(nominal_voltage)
        self.soc = float(soc_init)
        self.soh = float(soh_init)
        self.temp_c = float(temp_c)
        self.cycles_to_80 = float(cycles_to_80)
        self.ocv_age_delta_v_at_soh80 = float(ocv_age_delta_v_at_soh80)
        self.ocv_temp_coeff_v_per_c = float(ocv_temp_coeff_v_per_c)
        self.aging_enabled = aging_enabled
        self.max_discharge_current_a = (None if max_discharge_current_a is None
                                        else float(max_discharge_current_a))
        self.cutoff_voltage = (self.nominal_voltage * (3.0 / 3.7)
                               if cutoff_voltage is None else float(cutoff_voltage))
        self.max_substep_s = float(max_substep_s)
        self._ocv_table = (None if ocv_table is None
                           else self._validated_curve(ocv_table))
        if ocv_tables_by_soh is None:
            self._ocv_tables_by_soh = None
        else:
            if not ocv_tables_by_soh:
                raise ValueError('ocv_tables_by_soh cannot be empty')
            tables = []
            for health, curve in ocv_tables_by_soh.items():
                health = float(health)
                if not math.isfinite(health) or not 0.5 <= health <= 1:
                    raise ValueError('OCV table SOH keys must be in [0.5, 1]')
                tables.append((health, self._validated_curve(curve)))
            self._ocv_tables_by_soh = tuple(sorted(tables))

        # Reference components correspond to SOC=0.5, SOH=1, T=25 C.
        # Rough 5 Ah -> pack scaling; replace with pack HPPC fit when available.
        scale = (self.nominal_voltage / 14.8) * (5.0 / self.capacity_ah)
        self.r0_ref, self.r1_ref, self.r2_ref = 0.035 * scale, 0.010 * scale, 0.005 * scale
        self.c1_ref = 7.5 / self.r1_ref
        self.c2_ref = 30.0 / self.r2_ref

        self.v1 = self.v2 = 0.0
        self.elapsed_s = 0.0
        self.discharge_ah = 0.0
        self.load_energy_j = 0.0
        self.source_energy_j = 0.0
        self.heat_energy_j = 0.0
        self.rc_storage_energy_j = 0.0
        self.power_balance_error_j = 0.0
        self.last_info = {}

    @staticmethod
    def _validated_curve(curve):
        try:
            pts = tuple((float(soc), float(voltage)) for soc, voltage in curve)
        except (TypeError, ValueError) as exc:
            raise ValueError('OCV curve must be pairs of (SOC, pack voltage)') from exc
        if (len(pts) < 2 or not math.isclose(pts[0][0], 0.0, abs_tol=1e-12)
                or not math.isclose(pts[-1][0], 1.0, abs_tol=1e-12)):
            raise ValueError('OCV curve needs at least two SOC points, from 0 to 1')
        if any(not math.isfinite(s) or not math.isfinite(v) or v <= 0
               for s, v in pts):
            raise ValueError('OCV table values must be finite and voltages positive')
        if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
            raise ValueError('OCV SOC points must be strictly increasing')
        return pts

    @staticmethod
    def _curve_voltage(curve, soc):
        for (sa, va), (sb, vb) in zip(curve, curve[1:]):
            if soc <= sb:
                return va + (vb - va) * (soc - sa) / (sb - sa)
        return curve[-1][1]

    def ocv(self, soc=None, soh=None, temp_c=None):
        """Pack OCV(SOC,SOH,T). Default curves and aging are uncalibrated.

        The default aging parameter equals the mid-SOC voltage shift at
        SOH=0.8 (before nominal-voltage scaling). Its default -0.8 V preserves
        the original supplied synthetic curve at every SOC/SOH value.
        """
        s = self.soc if soc is None else float(soc)
        h = self.soh if soh is None else float(soh)
        t = self.temp_c if temp_c is None else float(temp_c)
        if not (math.isfinite(s) and 0 <= s <= 1 and math.isfinite(h) and 0.5 <= h <= 1
                and math.isfinite(t) and -10 <= t <= 50):
            raise ValueError('Invalid SOC, SOH, or temperature')
        if self._ocv_tables_by_soh is not None:
            tables = self._ocv_tables_by_soh
            if h <= tables[0][0]:
                voltage = self._curve_voltage(tables[0][1], s)
            elif h >= tables[-1][0]:
                voltage = self._curve_voltage(tables[-1][1], s)
            else:
                voltage = None
                for (ha, ca), (hb, cb) in zip(tables, tables[1:]):
                    if h <= hb:
                        va = self._curve_voltage(ca, s)
                        vb = self._curve_voltage(cb, s)
                        voltage = va + (vb - va) * (h - ha) / (hb - ha)
                        break
        elif self._ocv_table is not None:
            voltage = self._curve_voltage(self._ocv_table, s)
        else:
            cell_v = self._curve_voltage(self._CELL_OCV, s)
            fresh_v = cell_v * self.nominal_voltage / 3.7
            age_factor = (1.0 - h) / 0.2
            soc_shape = 4.0 * s * (1.0 - s)
            # Original: age_factor*(-0.2*s - 0.7*soc_shape).
            # It equals -0.8 V at SOC=0.5 and SOH=0.8 for 14.8 V pack.
            shape = 0.25 * s + 0.875 * soc_shape
            voltage = (fresh_v + age_factor * self.ocv_age_delta_v_at_soh80
                       * shape * self.nominal_voltage / 14.8)
        return voltage + self.ocv_temp_coeff_v_per_c * (t - 25.0)

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
        if self.max_discharge_current_a is not None and current > self.max_discharge_current_a:
            raise BatteryLimitError('Discharge current exceeds configured limit')
        if voltage < self.cutoff_voltage:
            raise BatteryLimitError('Terminal voltage below cutoff')
        return current, voltage, ocv, p

    @staticmethod
    def _rc_branch_step(v_old, current, resistance, capacitance, dt):
        """Exact constant-I, frozen-R/C RC step: (V_new, heat J, storage J)."""
        tau = resistance * capacitance
        decay = math.exp(-dt / tau)
        one_minus_decay = -math.expm1(-dt / tau)
        steady = current * resistance
        transient = v_old - steady
        v_new = steady + transient * decay
        # Exact integral of V(t)^2/R dt; includes heat at zero load.
        heat_j = (steady**2 * dt
                  + 2 * steady * transient * tau * one_minus_decay
                  + transient**2 * tau * (-math.expm1(-2 * dt / tau)) / 2) / resistance
        storage_j = 0.5 * capacitance * (v_new**2 - v_old**2)
        return v_new, max(heat_j, 0.0), storage_j

    def estimate_energy_drawn(self, power_w, duration_s=1.0):
        """Fixed-start-state power estimate; DOES NOT advance RC or SOH.

        source-minus-load is electrical overhead, not resistor heat. The
        instantaneous heat channel is diagnostic, NOT integrated transient heat.
        """
        if not (math.isfinite(power_w) and power_w >= 0
                and math.isfinite(duration_s) and duration_s > 0):
            raise ValueError('Power must be >= 0; duration must be > 0')
        i, v, ocv, p = self._electrical(power_w, self.soc, self.soh,
                                      self.v1, self.v2, self.temp_c)
        load_j = power_w * duration_s
        source_j = ocv * i * duration_s
        instantaneous_heat_w = i**2 * p.r0 + self.v1**2 / p.r1 + self.v2**2 / p.r2
        return dict(current_a=i, voltage_v=v,
                    load_energy_j=load_j, source_energy_j=source_j,
                    load_energy_wh=load_j / 3600.0, source_energy_wh=source_j / 3600.0,
                    energy_loss_j=source_j - load_j,
                    instantaneous_heat_power_w=instantaneous_heat_w)

    def energy_loss(self, power_w, duration_s=1.0):
        """Legacy fixed-state source-minus-load overhead (J), NOT heat (J)."""
        return self.estimate_energy_drawn(power_w, duration_s)['energy_loss_j']

    def step(self, power_w, dt=1.0, temp_c=None):
        """Apply constant terminal power for dt seconds; return (SOC, SOH).

        SOH follows an illustrative throughput aging law when enabled.
        On ANY battery limit violation, no battery state/energy is committed.
        All heat/storage fields integrate electrical substeps with frozen
        R/C. Their sum plus a signed residual closes source-to-load energy.
        """
        t = self.temp_c if temp_c is None else float(temp_c)
        if not (math.isfinite(power_w) and power_w >= 0 and math.isfinite(dt) and dt > 0
                and math.isfinite(t) and -10 <= t <= 50):
            raise ValueError('Invalid power, timestep, or temperature')
        n = math.ceil(dt / self.max_substep_s)
        if n > 100000:
            raise ValueError('Timestep too long; split it into smaller calls')
        ds = dt / n
        soc, soh, v1, v2 = self.soc, self.soh, self.v1, self.v2
        source_j = heat_j = rc_storage_j = discharge_ah = 0.0
        for _ in range(n):
            i, v, ocv, p = self._electrical(power_w, soc, soh, v1, v2, t)
            da = i * ds / 3600.0
            soc -= da / (self.capacity_ah * soh)
            if soc < 0.02:
                raise BatteryLimitError('SOC reserve reached')
            next_v1, heat1, store1 = self._rc_branch_step(v1, i, p.r1, p.c1, ds)
            next_v2, heat2, store2 = self._rc_branch_step(v2, i, p.r2, p.c2, ds)
            v1, v2 = next_v1, next_v2
            if self.aging_enabled:
                c_rate = i / (self.capacity_ah * soh)
                stress = 2 ** ((t - 25) / 10) * (1 + 0.2 * c_rate**2)
                soh -= (0.2 / self.cycles_to_80) * (da / self.capacity_ah) * stress
                if soh < 0.5:
                    raise BatteryLimitError('SOH below modeled range')
            source_j += ocv * i * ds
            heat_j += i**2 * p.r0 * ds + heat1 + heat2
            rc_storage_j += store1 + store2
            discharge_ah += da

        i, v, ocv, p = self._electrical(power_w, soc, soh, v1, v2, t)
        load_j = power_w * dt
        balance_j = source_j - load_j - heat_j - rc_storage_j
        # Commit only once, after the entire requested step is feasible.
        self.soc, self.soh, self.v1, self.v2, self.temp_c = soc, soh, v1, v2, t
        self.elapsed_s += dt
        self.discharge_ah += discharge_ah
        self.load_energy_j += load_j
        self.source_energy_j += source_j
        self.heat_energy_j += heat_j
        self.rc_storage_energy_j += rc_storage_j
        self.power_balance_error_j += balance_j
        self.last_info = dict(
            soc=soc, soh=soh, voltage_v=v, current_a=i, ocv_v=ocv,
            r0_ohm=p.r0, r1_ohm=p.r1, r2_ohm=p.r2, c1_f=p.c1, c2_f=p.c2,
            tau1_s=p.tau1, tau2_s=p.tau2,
            load_energy_j=load_j, source_energy_j=source_j,
            energy_loss_j=source_j - load_j,  # Electrical overhead, NOT heat.
            heat_energy_j=heat_j,
            rc_storage_energy_j=rc_storage_j,
            power_balance_error_j=balance_j,
            load_power_w=power_w,
            source_power_w=source_j / dt,
            internal_loss_power_w=heat_j / dt,
            rc_storage_power_w=rc_storage_j / dt,
            power_balance_error_w=balance_j / dt,
            discharge_ah=discharge_ah,
            cumulative_discharge_ah=self.discharge_ah,
            cumulative_load_energy_j=self.load_energy_j,
            cumulative_source_energy_j=self.source_energy_j,
            cumulative_heat_energy_j=self.heat_energy_j,
            cumulative_rc_storage_energy_j=self.rc_storage_energy_j,
            cumulative_power_balance_error_j=self.power_balance_error_j,
        )
        return soc, soh


def self_test():
    """Dependency-free contract and physical-accounting regression tests."""
    import copy

    def new(**kw):
        return Battery2RCECM(18.0, 14.8, **kw)

    # Discharge, scalar electrical power identity, positive/source energy.
    for power in (0, 5, 25, 42):
        b = new()
        for _ in range(100):
            previous_soc, previous_soh = b.soc, b.soh
            soc, soh = b.step(power)
            assert soc <= previous_soc and soh <= previous_soh
            assert math.isclose(b.last_info['current_a'] * b.last_info['voltage_v'],
                                power, abs_tol=1e-10)
        assert math.isclose(b.load_energy_j, power * 100)
        assert math.isclose(b.source_energy_j, b.load_energy_j + b.heat_energy_j
                            + b.rc_storage_energy_j + b.power_balance_error_j,
                            abs_tol=1e-8)
        e = b.estimate_energy_drawn(power, duration_s=10)
        assert math.isclose(e['source_energy_wh'], e['source_energy_j'] / 3600)
        assert math.isclose(e['load_energy_wh'], power * 10 / 3600)
        assert math.isclose(b.energy_loss(power, 10), e['energy_loss_j'])
        assert e['instantaneous_heat_power_w'] >= 0

    # Frozen aging, synthetic aging curve scale, and custom OCV curve support.
    b = new(aging_enabled=False)
    b.step(25)
    assert b.soh == 1
    reference_ocv = b.ocv(soc=0.5, soh=0.8)
    assert new(ocv_age_delta_v_at_soh80=0).ocv(soc=0.5, soh=0.8) > reference_ocv
    assert math.isclose(new(ocv_age_delta_v_at_soh80=-0.8).ocv(soc=0.5, soh=0.8),
                        reference_ocv)
    # Original synthetic curve has a -0.2 V shift at SOC=1, SOH=.8 (14.8V pack).
    assert math.isclose(new().ocv(soc=1, soh=1) - new().ocv(soc=1, soh=.8),
                        0.2, abs_tol=1e-12)
    measured = new(ocv_tables_by_soh={1.0: [(0, 12), (1, 16)],
                                      0.8: [(0, 12), (1, 15)]})
    assert math.isclose(measured.ocv(soc=.5, soh=.9), 13.75)
    assert math.isclose(measured.ocv(soc=.5, soh=.6), 13.5)
    assert new(ocv_table=[(0, 15), (1, 15)]).ocv(soc=.5, soh=.8) == 15
    assert math.isclose(new(ocv_temp_coeff_v_per_c=.005).ocv(soc=.5, temp_c=35)
                        - new().ocv(soc=.5, temp_c=25), .05, abs_tol=1e-12)

    # Limits must be transactional (no mutation even if near reserve on substep).
    b = new()
    prior = copy.deepcopy(b.__dict__)
    try:
        b.step(1e6)
    except BatteryLimitError:
        pass
    else:
        raise AssertionError('Overload accepted')
    assert b.__dict__ == prior
    b = new(soc_init=.0201, cutoff_voltage=1)
    prior = copy.deepcopy(b.__dict__)
    try:
        b.step(25, dt=10)
    except BatteryLimitError:
        pass
    else:
        raise AssertionError('SOC reserve violation accepted')
    assert b.__dict__ == prior
    b = new(max_discharge_current_a=2)
    prior = copy.deepcopy(b.__dict__)
    try:
        b.step(100)
    except BatteryLimitError:
        pass
    else:
        raise AssertionError('Current limit violated')
    assert b.__dict__ == prior

    fresh, old = new(aging_enabled=False), new(soh_init=.8, aging_enabled=False)
    for _ in range(600):
        fresh.step(30)
        old.step(30)
    assert old.soc < fresh.soc
    assert old.parameters().r0 > fresh.parameters().r0
    assert old.source_energy_j > fresh.source_energy_j

    # RC relaxation with zero load produces positive heat and negative storage.
    b = new(aging_enabled=False)
    b.step(25)
    previous_v1 = b.v1
    b.step(0)
    assert 0 < b.v1 < previous_v1
    assert b.last_info['heat_energy_j'] > 0
    assert b.last_info['rc_storage_energy_j'] < 0
    assert math.isclose(b.last_info['heat_energy_j']
                        + b.last_info['rc_storage_energy_j'], 0, abs_tol=1e-8)

    # Every ECM element depends on SOC, SOH, T; RC time constants stay positive.
    for name in ('r0', 'r1', 'r2', 'c1', 'c2'):
        base = getattr(b.parameters(soc=.5, soh=1, temp_c=25), name)
        assert getattr(b.parameters(soc=.2, soh=1, temp_c=25), name) != base
        assert getattr(b.parameters(soc=.5, soh=.8, temp_c=25), name) != base
        assert getattr(b.parameters(soc=.5, soh=1, temp_c=5), name) != base
    for soc in (.05, .3, .8):
        for soh in (.6, .85, 1.):
            for temp in (-10, 5, 25, 50):
                p = b.parameters(soc=soc, soh=soh, temp_c=temp)
                assert all(getattr(p, name) > 0 for name in
                           ('r0', 'r1', 'r2', 'c1', 'c2', 'tau1', 'tau2'))
                assert math.isclose(p.tau1, p.r1 * p.c1)
                assert math.isclose(p.tau2, p.r2 * p.c2)

    a, c = new(max_substep_s=.1), new(max_substep_s=.01)
    for power in (5, 42, 10, 30) * 20:
        a.step(power)
        c.step(power)
    assert abs(a.soc - c.soc) < 1e-5 and abs(a.v1 - c.v1) < 1e-4
    b = new(aging_enabled=False, ocv_table=[(0, 15), (1, 15)])
    initial_soc = b.soc
    b.step(25)
    assert math.isclose(initial_soc - b.soc, b.discharge_ah / (18 * b.soh),
                        abs_tol=1e-12)
    for bad in (-1, float('nan'), float('inf')):
        try:
            b.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid power accepted')
    print('PASS: SOC/SOH/T, OCV calibration lookup, RC dynamics, energy/heat/storage '
          'accounting, aging switch, current limit, rollback and convergence')


def main():
    parser = argparse.ArgumentParser(description='Uncalibrated 2RC battery simulation')
    parser.add_argument('--capacity-ah',type=float,default=18.0)
    parser.add_argument('--nominal-voltage',type=float,default=14.8)
    parser.add_argument('--soh-init',type=float,default=0.85)
    parser.add_argument('--soc-init',type=float,default=1.0)
    parser.add_argument('--temp-c',type=float,default=25.0)
    parser.add_argument('--dt',type=float,default=1.0)
    parser.add_argument('--max-current-a',type=float,default=None,
                        help='Optional limit; set from actual pack/BMS specification')
    parser.add_argument('--disable-aging',action='store_true')
    parser.add_argument('--max-substep-s',type=float,default=0.1)
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
    b = Battery2RCECM(args.capacity_ah,args.nominal_voltage,
                      soc_init=args.soc_init, soh_init=args.soh_init, temp_c=args.temp_c,
                      aging_enabled=not args.disable_aging,
                      max_discharge_current_a=args.max_current_a,
                      max_substep_s=args.max_substep_s)
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
        try: soc,soh = b.step(args.power_w, dt=args.dt)
        except BatteryLimitError as exc:
            print(f'# Episode terminated; failed step rolled back: {exc}')
            break
        print(f'{b.elapsed_s:g},{soc:.9f},{soh:.12f},'
              f'{b.last_info["current_a"]:.6f},{b.last_info["voltage_v"]:.6f}')


if __name__=='__main__':
    main()
