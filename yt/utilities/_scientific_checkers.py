"""Opt-in scientific runtime checks used by the SciBench pilot.

The module is inert unless ``SCIBENCH_TRIGGER_LOG`` names a log file. Checks
append stable IDs and never raise into yt, never change a return value and
never touch global random state. Re-calling checkers run with checking
suppressed (re-entrancy guard).

Tolerance convention (SANITIZER.md 5.8): ``_C * eps(dtype) * size * scale`` with
each factor justified at the call site. Laws are documented in the pilot's
``LAW_CANDIDATES.md``; IDs are ``YT-<AREA>-<NNN>``.

All imports of yt itself are lazy (inside functions) so that this module sits
outside the package import layering.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import warnings

import numpy as np

_ACTIVE = False
_EPS = float(np.finfo(np.float64).eps)
_C = 64.0
_ENV = "SCIBENCH_TRIGGER_LOG"
_MAX_ELEMENTS = 2_000_000
_MU_0 = 1.25663706212e-6  # CODATA 2018 vacuum permeability, N A^-2
_C_LIGHT_CM = 2.99792458e10  # defined


def enabled():
    return bool(os.environ.get(_ENV)) and not _ACTIVE


def trigger(checker_id):
    path = os.environ.get(_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"checker_id": checker_id}) + "\n")
    except Exception:
        _swallowed()


def trigger_if(condition, checker_id):
    _reach(checker_id)
    if bool(condition):
        trigger(checker_id)


SWALLOWED = []
REACHED = {}


def _reach(checker_id):
    """Curator-only: count predicate evaluations (observation reachability)."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        REACHED[checker_id] = REACHED.get(checker_id, 0) + 1


def _swallowed():
    """Curator-only: record exceptions swallowed inside a checker."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        import traceback

        SWALLOWED.append(traceback.format_exc())


def _guarded(fn):
    """Run ``fn`` once, never nested, never raising into yt."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _ACTIVE
        if _ACTIVE or not os.environ.get(_ENV):
            return None
        _ACTIVE = True
        logger = logging.getLogger("yt")
        level = logger.level
        logger.setLevel(logging.CRITICAL + 1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    return fn(*args, **kwargs)
        except Exception:
            _swallowed()
            return None
        finally:
            logger.setLevel(level)
            _ACTIVE = False

    return wrapper


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _a(x):
    """Unit-stripped float64 ndarray."""
    return np.asarray(getattr(x, "d", x), dtype=np.float64)


def _val(q, unit):
    return np.asarray(q.to_value(unit), dtype=np.float64)


_GL = {}


def _gl(n):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]


def _quad(f, a, b, panels=64, nodes=24):
    """Composite Gauss-Legendre of a vectorised ``f`` on the scalar interval ``[a, b]``."""
    x, w = _gl(nodes)
    edges = np.linspace(a, b, panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    pts = mid + half * x[None, :]
    return float(np.sum(half * w[None, :] * f(pts)))


def _wrap_pi(x):
    return (np.asarray(x, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _small(*arrays):
    return all(np.size(x) <= _MAX_ELEMENTS for x in arrays)


# ---------------------------------------------------------------------------
# A. cosmology (utilities/cosmology.py)
# ---------------------------------------------------------------------------


def _cosmo_ok(co):
    vals = (co.omega_matter, co.omega_lambda, co.omega_radiation, co.omega_curvature)
    return all(np.isfinite(v) for v in vals) and co.omega_matter > 0


def _h0_kms_mpc(co):
    return float(co.hubble_constant.to_value("km/s/Mpc"))


def _zs(*zs, limit=64):
    """Broadcast scalar/array redshifts to 1-D float arrays (None if too large)."""
    b = np.broadcast_arrays(*[np.asarray(_a(z), dtype=np.float64) for z in zs])
    if b[0].size > limit:
        return None
    return [np.atleast_1d(x).ravel() for x in b]


def _chi(co, z):
    """Dimensionless line-of-sight comoving distance int_0^z dz'/E for one redshift (checker's quadrature)."""
    if z <= 0:
        return 0.0

    def f(u):
        zz = np.exp(u) - 1.0
        return np.exp(u) / co.expansion_factor(zz)

    return _quad(f, 0.0, math.log1p(z), panels=32, nodes=24)


def _m_of_chi(co, chi):
    ok = co.omega_curvature
    if ok > 0:
        s = math.sqrt(ok)
        return math.sinh(s * chi) / s
    if ok < 0:
        s = math.sqrt(-ok)
        return math.sin(s * chi) / s
    return chi


def _closed_branch_ok(co, chi):
    ok = co.omega_curvature
    return ok >= 0 or math.sqrt(-ok) * chi < math.pi / 2


@_guarded
def check_critical_density(co, z, result):
    """YT-COS-001"""
    if not _cosmo_ok(co):
        return
    from yt.utilities.physical_ratios import rho_crit_g_cm3_h2

    (z,) = _zs(z)
    e = np.asarray(co.expansion_factor(z), dtype=np.float64)
    rho = np.atleast_1d(_val(result, "g/cm**3"))
    h = _h0_kms_mpc(co) / 100.0
    good = np.isfinite(e) & (e > 0) & np.isfinite(rho)
    if not good.any():
        return
    ratio = rho[good] / e[good] ** 2 / (rho_crit_g_cm3_h2 * h**2)
    trigger_if(np.any(np.abs(ratio - 1.0) > 2e-6), "YT-COS-001")


@_guarded
def check_hubble_distance(co, result):
    """YT-COS-002"""
    h0 = _h0_kms_mpc(co)
    if not (np.isfinite(h0) and h0 > 0):
        return
    expected = 299792.458 / h0
    got = float(_val(result, "Mpc"))
    trigger_if(abs(got / expected - 1.0) > 1e-9, "YT-COS-002")


@_guarded
def check_lookback_time(co, z_i, z_f, result):
    """YT-COS-003"""
    if not _cosmo_ok(co):
        return
    zz = _zs(z_i, z_f)
    if zz is None:
        return
    zi, zf = zz
    keep = (zi >= 0) & (zf > zi) & (zf <= 999.0)
    res = np.atleast_1d(_val(result, "s"))
    if res.size != zi.size:
        return
    for k in np.nonzero(keep)[0]:
        e = co.expansion_factor(np.array([zi[k], zf[k]]))
        if not np.all(np.isfinite(e)) or np.any(e <= 0):
            continue
        t_i = float(_val(co.t_from_z(zi[k]), "s"))
        t_f = float(_val(co.t_from_z(zf[k]), "s"))
        trigger_if(abs(res[k] - (t_i - t_f)) > 5e-5 * abs(t_i), "YT-COS-003")


@_guarded
def check_age_inverse(co, t, a_result):
    """YT-COS-004"""
    if not _cosmo_ok(co):
        return
    a = np.atleast_1d(_a(a_result))
    tt = np.atleast_1d(_val(t if hasattr(t, "to_value") else co.arr(t, "s"), "s"))
    if a.size > 256 or a.size != tt.size:
        return
    keep = np.isfinite(a) & (a >= 1e-2) & (a <= 10.0)
    for k in np.nonzero(keep)[0]:
        t2 = float(_val(co.t_from_a(a[k]), "s"))
        trigger_if(abs(t2 / tt[k] - 1.0) > 5e-5, "YT-COS-004")


@_guarded
def check_age_flat_lcdm(co, a, result):
    """YT-COS-005"""
    om, ol = co.omega_matter, co.omega_lambda
    if not (
        co.omega_radiation == 0
        and co.omega_curvature == 0
        and not co.use_dark_factor
        and om >= 0.05
        and ol >= 0.05
        and abs(om + ol - 1.0) < 1e-9
    ):
        return
    a = np.atleast_1d(_a(a))
    res = np.atleast_1d(_val(result, "s"))
    if a.size != res.size or a.size > 4096:
        return
    keep = np.isfinite(a) & (a >= 1e-2) & (a <= 10.0)
    h0 = float(co.hubble_constant.to_value("1/s"))
    t_an = 2.0 / (3.0 * h0 * math.sqrt(ol)) * np.arcsinh(math.sqrt(ol / om) * a**1.5)
    trigger_if(np.any(np.abs(res[keep] - t_an[keep]) > 5e-5 * t_an[keep]), "YT-COS-005")


@_guarded
def check_transverse_order(co, z_i, z_f, result):
    """YT-COS-006"""
    if not _cosmo_ok(co):
        return
    zz = _zs(z_i, z_f)
    if zz is None or np.any(zz[1] <= zz[0]):
        return
    dc = np.atleast_1d(_val(co.comoving_radial_distance(z_i, z_f), "Mpc"))
    dm = np.atleast_1d(_val(result, "Mpc"))
    tol = _C * _EPS * np.abs(dc)
    ok = co.omega_curvature
    if ok > 0:
        bad = dm < dc - tol
    elif ok < 0:
        bad = dm > dc + tol
    else:
        bad = np.abs(dm - dc) > tol
    trigger_if(np.any(bad & np.isfinite(dm) & np.isfinite(dc)), "YT-COS-006")


@_guarded
def check_angular_diameter(co, z_i, z_f, result):
    """YT-COS-007"""
    if not _cosmo_ok(co):
        return
    zz = _zs(z_i, z_f, limit=16)
    if zz is None:
        return
    zi, zf = zz
    res = np.atleast_1d(_val(result, "Mpc"))
    if res.size != zi.size:
        return
    d_h = 299792.458 / _h0_kms_mpc(co)
    ok = co.omega_curvature
    for k in range(zi.size):
        if not (0 <= zi[k] < zf[k] <= 1e4):
            continue
        c1, c2 = _chi(co, zi[k]), _chi(co, zf[k])
        if not (np.isfinite(c1) and np.isfinite(c2)) or not _closed_branch_ok(co, c2):
            continue
        m1, m2 = _m_of_chi(co, c1), _m_of_chi(co, c2)
        t1 = m2 * math.sqrt(1.0 + ok * m1**2)
        t2 = m1 * math.sqrt(1.0 + ok * m2**2)
        expected = d_h * (t1 - t2) / (1.0 + zf[k])
        tol = 1e-6 * d_h * (abs(t1) + abs(t2)) / (1.0 + zf[k])
        trigger_if(abs(res[k] - expected) > tol, "YT-COS-007")


@_guarded
def check_luminosity_reciprocity(co, z_i, z_f, result):
    """YT-COS-008"""
    if not _cosmo_ok(co):
        return
    zz = _zs(z_i, z_f, limit=16)
    if zz is None:
        return
    zi, zf = zz
    res = np.atleast_1d(_val(result, "Mpc"))
    if res.size != zi.size:
        return
    for k in range(zi.size):
        if not (0 <= zi[k] < zf[k]):
            continue
        d_a = float(_val(co.angular_diameter_distance(zi[k], zf[k]), "Mpc"))
        d_m2 = float(_val(co.comoving_transverse_distance(0, zf[k]), "Mpc"))
        d_m1 = float(_val(co.comoving_transverse_distance(0, zi[k]), "Mpc"))
        r2 = ((1.0 + zf[k]) / (1.0 + zi[k])) ** 2
        tol = 1e-9 * (abs(d_m2) * (1.0 + zf[k]) + abs(d_m1) * (1.0 + zi[k]))
        trigger_if(abs(res[k] - r2 * d_a) > tol, "YT-COS-008")


def _volume_independent(co, z_i, z_f):
    """4 pi D_H^3 int m(z)^2 / E dz from the checker's own nested quadrature, in Mpc^3 (None if outside the fence)."""
    d_h = 299792.458 / _h0_kms_mpc(co)
    u_i, u_f = math.log1p(z_i), math.log1p(z_f)
    x_out, w_out = _gl(12)
    edges = np.linspace(u_i, u_f, 33)
    lo, hi = edges[:-1, None], edges[1:, None]
    u = (0.5 * (lo + hi) + 0.5 * (hi - lo) * x_out[None, :]).ravel()
    w = (0.5 * (hi - lo) * w_out[None, :] * np.ones_like(lo)).ravel()
    x_in, w_in = _gl(48)
    up = 0.5 * u[:, None] * (x_in[None, :] + 1.0)
    chi = 0.5 * u * np.sum(w_in[None, :] * np.exp(up) / co.expansion_factor(np.exp(up) - 1.0), axis=1)
    ok = co.omega_curvature
    if ok < 0 and np.sqrt(-ok) * chi.max() >= math.pi / 2:
        return None
    if ok > 0:
        m = np.sinh(math.sqrt(ok) * chi) / math.sqrt(ok)
    elif ok < 0:
        m = np.sin(math.sqrt(-ok) * chi) / math.sqrt(-ok)
    else:
        m = chi
    integrand = m**2 * np.exp(u) / co.expansion_factor(np.exp(u) - 1.0)
    return 4.0 * math.pi * d_h**3 * float(np.sum(w * integrand))


@_guarded
def check_volume_integral(co, z_i, z_f, result):
    """YT-COS-009"""
    if not _cosmo_ok(co):
        return
    zz = _zs(z_i, z_f, limit=4)
    if zz is None:
        return
    zi, zf = zz
    res = np.atleast_1d(_val(result, "Mpc**3"))
    if res.size != zi.size:
        return
    for k in range(zi.size):
        if not (0 <= zi[k] < zf[k] <= 20.0):
            continue
        v = _volume_independent(co, zi[k], zf[k])
        if v is None or not np.isfinite(v) or v <= 0:
            continue
        trigger_if(abs(res[k] / v - 1.0) > 1e-5, "YT-COS-009")


@_guarded
def check_volume_additivity(co, z_i, z_f, result):
    """YT-COS-010"""
    if not _cosmo_ok(co):
        return
    zz = _zs(z_i, z_f, limit=4)
    if zz is None:
        return
    zi, zf = zz
    res = np.atleast_1d(_val(result, "Mpc**3"))
    if res.size != zi.size:
        return
    for k in range(zi.size):
        if not (0 <= zi[k] < zf[k] <= 20.0):
            continue
        if not _closed_branch_ok(co, _chi(co, zf[k])):
            continue
        zm = math.sqrt((1.0 + zi[k]) * (1.0 + zf[k])) - 1.0
        v1 = float(_val(co.comoving_volume(zi[k], zm), "Mpc**3"))
        v2 = float(_val(co.comoving_volume(zm, zf[k]), "Mpc**3"))
        if not (np.isfinite(v1) and np.isfinite(v2)) or res[k] == 0:
            continue
        trigger_if(abs((v1 + v2) / res[k] - 1.0) > 1e-6, "YT-COS-010")


@_guarded
def check_dark_factor(co, z, result):
    """YT-COS-011"""
    w0, wa = float(co.w_0), float(co.w_a)
    if not (abs(w0) <= 5 and abs(wa) <= 5):
        return
    z = np.atleast_1d(_a(z))
    res = np.atleast_1d(_a(result))
    if z.size > 256 or z.size != res.size:
        return
    for k in range(z.size):
        a = 1.0 / (1.0 + z[k])
        if not (1e-3 <= a <= 1.0):
            continue
        integral = _quad(lambda u: 1.0 + w0 + wa * (1.0 - np.exp(u)), math.log(a), 0.0, panels=8)
        expected = math.exp(3.0 * integral)
        trigger_if(abs(res[k] / expected - 1.0) > 1e-9, "YT-COS-011")


# ---------------------------------------------------------------------------
# B. physical-constant consistency (fields/{astro,fluid,magnetic}_fields)
# ---------------------------------------------------------------------------


@_guarded
def check_thomson(result_sigma_cm2):
    """YT-PHC-001: ``result_sigma_cm2`` is the Thomson cross section the field used (cm^2)."""
    from yt.utilities.physical_constants import (
        charge_proton_cgs,
        mass_electron_cgs,
        speed_of_light_cgs,
    )

    e = float(charge_proton_cgs.to_value("esu"))
    m = float(mass_electron_cgs.to_value("g"))
    c = float(speed_of_light_cgs.to_value("cm/s"))
    r_e = e**2 / (m * c**2)
    expected = 8.0 * math.pi / 3.0 * r_e**2
    trigger_if(abs(float(result_sigma_cm2) / expected - 1.0) > 1e-5, "YT-PHC-001")


@_guarded
def check_kt(temperature, result):
    """YT-PHC-002"""
    t = _val(temperature, "K")
    kt = _val(result, "keV")
    good = np.isfinite(t) & np.isfinite(kt) & (t > 0)
    if not good.any():
        return
    expected = t[good] / 1.160451812e7
    trigger_if(np.any(np.abs(kt[good] / expected - 1.0) > 2e-6), "YT-PHC-002")


@_guarded
def check_rotation_measure(b_los, n_e, result):
    """YT-PHC-003"""
    from yt.units import dimensions

    dims = b_los.units.dimensions
    if dims == dimensions.magnetic_field_cgs:
        b_gauss = _val(b_los, "gauss")
    elif dims == dimensions.magnetic_field_mks:
        b_gauss = _val(b_los, "T") * 1.0e4
    else:
        return
    n = _val(n_e, "cm**-3")
    e, m, c = 4.803204712570263e-10, 9.1093837015e-28, _C_LIGHT_CM
    k = e**3 / (2.0 * math.pi * m**2 * c**4)
    expected = k * b_gauss * n * 1.0e4  # rad cm^-2 per cm -> rad m^-2 per cm
    got = _val(result, "rad/m**2/cm")
    scale = np.abs(expected)
    good = np.isfinite(got) & np.isfinite(expected) & (scale > 0)
    if not good.any():
        return
    trigger_if(np.any(np.abs(got[good] - expected[good]) > 1e-5 * scale[good]), "YT-PHC-003")


# ---------------------------------------------------------------------------
# C. composition (utilities/chemical_formulas.py, fields/species_fields.py)
# ---------------------------------------------------------------------------


@_guarded
def check_mu_bounds(ion_state, result):
    """YT-CHM-001"""
    from yt.utilities.chemical_formulas import compute_mu

    a_h, a_he = 1.00794, 4.002602
    slack = 1e-3
    mu = float(result)
    mu_ion = float(compute_mu("ionized"))
    mu_neu = float(compute_mu("neutral"))
    if ion_state in ("ionized", None):
        lo, hi = a_h / 2.0, a_he / 3.0
    else:
        lo, hi = a_h, a_he
    bad = not (lo - slack <= mu <= hi + slack) or not mu_ion < mu_neu
    trigger_if(bad, "YT-CHM-001")


@_guarded
def check_species_atom_counts(data):
    """YT-CHM-002"""
    from yt.fields.species_fields import _get_all_elements, _get_element_multiple
    from yt.utilities.chemical_formulas import ChemicalFormula
    from yt.utilities.periodic_table import periodic_table

    for species in data.ds.field_info.species_names:
        if species.startswith("El"):
            continue
        try:
            weight = ChemicalFormula(species).weight
        except Exception:
            continue
        nucleus = species.split("_")[0]
        total = 0.0
        for element in _get_all_elements([nucleus]):
            total += _get_element_multiple(nucleus, element) * periodic_table[element].weight
        trigger_if(abs(total - weight) > 1e-12 * max(abs(weight), 1.0), "YT-CHM-002")


# ---------------------------------------------------------------------------
# D. fluid and MHD fields
# ---------------------------------------------------------------------------


@_guarded
def check_radial_mach(data, ftype, result):
    """YT-FLD-001"""
    mach = _a(_fetch(data, (ftype, "mach_number")))
    rad = _a(result)
    if mach.shape != rad.shape:
        return
    good = np.isfinite(mach) & np.isfinite(rad)
    trigger_if(np.any(rad[good] > mach[good] * (1.0 + _C * _EPS) + _C * _EPS), "YT-FLD-001")


@_guarded
def check_courant(data, ftype, result):
    """YT-FLD-002"""
    dt = _val(result, "s")
    d = np.minimum.reduce([_val(_fetch(data, (ftype, f"d{ax}")), "cm") for ax in "xyz"])
    cs = _val(_fetch(data, (ftype, "sound_speed")), "cm/s")
    v = np.sqrt(sum(_val(_fetch(data, (ftype, f"velocity_{ax}")), "cm/s") ** 2 for ax in "xyz"))
    if dt.shape != d.shape or dt.shape != cs.shape:
        return
    good = np.isfinite(dt) & np.isfinite(d) & np.isfinite(cs) & np.isfinite(v) & (cs > 0) & (d > 0)
    upper = d / cs
    lower = d / (cs + v)
    bad = (dt > upper * (1.0 + _C * _EPS)) | (dt < lower * (1.0 - _C * _EPS))
    trigger_if(np.any(bad & good), "YT-FLD-002")


def _b_tesla(data, b):
    """Magnetic field in tesla for gaussian-cgs or SI fields; None otherwise (incl. non-gaussian conventions)."""
    from yt.units import dimensions

    factor = getattr(data.ds, "_magnetic_factor", 4.0 * math.pi)
    dims = b.units.dimensions
    if dims == dimensions.magnetic_field_cgs:
        if abs(factor - 4.0 * math.pi) > 1e-12:
            return None
        return _val(b, "gauss") * 1.0e-4
    if dims == dimensions.magnetic_field_mks:
        return _val(b, "T")
    return None


@_guarded
def check_alfven_si(data, ftype, result):
    """YT-FLD-003"""
    b_t = _b_tesla(data, _fetch(data, (ftype, "magnetic_field_strength")))
    if b_t is None:
        return
    rho = _val(_fetch(data, (ftype, "density")), "kg/m**3")
    got = _val(result, "m/s")
    if b_t.shape != got.shape or rho.shape != got.shape:
        return
    good = np.isfinite(b_t) & np.isfinite(rho) & (rho > 0) & np.isfinite(got)
    expected = b_t / np.sqrt(_MU_0 * rho)
    scale = np.abs(expected)
    trigger_if(np.any((np.abs(got - expected) > 1e-8 * scale) & good & (scale > 0)), "YT-FLD-003")


@_guarded
def check_magnetic_energy_si(data, ftype, result):
    """YT-FLD-004"""
    b_t = _b_tesla(data, _fetch(data, (ftype, "magnetic_field_strength")))
    if b_t is None:
        return
    got = _val(result, "J/m**3")
    if b_t.shape != got.shape:
        return
    expected = b_t**2 / (2.0 * _MU_0)
    good = np.isfinite(got) & np.isfinite(expected) & (expected > 0)
    trigger_if(np.any((np.abs(got - expected) > 1e-8 * expected) & good), "YT-FLD-004")


@_guarded
def check_poloidal_toroidal(data, ftype, result):
    """YT-FLD-005"""
    b = _fetch(data, (ftype, "magnetic_field_strength"))
    tor = _fetch(data, (ftype, "magnetic_field_toroidal_magnitude"))
    unit = b.units
    bs, ts, ps = _val(b, unit), _val(tor, unit), _val(result, unit)
    if not (bs.shape == ts.shape == ps.shape):
        return
    good = np.isfinite(bs) & np.isfinite(ts) & np.isfinite(ps)
    err = np.abs(ps**2 + ts**2 - bs**2)
    trigger_if(np.any((err > _C * _EPS * bs**2) & good), "YT-FLD-005")


@_guarded
def check_four_velocity(data, result_ut):
    """YT-FLD-006"""
    u = _val(_fetch(data, ("gas", "four_velocity_magnitude")), "cm/s")
    ut = _val(result_ut, "cm/s")
    if u.shape != ut.shape:
        return
    gamma = ut / _C_LIGHT_CM
    err = np.abs(ut**2 - u**2 - _C_LIGHT_CM**2)
    tol = _C * _EPS * gamma**2 * _C_LIGHT_CM**2
    good = np.isfinite(u) & np.isfinite(ut) & (gamma >= 1.0)
    trigger_if(np.any((err > tol) & good), "YT-FLD-006")


@_guarded
def check_overdensity_normalization(data, ftype, result):
    """YT-FLD-007"""
    ds = data.ds
    co = ds.cosmology
    for name in ("omega_matter", "omega_lambda"):
        a, b = getattr(co, name), getattr(ds, name, None)
        if b is None or abs(a - b) > 1e-12 * max(1.0, abs(a)):
            return
    z = float(ds.current_redshift)
    om_z = co.omega_matter * (1.0 + z) ** 3 / float(co.expansion_factor(z)) ** 2
    od = _a(_fetch(data, (ftype, "overdensity")))
    md = _a(result)
    if od.shape != md.shape:
        return
    good = np.isfinite(od) & np.isfinite(md) & (md != 0)
    trigger_if(np.any((np.abs(od - md * om_z) > 1e-10 * np.abs(od)) & good), "YT-FLD-007")


def _fetch(data, key):
    """Read ``data[key]`` while ``data`` is locked for field generation.

    yt raises ``GenerationInProgress`` for any field that is not yet available while a field
    function runs (its dependency-discovery protocol). A checker needs extra fields without
    changing that bookkeeping for the production call, so the lock is lifted for this read only.
    """
    if key in data.field_data:
        return data.field_data[key]
    locked = getattr(data, "_locked", False)
    data._locked = False
    try:
        return data[key]
    finally:
        data._locked = locked


def _is_detector(data):
    """Field-detector objects feed dummy data to field functions for dependency discovery."""
    try:
        from yt.fields.field_detector import FieldDetector

        return isinstance(data, FieldDetector)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# E. coordinate geometry (utilities/math_utils.py, fields/geometric_fields.py, geometry/coordinates)
# ---------------------------------------------------------------------------


def _vec_shape_ok(arr):
    return arr.ndim in (2, 4) and arr.shape[0] == 3


@_guarded
def check_sph_cyl_consistency(coords, normal, theta):
    """YT-GEO-001"""
    from yt.utilities.math_utils import get_cyl_r, get_cyl_z, get_sph_r

    c = _a(coords)
    if not _vec_shape_ok(c) or not _small(c) or not np.any(_a(normal)):
        return
    th = _a(theta)
    r = _a(get_sph_r(c))
    z = _a(get_cyl_z(c, normal))
    rr = _a(get_cyl_r(c, normal))
    sin_t = np.abs(np.sin(th))
    good = np.isfinite(r) & (r > 0) & (sin_t >= 1e-4)
    tol = _C * _EPS * r / np.where(sin_t > 0, sin_t, 1.0)
    bad = (np.abs(r * np.cos(th) - z) > tol) | (np.abs(r * np.sin(th) - rr) > tol)
    trigger_if(np.any(bad & good), "YT-GEO-001")


@_guarded
def check_azimuth_equivariance(coords, normal, phi):
    """YT-GEO-002"""
    from yt.utilities.math_utils import get_sph_phi

    c = _a(coords)
    n = _a(normal).astype(np.float64)
    if not _vec_shape_ok(c) or not _small(c) or not np.any(n):
        return
    n = n / np.linalg.norm(n)
    nb = n.reshape((3,) + (1,) * (c.ndim - 1))
    alpha = 0.7
    cross = np.cross(np.broadcast_to(nb, c.shape), c, axisa=0, axisb=0, axisc=0)
    dot = np.sum(nb * c, axis=0)
    c2 = c * math.cos(alpha) + cross * math.sin(alpha) + nb * dot * (1.0 - math.cos(alpha))
    phi2 = _a(get_sph_phi(c2, normal))
    r = np.sqrt(np.sum(c**2, axis=0))
    big_r = np.sqrt(np.sum(cross**2, axis=0))
    good = np.isfinite(r) & (r > 0) & (big_r / np.where(r > 0, r, 1.0) >= 1e-4)
    err = np.abs(_wrap_pi(phi2 - _a(phi) - alpha))
    tol = 4 * _C * _EPS * r / np.where(big_r > 0, big_r, 1.0)
    trigger_if(np.any((err > tol) & good), "YT-GEO-002")


def _basis_components_norm(vectors, comps, tag):
    v2 = np.sum(_a(vectors) ** 2, axis=0)
    s2 = sum(_a(c) ** 2 for c in comps)
    good = np.isfinite(v2) & np.isfinite(s2)
    trigger_if(np.any((np.abs(s2 - v2) > 4 * _C * _EPS * v2) & good), tag)


@_guarded
def check_cyl_parseval(vectors, theta, normal):
    """YT-GEO-003"""
    from yt.utilities.math_utils import (
        get_cyl_r_component,
        get_cyl_theta_component,
        get_cyl_z_component,
    )

    v = _a(vectors)
    if not _vec_shape_ok(v) or not _small(v) or not np.any(_a(normal)):
        return
    comps = [
        get_cyl_r_component(vectors, theta, normal),
        get_cyl_theta_component(vectors, theta, normal),
        get_cyl_z_component(vectors, normal),
    ]
    _basis_components_norm(vectors, comps, "YT-GEO-003")


@_guarded
def check_sph_parseval(vectors, theta, phi, normal):
    """YT-GEO-004"""
    from yt.utilities.math_utils import (
        get_sph_phi_component,
        get_sph_r_component,
        get_sph_theta_component,
    )

    v = _a(vectors)
    if not _vec_shape_ok(v) or not _small(v) or not np.any(_a(normal)):
        return
    comps = [
        get_sph_r_component(vectors, theta, phi, normal),
        get_sph_theta_component(vectors, theta, phi, normal),
        get_sph_phi_component(vectors, phi, normal),
    ]
    _basis_components_norm(vectors, comps, "YT-GEO-004")


@_guarded
def check_periodic_dist(a, b, period, periodicity, result):
    """YT-GEO-005"""
    from yt.utilities.math_utils import periodic_dist

    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 1 or a.shape[0] != 3 or not _small(a):
        return
    per = np.array(period, dtype=np.float64).ravel()
    per = np.full(3, per[0]) if per.size == 1 else per
    if per.size != 3:
        return
    per = per.reshape((3,) + (1,) * (a.ndim - 1))
    diff = np.abs(a - b)
    if np.any(diff > per) or len(periodicity) != 3:
        return
    p = np.array(periodicity, dtype=bool).reshape((3,) + (1,) * (a.ndim - 1))
    bound = np.sqrt(np.sum(np.where(p, (per / 2.0) ** 2, per**2) * np.ones_like(diff), axis=0))
    euclid = np.sqrt(np.sum(diff**2, axis=0))
    d = np.asarray(result, dtype=np.float64)
    d_swap = np.asarray(periodic_dist(b, a, period, periodicity), dtype=np.float64)
    tol = _C * _EPS * np.maximum(d, 1e-300)
    bad = (d > euclid * (1 + _C * _EPS)) | (d > bound * (1 + _C * _EPS)) | (np.abs(d - d_swap) > tol)
    trigger_if(np.any(bad & np.isfinite(d)), "YT-GEO-005")


@_guarded
def check_periodic_position(pos, ds, result):
    """YT-GEO-006"""
    unit = "code_length"

    def conv(x):
        return np.asarray(x.to_value(unit) if hasattr(x, "to_value") else x, dtype=np.float64)

    p, out = conv(pos), conv(result)
    le, re = conv(ds.domain_left_edge), conv(ds.domain_right_edge)
    dw = re - le
    if p.shape != out.shape or p.shape[-1:] != (3,) or not np.all(np.isfinite(p)):
        return
    s = _C * _EPS * (np.abs(p) + dw)
    k = (p - out) / dw
    bad = (out < le - s) | (out > re + s) | (np.abs(k - np.round(k)) > s / dw)
    trigger_if(np.any(bad), "YT-GEO-006")


@_guarded
def check_radius_cross_method(data, ftype, result):
    """YT-GEO-007"""
    if _is_detector(data):
        return
    r1 = _val(result, "cm")
    r2 = _val(_fetch(data, ("index", "spherical_radius")), "cm")
    if r1.shape != r2.shape:
        return
    width = float(np.max(_val(data.ds.domain_width, "cm")))
    good = np.isfinite(r1) & np.isfinite(r2)
    tol = _C * _EPS * (np.abs(r1) + width)
    trigger_if(np.any((np.abs(r1 - r2) > tol) & good), "YT-GEO-007")


_J_NODES = None


def _cell_quad_nodes():
    x_r, w_r = _gl(3)
    x_t, w_t = _gl(10)
    return x_r, w_r, x_t, w_t


@_guarded
def check_spherical_volume(r, dr, theta, dtheta, dphi, result, tag="YT-GEO-008"):
    """YT-GEO-008 / YT-GEO-010 (same Jacobian, different handlers)."""
    r_, dr_ = _val(r, "code_length"), _val(dr, "code_length")
    t_, dt_ = _a(theta), _a(dtheta)
    dp_ = _a(dphi)
    v = _val(result, "code_length**3")
    if not (r_.shape == dr_.shape == t_.shape == dt_.shape == dp_.shape == v.shape):
        return
    n = min(v.size, 100_000)
    r_, dr_, t_, dt_, dp_, v = (x.ravel()[:n] for x in (r_, dr_, t_, dt_, dp_, v))
    x_r, w_r, x_t, w_t = _cell_quad_nodes()
    rr = r_[:, None, None] + 0.5 * dr_[:, None, None] * x_r[None, :, None]
    tt = t_[:, None, None] + 0.5 * dt_[:, None, None] * x_t[None, None, :]
    wgt = w_r[None, :, None] * w_t[None, None, :]
    q = np.sum(wgt * rr**2 * np.sin(tt), axis=(1, 2)) * (0.5 * dr_) * (0.5 * dt_) * dp_
    scale = np.abs(q)
    cond = 1.0 + np.abs(r_) / np.where(dr_ > 0, dr_, 1.0) + 1.0 / np.maximum(
        np.abs(dt_) * np.abs(np.sin(t_)), 1e-300
    )
    good = np.isfinite(v) & np.isfinite(q) & (scale > 0) & (dr_ > 0) & (dt_ > 0) & (r_ > 0)
    trigger_if(np.any((np.abs(v - q) > _C * _EPS * cond * scale) & good), tag)


@_guarded
def check_cylindrical_volume(r, dr, dtheta, dz, result):
    """YT-GEO-009"""
    r_, dr_, dz_ = (_val(x, "code_length") for x in (r, dr, dz))
    dt_ = _a(dtheta)
    v = _val(result, "code_length**3")
    if not (r_.shape == dr_.shape == dt_.shape == dz_.shape == v.shape):
        return
    n = min(v.size, 100_000)
    r_, dr_, dt_, dz_, v = (x.ravel()[:n] for x in (r_, dr_, dt_, dz_, v))
    x, w = _gl(3)
    rr = r_[:, None] + 0.5 * dr_[:, None] * x[None, :]
    q = np.sum(w[None, :] * rr, axis=1) * (0.5 * dr_) * dt_ * dz_
    scale = np.abs(q)
    cond = 1.0 + np.abs(r_) / np.where(dr_ > 0, dr_, 1.0)
    good = np.isfinite(v) & np.isfinite(q) & (scale > 0) & (dr_ > 0) & (r_ > 0)
    trigger_if(np.any((np.abs(v - q) > _C * _EPS * cond * scale) & good), "YT-GEO-009")


# ---------------------------------------------------------------------------
# F. rotations and reference frames (utilities/math_utils.py, utilities/orientation.py)
# ---------------------------------------------------------------------------


@_guarded
def check_rotation_matrix(theta, rot_vector, rot):
    """YT-ROT-001"""
    a = np.asarray(rot_vector, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(a))
    r = np.asarray(rot, dtype=np.float64)
    if a.size != 3 or not np.isfinite(norm) or abs(norm - 1.0) > 1e-12 or r.shape != (3, 3):
        return
    tol = _C * _EPS + 8.0 * abs(norm - 1.0)
    orth = np.max(np.abs(r.T @ r - np.eye(3)))
    det = abs(np.linalg.det(r) - 1.0)
    axis = np.max(np.abs(r @ a - a))
    trigger_if(max(orth, det, axis) > tol, "YT-ROT-001")


@_guarded
def check_rotate_vector(a, dim, angle, result):
    """YT-ROT-002"""
    v = np.asarray(a, dtype=np.float64)
    out = np.asarray(result, dtype=np.float64)
    if v.shape != out.shape or v.shape[-1] != 3 or not np.all(np.isfinite(v)):
        return
    n_in = np.sqrt(np.sum(v**2, axis=-1))
    n_out = np.sqrt(np.sum(out**2, axis=-1))
    axial = np.abs(out[..., dim] - v[..., dim])
    tol = _C * _EPS * n_in
    trigger_if(np.any((np.abs(n_out - n_in) > tol) | (axial > tol)), "YT-ROT-002")


@_guarded
def check_quat_to_matrix(quaternion, rot):
    """YT-ROT-003 (quaternion -> matrix)"""
    from yt.utilities.math_utils import rotation_matrix_to_quaternion

    q = np.asarray(quaternion, dtype=np.float64).ravel()
    r = np.asarray(rot, dtype=np.float64)
    if q.size != 4 or r.shape != (3, 3):
        return
    dev = abs(float(np.linalg.norm(q)) - 1.0)
    if not np.isfinite(dev) or dev > 1e-12:
        return
    tol = 4 * _C * _EPS + 16.0 * dev
    orth = max(np.max(np.abs(r.T @ r - np.eye(3))), abs(np.linalg.det(r) - 1.0))
    q2 = np.asarray(rotation_matrix_to_quaternion(r), dtype=np.float64)
    sign = 1.0 if float(q @ q2) >= 0 else -1.0
    trigger_if(max(orth, np.max(np.abs(sign * q2 - q))) > tol, "YT-ROT-003")


@_guarded
def check_matrix_to_quat(rot, quaternion):
    """YT-ROT-003 (matrix -> quaternion)"""
    from yt.utilities.math_utils import quaternion_to_rotation_matrix

    r = np.asarray(rot, dtype=np.float64)
    q = np.asarray(quaternion, dtype=np.float64).ravel()
    if r.shape != (3, 3) or q.size != 4 or not np.all(np.isfinite(r)):
        return
    dev = max(np.max(np.abs(r.T @ r - np.eye(3))), abs(np.linalg.det(r) - 1.0))
    if dev > 1e-12:
        return
    tol = 4 * _C * _EPS + 16.0 * dev
    unit = abs(float(np.linalg.norm(q)) - 1.0)
    back = np.max(np.abs(quaternion_to_rotation_matrix(q) - r))
    trigger_if(max(unit, back) > tol, "YT-ROT-003")


@_guarded
def check_modify_frame(com, l_in, p_in, v_in, l_out, p_out, v_out):
    """YT-ROT-004"""
    l0 = np.asarray(l_in, dtype=np.float64)
    lo = np.asarray(l_out, dtype=np.float64)
    nl = float(np.linalg.norm(l0))
    if l0.shape != (3,) or not np.isfinite(nl) or nl == 0:
        return
    bad = np.max(np.abs(lo - np.array([0.0, 0.0, nl]))) > 1e-6 * nl
    pc = vv = None
    if p_in is not None and p_out is not None:
        pc = np.asarray(p_in, dtype=np.float64) - np.asarray(com, dtype=np.float64)
        po = np.asarray(p_out, dtype=np.float64)
        n_in, n_out = np.linalg.norm(pc, axis=-1), np.linalg.norm(po, axis=-1)
        bad |= bool(np.any(np.abs(n_in - n_out) > _C * _EPS * np.maximum(n_in, 1e-300)))
    if v_in is not None and v_out is not None:
        vv = np.asarray(v_in, dtype=np.float64)
        vo = np.asarray(v_out, dtype=np.float64)
        n_in, n_out = np.linalg.norm(vv, axis=-1), np.linalg.norm(vo, axis=-1)
        bad |= bool(np.any(np.abs(n_in - n_out) > _C * _EPS * np.maximum(n_in, 1e-300)))
    if pc is not None and vv is not None:
        t_in = np.sum(np.cross(pc, vv) * l0, axis=-1)
        t_out = np.sum(np.cross(po, vo) * lo, axis=-1)
        scale = np.linalg.norm(pc, axis=-1) * np.linalg.norm(vv, axis=-1) * nl
        bad |= bool(np.any(np.abs(t_in - t_out) > 1e-6 * scale))
    trigger_if(bad, "YT-ROT-004")


@_guarded
def check_ortho_find(vec1, out):
    """YT-ROT-005"""
    v = np.asarray(vec1, dtype=np.float64)
    e1, e2, e3 = (np.asarray(x, dtype=np.float64) for x in out)
    nv = float(np.linalg.norm(v))
    if v.shape != (3,) or not np.isfinite(nv) or nv == 0:
        return
    tol = _C * _EPS
    errs = [
        abs(np.linalg.norm(e1) - 1),
        abs(np.linalg.norm(e2) - 1),
        abs(np.linalg.norm(e3) - 1),
        abs(e1 @ e2),
        abs(e1 @ e3),
        abs(e2 @ e3),
        np.max(np.abs(e1 - v / nv)),
        np.max(np.abs(e3 - np.cross(e1, e2))),
    ]
    trigger_if(max(errs) > tol, "YT-ROT-005")


@_guarded
def check_velocity_decomposition(com, l_in, p_in, v_in):
    """YT-ROT-006"""
    from yt.utilities.math_utils import (
        compute_cylindrical_radius,
        compute_parallel_velocity,
        compute_radial_velocity,
        compute_rotational_velocity,
    )

    p = np.asarray(p_in, dtype=np.float64)
    v = np.asarray(v_in, dtype=np.float64)
    l0 = np.asarray(l_in, dtype=np.float64)
    if p.ndim != 2 or p.shape != v.shape or p.shape[1] != 3 or l0.shape != (3,) or not l0.any():
        return
    if not _small(p) or p.shape[0] > 20_000:
        return
    rot = np.asarray(compute_rotational_velocity(com, l_in, p_in, v_in), dtype=np.float64)
    rad = np.asarray(compute_radial_velocity(com, l_in, p_in, v_in), dtype=np.float64)
    par = np.asarray(compute_parallel_velocity(com, l_in, p_in, v_in), dtype=np.float64)
    cyl = np.asarray(compute_cylindrical_radius(com, l_in, p_in, v_in), dtype=np.float64)
    scale = float(np.max(np.linalg.norm(p - np.asarray(com, dtype=np.float64), axis=1)))
    v2 = np.sum(v**2, axis=1)
    good = np.isfinite(rot) & np.isfinite(rad) & np.isfinite(par) & (cyl > 1e-9 * scale)
    err = np.abs(rot**2 + rad**2 + par**2 - v2)
    trigger_if(np.any((err > 4 * _C * _EPS * v2) & good), "YT-ROT-006")


@_guarded
def check_orientation(orient, snapshot):
    """YT-ORI-001"""
    normal_in, north_in = snapshot
    m = np.asarray(_a(orient.unit_vectors), dtype=np.float64)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        return
    n_in = normal_in / np.linalg.norm(normal_in)
    sin_a = 1.0
    if north_in is not None:
        nn = north_in / np.linalg.norm(north_in)
        sin_a = float(np.linalg.norm(np.cross(nn, n_in)))
        if sin_a < 1e-6:
            return
    tol = _C * _EPS / sin_a
    errs = [
        np.max(np.abs(m @ m.T - np.eye(3))),
        abs(np.linalg.det(m) - 1.0),
        np.max(np.abs(m[2] - n_in)),
    ]
    if north_in is not None:
        proj = north_in - (north_in @ n_in) * n_in
        errs.append(np.max(np.abs(m[1] - proj / np.linalg.norm(proj))))
    trigger_if(max(errs) > tol, "YT-ORI-001")


def snapshot_orientation(normal_vector, north_vector):
    try:
        n = np.array(_a(normal_vector), dtype=np.float64)
        v = None if north_vector is None else np.array(_a(north_vector), dtype=np.float64)
        return n, v
    except Exception:
        _swallowed()
        return None


# ---------------------------------------------------------------------------
# G. derived quantities (data_objects/derived_quantities.py)
# ---------------------------------------------------------------------------


def _gather_weighted(ad, specs):
    """Concatenate ``(values_by_axis, weights)`` sources; ``specs`` = [([fields...], weight_field)].

    Returns (values list per component, weights) as float64 arrays in the first source's units,
    or None when any weight is negative, the total is not positive or the data are too large.
    """
    comps = None
    ws = []
    for fields, wfield in specs:
        w = ad[wfield]
        if w.size > _MAX_ELEMENTS:
            return None
        ws.append(_a(w))
        vals = [ad[f] for f in fields]
        if comps is None:
            comps = [[] for _ in fields]
            units = [v.units for v in vals]
        for i, v in enumerate(vals):
            comps[i].append(_val(v, units[i]))
    w = np.concatenate(ws)
    if not (np.all(np.isfinite(w)) and np.all(w >= 0) and w.sum() > 0):
        return None
    return [np.concatenate(c) for c in comps], w, units


def _mean_in_range(vals, w, results, tag):
    keep = w > 0
    n = int(keep.sum())
    for v, res in zip(vals, results, strict=True):
        lo, hi = float(v[keep].min()), float(v[keep].max())
        slack = _C * _EPS * max(n, 1) * max(abs(lo), abs(hi))
        trigger_if(not (lo - slack <= res <= hi + slack), tag)


@_guarded
def check_center_of_mass(dq, result):
    """YT-DQ-001"""
    args = getattr(dq, "_sc_args", None)
    if args is None:
        return
    use_gas, use_particles, ptype = args
    ad = dq.data_source
    specs = []
    if dq.use_gas:
        specs.append(([("gas", ax) for ax in "xyz"], ("gas", "mass")))
    if dq.use_particles:
        specs.append(([(ptype, f"particle_position_{ax}") for ax in "xyz"], (ptype, "particle_mass")))
    if not specs:
        return
    got = _gather_weighted(ad, specs)
    if got is None:
        return
    vals, w, units = got
    res = _val(result, units[0])
    _mean_in_range(vals, w, res, "YT-DQ-001")


@_guarded
def check_weighted_mean(dq, result):
    """YT-DQ-002 (WeightedAverageQuantity)"""
    args = getattr(dq, "_sc_args", None)
    if args is None:
        return
    fields, weight = args
    res = result if isinstance(result, list) else [result]
    if len(res) != len(fields):
        return
    ad = dq.data_source
    w = ad[weight]
    if w.size > _MAX_ELEMENTS:
        return
    wv = _a(w)
    if not (np.all(np.isfinite(wv)) and np.all(wv >= 0) and wv.sum() > 0):
        return
    for f, r in zip(fields, res, strict=True):
        q = ad[f]
        v = _val(q, q.units)
        _mean_in_range([v], wv, [float(_val(r, q.units))], "YT-DQ-002")


@_guarded
def check_bulk_velocity(dq, result):
    """YT-DQ-002 (BulkVelocity)"""
    args = getattr(dq, "_sc_args", None)
    if args is None:
        return
    use_gas, use_particles, ptype = args
    ad = dq.data_source
    specs = []
    if use_gas:
        specs.append(([("gas", f"velocity_{ax}") for ax in "xyz"], ("gas", "mass")))
    if use_particles and "nbody" in ad.ds.particle_types:
        specs.append(([(ptype, f"particle_velocity_{ax}") for ax in "xyz"], (ptype, "particle_mass")))
    if not specs:
        return
    got = _gather_weighted(ad, specs)
    if got is None:
        return
    vals, w, units = got
    res = _val(result, units[0])
    _mean_in_range(vals, w, res, "YT-DQ-002")


@_guarded
def check_weighted_std(dq, result):
    """YT-DQ-003"""
    args = getattr(dq, "_sc_args", None)
    if args is None:
        return
    fields, weight = args
    if len(result) != len(fields):
        return
    ad = dq.data_source
    w = ad[weight]
    if w.size > _MAX_ELEMENTS:
        return
    wv = _a(w)
    if not (np.all(np.isfinite(wv)) and np.all(wv >= 0) and wv.sum() > 0):
        return
    keep = wv > 0
    n = int(keep.sum())
    for f, r in zip(fields, result, strict=True):
        v = _a(ad[f])[keep]
        std, mean = float(r[0]), float(r[1])
        lo, hi = float(v.min()), float(v.max())
        mx = max(abs(lo), abs(hi))
        slack = _C * _EPS * max(n, 1) * mx
        bad = (
            std < 0
            or std**2 > ((hi - lo) / 2.0) ** 2 + _C * _EPS * max(n, 1) * mx**2
            or not (lo - slack <= mean <= hi + slack)
        )
        trigger_if(bad, "YT-DQ-003")


@_guarded
def check_spin_dimensionless(result):
    """YT-DQ-004"""
    from unyt.dimensions import dimensionless

    units = getattr(result, "units", None)
    if units is None:
        return
    trigger_if(units.dimensions != dimensionless, "YT-DQ-004")


# ---------------------------------------------------------------------------
# H. profiles (data_objects/profiles.py)
# ---------------------------------------------------------------------------


def _profile_bins(profile):
    names = [n for n in ("x_bins", "y_bins", "z_bins") if hasattr(profile, n)]
    return [getattr(profile, n) for n in names]


@_guarded
def check_profile(profile, fields):
    """YT-PRF-001/002/003"""
    if getattr(profile, "comm", None) is not None and getattr(profile.comm, "size", 1) > 1:
        return
    bins = _profile_bins(profile)
    if not bins or len(bins) != len(profile.bin_fields):
        return
    ds = profile.ds
    ad = profile.data_source
    xs = [[] for _ in bins]
    fs = [[] for _ in fields]
    ws = []
    n_tot = 0
    for chunk in ad.chunks([], "io"):
        sh = chunk[profile.bin_fields[0]].shape
        n_tot += int(np.prod(sh))
        if n_tot > _MAX_ELEMENTS:
            return
        for i, (bf, b) in enumerate(zip(profile.bin_fields, bins, strict=True)):
            xs[i].append(_val(chunk[bf], b.units).ravel())
        for i, f in enumerate(fields):
            fs[i].append(_val(chunk[f], ds.field_info[f].output_units).ravel())
        if profile.weight_field is not None:
            ws.append(_val(chunk[profile.weight_field], ds.field_info[profile.weight_field].output_units).ravel())
    if not xs[0]:
        return
    x = [np.concatenate(c) for c in xs]
    f = [np.concatenate(c) for c in fs]
    edges = [(_a(b).min(), _a(b).max()) for b in bins]
    inside = np.ones(x[0].shape, dtype=bool)
    closed = np.ones(x[0].shape, dtype=bool)
    for xi, (lo, hi) in zip(x, edges, strict=True):
        inside &= (xi > lo) & (xi < hi)
        closed &= (xi >= lo) & (xi <= hi)
    edge = closed & ~inside
    n = int(closed.sum())
    if profile.weight_field is None:
        for fi, field in zip(f, fields, strict=True):
            got = float(np.sum(_val(profile.field_data[field], ds.field_info[field].output_units)))
            expect = float(fi[inside].sum())
            allowance = float(np.abs(fi[edge]).sum()) + _C * _EPS * max(n, 1) * float(np.abs(fi[closed]).sum())
            trigger_if(abs(got - expect) > allowance, "YT-PRF-001")
        return
    w = np.concatenate(ws)
    if not (np.all(np.isfinite(w[closed])) and np.all(w[closed] >= 0)):
        return
    unit_w = ds.field_info[profile.weight_field].output_units
    bin_w = _val(profile.weight, unit_w)
    tw = float(w[inside].sum())
    w_allow = float(w[edge].sum()) + _C * _EPS * max(n, 1) * float(w[closed].sum())
    for fi, field in zip(f, fields, strict=True):
        unit_f = ds.field_info[field].output_units
        m_b = _val(profile.field_data[field], unit_f)
        s_b = _val(profile.standard_deviation[field], unit_f)
        sum_wf = float((w[inside] * fi[inside]).sum())
        mom_allow = float((w[edge] * np.abs(fi[edge])).sum()) + _C * _EPS * max(n, 1) * float(
            (w[closed] * np.abs(fi[closed])).sum()
        )
        trigger_if(abs(float(bin_w.sum()) - tw) > w_allow, "YT-PRF-002")
        trigger_if(abs(float((bin_w * m_b).sum()) - sum_wf) > mom_allow, "YT-PRF-002")
        if tw > 0:
            mean = sum_wf / tw
            var_data = float((w[inside] * (fi[inside] - mean) ** 2).sum())
            var_bins = float((bin_w * (s_b**2 + (m_b - mean) ** 2)).sum())
            var_allow = (
                float((w[edge] * (fi[edge] - mean) ** 2).sum())
                + _C * _EPS * max(n, 1) * float((w[closed] * (fi[closed] - mean) ** 2).sum())
                + _C * _EPS * max(n, 1) * tw * mean**2
            )
            trigger_if(abs(var_bins - var_data) > var_allow, "YT-PRF-003")
