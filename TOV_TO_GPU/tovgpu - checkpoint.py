
#from scipy.integrate import odeint
#from scipy.interpolate import PchipInterpolator
from scipy.optimize import root
import matplotlib.pyplot as plt
import pkg_resources
import numpy as np
import cupy as cp
from constants import *
import time
class TOVGPU:
  def rk4d(f, y0, t, args=(), rtol=None):
    if rtol is None:
        raise ValueError("A relative tolerance (rtol) must be provided.")
    y0 = cp.asarray(y0, dtype=cp.float64).ravel()
    y = cp.zeros((len(t), y0.size), dtype=y0.dtype)
    y[0] = y0  
    t_array = cp.asarray(t, dtype=cp.float64)  
    for i in range(1, len(t_array)):
        h = t_array[i] - t_array[i - 1]
        ti = t_array[i - 1]
        yi = y[i - 1]
        k1 = cp.asarray(f(yi, ti, *args))
        k2 = cp.asarray(f(yi + h * k1 / 2, ti + h / 2, *args))
        k3 = cp.asarray(f(yi + h * k2 / 2, ti + h / 2, *args))
        k4 = cp.asarray(f(yi + h * k3, ti + h, *args))
        y_next = yi + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        error = cp.max(cp.abs((y_next - yi) - h * (k1 + 2*k2 + 2*k3 + k4) / 6))
        if error > rtol:
            h *= 0.5
        elif error < rtol / 10:
            h *= 1.5
        y[i, :] = y_next
    return y
  class PchipInterpolator:
      def __init__(self, x, y):
        self.x = cp.asarray(x).ravel()
        self.y = cp.asarray(y).ravel()
        if self.x.size != self.y.size:
            raise ValueError("x and y must have the same length.")
        if not cp.all(self.x[1:] > self.x[:-1]):
            raise ValueError("x must be strictly increasing.")

      def __call__(self, xp):
        scalar_input = cp.isscalar(xp)
        xp = cp.asarray(xp).ravel()
        xp = cp.clip(xp, self.x[0], self.x[-1])
        indices = cp.searchsorted(self.x, xp, side='right') - 1
        indices = cp.clip(indices, 0, self.x.size - 2)
        x0 = self.x[indices]
        x1 = self.x[indices + 1]
        y0 = self.y[indices]
        y1 = self.y[indices + 1]
        slope = (y1 - y0) / (x1 - x0)
        result = y0 + slope * (xp - x0)
        if scalar_input:
          return result.item() 
        return result
  """
  Instance of the TOV solver
  """
  def __init__(self, en_arr, p_arr, add_crust=True, plot_eos=False):
    """
    Initializes TOVsolver, EOS should be provided by passing pressure (p_arr)
    as function of energy density (en_arr). p_arr and en_arr are given in nuclear
    units (MeV/fm^3). By default adds nuclear crust from:
    G. Baym, C. Pethick and P. Sutherland,
    ``The Ground state of matter at high densities: Equation of state and stellar models,''
    Astrophys. J. \textbf{170}, 299-317 (1971), doi:10.1086/151216.

    Parameters
    ----------
    en_arr : array_like, MeV / fm^3
             Array with range of energy densities of provides EOS, 
             should be provided in MeV/fm^3.
    p_arr  : array_like, MeV / fm^3
             Array with pressure as fucntion of en_arr,
             should be provided in MeV/fm^3.
    add_crust : bool, optional
             Merge EOS with crust EOS from nuclear statistical equilibrium.
    plot_eos  : bool, optional
             Output a plot with the EOS, optional.

    Returns
    -------
    solver : TOV
             Instance of tovsolver.
    """  
    en_arr *= MeV_fm3_to_pa_cgs / c**2
    p_arr  *= MeV_fm3_to_pa_cgs

    sort_ind = cp.argsort(p_arr)
    self.en_dens = TOVGPU.PchipInterpolator((p_arr[sort_ind]),
                                     (en_arr[sort_ind])) 

    sort_ind = cp.argsort(en_arr)
    self.press = TOVGPU.PchipInterpolator((en_arr[sort_ind]),
                                   (p_arr[sort_ind]))

    self.__en_arr = en_arr
    self.__p_arr = p_arr

    self.min_dens = cp.min(en_arr)
    self.max_dens = cp.max(en_arr)

    self.min_p = cp.min(p_arr)
    self.max_p = cp.max(p_arr)

    if add_crust:
      if plot_eos:
        plt.plot(cp.asnumpy(self.__en_arr / (MeV_fm3_to_pa_cgs / c**2)),
                 cp.asnumpy(self.__p_arr / MeV_fm3_to_pa_cgs),
                 linestyle='-', label='original EOS')
      self.add_crust()
    if plot_eos:
      plt.plot(cp.asnumpy(self.__en_arr / (MeV_fm3_to_pa_cgs / c**2)),
               cp.asnumpy(self.__p_arr / MeV_fm3_to_pa_cgs),
               linestyle='--', label='EOS with crust')
      plt.xscale('log')
      plt.yscale('log')
      plt.xlabel(r'${\rm \varepsilon~(MeV/fm^{3}) }$')
      plt.ylabel(r'${\rm P~(MeV/fm^{3}) }$')
      plt.legend()
      plt.show()

  def add_crust(self):
    """
    Adds Nuclear Statistical Equilibrium crust EOS from:
    G. Baym, C. Pethick and P. Sutherland,
    ``The Ground state of matter at high densities: Equation of state and stellar models,''
    Astrophys. J. \textbf{170}, 299-317 (1971), doi:10.1086/151216.
    Finds an intersection point between provided EOS and NSE EOS, then the two EOS are merged,
    so at lower densities crust EOS is used and at higher -- the provided one.

    Parameters
    ----------

    Returns
    -------
    """
    crust_loc = pkg_resources.resource_filename(__name__, 'data/')
    # dir_name = os.path.dirname(__file__)
    
    baym_eos = np.genfromtxt(crust_loc + "Baym_eos.dat", 
                             dtype=float, skip_header=1,
                             names=["en", "p", "nB",])
    P_crust = TOVGPU.PchipInterpolator(baym_eos["en"], baym_eos["p"])

    def eq_glue(n):
      return float(P_crust(n) - self.press(n))

    g = root(eq_glue, [44.*(MeV_fm3_to_pa_cgs / c**2)], options = {'maxfev' : 200})
    n_glue = g['x'][0]

    en_gpu = self.__en_arr
    p_gpu = self.__p_arr

    mask_crust = en_gpu < n_glue
    mask_core  = en_gpu >= n_glue

    en_crust = en_gpu[mask_crust]
    p_crust  = p_gpu[mask_crust]
    en_core  = en_gpu[mask_core]
    p_core   = p_gpu[mask_core]

    en_arr = cp.concatenate((en_crust, en_core))
    p_arr  = cp.concatenate((p_crust, p_core))

    self.min_dens = en_arr.min()
    self.min_p = p_arr.min()
    self.en_dens = TOVGPU.PchipInterpolator(p_arr, en_arr)
    self.press = TOVGPU.PchipInterpolator(en_arr,p_arr)

    self.__en_arr = en_arr
    self.__p_arr  = p_arr

    return

  def dedp(self, r, R_dep):
    e_R, p_R, m_R = R_dep

    p = p_R(r)
    dp = p * 0.005

    el_3 = self.en_dens(p - 3 * dp)
    el_2 = self.en_dens(p - 2 * dp)
    el_1 = self.en_dens(p - 1 * dp)
    er_3 = self.en_dens(p + 3 * dp)
    er_2 = self.en_dens(p + 2 * dp)
    er_1 = self.en_dens(p + 1 * dp)
    de_dp = (-1 / 60 * el_3 + 3 / 20 * el_2 - 3 / 4 * el_1 + 3 / 4 * er_1 - 3 / 20 * er_2 + 1 / 60 * er_3) / dp

    return de_dp

  def love_eq(self, param, r, R_dep):
    beta, H = param
    e_R, p_R, m_R = R_dep

    try:
      dummy = p_R(r)
    except ValueError:
      return [100000, 100000]

    de_dp = self.dedp(r, R_dep)

    dbetadr = H * (-2 * pi * G / c ** 2 * (
        5 * e_R(r) + 9 * p_R(r) / c ** 2 + de_dp * c ** 2 * (e_R(r) + p_R(r) / c ** 2)) \
                   + 3 / r ** 2 \
                   + 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1) * (
                       m_R(r) / r ** 2 * km_to_mSun + G / c ** 4 * 4 * pi * r * p_R(r)) ** 2) \
              + beta / r * (
                  -1 + m_R(r) / r * km_to_mSun + 2 * pi * r ** 2 * G / c ** 2 * (e_R(r) - p_R(r) / c ** 2))
    dbetadr *= 2 * (1 - 2 * m_R(r) / r * km_to_mSun) ** (-1)

    dHdr = beta
    return [dbetadr, dHdr]

  def tov_eq(self, y, r):
    P, m = y
    if isinstance(P, cp.ndarray):
        P = P.item()
    if P < self.min_p or P > self.max_p:
        return [0., 0.]


    eden = self.en_dens(P)

    dPdr = -G * (eden + P / c ** 2) * (m + 4.0 * pi * r ** 3 * P / c ** 2)
    dPdr = dPdr / (r * (r - 2.0 * G * m / c ** 2))

    dmdr = 4.0 * pi * r ** 2 * eden

    return [dPdr, dmdr]

  def check_density(self, dens):
    if dens < self.min_dens or dens > self.max_dens:
      raise Exception('Central density: %8.4E is outside of the EoS range. \n' 
                        %(dens/(MeV_fm3_to_pa_cgs / c ** 2)) +
                        'min density is: %8.4E, max density is:%8.4E'
                         %(self.min_dens/(MeV_fm3_to_pa_cgs / c ** 2), 
                         self.max_dens/(MeV_fm3_to_pa_cgs / c ** 2)))



  def solve(self, c_dens, rmax=50e5, rtol=1.0e-5, dmrel=10e-12, dr=100):
    """
    Solves TOV equation for neutron star with given central density c_dens.
    Parameters
    ----------
    c_dens : float, MeV / fm^3
    rmax : float, cm, optional
      Maximal distance from the star center along which star profile is calculated.
      If rmax is smaller then actual radius of the star, then Mass and Radius 
      will be calculated wrong.
    rtol : float, optional
      Relative accuracy of ODE solver.
    dmrel : float, optional
      Relative mass increase by which star boundary is estimated. 
    dr : float, cm, optinal
      Stepsize for ODE solver.
    Returns
    -------
    R : float, km
      Calculated radius of the star.
    M : float, Msun
      Calculated mass of the star.
    tuple (r, e_R, p_R, m_R) :
      Neutron star profile.
      r : numpy.array, cm
        Array of points along which profile is calculated.
      e_R : numpy.array
        Energy density of the star along r.
      p_R : numpy.array
        Pressure of the star along r.
      m_R : numpy.array
        Integrated mass of the star along r.
    """
    c_dens = c_dens.astype(cp.float64) * MeV_fm3_to_pa_cgs / c ** 2

    self.check_density(c_dens)

    r = cp.arange(dr, rmax + dr, dr)

    P = self.press(c_dens)
    ll= P
    eden = self.en_dens(P)
    m = 4.0 * pi * r[0] ** 3 * eden
    kkk= m
    psol = TOVGPU.rk4d(self.tov_eq, [P, m], r, rtol=rtol) #tentar manualmente?
    p_R, m_R = psol[:, 0], psol[:, 1]

    # find the boundary of the star by finding point
    # where the mass stops to increase

    diff = (m_R[1:] - m_R[:-1])/m_R[1:]
    ind = -1
    for i, dm in enumerate(diff):
      if dm < dmrel and m_R[i] != 0:
        ind = i
        break

    M = m_R[ind - 1]
    R = r[ind - 1]

    r   = r[:ind]
    p_R = p_R[:ind]
    m_R = m_R[:ind]
    
    e_R = self.en_dens(p_R)
    
    return R / 1e5, M / Msun, (r, e_R, p_R, m_R) 

  def solve_tidal(self, c_dens, rmax=30e5, rtol=1.0e-4, dmrel=10e-12, dr=100):
    """
    Solves TOV equation and calculates tidal properties 
    for neutron star with given central density c_dens.

    Parameters
    ----------
    c_dens : float, MeV / fm^3
    rmax : float, cm, optional
      Maximal distance from the star center along which star profile is calculated.
      If rmax is smaller then actual radius of the star, then Mass and Radius 
      will be calculated wrong.
    rtol : float, optional
      Relative accuracy of ODE solver.
    dmrel : float, optional
      Relative mass increase by which star boundary is estimated. 
    dr : float, cm, optinal
      Stepsize for ODE solver.

    Returns
    -------
    [R, M, C, k2, y, beta, H], float array_like:
      R : float, km
        Calculated radius of the star.
      M : float, Msun
        Calculated mass of the star.
      C : float, unitless
        Compactness.
      k2 : float, unitless
        Second Love number.
      y : float, unitless
      beta : float, unitless
      H : float, unitless
    """
    R, M, R_dep  = self.solve(c_dens, rmax=rmax, rtol=rtol, dmrel=dmrel, dr=dr)
    r, e_R, p_R, m_R = R_dep

    R *= 1e5
    M *= Msun

    e_R = TOVGPU.PchipInterpolator(r, e_R)
    p_R = TOVGPU.PchipInterpolator(r, p_R)
    m_R = TOVGPU.PchipInterpolator(r, m_R)

    beta0 = 2 * r[0]
    H0 = r[0] ** 2

    solution = TOVGPU.rk4d(self.love_eq, [beta0, H0], r, args=([e_R, p_R, m_R],), rtol=rtol)
    beta = solution[-1, 0]
    H = solution[-1, 1]

    y = R * beta / H

    C = compactness = M / R * km_to_mSun

    k2 = 8 / 5 * C ** 5 * (1 - 2 * C) ** 2 * (2 + 2 * C * (y - 1) - y) * (
          2 * C * (6 - 3 * y + 3 * C * (5 * y - 8)) + 4 * C ** 3 * (
            13 - 11 * y + C * (3 * y - 2) + 2 * C ** 2 * (1 + y)) + 3 * (1 - 2 * C) ** 2 * (2 - y + 2 * C * (y - 1)) * (
            cp.log(1 - 2 * C))) ** (-1)

    return cp.array([R / 1e5, M / Msun, C, k2, y, beta, H])

#---------------------------------------------------------
start_time = time.perf_counter()
central_densities = cp.full(10, 200)
R_arr = cp.zeros(10)
M_arr = cp.zeros(10)
profile_list = []
en_arr = cp.linspace(10, 1000, 5000000)  
p_arr = 0.1 * en_arr ** 1.5
solver = TOVGPU(en_arr, p_arr)
for i in range(10):
    R, M, profile = solver.solve(central_densities[i])  
    R_arr[i] = R
    M_arr[i] = M
    profile_list.append(profile)
    print(f"[{i}] Raio da estrela: {R:.2f} km")
    print(f"[{i}] Massa da estrela: {M:.2f} M☉")
elapsed_time = time.perf_counter() - start_time
print(f"Tempo de execução: {elapsed_time:.6f} segundos")




