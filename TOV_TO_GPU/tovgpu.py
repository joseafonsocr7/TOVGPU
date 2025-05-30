
#from scipy.integrate import odeint
from scipy.interpolate import PchipInterpolator as PchipCPU
from scipy.optimize import root
import matplotlib.pyplot as plt
import pkg_resources
import numpy as np
import cupy as cp
from constants import *
import time
class TOVGPU:
#Pchip begins here
  _compute_derivatives_kernel = cp.RawKernel(r'''
  extern "C" __global__
  void compute_derivatives(const double* x, const double* y, const double* h, const double* delta, 
                          double* m, int n, int batch_size, int stride) 
  {
      int batch_idx = blockIdx.y;
      int i = blockDim.x * blockIdx.x + threadIdx.x + 1; 
      
      if (batch_idx < batch_size && i < n - 1) { 
          int idx = batch_idx * stride + i;
          int prev_idx = idx - 1;
          
          if (delta[prev_idx] * delta[idx] > 0) {
              double w1 = 2 * h[idx] + h[prev_idx];
              double w2 = h[idx] + 2 * h[prev_idx];
              m[idx] = (w1 + w2) / (w1 / delta[prev_idx] + w2 / delta[idx]);
          } else {
              m[idx] = 0.0;
          }
      }
  }
  ''', 'compute_derivatives')

  _endpoint_kernel = cp.RawKernel(r'''
  extern "C" __global__
  void compute_endpoint(const double* h, const double* delta, double* m, 
                      int n, bool is_first, int batch_size, int stride) 
  {
      int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
      
      if (batch_idx < batch_size) {
          int base_idx = batch_idx * stride;
          double h1, h2, d1, d2;
          int idx;
          
          if (is_first) {
              h1 = h[base_idx];
              h2 = h[base_idx + 1];
              d1 = delta[base_idx];
              d2 = delta[base_idx + 1];
              idx = base_idx;
          } else {
              h1 = h[base_idx + n - 2];
              h2 = h[base_idx + n - 3];
              d1 = delta[base_idx + n - 2];
              d2 = delta[base_idx + n - 3];
              // Reverse for last endpoint
              double temp = d1;
              d1 = d2;
              d2 = temp;
              temp = h1;
              h1 = h2;
              h2 = temp;
              idx = base_idx + n - 1;
          }
          
          if (d1 * d2 <= 0) {
              m[idx] = 0.0;
              return;
          }
          
          double numerator = (2 * h1 + h2) * d1 - h1 * d2;
          double denominator = h1 + h2;
          
          if (denominator == 0) {
              m[idx] = 0.0;
              return;
          }
          
          double slope = numerator / denominator;
          
          if (slope * d1 <= 0) {
              m[idx] = 0.0;
          } else if (fabs(slope) > 3 * fabs(d1)) {
              m[idx] = 3 * d1;
          } else {
              m[idx] = slope;
          }
      }
  }
  ''', 'compute_endpoint')

  _interpolate_kernel = cp.RawKernel(r'''
  extern "C" __global__
  void interpolate(const double* x, const double* y, const double* h, const double* m, 
                  double* x_new, double* y_new, int n, int x_new_len, 
                  int batch_size, int stride, int output_stride) 
  {
      int batch_idx = blockIdx.y;
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      
      if (batch_idx < batch_size && i < x_new_len) {
          int x_base = batch_idx * stride;
          int y_base = batch_idx * output_stride;
          
          double xi = x_new[y_base + i];
          int j = 0;
          
          if (xi <= x[x_base]) {
              j = 0;
          } else if (xi >= x[x_base + n - 1]) {
              j = n - 2;
          } else {
              int left = 0;
              int right = n - 2;
              
              while (left <= right) {
                  int mid = (left + right) / 2;
                  if (x[x_base + mid] <= xi && xi < x[x_base + mid + 1]) {
                      j = mid;
                      break;
                  } else if (xi < x[x_base + mid]) {
                      right = mid - 1;
                  } else {
                      left = mid + 1;
                  }
              }
          }
          
          int idx = x_base + j;
          double h_val = h[idx];
          double t = (xi - x[idx]) / h_val;
          double yj = y[idx];
          double yjp1 = y[idx + 1];
          double mj = m[idx];
          double mjp1 = m[idx + 1];
          
          double h00 = 2*t*t*t - 3*t*t + 1;
          double h10 = t*t*t - 2*t*t + t;
          double h01 = -2*t*t*t + 3*t*t;
          double h11 = t*t*t - t*t;
          
          y_new[y_base + i] = h00 * yj + h10 * h_val * mj + h01 * yjp1 + h11 * h_val * mjp1;
      }
  }
  ''', 'interpolate')

  class PchipInterpolator:
      def __init__(self, x, y):
          self.x = np.asarray(x, dtype=np.float64)
          self.y = np.asarray(y, dtype=np.float64)
          
          if self.x.ndim != 2 or self.y.ndim != 2:
              raise ValueError("x and y must be 2D arrays with shape (batch_size, n_points)")
          
          if self.x.shape != self.y.shape:
              raise ValueError("x and y must have the same shape")
          
          self.batch_size, self.n_points = self.x.shape
        

          for i in range(self.batch_size):
              if np.any(np.diff(self.x[i]) <= 0):
                  raise ValueError(f"x must be strictly increasing for batch {i}")
          
          self.h = np.zeros_like(self.x)
          self.delta = np.zeros_like(self.x)
          
          for i in range(self.batch_size):
              self.h[i, :-1] = np.diff(self.x[i])
              self.delta[i, :-1] = np.diff(self.y[i]) / self.h[i, :-1]
          
          self.m = self._compute_derivatives()
          
          self.d_x = cp.asarray(self.x)
          self.d_y = cp.asarray(self.y)
          self.d_h = cp.asarray(self.h)
          self.d_m = cp.asarray(self.m)
          
      def _compute_derivatives(self):
          m = cp.zeros((self.batch_size, self.n_points), dtype=cp.float64)
          
          d_x = cp.asarray(self.x)
          d_y = cp.asarray(self.y)
          d_h = cp.asarray(self.h)
          d_delta = cp.asarray(self.delta)
          
          threads_per_block = 256
          blocks_x = (self.n_points - 2 + threads_per_block - 1) // threads_per_block
          TOVGPU._compute_derivatives_kernel(
              (blocks_x, self.batch_size), 
              (threads_per_block, 1), 
              (d_x.reshape(-1), d_y.reshape(-1), d_h.reshape(-1), 
              d_delta.reshape(-1), m.reshape(-1), self.n_points, 
              self.batch_size, self.n_points)
          )
          
          threads = min(256, self.batch_size)
          blocks = (self.batch_size + threads - 1) // threads
          
          TOVGPU._endpoint_kernel(
              (blocks,), (threads,), 
              (d_h.reshape(-1), d_delta.reshape(-1), m.reshape(-1), 
              self.n_points, True, self.batch_size, self.n_points)
          )
          
          TOVGPU._endpoint_kernel(
              (blocks,), (threads,), 
              (d_h.reshape(-1), d_delta.reshape(-1), m.reshape(-1), 
              self.n_points, False, self.batch_size, self.n_points)
          )
          
          return cp.asnumpy(m)
          
      def __call__(self, x_new):
          x_new = np.asarray(x_new, dtype=np.float64)
          
          if x_new.ndim == 0:  # scalar
              x_new = np.array([[x_new]])
              scalar_result = True
          elif x_new.ndim == 1:  # 1D array - apply to all batches
              x_new = np.tile(x_new, (self.batch_size, 1))
              scalar_result = False
          else:  # 2D array - already batched
              if x_new.shape[0] != self.batch_size:
                  raise ValueError(f"First dimension of x_new must match batch size ({self.batch_size})")
              scalar_result = False
          
          d_x_new = cp.asarray(x_new)
          x_new_len = x_new.shape[1]
          d_y_new = cp.zeros_like(d_x_new)
          
          threads_per_block = 256
          blocks_x = (x_new_len + threads_per_block - 1) // threads_per_block
          TOVGPU._interpolate_kernel(
              (blocks_x, self.batch_size), 
              (threads_per_block, 1), 
              (self.d_x.reshape(-1), self.d_y.reshape(-1), self.d_h.reshape(-1), self.d_m.reshape(-1),
              d_x_new.reshape(-1), d_y_new.reshape(-1), self.n_points, x_new_len,
              self.batch_size, self.n_points, x_new_len)
          )
          
          y_new = cp.asnumpy(d_y_new)
          
          return y_new[0, 0] if scalar_result else y_new
#Pchip ends here    
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
    en_arr *= MeV_fm3_to_pa_cgs / c**2 #será transformado num array bi dimensional
    p_arr  *= MeV_fm3_to_pa_cgs #será transformado num array bi dimensional

    sort_ind = cp.argsort(p_arr) 
    sort_arrays_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void apply_sort_ind(const double* input_p_arr, const double* input_en_arr,
                        double* output_p_arr, double* output_en_arr,
                        const int* sort_indices, int array_length) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i < array_length) {
            int sort_idx = sort_indices[i];
            output_p_arr[i] = input_p_arr[sort_idx];
            output_en_arr[i] = input_en_arr[sort_idx];
        }
    }
    ''', 'apply_sort_ind')
    sorted_p_arr = cp.empty_like(p_arr)
    sorted_en_arr = cp.empty_like(en_arr)

    array_length = p_arr.size
    threads_per_block = 256
    blocks = (array_length + threads_per_block - 1) // threads_per_block
    sort_arrays_kernel((blocks,), (threads_per_block,),
                  (p_arr, en_arr, sorted_p_arr, sorted_en_arr, sort_ind, array_length))
    self.en_dens= TOVGPU.PchipInterpolator(cp.asnumpy(sorted_p_arr), cp.asnumpy(sorted_en_arr))

    sorted_p_arr = cp.empty_like(p_arr)
    sorted_en_arr = cp.empty_like(en_arr)

    array_length = p_arr.size
    threads_per_block = 256
    blocks = (array_length + threads_per_block - 1) // threads_per_block
    sort_arrays_kernel((blocks,), (threads_per_block,),
                  (p_arr, en_arr, sorted_p_arr, sorted_en_arr, sort_ind, array_length))
    self.press= TOVGPU.PchipInterpolator(cp.asnumpy(sorted_p_arr), cp.asnumpy(sorted_en_arr))

    
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
    #somehow tem de ser transformado em array bi dimensional, afinal não
    baym_eos = np.genfromtxt(crust_loc + "Baym_eos.dat", 
                             dtype=float, skip_header=1,
                             names=["en", "p", "nB",])
    P_crust = PchipCPU(baym_eos["en"], baym_eos["p"])

    def eq_glue(n):
      p_crust_value = P_crust(n)
  
      p_batch = self.press(n)
      
      p_crust_batch = cp.full_like(p_batch, p_crust_value)
      
      return p_crust_batch - p_batch
    
      init_n_values = np.array([44.0 * (MeV_fm3_to_pa_cgs / c**2)] * batch_size)  # Initial guess for each batch
      
      glue_points = cp.zeros(batch_size, dtype=cp.float64)
      
      for i in range(batch_size):
        g = root(lambda n: float(eq_glue(n)[i]), init_n_values[i], options={'maxfev': 200})
        glue_points[i] = g['x'][0]
      
    
      en_arr_batched = []
      p_arr_batched = []
      
      for i in range(batch_size):
        n_glue = glue_points[i]
        
        
        mask_crust = self.__en_arr[i] < n_glue
        mask_core = self.__en_arr[i] >= n_glue
        
        en_crust = self.__en_arr[i][mask_crust]
        p_crust = self.__p_arr[i][mask_crust]
        en_core = self.__en_arr[i][mask_core]
        p_core = self.__p_arr[i][mask_core]
        
       
        en_arr_batched.append(cp.concatenate((en_crust, en_core)))
        p_arr_batched.append(cp.concatenate((p_crust, p_core)))
      
    
      en_arr = cp.stack(en_arr_batched)
      p_arr = cp.stack(p_arr_batched)
      
      self.min_dens = cp.min(en_arr, axis=1)
      self.min_p = cp.min(p_arr, axis=1)
    self.en_dens = TOVGPU.PchipInterpolator(p_arr, en_arr)
    self.press = TOVGPU.PchipInterpolator(en_arr,p_arr)

    self.__en_arr = en_arr
    self.__p_arr  = p_arr

    return
  def dedp(self, r, R_dep):
    batch_size = len(r)
    
    p_values = cp.zeros(batch_size)
    for i in range(batch_size):
      e_R, p_R, m_R = R_dep[i]
      p = p_R(r[i])
      p_values[i] = p
    dp_values = p_values * 0.005

    el_3 = self.en_dens(p_values - 3 * dp_values)
    el_2 = self.en_dens(p_values - 2 * dp_values)
    el_1 = self.en_dens(p_values - 1 * dp_values)
    er_3 = self.en_dens(p_values + 3 * dp_values)
    er_2 = self.en_dens(p_values + 2 * dp_values)
    er_1 = self.en_dens(p_values + 1 * dp_values)
    de_dp = (-1 / 60 * el_3 + 3 / 20 * el_2 - 3 / 4 * el_1 + 3 / 4 * er_1 - 3 / 20 * er_2 + 1 / 60 * er_3) / dp_values

    return de_dp

  def love_eq(self, param, r, R_dep):
    batch_size = len(r)
      
    dbetadr = cp.zeros(batch_size, dtype=cp.float64)
    dHdr = cp.zeros(batch_size, dtype=cp.float64)
      
      
    d_beta = cp.array([p[0] for p in param], dtype=cp.float64)
    d_H = cp.array([p[1] for p in param], dtype=cp.float64)
    d_r = cp.array(r, dtype=cp.float64)
      
    e_values = cp.zeros(batch_size, dtype=cp.float64)
    p_values = cp.zeros(batch_size, dtype=cp.float64)
    m_values = cp.zeros(batch_size, dtype=cp.float64)
      
    try:
      for i in range(batch_size):
        e_R, p_R, m_R = R_dep[i]
        e_values[i] = e_R(r[i])
        p_values[i] = p_R(r[i])
        m_values[i] = m_R(r[i])
    except ValueError:
      return [cp.full(batch_size, 100000), cp.full(batch_size, 100000)]
      
     
    de_dp_values = self.dedp(r, R_dep)
      
    love_eq_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void love_eq_kernel(const double* beta, const double* H, const double* r,
              const double* e_values, const double* p_values, 
              const double* m_values, const double* de_dp_values,
              double* dbetadr, double* dHdr,
              const double G, const double c, const double pi,
              const double km_to_mSun, const int batch_size) {
      int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
      if (idx < batch_size) {
        double r_val = r[idx];
        double beta_val = beta[idx];
        double H_val = H[idx];
        double e_val = e_values[idx];
        double p_val = p_values[idx];
        double m_val = m_values[idx];
        double de_dp_val = de_dp_values[idx];
          
        double term1 = -2 * pi * G / (c * c) * (
          5 * e_val + 9 * p_val / (c * c) + 
          de_dp_val * c * c * (e_val + p_val / (c * c))
        );
          
        double term2 = 3 / (r_val * r_val);
          
        double mOverR = m_val / r_val * km_to_mSun;
        double oneMinusTwoMOverR = 1 - 2 * mOverR;
          
        double inner_term = mOverR / r_val + G / (c * c * c * c) * 4 * pi * r_val * p_val;
        double term3 = 2 / oneMinusTwoMOverR * inner_term * inner_term;
          
        double term4 = beta_val / r_val * (
          -1 + mOverR + 2 * pi * r_val * r_val * G / (c * c) * (e_val - p_val / (c * c))
        );
          
        dbetadr[idx] = (H_val * (term1 + term2 + term3) + term4) * 2 / oneMinusTwoMOverR;
        dHdr[idx] = beta_val;
      }
    }
    ''', 'love_eq_kernel')
      
    threads_per_block = 256
    blocks = (batch_size + threads_per_block - 1) // threads_per_block
      
    love_eq_kernel((blocks,), (threads_per_block,),
            (d_beta, d_H, d_r, e_values, p_values, m_values, de_dp_values,
              dbetadr, dHdr, G, c, pi, km_to_mSun, batch_size))
    
      
    return [dbetadr, dHdr]
  

  def solve_love_eq_batch(self, init_params, r_array, args=None, rtol=None):
      beta, H = init_params
      batch_size = 1 
      if hasattr(beta, "__len__"):
          batch_size = len(beta)
          
      n_steps = len(r_array)
    
      results = np.zeros((batch_size, n_steps, 2), dtype=np.float64)
      results[:, 0, 0] = beta  # Initial beta
      results[:, 0, 1] = H     # Initial H
      
      d_results = cp.asarray(results)
      d_r_array = cp.asarray(r_array)
      
      
      R_dep = args[0] if args is not None else None
      
      for i in range(1, n_steps):            
          h = float(r_array[i] - r_array[i-1])
          r = float(r_array[i-1])
          r_target = float(r_array[i])
          
          beta = d_results[:, i-1, 0]
          H = d_results[:, i-1, 1]
          
          while r < r_target:
              h = min(h, r_target - r)
              
              params = [(beta[j], H[j]) for j in range(batch_size)]
              
              k1_beta, k1_H = self.love_eq(params, [r] * batch_size, R_dep)
              
              beta_k2 = beta + 0.5 * h * k1_beta
              H_k2 = H + 0.5 * h * k1_H
              r_k2 = r + 0.5 * h
              params_k2 = [(beta_k2[j], H_k2[j]) for j in range(batch_size)]
              k2_beta, k2_H = self.love_eq(params_k2, [r_k2] * batch_size, R_dep)
              
              beta_k3 = beta + 0.5 * h * k2_beta
              H_k3 = H + 0.5 * h * k2_H
              params_k3 = [(beta_k3[j], H_k3[j]) for j in range(batch_size)]
              k3_beta, k3_H = self.love_eq(params_k3, [r_k2] * batch_size, R_dep)
              
              beta_k4 = beta + h * k3_beta
              H_k4 = H + h * k3_H
              r_k4 = r + h
              params_k4 = [(beta_k4[j], H_k4[j]) for j in range(batch_size)]
              k4_beta, k4_H = self.love_eq(params_k4, [r_k4] * batch_size, R_dep)
              
              # RK4 update
              beta_new = beta + h * (k1_beta + 2*k2_beta + 2*k3_beta + k4_beta) / 6
              H_new = H + h * (k1_H + 2*k2_H + 2*k3_H + k4_H) / 6
              
              # Error estimation
              beta_error = cp.max(cp.abs(beta_new - beta - h * (k1_beta + 2*k2_beta + 2*k3_beta + k4_beta) / 6))
              H_error = cp.max(cp.abs(H_new - H - h * (k1_H + 2*k2_H + 2*k3_H + k4_H) / 6))
              max_error = max(float(beta_error), float(H_error))
              
              if rtol and max_error > rtol:
                  h *= 0.5  
                  continue  
              elif rtol and max_error < rtol / 10:
                  h *= 1.5  
              
              beta = beta_new
              H = H_new
              r = r + h
              
              if r >= r_target:
                  break
              
          d_results[:, i, 0] = beta
          d_results[:, i, 1] = H
      
      return cp.asnumpy(d_results)
  _tov_eq_batch_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void tov_eq_batch(const double* P, const double* m, const double r, double* dPdr, double* dmdr, 
                    const double* eden, const double min_p, const double max_p,
                    const double G, const double c, const double pi, const int batch_size) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (idx < batch_size) {
            if (P[idx] < min_p || P[idx] > max_p) {
                dPdr[idx] = 0.0;
                dmdr[idx] = 0.0;
                return;
            }
            
            double term1 = -G * (eden[idx] + P[idx] / (c * c));
            double term2 = m[idx] + 4.0 * pi * r * r * r * P[idx] / (c * c);
            double term3 = r * (r - 2.0 * G * m[idx] / (c * c));
            
            dPdr[idx] = term1 * term2 / term3;
            dmdr[idx] = 4.0 * pi * r * r * eden[idx];
        }
    }
    ''', 'tov_eq_batch')
  
  def solve_tov_batch(self, init_pressures, r_array, rtol=None):
        P, m = init_pressures
        batch_size = 1 
        if hasattr(P, "__len__"):
            batch_size = len(P)
            
        n_steps = len(r_array)
      
        results = np.zeros((batch_size, n_steps, 2), dtype=np.float64)
        results[:, 0, 0] = P  # Initial pressure
        results[:, 0, 1] = m  # Initial mass
        
        # Transfer data to GPU
        d_results = cp.asarray(results)
        d_r_array = cp.asarray(r_array)
        
        threads_per_block = 256
        blocks = (batch_size + threads_per_block - 1) // threads_per_block
        
        for i in range(1, n_steps):            
            h = float(r_array[i] - r_array[i-1])
            r = float(r_array[i-1])
            r_target = float(r_array[i])
            
            P = d_results[:, i-1, 0]
            m = d_results[:, i-1, 1]
            
            dPdr = cp.zeros(batch_size, dtype=cp.float64)
            dmdr = cp.zeros(batch_size, dtype=cp.float64)
            
            while r < r_target:
                h = min(h, r_target - r)
                
                eden = cp.asarray(self.en_dens(cp.asnumpy(P)))
                
                TOVGPU._tov_eq_batch_kernel((blocks,), (threads_per_block,), 
                                    (P, m, r, dPdr, dmdr, eden, self.min_p, self.max_p, G, c, pi, batch_size))
                k1_P = dPdr.copy()
                k1_m = dmdr.copy()
                
                P_k2 = P + 0.5 * h * k1_P
                m_k2 = m + 0.5 * h * k1_m
                r_k2 = r + 0.5 * h
                eden = cp.asarray(self.en_dens(cp.asnumpy(P_k2)))
                TOVGPU._tov_eq_batch_kernel((blocks,), (threads_per_block,), 
                                    (P_k2, m_k2, r_k2, dPdr, dmdr, eden, self.min_p, self.max_p, G, c, pi, batch_size))
                k2_P = dPdr.copy()
                k2_m = dmdr.copy()
                
                P_k3 = P + 0.5 * h * k2_P
                m_k3 = m + 0.5 * h * k2_m
                eden = cp.asarray(self.en_dens(cp.asnumpy(P_k3)))
                TOVGPU._tov_eq_batch_kernel((blocks,), (threads_per_block,), 
                                    (P_k3, m_k3, r_k2, dPdr, dmdr, eden, self.min_p, self.max_p, G, c, pi, batch_size))
                k3_P = dPdr.copy()
                k3_m = dmdr.copy()
                
                P_k4 = P + h * k3_P
                m_k4 = m + h * k3_m
                r_k4 = r + h
                eden = cp.asarray(self.en_dens(cp.asnumpy(P_k4)))
                TOVGPU._tov_eq_batch_kernel((blocks,), (threads_per_block,), 
                                    (P_k4, m_k4, r_k4, dPdr, dmdr, eden, self.min_p, self.max_p, G, c, pi, batch_size))
                k4_P = dPdr.copy()
                k4_m = dmdr.copy()
                
                #RK4 update
                P_new = P + h * (k1_P + 2*k2_P + 2*k3_P + k4_P) / 6
                m_new = m + h * (k1_m + 2*k2_m + 2*k3_m + k4_m) / 6
                
                #Error estimation
                P_error = cp.max(cp.abs(P_new - P - h * (k1_P + 2*k2_P + 2*k3_P + k4_P) / 6))
                m_error = cp.max(cp.abs(m_new - m - h * (k1_m + 2*k2_m + 2*k3_m + k4_m) / 6))
                max_error = max(float(P_error), float(m_error))
                
                if max_error > rtol:
                    h *= 0.5  
                    continue  
                elif max_error < rtol / 10:
                    h *= 1.5  
                
                P = P_new
                m = m_new
                r = r + h
                
                if r >= r_target:
                    break
                
            
            d_results[:, i, 0] = P
            d_results[:, i, 1] = m
        
        return cp.asnumpy(d_results)


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
    c_dens *= MeV_fm3_to_pa_cgs / c ** 2

    self.check_density(c_dens)

    r = cp.arange(dr, rmax + dr, dr) 

    P = self.press(c_dens)
    eden = self.en_dens(P)
    m = 4.0 * pi * r[0] ** 3 * eden

    psol = TOVGPU.solve_tov_batch(self, [P, m], r, rtol=rtol)
    p_R, m_R = psol[:,:,0], psol[:,:,1] 

    # find the boundary of the star by finding point
    # where the mass stops to increase

    batch_size = m_R.shape[0] 
    d_ind_array = cp.zeros(batch_size, dtype=cp.int32)

    find_star_boundary = cp.RawKernel(r'''
    extern "C" __global__
    void batch_find_star_boundary(const double* m_R, int n_points, int batch_size, double dmrel, int* ind_array) {
      int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
      
      if (batch_idx < batch_size) {
        ind_array[batch_idx] = -1;
        
        const double* batch_m_R = m_R + batch_idx * n_points;
        
        for (int i = 1; i < n_points; i++) {
          double curr = batch_m_R[i];
          double prev = batch_m_R[i-1];
          
          if (curr != 0) {
            double diff = (curr - prev) / curr;
            if (diff < dmrel) {
              ind_array[batch_idx] = i;
              break;
            }
          }
        }
        
        if (ind_array[batch_idx] == -1) {
          ind_array[batch_idx] = n_points - 1;
        }
      }
    }
    ''', 'find_star_boundary')

    threads_per_block = 256
    blocks = (batch_size + threads_per_block - 1) // threads_per_block
    find_star_boundary(
        (blocks,), (threads_per_block,),
        (m_R, len(r), batch_size, dmrel, d_ind_array)
    )

    ind_array = cp.asnumpy(d_ind_array)
    extract_values_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void extract_star_values(const double* m_R, const double* r, 
          const int* ind_array, double* M_values, double* R_values,
          int n_points, int batch_size) {
      int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
      
      if (batch_idx < batch_size) {
      int ind = ind_array[batch_idx];
      if (ind > 0 && ind < n_points) {
        M_values[batch_idx] = m_R[batch_idx * n_points + ind - 1];
        R_values[batch_idx] = r[ind - 1];
      } else {
        M_values[batch_idx] = 0.0;
        R_values[batch_idx] = 0.0;
      }
      }
    }
    ''', 'extract_star_values')

    d_M_values = cp.zeros(batch_size, dtype=cp.float64)
    d_R_values = cp.zeros(batch_size, dtype=cp.float64)

    threads_per_block = 256
    blocks = (batch_size + threads_per_block - 1) // threads_per_block
    extract_values_kernel((blocks,), (threads_per_block,),
           (m_R, r, ind_array, d_M_values, d_R_values, 
          len(r), batch_size))

    M = cp.asnumpy(d_M_values)
    R = cp.asnumpy(d_R_values)

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

    solution = TOVGPU.solve_love_eq_batch(self, [beta0, H0], r, args=([e_R, p_R, m_R],), rtol=rtol)
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
en_arr1 = cp.linspace(10, 1000, 5000000)
nn=100
lmao=np.linspace(0.1,nn/100*2,nn)
en_arr = cp.zeros((nn, len(en_arr1)))
for i in range(nn):
    en_arr[i] = en_arr1 * lmao[i]
p_arr = 0.1 * en_arr ** 1.5
solver = TOVGPU(en_arr, p_arr)
R_arr, M_arr, profiles = solver.solve(central_densities)
print(f"Raios das estrelas (km): {R_arr}")
print(f"Massas das estrelas (M☉): {M_arr}")
elapsed_time = time.perf_counter() - start_time
print(f"Tempo de execução: {elapsed_time:.6f} segundos")




