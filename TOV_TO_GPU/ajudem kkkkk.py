import cupy as cp

rk4d = r """
extern "C" __global__ 
void rk4d_kernel( float* y, float* t, float h, double rtol, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i => =n) return;
    float yi = y[i];
    float ti = t[i];
    
        float k1 = f(yi, ti);
        float k2 = f(yi +h*k1/2, ti +h/2);
        float k3 = f(yi +h*k2/2, ti +h/2);
        float k4 = f(yi +h*k3, ti +h);
        float y_next = yi +h*(k1 +2*k2 +2*k3 +k4)/6;
        error = cp
    
}
