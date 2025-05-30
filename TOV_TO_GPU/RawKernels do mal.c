#include <stdio.h>
#include <stdlib.h> 
#include <math.h>

#define N  10000
#define T_MAX 5
float h = 1.0/N;

float f(float y, float t) {
    return -y + sin(t) + cos(t);
}

int rk4d(float y, float t, float h) {
    while(t < T_MAX) {
        float k1 = h * f(y, t);
        float k2 = h * f(y + k1 / 2, t + h / 2);
        float k3 = h * f(y + k2 / 2, t + h / 2);
        float k4 = h * f(y + k3, t + h);
        y += (k1 + 2*k2 + 2*k3 + k4) / 6;
        t += h;
    }
    return y;
}

int main(){
    float y0 = 0.0;
    float t0 = 0.0;
    float y = rk4d(y0, t0, h);
    printf("Final value of y: %f\n", y);
    return 0;


}