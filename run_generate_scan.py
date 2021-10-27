from generate_scan import generate_scan

nside = 64
alpha = 45.
prec_hr = 3.2058
beta = 50.
spin_rpm = 0.05
det_angle = 0.
days = 365. * 3.
days_out = 15.
sampling_hz = 19.
hwp_rpm = 46. # = 0. for no HWP 

generate_scan(nside, alpha, prec_hr, beta, spin_rpm, det_angle, days, days_out, sampling_hz, hwp_rpm)
