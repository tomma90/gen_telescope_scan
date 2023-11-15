import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
import time
from numba import jit

# Normalize Vector or Quaternion
@jit
def normalize(v):
	size = v.size
	if size == 3:
		lenght = np.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

	if size == 4:
		lenght = np.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3])
	
	return v/lenght

# Quaternion Product
@jit
def q_mult(q1, q2):
	w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
	x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
	y = q1[0] * q2[2] + q1[2] * q2[0] + q1[3] * q2[1] - q1[1] * q2[3]
	z = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
	return np.array([w, x, y, z])

# Quaternion Conjugate: q-1
@jit
def q_conjugate(q):
	return np.array([q[0], -q[1], -q[2], -q[3]])

# Rotate a Vector Using a Quaternion: qvq-1
@jit
def rotate_vector(q1, v1):
	q2 = np.concatenate((np.array([0.]),v1))
	return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

# Create a Quaternion that defines a rotation by an angle theta (deg) around a vector [x,y,z]
@jit
def rotaxis_to_q(v, theta):
	v = normalize(v)
	theta = np.deg2rad(theta)
	w = np.cos(theta/2.)
	x = v[0]*np.sin(theta/2.)
	y = v[1]*np.sin(theta/2.)
	z = v[2]*np.sin(theta/2.)
	return np.array([w, x, y, z])

# Define rotation around the Sun: 1 deg per day around vector [0,0,1] (z axis)
@jit
def rot_around_sun(v, t):
	nu_around_sun = 1./365./24./60./60.
	z = np.array([0.,0.,1.])
	q = rotaxis_to_q(z, 360.*nu_around_sun*t)
	r = rotate_vector(q, v)
	return r

# Define satellite precession around anti-sun direction: 
# initialized with anti-sun direction aligned with x axis [1,0,0]
# precession defined by the period in hour
@jit
def rot_around_antisun_dir(v, precession_hr, t):
	nu_around_antisun_dir = 1./precession_hr/60./60.
	x = np.array([1.,0.,0.])
	q = rotaxis_to_q(x, 360.*nu_around_antisun_dir*t)
	r = rotate_vector(q, v)
	return r

# Define spin around spin-axis tilted by 45 deg with respect to anti-sun direction: 
# initialized with spin-axis oriented at 45 deg from x axis in 
# the xz plane [1./np.sqrt(2),0.,1./np.sqrt(2)] spin defined by the frequency in rpm
@jit
def rot_around_spin_ax(v, spin_rpm, t):
	nu_around_spin_ax = spin_rpm/60.
	xz45 = np.array([1./np.sqrt(2),0.,1./np.sqrt(2)])
	q = rotaxis_to_q(xz45, 360.*nu_around_spin_ax*t)
	r = rotate_vector(q, v)
	return r

# Define HWP rotation around boresight direction: 
# boresight direction defined by the axis variable 
# rotation defined by the HWP frequency in rpm
@jit
def rot_HWP(v, hwp_rpm, t, axis):
	nu_around_boresight = hwp_rpm/60.
	q = rotaxis_to_q(axis, 360.*4.*nu_around_boresight*t)
	r = rotate_vector(q, v)
	return r

# Generate scan:
# nside = nside of output maps
# alpha = spin axis precession angle
# precession_hr = precession period in hours
# beta = boresight angle from spin axis
# spin_rpm = spin rate in rpm
# det_gamma = detector orientation in deg
# days = number of days to simulate
# days_out = output map interval in days
# sampling_hz = sampling rate in Hz
# hwp_rpm = HWP rotation rate in rpm (if 0 the HWP is off)
def generate_scan(nside, alpha, precession_hr, beta, spin_rpm, det_gamma, days, days_out, sampling_hz, hwp_rpm):

	z = np.array([0.,0.,1.])
	beta = np.deg2rad(beta)
	alpha = np.deg2rad(alpha)
	sec_in_day = 60.*60.*24.
	steps = int(sec_in_day*sampling_hz)

	cos4 = np.zeros(hp.nside2npix(nside))
	sin4 = np.zeros(hp.nside2npix(nside))
	cos2 = np.zeros(hp.nside2npix(nside))
	sin2 = np.zeros(hp.nside2npix(nside))
	nhit = np.zeros(hp.nside2npix(nside))
	
	# Initialization of the boresight axis in the xz plane at alpha+beta deg from the x axis  
	boresight = np.array([np.sin(np.pi/2.-alpha-beta), 0., np.cos(np.pi/2.-alpha-beta)])

	# Quaternion rotation around boresight by angle det_gamma (orientation of the detector in 
	# the plane perpendicular to the boresight)
	q_det_gamma = rotaxis_to_q(boresight, det_gamma)

	# Zeroth orientation of the polarization sensitive detector co-planar to the xz plane
	detector_orientation_0 = np.array([-boresight[2], 0., boresight[0]])

	# Rotate the detector to det_gamma angle
	detector_orientation = rotate_vector(q_det_gamma, detector_orientation_0)

	for i in range(int(days+1.)):

		# Save hit map and cross-linking maps
		if i == 1 or i%days_out == 0. or i == 365 or i == 365*2 or i == 365*3 or i == int(days):
			hp.write_map('nhit_day_%d.fits'%i, nhit)
			hp.write_map('cos2_day_%d.fits'%i, cos2)
			hp.write_map('sin2_day_%d.fits'%i, sin2)
			hp.write_map('cos4_day_%d.fits'%i, cos4)
			hp.write_map('sin4_day_%d.fits'%i, sin4)

		# Initial time
		t0 = i * sec_in_day

		for j  in range(int(steps)):

			# Increment time
			t = t0+(j+1.)/sampling_hz
			
			# Rotate boresight
			v_tmp = rot_around_spin_ax(boresight, spin_rpm, t)
			v_tmp = rot_around_antisun_dir(v_tmp, precession_hr, t)
			v_new = rot_around_sun(v_tmp, t)

			# Rotate detector vector
			det_tmp = rot_HWP(detector_orientation, hwp_rpm, t, boresight)
			det_tmp = rot_around_spin_ax(det_tmp, spin_rpm, t)
			det_tmp = rot_around_antisun_dir(det_tmp, precession_hr, t)
			det_new = rot_around_sun(det_tmp, t)

			# Great circle plane
			plane_axis = np.cross(v_new, z)

			# Pixel index
			ipix = hp.vec2pix(nside, v_new[0], v_new[1], v_new[2])

			# Detector angle on the sky with respect to the great circle
			sin = np.inner(det_new, plane_axis)/np.linalg.norm(det_new)/np.linalg.norm(plane_axis)
			sin = np.clip(sin, -1.0, 1.0)
			angle = np.arcsin(sin)
			angle = np.pi/2 - angle
			
			# Update maps
			cos4[ipix] = cos4[ipix] + np.cos(4.*angle)
			sin4[ipix] = sin4[ipix] + np.sin(4.*angle)
			cos2[ipix] = cos2[ipix] + np.cos(2.*angle)
			sin2[ipix] = sin2[ipix] + np.sin(2.*angle)
			nhit[ipix] = nhit[ipix] + 1.
		
