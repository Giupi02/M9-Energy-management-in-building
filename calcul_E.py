# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 22:26:59 2025

@author: Samsung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dm4bem import read_epw, sol_rad_tilt_surf

np.set_printoptions(precision=1)

# Data
# ====
# dimensions
L, H, Wbath, Wliving = 5, 3, 5, 2 # m

# thermo-physical propertites
λ =    1.7          # W/(m K) wall thermal conductivity
ρ, c = 1.2, 1000    # kg/m3, J/(kg K) density, specific heat air
hi, ho = 8, 25      # W/(m2 K) convection coefficients in, out

# short-wave solar radiation absorbed by each wall
filename = r'FRA_AR_Lyon-Bron.AP.074800_TMYx.2004-2018.epw'
# filename = '../weather_data/FRA_AR_Lyon-Bron.AP.074800_TMYx.2004-2018.epw'            # W/m2 

[data, meta] = read_epw(filename, coerce_year=None)
data
# Extract the month and year from the DataFrame index with the format 'MM-YYYY'
month_year = data.index.strftime('%m-%Y')

# Create a set of unique month-year combinations
unique_month_years = sorted(set(month_year))

# Create a DataFrame from the unique month-year combinations
pd.DataFrame(unique_month_years, columns=['Month-Year'])

# select columns of interest
weather_data = data[["temp_air", "dir_n_rad", "dif_h_rad"]]

# replace year with 2000 in the index 
weather_data.index = weather_data.index.map(
    lambda t: t.replace(year=2000))

weather_data.loc['2000-06-29 12:00']
# Define start and end dates
start_date = '2000-06-29 12:00'
end_date = '2000-07-02'         # time is 00:00 if not indicated

# Filter the data based on the start and end dates
weather_data = weather_data.loc[start_date:end_date]
del data
weather_data
weather_data['temp_air'].plot()
plt.xlabel("Time")
plt.ylabel("Dry-bulb air temperature, θ / °C")
plt.legend([])
plt.show()

weather_data[['dir_n_rad', 'dif_h_rad']].plot()
plt.xlabel("Time")
plt.ylabel("Solar radiation, Φ / (W·m⁻²)")
plt.legend(['$Φ_{direct}$', '$Φ_{diffuse}$'])
plt.show()

surface_orientation_south = {'slope': 90,     # 90° is vertical; > 90° downward
                       'azimuth': 0,    # 0° South, positive westward
                       'latitude': 45} # °, North Pole 90° positive
surface_orientation_west = {'slope': 90,     # 90° is vertical; > 90° downward
                       'azimuth': -90,    # 0° South, positive westward
                       'latitude': 45}# °, North Pole 90° positive
surface_orientation_north = {'slope': 90,     # 90° is vertical; > 90° downward
                       'azimuth': 180,    # 0° South, positive westward
                       'latitude': 45}# °, North Pole 90° positive
albedo = 0.45
E=[0,0,0] 
E[0]=rad_surf = sol_rad_tilt_surf(weather_data, surface_orientation_south, albedo)
E[1]=rad_surf1 = sol_rad_tilt_surf(weather_data, surface_orientation_west, albedo)
E[2]=rad_surf2 = sol_rad_tilt_surf(weather_data, surface_orientation_north, albedo)

# E[0].plot()
# plt.xlabel("Time")
# plt.ylabel("Solar irradiance,  Φ / (W·m⁻²)")
# plt.show()

# E[1].plot()
# plt.xlabel("Time")
# plt.ylabel("Solar irradiance,  Φ / (W·m⁻²)")
# plt.show()

# E[2].plot()
# plt.xlabel("Time")
# plt.ylabel("Solar irradiance,  Φ / (W·m⁻²)")
# plt.show()

#print(f"{rad_surf.loc['2000-06-29 12:00']['direct']:.0f} W/m²")

# print(f"Mean. direct irradiation: {rad_surf['direct'].mean():.0f} W/m²")
# print(f"Max. direct irradiation:  {rad_surf['direct'].max():.0f} W/m²")
# print(f"Direct solar irradiance is maximum on {rad_surf['direct'].idxmax()}")

# print(f"{rad_surf1.loc['2000-06-29 12:00']['direct']:.0f} W/m²")

# print(f"Mean.1 direct irradiation: {rad_surf1['direct'].mean():.0f} W/m²")
# print(f"Max.1 direct irradiation:  {rad_surf1['direct'].max():.0f} W/m²")
# print(f"Direct solar irradiance 1 is maximum on {rad_surf1['direct'].idxmax()}")

# print(f"{rad_surf2.loc['2000-06-29 12:00']['direct']:.0f} W/m²")

# print(f"Mean.2 direct irradiation: {rad_surf2['direct'].mean():.0f} W/m²")
# print(f"Max.2 direct irradiation:  {rad_surf2['direct'].max():.0f} W/m²")
# print(f"Direct solar irradiance is maximum on {rad_surf2['direct'].idxmax()}")
β = surface_orientation_south['slope']
γ = surface_orientation_south['azimuth']
ϕ = surface_orientation_south['latitude']

β1 = surface_orientation_west['slope']
γ1 = surface_orientation_west['azimuth']
ϕ1 = surface_orientation_west['latitude']

β2 = surface_orientation_north['slope']
γ2 = surface_orientation_north['azimuth']
ϕ2 = surface_orientation_north['latitude']

# Transform degrees in radians
β = β * np.pi / 180
γ = γ * np.pi / 180
ϕ = ϕ * np.pi / 180

β1 = β1 * np.pi / 180
γ1 = γ1 * np.pi / 180
ϕ1 = ϕ1 * np.pi / 180

β2 = β2 * np.pi / 180
γ2 = γ2 * np.pi / 180
ϕ2 = ϕ2 * np.pi / 180


n = weather_data.index.dayofyear

declination_angle = 23.45 * np.sin(360 * (284 + n) / 365 * np.pi / 180)
δ = declination_angle * np.pi / 180

hour = weather_data.index.hour
minute = weather_data.index.minute + 60
hour_angle = 15 * ((hour + minute / 60) - 12)   # deg
ω = hour_angle * np.pi / 180       

theta = np.sin(δ) * np.sin(ϕ) * np.cos(β) \
    - np.sin(δ) * np.cos(ϕ) * np.sin(β) * np.cos(γ) \
    + np.cos(δ) * np.cos(ϕ) * np.cos(β) * np.cos(ω) \
    + np.cos(δ) * np.sin(ϕ) * np.sin(β) * np.cos(γ) * np.cos(ω) \
    + np.cos(δ) * np.sin(β) * np.sin(γ) * np.sin(ω)

theta1 = np.sin(δ) * np.sin(ϕ1) * np.cos(β1) \
    - np.sin(δ) * np.cos(ϕ1) * np.sin(β1) * np.cos(γ1) \
    + np.cos(δ) * np.cos(ϕ1) * np.cos(β1) * np.cos(ω) \
    + np.cos(δ) * np.sin(ϕ1) * np.sin(β1) * np.cos(γ1) * np.cos(ω) \
    + np.cos(δ) * np.sin(β1) * np.sin(γ1) * np.sin(ω)
    
theta2 = np.sin(δ) * np.sin(ϕ2) * np.cos(β2) \
        - np.sin(δ) * np.cos(ϕ2) * np.sin(β2) * np.cos(γ2) \
        + np.cos(δ) * np.cos(ϕ2) * np.cos(β2) * np.cos(ω) \
        + np.cos(δ) * np.sin(ϕ2) * np.sin(β2) * np.cos(γ2) * np.cos(ω) \
        + np.cos(δ) * np.sin(β2) * np.sin(γ2) * np.sin(ω)
        
theta = np.array(np.arccos(theta))
theta1 = np.array(np.arccos(theta1))
theta2 = np.array(np.arccos(theta2))
theta = np.minimum(theta, np.pi / 2)  
theta1 = np.minimum(theta1, np.pi / 2)    
theta2 = np.minimum(theta2, np.pi / 2)             

dir_rad = weather_data["dir_n_rad"] * np.cos(theta)
dir_rad[dir_rad < 0] = 0
dif_rad = weather_data["dif_h_rad"] * (1 + np.cos(β)) / 2

dir_rad = weather_data["dir_n_rad"] * np.cos(theta1)
dir_rad[dir_rad < 0] = 0
dif_rad = weather_data["dif_h_rad"] * (1 + np.cos(β1)) / 2

dir_rad = weather_data["dir_n_rad"] * np.cos(theta2)
dir_rad[dir_rad < 0] = 0
dif_rad = weather_data["dif_h_rad"] * (1 + np.cos(β2)) / 2

gamma = np.cos(δ) * np.cos(ϕ) * np.cos(ω) \
    + np.sin(δ) * np.sin(ϕ)
    
gamma1 = np.cos(δ) * np.cos(ϕ1) * np.cos(ω) \
    + np.sin(δ) * np.sin(ϕ1)
    
gamma2 = np.cos(δ) * np.cos(ϕ2) * np.cos(ω) \
    + np.sin(δ) * np.sin(ϕ2)

gamma = np.array(np.arcsin(gamma))
gamma[gamma < 1e-5] = 1e-5

gamma1 = np.array(np.arcsin(gamma1))
gamma1[gamma1 < 1e-5] = 1e-5

gamma2 = np.array(np.arcsin(gamma2))
gamma2[gamma2 < 1e-5] = 1e-5

dir_h_rad = weather_data["dir_n_rad"] * np.sin(gamma)

ref_rad = (dir_h_rad + weather_data["dif_h_rad"]) * albedo \
        * (1 - np.cos(β) / 2)  # rad
        
dir_h_rad = weather_data["dir_n_rad"] * np.sin(gamma1)

ref_rad = (dir_h_rad + weather_data["dif_h_rad"]) * albedo \
        * (1 - np.cos(β1) / 2)  # rad
        
dir_h_rad = weather_data["dir_n_rad"] * np.sin(gamma2)

ref_rad = (dir_h_rad + weather_data["dif_h_rad"]) * albedo \
        * (1 - np.cos(β2) / 2)  # rad
        
# outdoor temperature
To = 0              # °C

# ventilation rate (air-changes per hour)
ACH = 1             # volume/h

# volume and mass for the air rate change 
V_Bath = L * Wbath * H * ACH / 3600 
V_Livingroom = L * Wliving * H * ACH / 3600 # volumetric air flow rate
m_bath = ρ * V_Bath
m_living =  ρ * V_Livingroom # mass air flow rate

#  number of flow branches and temp nodes 
nq = 16
np1 = 22

A = np.zeros([nq, np1])

Gp= np.array(A.shape[0])


Es = rad_surf['direct'].max()
Ew = rad_surf1['direct'].max()
En = rad_surf2['direct'].max()

# calcul_E.py
E_f = [Es, Ew, En]  # dentro una lista
