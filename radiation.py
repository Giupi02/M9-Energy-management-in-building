import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

l = 3               # m length of the cubic room
Sg = l**2           # m² surface area of the glass wall
Sc = Si = 5 * Sg    # m² surface area of concrete & insulation of the 5 walls

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

concrete_intern_salon = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 960,           # J/(kg⋅K)
            'Width': 0.075,                   # m
            'Surface': 5*3}            # m²

concrete_extern_salon = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 960,           # J/(kg⋅K)
            'Width': 0.2,                   # m
            'Surface': 5*3*2}            # m²

insulation_salon = {'Conductivity': 0.027,        # W/(m·K)
              'Density': 50.0,              # kg/m³
              'Specific heat': 840,        # J/(kg⋅K)
              'Width': 0.1,                # m
              'Surface': 5*2*3}          # m²

glass_salon = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.024,                     # m
         'Surface': 5*3}                   # m²

concrete_intern_bathroom = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 960,           # J/(kg⋅K)
            'Width': 0.075,                   # m
            'Surface': 5*3}            # m²

concrete_extern_bathroom = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 960,           # J/(kg⋅K)
            'Width': 0.2,                   # m
            'Surface': 5*3+2*3}            # m²

insulation_bathroom = {'Conductivity': 0.027,        # W/(m·K)
              'Density': 50.0,              # kg/m³
              'Specific heat': 840,        # J/(kg⋅K)
              'Width': 0.1,                # m
              'Surface': 5*3+2*3}          # m²

glass_bathroom = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.024,                     # m
         'Surface': 2*3}                   # m²

ceiling_soil_bathroom = {                    # m
         'Surface': 2*5*2}   

ceiling_soil_salon = {                    # m
         'Surface': 2*5*5}   
door = {'Conductivity' : 0.17,
        'Width' : 5,
        'Surface' : 0.9*2.1,
        'Specific heat' : 2000,
        'Density' : 800 }

wall_salon = pd.DataFrame.from_dict({'Layer_out': concrete_extern_salon,
                                     'Layer_out1': concrete_intern_bathroom,
                               'Layer_in': insulation_salon,
                               'Layer_in1': insulation_bathroom,
                               'Glass': glass_salon,
                               'Glass1': glass_bathroom,
                               'Layer_out2' : concrete_extern_bathroom,
                               'roof' : ceiling_soil_bathroom, 
                               'roof1' : ceiling_soil_salon,
                               'door' : door
                               },
                              orient='index')
wall_salon

wall_bathroom = pd.DataFrame.from_dict({'Layer_out': concrete_extern_salon,
                                     'Layer_out1': concrete_intern_bathroom,
                               'Layer_in': insulation_salon,
                               'Layer_in1': insulation_bathroom,
                               'Glass': glass_salon,
                               'Glass1': glass_bathroom,
                               'Layer_out2' : concrete_extern_bathroom,
                               'roof' : ceiling_soil_bathroom, 
                               'roof1' : ceiling_soil_salon,
                               'door' : door
                               },
                              orient='index')
wall_bathroom

# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant

h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)
h

G = np.zeros(27)

# ventilation flow rate
Va = l**3                   # m³, volume of air
ACH = 1                     # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration

# ventilation & advection
G[0,0]=G[26,26] = air['Density'] * air['Specific heat'] * Va_dot
G[2,2] = h * (wall_bathroom['Surface'].iloc[6]-3*2)     # on enlève la partie du mur négligée 
G[3,3] = 2*(wall_bathroom['Surface'].iloc[6]-3*2)*wall_bathroom['Conductivity'].iloc[6]/wall_bathroom['Width'].iloc[6]    # on enlève la partie du mur négligée 
G[4,4] = G[3,3]
G[5,5]= 2*(wall_bathroom['Surface'].iloc[3]-3*2)*wall_bathroom['Conductivity'].iloc[3]/wall_bathroom['Width'].iloc[3]     # on enlève la partie du mur négligée 
G[6,6] = G[5,5]
G[7,7]= h * (wall_bathroom['Surface'].iloc[3]-3*2)  # on enlève la partie du mur négligée 
G[8,8]=h * wall_bathroom['Surface'].iloc[1]
G[9,9]=2*wall_bathroom['Surface'].iloc[1]*wall_bathroom['Conductivity'].iloc[1]/wall_bathroom['Width'].iloc[1]
G[10,10]=2*wall_bathroom['Surface'].iloc[1]*wall_bathroom['Conductivity'].iloc[1]/wall_bathroom['Width'].iloc[1]
G[11,11]= h * wall_salon['Surface'].iloc[1]
G[12,12]= h * wall_salon['Surface'].iloc[4]
G[13,13]= h * wall_salon['Surface'].iloc[4]
G[14,14]= h * wall_bathroom['Surface'].iloc[5] 
G[15,15]= h * wall_bathroom['Surface'].iloc[9] 
G[16,16]= 2*wall_bathroom['Surface'].iloc[9]*wall_bathroom['Conductivity'].iloc[9]/wall_bathroom['Width'].iloc[9]
G[17,17]=G[16,16]
G[18,18]= h * wall_salon['Surface'].iloc[9] 
G[19,19]= h * (wall_salon['Surface'].iloc[2]-5*3)
G[20,20]= h * wall_bathroom['Surface'].iloc[5] 
G[21,21]= 2*(wall_salon['Surface'].iloc[2]-5*3)*wall_bathroom['Conductivity'].iloc[2]/wall_bathroom['Width'].iloc[2]
G[22,22]=G[21,21]
G[23,23]=2*(wall_salon['Surface'].iloc[0]-5*3)*wall_bathroom['Conductivity'].iloc[0]/wall_bathroom['Width'].iloc[0]
G[24,24]= G[23,23]
G[25,25]=h * wall_salon['Surface'].iloc[0] 

# view factor
Fw_externgbathroom = (concrete_extern_bathroom['Surface']-3*2) / (glass_bathroom['Surface']+concrete_intern_bathroom['Surface']+ceiling_soil_bathroom['Surface'])
Fw_interngbathroom = (concrete_intern_bathroom['Surface']) / (glass_bathroom['Surface']+concrete_extern_bathroom['Surface']+ceiling_soil_bathroom['Surface'])
Fw_externgsalon = (concrete_extern_salon['Surface']-3*5) / (glass_salon['Surface']+concrete_intern_salon['Surface']+ceiling_soil_salon['Surface'])
Fw_interngsalon = (concrete_intern_salon['Surface']) / (glass_salon['Surface']+concrete_extern_salon['Surface']+ceiling_soil_salon['Surface'])



T_int = 273.15 + np.array([0, 40])
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For 0°C < (T/K - 273.15)°C < 40°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

T_int = 273.15 + np.array([10, 30])
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For 10°C < (T/K - 273.15)°C < 30°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

T_int = 273.15 + 20
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For (T/K - 273.15)°C = 20°C, 4σT³ = {4 * σ * T_int**3:.1f} W/(m²·K)')

#For 0°C < (T/K - 273.15)°C < 40°C, 4σT³/[W/(m²·K)] ∈ [4.6 7. ]
#For 10°C < (T/K - 273.15)°C < 30°C, 4σT³/[W/(m²·K)] ∈ [5.1 6.3]
#For (T/K - 273.15)°C = 20°C, 4σT³ = 5.7 W/(m²·K)

nr = 27
nc= 20

A = np.zeros([nr, nc])



A[0, 5] = 1
A[1, 9] = 1
A[2, 0] = 1

j=0
for i in range(3,13):
    A[i,j], A[i,j+1] = -1, 1
    j+=1
    
A[13, 10] = 1
A[14, 14], A[14, 5] = -1, 1
A[14, 14], A[14, 5] = -1, 1
A[15, 5], A[15, 11] = -1, 1
A[16, 11], A[16, 12] = -1, 1
A[17, 12], A[17, 13] = -1, 1
A[18, 13], A[18, 9] = -1, 1
A[19, 15], A[19, 9] = -1, 1
A[20, 14] = 1
 
j=15
for i in range(21,25):
    A[i,j+1], A[i,j] = -1, 1
    j+=1
    
A[25, 19] = 1
A[26, 9] = 1



