import numpy as np
import pandas as pd


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

σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant

h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)
h

G = np.zeros((27,27))

# ventilation flow rate
Va = l**3                   # m³, volume of air
ACH = 1                     # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration
Kp = 5 # W/K

# ventilation & advection
G[0,0]=G[26,26] = air['Density'] * air['Specific heat'] * Va_dot
G[1,1]=Kp
G[2,2] = h['out'] * (wall_bathroom['Surface'].iloc[6]-3*2)     # on enlève la partie du mur négligée
G[3,3] = 2*(wall_bathroom['Surface'].iloc[6]-3*2)*wall_bathroom['Conductivity'].iloc[6]/wall_bathroom['Width'].iloc[6]    # on enlève la partie du mur négligée
G[4,4] = G[3,3]
G[5,5]= 2*(wall_bathroom['Surface'].iloc[3]-3*2)*wall_bathroom['Conductivity'].iloc[3]/wall_bathroom['Width'].iloc[3]     # on enlève la partie du mur négligée
G[6,6] = G[5,5]
G[7,7]= h['in'] * (wall_bathroom['Surface'].iloc[3]-3*2)  # on enlève la partie du mur négligée
G[8,8]=h['in'] * wall_bathroom['Surface'].iloc[1]
G[9,9]=2*wall_bathroom['Surface'].iloc[1]*wall_bathroom['Conductivity'].iloc[1]/wall_bathroom['Width'].iloc[1]
G[10,10]=2*wall_bathroom['Surface'].iloc[1]*wall_bathroom['Conductivity'].iloc[1]/wall_bathroom['Width'].iloc[1]
G[11,11]= h['in'] * wall_salon['Surface'].iloc[1]
G[12,12]= h['in'] * wall_salon['Surface'].iloc[4]
G[13,13]= h['out'] * wall_salon['Surface'].iloc[4]
G[14,14]= h['in'] * wall_bathroom['Surface'].iloc[5]
G[15,15]= h['in'] * wall_bathroom['Surface'].iloc[9]
G[16,16]= 2*door['Surface']*door['Conductivity']/door['Width']
G[17,17]=G[16,16]
G[18,18]= h['in'] * wall_salon['Surface'].iloc[9]
G[19,19]= h['in'] * (wall_salon['Surface'].iloc[2]-5*3)
G[20,20]= h['out'] * wall_bathroom['Surface'].iloc[5]
G[21,21]= 2*(wall_salon['Surface'].iloc[2]-5*3)*wall_bathroom['Conductivity'].iloc[2]/wall_bathroom['Width'].iloc[2]
G[22,22]=G[21,21]
G[23,23]=2*(wall_salon['Surface'].iloc[0]-5*3)*wall_bathroom['Conductivity'].iloc[0]/wall_bathroom['Width'].iloc[0]
G[24,24]= G[23,23]
G[25,25]=h['out'] * wall_salon['Surface'].iloc[0]


    
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






f = np.zeros(20)  




# flow-rate sources f for steady state

# ----- C matrix -----

T0=15 #données météo
Tisp=19 # à déinir
air_bathroom = {'Conductivity': 1.4,               # W/(m·K)
          'Density': 1.2,                      # kg/m³                  
          'Specific heat': 1000,             # J/(kg⋅K)
          'Volume': 30, }                    # m^3

air_salon = {'Conductivity': 1.4,               # W/(m·K)
          'Density': 1.2,                      # kg/m³                  
          'Specific heat': 1000,             # J/(kg⋅K)
          'Volume': 75, }                    # m^3
                  

C1=concrete_extern_bathroom['Specific heat']*concrete_extern_bathroom['Density']*concrete_extern_bathroom['Width']*concrete_extern_bathroom['Surface']
C3=insulation_bathroom['Specific heat']*insulation_bathroom['Density']*insulation_bathroom['Width']*insulation_bathroom['Surface']
C5=air_bathroom['Specific heat']*air_bathroom['Density']*air_bathroom['Volume']
C7=concrete_intern_bathroom['Specific heat']*concrete_intern_bathroom['Density']*concrete_intern_bathroom['Width']*concrete_intern_bathroom['Surface']
C9=air_salon['Specific heat']*air_salon['Density']*air_salon['Volume']
C10=glass_salon['Specific heat']*glass_salon['Density']*glass_salon['Width']*glass_salon['Surface']
C14=glass_bathroom['Specific heat']*glass_bathroom['Density']*glass_bathroom['Width']*glass_bathroom['Surface']
C16=insulation_salon['Specific heat']*insulation_salon['Density']*insulation_salon['Width']*insulation_salon['Surface']
C18=concrete_extern_salon['Specific heat']*concrete_extern_salon['Density']*concrete_extern_salon['Width']*concrete_extern_salon['Surface']

Cdiag=[0, C1, 0, C3, 0, C5, 0, C7, 0, C9, C10, 0, 0, 0, C14, 0, C16, 0, C18, 0]
C=np.zeros((20,20))
for k in range(20):
    C[k,k]=Cdiag[k]
# ----- b and f vectors -----

b = np.zeros(27)
b[0]=b[2]=b[13]=b[20]=b[25]=T0
b[1]=Tisp    

Es = 307 #W/m²
Ew = 391 #W/m²
En = 150 #W/m²

# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

# view factor
Fwall_ground = 0.5
Fwall_sky = 0.5
Fw_externgbathroom = (concrete_extern_bathroom['Surface']-3*2) / (glass_bathroom['Surface']+concrete_intern_bathroom['Surface']+ceiling_soil_bathroom['Surface'])
Fw_interngbathroom = (concrete_intern_bathroom['Surface']) / (glass_bathroom['Surface']+concrete_extern_bathroom['Surface']+ceiling_soil_bathroom['Surface'])
Fw_externgsalon = (concrete_extern_salon['Surface']-3*5) / (glass_salon['Surface']+concrete_intern_salon['Surface']+ceiling_soil_salon['Surface'])
Fw_interngsalon = (concrete_intern_salon['Surface']) / (glass_salon['Surface']+concrete_extern_salon['Surface']+ceiling_soil_salon['Surface'])

phi0 = Fwall_sky*En*(concrete_extern_bathroom['Surface']-2*3)
phi14 = Fwall_sky*Ew*glass_bathroom['Surface']
phi10 = Fwall_sky*Es*glass_salon['Surface']
phi19 = Fwall_sky*Ew*concrete_extern_salon['Surface']/2
phi4 = Fw_externgbathroom*phi14*τ_gSW*α_wSW
phi6 = Fw_interngbathroom*phi14*τ_gSW*α_wSW
phi8 = Fw_interngsalon*phi10*τ_gSW*α_wSW
phi15 = Fw_externgsalon*phi10*τ_gSW*α_wSW
Qa1 = 90+70 # one standing people and some machine in W
Qa2 = 90 # one standing people in W

f=[phi0, 0, 0, 0, phi4, Qa2, phi6, 0, phi8, Qa1, phi10, 0, 0, 0, phi14, phi15, 0, 0, 0, phi19 ]


# temperature nodes
nθ = 20      # number of temperature nodes
θ = [f'θ{i}' for i in range(nθ)]

# flow-rate branches
nq = 27     # number of flow branches
q = [f'q{i}' for i in range(nq)]


θ = np.linalg.inv(A.T @ G @ A ) @ (A.T @ G @ b + f)
print(f'θ = {np.around(θ, 1)} °C')

print(f'Bathroom = {np.around(θ[5], 1)} °C')
print(f'Living room = {np.around(θ[9], 1)} °C')