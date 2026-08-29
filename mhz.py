import numpy as np
import scipy.special as special
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import warnings

MPl = 1
HI = (1e-5) * MPl
L = 1e26
l = 2.0
j = 1j

f_array = np.logspace(3, 10, 50) 
c_values = [3.6, 3.45, 3.3]
colors = ['red', 'blue', 'green']

def calculate_Y(model_name, c_val, f_array):
    Y_vals = []
    N_tot = (l / 8) * (c_val**2 - 2)
    
    for f in f_array:
        k = 2*np.pi*f  
        if model_name == 'RH2S2':
            k1, k2, k3 = k / np.sqrt(6), k / np.sqrt(6), np.sqrt(2/3) * k              
            W = 0                                                                      
        elif model_name == 'UH2':
            k1, k2, k3 = k / np.sqrt(3), k / np.sqrt(3), k / np.sqrt(3)                 
            W = -(j/L)*(k2 + k3)                                                       
        elif model_name == 'Nil':
            k1, k2, k3 = np.sqrt(2/3) * k, k / np.sqrt(6), k / np.sqrt(6)               
            W = (j/L)*(k2 + k3)                                                        
        elif model_name == 'Solv':
            k1, k2, k3 = k / np.sqrt(6), k / np.sqrt(6), np.sqrt(2/3) * k              
            W = -(2*j/L)*(k1 + k2)                                                     

        Omega = np.sqrt(k**2 - W)
        Omega_over_k = Omega/k
        z = (9.1e-11)*k
        Pvk = 2*Omega_over_k*(HI/(np.pi * MPl))**2
        Ne_minus_Nk = -np.log(z)
        in_valid_window = (0 <= Ne_minus_Nk <= N_tot)
        
        if in_valid_window:
        
            tau = -z/Omega

#---------------------------------------------------------------------------------------------
            def hankel1(nu_val, x):
                return special.hankel1(nu_val, x)

            def hankel2(nu_val, x):
                return special.hankel2(nu_val, x)

#---------------------------------------------------------------------------------------------
            def calculate_nu(z_p):
                log_term = np.log(z/z_p)
                denom = l - 4*log_term
                if denom <= 0:
                    return 0.0 
                return c_val*np.sqrt((2*l) / denom) - 0.5

#---------------------------------------------------------------------------------------------
            def G_func(tau_p):
                z_p = -tau_p * Omega
                nu_val = calculate_nu(z_p)
                nu_plus = nu_val + 1
                nu_minus = nu_val - 1

                Green_num = k * np.sqrt(tau * tau_p) * (hankel1(nu_val, -k * tau_p) * hankel2(nu_val, -k * tau) - hankel1(nu_val, -k * tau) * hankel2(nu_val, -k * tau_p))
                Green_denom = 0.5 * k * tau * hankel1(nu_val, -k * tau) * (hankel2(nu_plus, -k * tau) -  hankel2(nu_minus, -k * tau)) - nu_val * hankel1(nu_val, -k * tau) * hankel2(nu_val,  -k * tau) - (k * tau) * hankel1(nu_plus, -k * tau) * hankel2(nu_val, -k * tau)
                
                return Green_num / Green_denom        

#---------------------------------------------------------------------------------------------
            def Integrand_func(z_p):
                nu_val = calculate_nu(z_p)
                if nu_val <= 2.5:
                    return 0.0
            
                q_inv_val = z*np.exp((l/8)*(c_val**2 - 2))
                gamma_term = special.gamma(nu_val + 0.5)                               
            
                term1 = ((2**(2*nu_val - 1)) * (gamma_term**2)) / np.sqrt(nu_val - 2)
                term2 = (q_inv_val)**(nu_val - 2)
                term3 = (z_p)**(-2*nu_val + 3)

                return term1 * term2 * term3

#---------------------------------------------------------------------------------------------
            def complex_quad(func, a, b):
                real_integral = integrate.quad(lambda x: np.real(func(x)), a, b, limit=200)[0]
                imag_integral = integrate.quad(lambda x: np.imag(func(x)), a, b, limit=200)[0]
                return real_integral + 1j * imag_integral

            nu_z = calculate_nu(z)
            q_inv = z*np.exp((l/8) * (c_val**2 - 2))

            C = (0.5-nu_z)/((k*q_inv)**2)
            A = 1

#---------------------------------------------------------------------------------------------
            if model_name == 'Solv':
                S = (2*j*C*k1*(k3**3)*A) / (L + 2*j*C*k3)                              
            else:
                S = 0                                                                  
        
            if f == f_array[0] and c_val == c_values[0]:
                geom_dev = np.abs(W)/(k**2) if W != 0 else 0.0
                source_mod = np.abs(S)
                print(f"Model: {model_name:<5} | Geometric Deviation (|W|/k^2): {geom_dev:.5e} | Source Modification (|S|): {source_mod:.5e}")

#---------------------------------------------------------------------------------------------
            try:
                PI = complex_quad(lambda tau_p: G_func(tau_p) * S, -100, np.abs(tau))
            except Exception:
                PI = 0

#---------------------------------------------------------------------------------------------
            S_bar_denom = (1 / np.sqrt(2 * k)) * np.sqrt(-np.pi * k * tau / 2) * hankel1(nu_z, -k * tau)
            S_bar = PI / S_bar_denom
            Psk_factor = ((4/3)**2)*((1/Omega_over_k)**3)*((HI/MPl)**4)

#---------------------------------------------------------------------------------------------
            z_cutoff = z * np.exp((l / 4) * ((2 * c_val**2 / 9) - 1))                           #Obtained from nu(min) = 2.5
            try:
                pts = [z_cutoff] if z < z_cutoff < np.abs(Omega_over_k) else None
                Integral = integrate.quad(Integrand_func, z, np.abs(Omega_over_k), points=pts, limit=500)[0]
            except Exception:
                Integral = 0

#---------------------------------------------------------------------------------------------
            Psk_source = (np.abs(1 + S_bar))**4
            Psk = Psk_factor * (np.abs(Integral)**2) * Psk_source    
        else:
            Psk = 0.0
        
        Pk = Pvk + Psk
#---------------------------------------------------------------------------------------------
        h0 = 0.7
        OmR = 1e-5
        g1 = 106.75
        g2 = 106.75
        
        Y_prefactor = (1/24)*h0*h0*OmR*(g1/3.363)*(3.909/g2)**(4/3)
        Y = Y_prefactor * np.abs(Pk)
        Y_vals.append(Y)
        
    return np.array(Y_vals)

#---------------------------------------------------------------------------------------------
target_models = ['RH2S2', 'UH2', 'Nil', 'Solv']

for target_model in target_models:
    plt.figure(figsize=(8, 6))
    
    for c_val, color in zip(c_values, colors):
        Y_model = calculate_Y(target_model, c_val, f_array)
        
        if target_model == 'RH2S2':
            with np.errstate(divide='ignore'): 
                plt.plot(np.log10(f_array), np.log10(Y_model), color=color, label=f'c = {c_val}')
        else:
            Y_base = calculate_Y('RH2S2', c_val, f_array)
            fractional_diff = (Y_model - Y_base) / Y_base
            plt.plot(np.log10(f_array), fractional_diff, color=color, label=f'c = {c_val}')

    plt.xlabel(r"$\log_{10}(f)$")
    
    if target_model == 'RH2S2':
        h0, OmR, g1, g2 = 0.7, 1e-5, 106.75, 106.75
        Y_prefactor = (1/24)*h0*h0*OmR*(g1/3.363)*(3.909/g2)**(4/3)
        Pvk_constant = 2 * 1 * (HI / (np.pi * MPl))**2  
        Y_vac_true = Y_prefactor * Pvk_constant
        
        plt.plot(np.log10(f_array), np.full_like(f_array, np.log10(Y_vac_true)), 
                 color='yellow', label='Pure Tensor Vacuum', linestyle='--')
            
        plt.ylabel(r"$\log_{10}(h_0^2 \Omega_{GW})$")
        plt.title(r"$\mathbb{R} \times \mathbb{H}^2 / S^2 \left(\ell = 2.0 \right)$")
        #plt.ylim(-20, 50)
    else:
        plt.ylabel(r"Fractional difference $\left(\frac{\Delta \Omega_{GW}}{\Omega_{GW}}\right)$")
        
        if target_model == 'UH2':
            display_name = r"$\widetilde{U \left(\mathbb{H}^2 \right)}$"
        else:
            display_name = target_model

        #plt.title(f"Deviation of {display_name} geometry from FLRW")
        plt.title(r"Common deviation of $\widetilde{U \left(\mathbb{H}^2 \right)}$, Nil & Solv from FLRW $\left(\ell = 2.0 \right)$")

    k_box = 2*np.pi*f_array[0]
    A_box = 1
    box_text = r"$\mathbf{Characterizations}$" + "\n"

#---------------------------------------------------------------------------------------------
    if target_model == 'RH2S2':
        g_dev = 0.0
    elif target_model == 'UH2':
        g_dev = np.abs(-(1j/L) * (2*k_box/np.sqrt(3))) / (k_box**2)
    elif target_model == 'Nil':
        g_dev = np.abs((1j/L) * (2*k_box/np.sqrt(6))) / (k_box**2)
    elif target_model == 'Solv':
        g_dev = np.abs(-(2*1j/L) * (2*k_box/np.sqrt(6))) / (k_box**2)
        
    box_text += f"$\\Delta$(Geometry) = {g_dev:.3e}\n\n"
    box_text += r"$\mathbf{Sourcing}$" + "\n"

#---------------------------------------------------------------------------------------------
    for c_val_box in c_values:
        nu_box = c_val_box * np.sqrt(2) - 0.5                   #Baseline (boundary) value (at z-prime = z)
        z_box = 9.1e-11 * k_box
        q_inv_box = z_box * np.exp((l/8)*(c_val_box**2 - 2))
        C_box = (0.5 - nu_box) / ((k_box * q_inv_box)**2)
        
        if target_model in ['RH2S2', 'UH2', 'Nil']:
            s_mod = 0.0
        elif target_model == 'Solv':
            s_mod = np.abs((2*1j * C_box * (k_box/np.sqrt(6)) * (np.sqrt(2/3)*k_box)**3 * A_box) / (L + 2*1j * C_box * (np.sqrt(2/3)*k_box)))
            
        box_text += f"c={c_val_box}: {s_mod:.3e}\n"

    box_text = box_text.strip()

    #plt.gca().text(0.05, 0.05, box_text, transform=plt.gca().transAxes, fontsize=10,
                   #verticalalignment='bottom', horizontalalignment='left',
                   #bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

    plt.grid(True)
    plt.legend()
    plt.savefig(f'mhz_{target_model}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("All plots generated and saved successfully!")
