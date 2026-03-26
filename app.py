import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from semiCirclesEISModel import semiCirclesEISModel

# Page config
st.set_page_config(page_title="EIS Parameter Extractor", layout="centered")
st.title("Impedance Spectroscopy (EIS) Extractor")
st.write("Upload your impedance text file to automatically extract the Polarization Resistance and Heterogeneity Parameter.")

# File uploader
uploaded_file = st.file_uploader("Upload Impedance File (.txt)", type=["txt", "csv", "mpt"])

if uploaded_file is not None:
    try:
        # 1. Read the uploaded file directly (skip the first row/header)
        data_is = np.loadtxt(uploaded_file, skiprows=1)
        
        # 2. Extract frequency, Z_real, and Z_imag based on your pro_data structure
        freq = data_is[:, 1]
        Z_re = data_is[:, 2]
        Z_im = data_is[:, 3]
        
        # 3. Create the complex impedance array
        Zc = Z_re - 1j * Z_im
        
        # 4. Instantiate your existing model
        fit = semiCirclesEISModel(freq, Zc)
        real_min, real_max, l = fit.search_min_max()
        
        # 5. Fit first dispersion
        if len(real_min) > 0:
            p1 = fit.optCircleParameters(real_min=0, real_max=real_min[0])
        else:
            p1 = fit.optCircleParameters(real_min=0, real_max=np.inf)
            
        r0 = p1[1]
        x0 = p1[0][0]
        y0 = p1[0][1]
        
        # 6. Calculate parameters
        polarization_resistance = np.sqrt(r0**2 - y0**2) - x0
        
        s1 = fit.real[l] > real_min[-1] if len(real_min) > 0 else fit.real[l] > 0
        lin_re = stats.linregress(fit.real[l][s1], fit.imag[l][s1])
        m = lin_re[0]
        alpha = 2 / np.pi * np.arctan(m)
        
        # --- UI OUTPUT ---
        st.divider()
        st.subheader("Calculated Parameters")
        
        col1, col2 = st.columns(2)
        col1.metric(label="Polarization Resistance (Rp)", value=f"{polarization_resistance:.2f} Ω")
        col2.metric(label="Heterogeneity Parameter (ϕ)", value=f"{alpha:.4f}")
        
        # --- PLOTTING ---
        st.subheader("Nyquist Plot & Fit")
        fig = plt.figure(dpi=300)
        
        # Call the plot function from your model
        fit.plotNyquist(show_title=True)
        
        # Add the linear fit line
        plt.plot(fit.real[l][s1], lin_re[0] * fit.real[l][s1] + lin_re[1], 'k:', label='Linear fit')
        plt.legend()
        
        # Render plot in Streamlit
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Oops! Something went wrong reading the file. Make sure it matches the expected format. Error details: {e}")