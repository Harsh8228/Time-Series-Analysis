# Importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Reading data
df = pd.read_csv("airline2.csv")

# Converting Date to datetime datatype to extract month, days and year from it
df['Date'] = pd.to_datetime(df['Date'])

# Extracting individual values from date and adding them as a column in dataframe
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year

# Calculating fourier coefficients for the Number
fft_coeffs = np.fft.fft(df['Number'])

# Calculating the frequencies for which coefficients were calculated
frequencies = np.fft.fftfreq(len(df['Number']))

# Filtering the coefficients to first 8 terms only
n_terms = 8
fft_coeffs_filtered = np.zeros_like(fft_coeffs)
fft_coeffs_filtered[:n_terms] = fft_coeffs[:n_terms]

# Reconstructing the fourier series with filtered coefficients and keeping only the real value part
reconstructed_signal = np.fft.ifft(fft_coeffs_filtered).real

# Adding fourier estimated values into the dataframe across the dates
df['Fourier_signal'] = reconstructed_signal

# Calculating monthly average of Passengers over entire 3 years period
month_avg = df.groupby('Month')['Number'].sum()
month_avg = month_avg/3

# Calculating monthly average of Fourier Series over entire 3 years period
month_favg = df.groupby('Month')['Fourier_signal'].sum()
month_favg = month_favg/3

# Plotting monthly average as bar and Fourier series as line
plt.figure( dpi = 144)

plt.bar(df['Month'].unique(), month_avg, color = 'Green', edgecolor = 'black')
plt.plot(np.arange(1,13,1), month_favg, label="Fourier Series", color="red", linewidth=2)

plt.text(0.5, 600, 'Student ID: 23089625', fontsize=12)

plt.xlabel('Month (Jan - Dec)', fontsize = 17)
plt.ylabel('Average Travellers', fontsize = 17)
plt.title('Fourier Series', fontsize = 17)
plt.xticks(np.arange(1, 13, 1))
plt.ylim(0, 650)

plt.legend(labels=['Fourier Series', 'Travellers'], fontsize=12)

plt.show()

# Calculating power to plot the power spectrum
power = np.abs(fft_coeffs)**2

# Taking only positive frequencies
freq = frequencies[frequencies > 0]

# Plotting the Power Spectrum for time period of 1 day to 1 year
plt.figure(dpi=120)

a = len(freq)

plt.plot(1/freq, power[:a])

plt.xlabel("Period (Days)", fontsize = 17)
plt.ylabel("Power (log)", fontsize = 17)
plt.title("Power Spectrum", fontsize = 17)
plt.yscale('log')
plt.xlim(0, 370)

plt.text(25, 100000000, 'Student ID: 23089625', fontsize=12)

# Calculating 2 highest peaks on the plotted line
peaks, properties = find_peaks(power[0:a])

# Sort peaks by height and get the two highest
sorted_peaks = sorted(peaks, key=lambda x: power[x], reverse=True)
highest_peaks = sorted_peaks[:2]

# Get the x, y values of the two highest peaks
peak_values = [(1 / freq[peak], power[peak]) for peak in highest_peaks]

# Mark the two highest peaks on the plot
j = 1
for x, y in peak_values:
    plt.scatter(x, y, color='purple', zorder=5, label = f'{j} Highest Peaks')
    j+=1

plt.legend(fontsize=12)

plt.show()

# Printing the X, Y values of 2 peaks
i=1
for peak in peak_values:
  x, y = peak
  print(f"For Peak{i}, Time Period(Day) = {x:.2f} and Power = {y:.2f}")
  i+=1
