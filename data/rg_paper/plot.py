import matplotlib.pyplot as plt
import numpy as np 

reservoirs = ["critical", "control1", "control2"]

avg_err = []

for r in reservoirs:
    data = None

    for i in range(5):
        file_path = "sine_wave/avg_errors_" + r + "_renorm_" + str(i) + ".dat"

        with open(file_path, 'r') as file:
            # Read and convert lines to float
            current_data = [float(line.strip()) for line in file]

            # Initialize `data` on the first iteration
            if data is None:
                data = np.array(current_data)
            else:
                # Add the current data to the cumulative data
                data = np.array(current_data)

            print(f"Length of data in file {i}: {len(current_data)}")

    # Plot the averaged data
    plt.plot(data, label=r)
print(avg_err)

plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.title('Averaged Sine Wave Errors')
plt.show()

