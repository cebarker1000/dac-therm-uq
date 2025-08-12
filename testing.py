import numpy as np
import matplotlib.pyplot as plt
from train_surrogate_models import FullSurrogateModel

surr = FullSurrogateModel.load_model("outputs/edmund1/full_surrogate_model_narrow_k.pkl")
names = surr.parameter_names       # Should contain 'k_int'
print("Parameter order in surrogate:", names)

theta = np.array([[2.9e-6, 7.0725e6, 2.62131e6,
                    4e-6, 4.2e-6, 7e-6,
                    40.0, 4.0, 1.0]])          # k_sample, k_ins, k_int
y_pred, *_ = surr.predict_temperature_curves(theta)
plt.plot(y_pred[0]); plt.show()