from calibration_methods.temperature_scaling import TemperatureScaling
from calibration_methods.irova import IROvA

from tqdm import tqdm

class IROvATS():
    
    def __init__(self):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        
    # Find the temperature
    def fit(self, logits, labels):
        self.ts_calibrator = TemperatureScaling(loss='mse')
        self.ts_calibrator.fit(logits, labels)
        self.irova_calibrator = IROvA()
        self.irova_calibrator.fit(self.ts_calibrator.predict(logits), labels)

        
    def predict_proba(self, logits):
        """
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        return self.irova_calibrator.predict_proba(self.ts_calibrator.predict(logits))