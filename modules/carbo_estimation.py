import pandas as pd

class CarboEstimator:

    def __init__(self, food_nutrients_file='data/food_nutrients.csv'):
        self.food_nutrients_file = food_nutrients_file


    def __call__(self, food_volumes: dict, display=False):
        self.food_volumes = food_volumes    # in milliliters
        self.display = display
        return self.estimate_carbo()
    
    def estimate_carbo(self):
        # Read food nutrients file
        df = pd.read_csv(self.food_nutrients_file, sep=',', header=0, index_col=0)
        
        # Compute the carbohydrate quantity for each food in the dictionary
        food_masses = {}
        self.food_carbs = {}
        for food, volume in self.food_volumes.items():
            # Discard background
            if food == 'background':
                continue
            
            # Compute mass
            density = df.loc[food, 'density(g/mL)']
            mass = volume * density
            food_masses[food] = mass

            # Compute carbohydrate quantity
            carb_100g = df.loc[food, 'carbohydrate(100g)']
            carb = carb_100g * mass / 100
            self.food_carbs[food] = carb

        print("Food masses: ", food_masses)     # DEBUG
        print("Food carbs: ", self.food_carbs)  # DEBUG

        return self.food_carbs