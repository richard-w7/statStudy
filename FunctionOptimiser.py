import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.widgets import RadioButtons
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import random
import time
import re
import threading

class FunctionOptimiser:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.35)
        
        # Initialise variables
        self.function_str = "x**2"
        self.functions = self.parse_functions(self.function_str)
        self.x_min, self.x_max = -10, 10
        self.current_algorithm = "Genetic Algorithm"
        self.sample_points = {}
        self.N_steps = 20
        self.animation_running = False
        self.current_points = []
        self.stop_signal = False
        self.optimisation_thread = None
        
        # Setup UI elements
        self.setup_ui()
        
        # Initial plot
        self.update_plot()
    
    def parse_functions(self, function_str):
        """Parse the function string into piecewise functions"""
        functions = []
        # Split by | but not within brackets
        parts = re.split(r'\s*\|\s*(?![^()]*\))', function_str)
        
        for part in parts:
            # Split into function and condition
            if ',' in part:
                func_part, cond_part = part.rsplit(',', 1)
                func_part = func_part.strip()
                cond_part = cond_part.strip()
                
                # Parse condition
                if '<=' in cond_part:
                    lower, upper = re.split(r'<=x<', cond_part)
                    lower = float(lower)
                    upper = float(upper)
                elif '<' in cond_part:
                    lower, upper = re.split(r'<x<', cond_part)
                    lower = float(lower)
                    upper = float(upper)
                else:
                    raise ValueError("Invalid condition format")
                
                functions.append({
                    'function': lambda x, f=func_part: eval(f, {'x': x, 'np': np}),
                    'range': (lower, upper),
                    'str': f"{func_part}, {lower}≤x<{upper}"
                })
            else:
                # Default case (no condition)
                functions.append({
                    'function': lambda x, f=part.strip(): eval(f, {'x': x, 'np': np}),
                    'range': (None, None),
                    'str': part.strip()
                })
        
        return functions
    
    def evaluate_function(self, x):
        # Evaluate the piecewise function at x
        for func in self.functions:
            lower, upper = func['range']
            if (lower is None and upper is None) or (lower <= x < upper):
                try:
                    return func['function'](x)
                except:
                    return np.nan
        return np.nan
    
    def setup_ui(self):
        # Function input
        ax_func = plt.axes([0.2, 0.25, 0.6, 0.05])
        self.func_input = TextBox(ax_func, 'Function(s) e.g.[x**2, 0<=x<3 | x**3, 3<=x<5]:', initial=self.function_str)
        self.func_input.on_submit(self.update_function)
        
        # Range inputs
        ax_xmin = plt.axes([0.2, 0.2, 0.2, 0.05])
        ax_xmax = plt.axes([0.6, 0.2, 0.2, 0.05])
        self.xmin_input = TextBox(ax_xmin, 'x min:', initial=str(self.x_min))
        self.xmax_input = TextBox(ax_xmax, 'x max:', initial=str(self.x_max))
        self.xmin_input.on_submit(self.update_range)
        self.xmax_input.on_submit(self.update_range)
        
        algorithms = ["Genetic Algorithm", 
                     "MLP + Gradient Descent", 
                     "Gradient Descent",
                     "Particle Swarm (PSO)",
                     "Simulated Annealing",
                     "Bayesian Optimisation",
                     "Nelder-Mead",
                     "Differential Evolution"]
        
        # Buttons for algorithm selection
        ax_algo = plt.axes([0.2, 0.05, 0.6, 0.15])
        self.algo_radio = RadioButtons(ax_algo, algorithms, active=algorithms.index(self.current_algorithm))
        self.algo_radio.on_clicked(self.set_algorithm)
        
        # Run button
        ax_run = plt.axes([0.3, 0.02, 0.2, 0.05])
        self.run_button = Button(ax_run, 'Run Optimisation')
        self.run_button.on_clicked(self.run_optimisation)
        
        # Stop button
        ax_stop = plt.axes([0.55, 0.02, 0.2, 0.05])
        self.stop_button = Button(ax_stop, 'Stop')
        self.stop_button.on_clicked(self.stop_optimisation)
    
    def update_function(self, text):
        try:
            # Test the function
            test_x = 1.0
            self.function_str = text
            self.functions = self.parse_functions(text)
            self.evaluate_function(test_x)  # Test evaluation
            self.sample_points = {}  # Clear previous points
            self.update_plot()
        except Exception as e:
            print(f"Invalid function expression: {e}")
    
    def update_range(self, text):
        try:
            self.x_min = float(self.xmin_input.text)
            self.x_max = float(self.xmax_input.text)
            if self.x_min >= self.x_max:
                raise ValueError("x_min must be less than x_max")
            self.sample_points = {}  # Clear previous points
            self.update_plot()
        except:
            print("Invalid range values")
    
    def set_algorithm(self, algorithm):
        self.current_algorithm = algorithm
        print(f"Algorithm set to: {algorithm}")
    
    def stop_optimisation(self, event):
        self.stop_signal = True
        if self.optimisation_thread and self.optimisation_thread.is_alive():
            self.optimisation_thread.join()
        self.animation_running = False
    
    def run_optimisation(self, event):
        if self.animation_running:
            return
        
        # Clear all previous points
        self.sample_points = {self.current_algorithm: []}
        self.current_points = []
        self.stop_signal = False
        self.update_plot()
        
        # Start optimisation in a separate thread
        self.optimisation_thread = threading.Thread(target=self.run_optimisation_thread)
        self.optimisation_thread.start()
        
        # Start animation
        self.animate_points()
    
    def run_optimisation_thread(self):
        """Run optimisation in a separate thread"""
        if self.current_algorithm == "Genetic Algorithm":
            self.genetic_algorithm()
        elif self.current_algorithm == "MLP + Gradient Descent":
            self.mlp_optimisation()
        elif self.current_algorithm == "Gradient Descent":
            self.gradient_descent()
        elif self.current_algorithm == "Particle Swarm (PSO)":
            self.particle_swarm_optimisation()
        elif self.current_algorithm == "Simulated Annealing":
            self.simulated_annealing()
        elif self.current_algorithm == "Bayesian Optimisation":
            self.bayesian_optimisation()
        elif self.current_algorithm == "Nelder-Mead":
            self.nelder_mead()
        elif self.current_algorithm == "Differential Evolution":
            self.differential_evolution()
    
    def genetic_algorithm(self):
        population_size = 10
        mutation_rate = 0.1
        
        # Initialise population
        population = np.random.uniform(self.x_min, self.x_max, population_size)
        
        while not self.stop_signal:
            # Evaluate fitness for one individual at a time
            for i in range(population_size):
                if self.stop_signal:
                    break
                    
                x = population[i]
                y = self.evaluate_function(x)
                self.sample_points.setdefault("Genetic Algorithm", []).append((x, y))
                time.sleep(0.3)
                
            if self.stop_signal:
                break
                
            # Selection - tournament selection
            new_population = []
            for _ in range(population_size):
                if self.stop_signal: break
                # Pick 2 random individuals
                idx1, idx2 = np.random.randint(0, population_size, 2)
                # Select the better one
                winner = population[idx1] if self.evaluate_function(population[idx1]) < self.evaluate_function(population[idx2]) else population[idx2]
                new_population.append(winner)
            
            # Crossover - uniform crossover
            population = []
            for i in range(0, population_size, 2):
                if self.stop_signal: break
                if i+1 >= len(new_population):
                    break
                parent1, parent2 = new_population[i], new_population[i+1]
                child1 = parent1 if random.random() < 0.5 else parent2
                child2 = parent2 if random.random() < 0.5 else parent1
                population.extend([child1, child2])
            
            # Mutation
            for i in range(population_size):
                if self.stop_signal: break
                if random.random() < mutation_rate:
                    population[i] += np.random.normal(0, (self.x_max-self.x_min)/10)
                    population[i] = np.clip(population[i], self.x_min, self.x_max)
    
    def mlp_optimisation(self):
        # First generate some random samples to train the MLP
        n_samples = 50
        X = np.random.uniform(self.x_min, self.x_max, n_samples).reshape(-1, 1)
        y = np.array([self.evaluate_function(x) for x in X]).reshape(-1, 1)
        
        # Train MLP
        mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation='tanh', 
                          max_iter=1000, learning_rate_init=0.01)
        mlp.fit(X, y.ravel())
        
        # Now use gradient descent on the MLP approximation
        x = np.random.uniform(self.x_min, self.x_max)
        learning_rate = 0.1
        
        while not self.stop_signal:
            # Compute gradient numerically
            eps = 1e-5
            grad = (mlp.predict([[x + eps]]) - mlp.predict([[x - eps]])) / (2 * eps)
            
            # Update x
            x -= learning_rate * grad[0]
            x = np.clip(x, self.x_min, self.x_max)
            
            # Store point
            y = self.evaluate_function(x)
            self.sample_points.setdefault("MLP + Gradient Descent", []).append((x, y))
            
            time.sleep(0.3)
    
    def gradient_descent(self):
        x = np.random.uniform(self.x_min, self.x_max)
        learning_rate = 0.1
        
        while not self.stop_signal:
            # Compute gradient numerically
            eps = 1e-5
            grad = (self.evaluate_function(x + eps) - self.evaluate_function(x - eps)) / (2 * eps)
            
            # Update x
            x -= learning_rate * grad
            x = np.clip(x, self.x_min, self.x_max)
            
            # Store point
            y = self.evaluate_function(x)
            self.sample_points.setdefault("Gradient Descent", []).append((x, y))
            
            time.sleep(0.3)
    
    def particle_swarm_optimisation(self):
        n_particles = 20
        w = 0.7  # inertia
        c1 = 1.4  # cognitive coefficient
        c2 = 1.4  # social coefficient
        
        # Initialise particles
        particles = np.random.uniform(self.x_min, self.x_max, n_particles)
        velocities = np.zeros(n_particles)
        best_positions = particles.copy()
        best_scores = np.array([self.evaluate_function(p) for p in particles])
        global_best_pos = particles[np.argmin(best_scores)]
        global_best_score = np.min(best_scores)
        
        while not self.stop_signal:
            for i in range(n_particles):
                if self.stop_signal: break
                
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (best_positions[i] - particles[i]) + 
                               c2 * r2 * (global_best_pos - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.x_min, self.x_max)
                
                # Evaluate
                current_score = self.evaluate_function(particles[i])
                self.sample_points.setdefault("Particle Swarm (PSO)", []).append((particles[i], current_score))
                
                # Update personal best
                if current_score < best_scores[i]:
                    best_positions[i] = particles[i]
                    best_scores[i] = current_score
                    
                    # Update global best
                    if current_score < global_best_score:
                        global_best_pos = particles[i]
                        global_best_score = current_score
            
            time.sleep(0.3)
    
    def simulated_annealing(self):
        current_x = np.random.uniform(self.x_min, self.x_max)
        current_y = self.evaluate_function(current_x)
        T = 100.0  # Initial temperature
        cooling_rate = 0.95
        min_temp = 1e-3
        
        while T > min_temp and not self.stop_signal:
            # Generate neighbour
            new_x = current_x + np.random.normal(0, (self.x_max-self.x_min)/10)
            new_x = np.clip(new_x, self.x_min, self.x_max)
            new_y = self.evaluate_function(new_x)
            
            # Acceptance probability
            delta = new_y - current_y
            if delta < 0 or np.random.rand() < np.exp(-delta/T):
                current_x, current_y = new_x, new_y
                self.sample_points.setdefault("Simulated Annealing", []).append((current_x, current_y))
            
            # Cool down
            T *= cooling_rate
            time.sleep(0.3)
    
    def bayesian_optimisation(self):
        n_init = 5
        X = np.random.uniform(self.x_min, self.x_max, n_init).reshape(-1, 1)
        y = np.array([self.evaluate_function(x[0]) for x in X])
        
        kernel = RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel)
        
        while not self.stop_signal:
            # Fit GP
            gp.fit(X, y)
            
            # Find next point using Expected Improvement
            test_points = np.linspace(self.x_min, self.x_max, 100).reshape(-1, 1)
            pred_mean, pred_std = gp.predict(test_points, return_std=True)
            
            # Expected Improvement calculation
            current_best = np.min(y)
            z = (current_best - pred_mean) / pred_std
            ei = (current_best - pred_mean) * norm.cdf(z) + pred_std * norm.pdf(z)
            next_x = test_points[np.argmax(ei)][0]
            
            # Evaluate and add to dataset
            next_y = self.evaluate_function(next_x)
            self.sample_points.setdefault("Bayesian Optimisation", []).append((next_x, next_y))
            X = np.vstack([X, [[next_x]]])
            y = np.append(y, next_y)
            
            time.sleep(0.3)
    
    def nelder_mead(self):
        # Initialise simplex (triangle in 1D case)
        x1 = np.random.uniform(self.x_min, self.x_max)
        x2 = x1 + 0.1*(self.x_max-self.x_min)
        x2 = min(x2, self.x_max)
        
        while not self.stop_signal:
            # Evaluate vertices
            y1 = self.evaluate_function(x1)
            y2 = self.evaluate_function(x2)
            self.sample_points.setdefault("Nelder-Mead", []).extend([(x1, y1), (x2, y2)])
            
            # Order points
            if y1 < y2:
                best_x, best_y = x1, y1
                other_x, other_y = x2, y2
            else:
                best_x, best_y = x2, y2
                other_x, other_y = x1, y1
            
            # Calculate centroid (trivial in 1D)
            centroid = best_x
            
            # Reflection
            reflected = centroid + (centroid - other_x)
            reflected = np.clip(reflected, self.x_min, self.x_max)
            reflected_y = self.evaluate_function(reflected)
            
            if reflected_y < best_y:
                # Expansion
                expanded = centroid + 2*(centroid - other_x)
                expanded = np.clip(expanded, self.x_min, self.x_max)
                expanded_y = self.evaluate_function(expanded)
                if expanded_y < reflected_y:
                    x1, x2 = best_x, expanded
                else:
                    x1, x2 = best_x, reflected
            else:
                # Contraction
                contracted = centroid + 0.5*(other_x - centroid)
                contracted = np.clip(contracted, self.x_min, self.x_max)
                contracted_y = self.evaluate_function(contracted)
                if contracted_y < other_y:
                    x1, x2 = best_x, contracted
                else:
                    # Shrink
                    x2 = best_x + 0.5*(x2 - best_x)
            
            time.sleep(0.3)
    
    def differential_evolution(self):
        pop_size = 10
        F = 0.8  # differential weight
        cr = 0.9  # crossover probability
        
        population = np.random.uniform(self.x_min, self.x_max, pop_size)
        
        while not self.stop_signal:
            for i in range(pop_size):
                # Select three random individuals
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Mutation
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.x_min, self.x_max)
                
                # Crossover
                if np.random.rand() < cr:
                    trial = mutant
                else:
                    trial = population[i]
                
                # Selection
                if self.evaluate_function(trial) < self.evaluate_function(population[i]):
                    population[i] = trial
                    self.sample_points.setdefault("Differential Evolution", []).append((trial, self.evaluate_function(trial)))
            
            time.sleep(0.3)
    
    def animate_points(self):
        if self.current_algorithm not in self.sample_points:
            return
            
        self.animation_running = True
        color_map = {
            "Genetic Algorithm": 'red',
            "MLP + Gradient Descent": 'green',
            "Gradient Descent": 'purple',
            "Particle Swarm (PSO)": 'blue',
            "Simulated Annealing": 'orange',
            "Bayesian Optimisation": 'cyan',
            "Nelder-Mead": 'magenta',
            "Differential Evolution": 'brown'
        }
        color = color_map.get(self.current_algorithm, 'black')
        
        def update(frame):
            if not self.animation_running:
                return
            
            # Get current points (last 10 points)
            points = self.sample_points.get(self.current_algorithm, [])
            start_idx = max(0, len(points) - 10)
            self.current_points = points[start_idx:]
            
            # Update plot
            self.ax.clear()
            
            # Plot the piecewise function
            x_vals = np.linspace(self.x_min, self.x_max, 1000)
            y_vals = np.array([self.evaluate_function(x) for x in x_vals])
            self.ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Function')
            
            # Plot current algorithm's points
            if self.current_points:
                # Calculate alpha values for fading effect
                n_points = len(self.current_points)
                alphas = np.linspace(0.2, 1, n_points)
                
                for i, (x, y) in enumerate(self.current_points):
                    alpha = alphas[i]
                    self.ax.scatter([x], [y], color=color, s=50, alpha=alpha)
                    
                    # Connect to next point if exists
                    if i < n_points - 1:
                        next_x, next_y = self.current_points[i+1]
                        self.ax.plot([x, next_x], [y, next_y], 
                                    color=color, linestyle='--', alpha=alpha)
            
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            
            # Display function pieces
            func_str = "\n".join([f['str'] for f in self.functions])
            self.ax.set_title(f'Function:\n{func_str}')
            
            self.ax.grid(True)
            self.ax.legend()
            
            # Add current algorithm info
            self.ax.text(0.02, 0.98, f'Current Algorithm: {self.current_algorithm}', 
                        transform=self.ax.transAxes, verticalalignment='top')
            
            self.fig.canvas.draw()
            
            # Continue animation if not stopped
            if not self.stop_signal:
                self.fig.canvas.flush_events()
                time.sleep(0.1)
                update(frame + 1)
        
        # Start animation
        update(0)
    
    def update_plot(self):
        self.ax.clear()
        
        # Plot the piecewise function
        x_vals = np.linspace(self.x_min, self.x_max, 1000)
        y_vals = np.array([self.evaluate_function(x) for x in x_vals])
        self.ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Function')
        
        # Plot current points if any
        if self.current_points:
            color_map = {
                "Genetic Algorithm": 'red',
                "MLP + Gradient Descent": 'green',
                "Gradient Descent": 'purple',
                "Particle Swarm (PSO)": 'blue',
                "Simulated Annealing": 'orange',
                "Bayesian Optimisation": 'cyan',
                "Nelder-Mead": 'magenta',
                "Differential Evolution": 'brown'
            }
            color = color_map.get(self.current_algorithm, 'black')
            n_points = len(self.current_points)
            alphas = np.linspace(0.2, 1, n_points)
            
            for i, (x, y) in enumerate(self.current_points):
                alpha = alphas[i]
                self.ax.scatter([x], [y], color=color, s=50, alpha=alpha)
                
                # Connect to next point if exists
                if i < n_points - 1:
                    next_x, next_y = self.current_points[i+1]
                    self.ax.plot([x, next_x], [y, next_y], 
                               color=color, linestyle='--', alpha=alpha)
        
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        
        # Display function pieces
        func_str = "\n".join([f['str'] for f in self.functions])
        self.ax.set_title(f'Function:\n{func_str}')
        
        self.ax.grid(True)
        self.ax.legend()
        
        # Add current algorithm info
        self.ax.text(0.02, 0.98, f'Current Algorithm: {self.current_algorithm}', 
                    transform=self.ax.transAxes, verticalalignment='top')
        
        self.fig.canvas.draw()

if __name__ == "__main__":
    optimiser = FunctionOptimiser()
    plt.show()