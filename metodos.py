from pickle import NONE
from utils import closest_point_index, closest_2_points_index
import numpy as np


class BlackScholes:

	def __init__(self, N: int, K: int, L: int, sigma: float, r: float, T:float, u_grid_method='fast'):
		"""Starts a Black Scholes model with the given parameters and builds the required grids.

		Args:
			N (int): space discretization
			K (int): strike price
			L (int): regulates the domain of the x variable
			sigma (float): volatility
			r (float): risk-free interest rate
			T (float): expiry time
			u_grid_method (str, optional): switches between the 'naive' and the 'fast' way of building the u grid. Defaults to 'fast'.
		"""

		self.N = N
		self.L = L

		self.K = K
		self.sigma = sigma
		self.T = T
		self.r = r
	
		self.delta_x = (2*self.L)/self.N
		
		# multiply by 2 to get a bigger grid
		self.M = 2*int(np.ceil((self.sigma**2*self.T)/(self.delta_x**2)))
		
		self.delta_tau = self.T/self.M
		
		self._build_initial_grids()
		self._build_u_grid(u_grid_method)
		self._build_V_grid()
	

	def _build_initial_grids(self) -> None:
		"""Builds the initial grids for the x and tau heat equation variables
		"""

		self.x_grid = np.zeros(self.N+1)

		for i in range(len(self.x_grid)):
			self.x_grid[i] = i*self.delta_x - self.L
		
		self.tau_grid = np.zeros(self.M+1)
		
		for j in range(len(self.tau_grid)):
			self.tau_grid[j] = j*self.delta_tau


	def _build_u_grid(self, method='fast') -> None:
		"""Builds the u heat equation grid.

		Args:
			method (str, optional): 'naive': two standard for loops, 'fast': vectorizes one of the loops using numpy. Defaults to 'fast'.
		"""

		u = np.empty(shape=(self.N+1, self.M+1))		

		if method == 'naive':
			
			for i in range(self.N):
				u[i, 0] = self.K*np.maximum(np.exp(self.x_grid[i])-1, 0)

			u[0, 0]	= 0

			u[self.N, 0] = self.K*np.exp(self.L+self.sigma**2*self.tau_grid[0]/2)

			for j in range(1, self.M+1): #  range(1, len(tau_grid))
				
				u[0, j]	= 0

				u[self.N, j] = self.K*np.exp(self.L+self.sigma**2*self.tau_grid[j]/2)
				
				for i in range(1, self.N): # range(1, len(x_grid)-1)
					u[i, j] = u[i, j-1] + \
							  (self.delta_tau/(self.delta_x**2))* \
							  ((self.sigma**2)/2)* \
							  (u[i-1, j-1] - 2*u[i, j-1] + u[i+1, j-1])

		# TODO vectorize operations over columns as well
		elif method == 'fast':
			
			# build matrix borders (top row, left column, bottom row)
			u[:, 0] = self.K*np.maximum(np.exp(self.x_grid)-1, 0)
			u[0, :] = 0
			u[self.N, :] = self.K*np.exp(self.L+self.sigma**2*self.tau_grid/2)
			
			# slicing shenaningans
			for j in range(1, self.M+1): # range(1, len(tau_grid))
				u[1:-1, j] = u[1:-1, j-1] + \
						     (self.delta_tau/(self.delta_x**2))* \
							 ((self.sigma**2)/2)* \
							 (u[:-2, j-1] - 2*u[1:-1, j-1] + u[2:, j-1])
			
		self.u_grid = u


	def _build_V_grid(self) -> None:
		"""Builds the V grid (option price)
		"""

		self.V_grid = self.u_grid*np.exp(-self.r*self.tau_grid)


	def get_u(self, S: float, t: float) -> float:
		"""Get the value of the u heat equation variable at an given point.

		Args:
			S (float): Spot Price of the asset at time t
			t (float): time

		Returns:
			float: u(S, t)
		"""
		
		tau = self.T - t
		x = np.log(S/self.K)+(self.r-(self.sigma**2/2))*tau

		i = closest_point_index(self.x_grid, x)
		j = closest_point_index(self.tau_grid, tau)

		u = self.u_grid[i, j]
		return u


	def get_V(self, S: float, t: float, interpolation=False) -> float:
		"""Get the value of the option price at an given point.

		Args:
			S (float): Spot Price of the asset at time t
			t (float): time
			interpolation (bool): whether to interpolate the option price or get the closest point in the grid. Defaults to False.

		Returns:
			float: V(S, t)
		"""
		
		tau = self.T - t
		x = np.log(S/self.K)+(self.r-(self.sigma**2/2))*tau
		
		if not interpolation:

			i = closest_point_index(self.x_grid, x)
			j = closest_point_index(self.tau_grid, tau)

			V = self.V_grid[i, j]
			return V

		if interpolation:

			i, i_plus1  = closest_2_points_index(self.x_grid, x)
			j = closest_point_index(self.tau_grid, tau)

			assert (x < self.x_grid[i_plus1]) and (x > self.x_grid[i])

			V = (((self.x_grid[i_plus1]-x)*self.V_grid[i, j]) - \
				((self.x_grid[i]-x)*self.V_grid[i_plus1, j])) / \
				(self.x_grid[i_plus1]-self.x_grid[i])
			return V


	def get_u_analytical(self, S: float, t: float) -> float:
		"""Get the value of the option price at an given point using the analytical solution.

		Args:
			S (float): Spot Price of the asset at time t
			t (float): time

		Returns:
			float: u(S, t)
		
		Obs:
			Requires scipy
		"""

		from scipy.stats import norm

		tau = self.T - t
		x = np.log(S/self.K)+(self.r-(self.sigma**2/2))*tau

		d1 = (x + tau*(self.sigma**2))/(self.sigma*np.sqrt(tau))
		d2 = x/(self.sigma*np.sqrt(tau))

		return self.K*np.exp(x+(self.sigma**2)*tau/2)*norm.cdf(d1) - self.K*norm.cdf(d2)


	def get_V_analytical(self, S: float, t: float) -> float:
		"""Get the value of the option price at an given point using the analytical solution.

		Args:
			S (float): Spot Price of the asset at time t
			t (float): time

		Returns:
			float: V(S, t)
		
		Obs:
			Requires scipy
		"""
		from scipy.stats import norm

		tau = self.T - t
		x = np.log(S/self.K)+(self.r-(self.sigma**2/2))*tau

		d1 = (x + tau*(self.sigma**2))/(self.sigma*np.sqrt(tau))
		d2 = x/(self.sigma*np.sqrt(tau))

		return S*norm.cdf(d1) - self.K*np.exp((-self.r)*(self.T - t))*norm.cdf(d2)


if __name__ == '__main__':
	K = 1 ; sigma = 0.01 ; T = 1 ; r = 0.01 ; N = 10000 ; L = 10
	S = 1 ; t = 0.5

	bs = BlackScholes(N, K, L, sigma, r, T)
	bs.get_V(S, 0.5, interpolation=False)
	bs.get_V(S, 0.5, interpolation=True)
	bs.get_V_analytical(S, 0.5)