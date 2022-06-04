from black_scholes import BlackScholes
import json
import time
from utils import print_and_write
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def experiment0():
	
	K = 1 ; sigma = 0.01 ; T = 1 ; r = 0.01 ; N = 10000 ; L = 10

	init_n = time.perf_counter()
	BlackScholes(N, K, L, sigma, r, T, u_grid_method='naive')
	end_n = time.perf_counter()

	init_f = time.perf_counter()
	BlackScholes(N, K, L, sigma, r, T, u_grid_method='fast')
	end_f = time.perf_counter()

	with open(f"results/exp0/report.txt", "w") as file:
		print_and_write("---- using two for loops vs vectorizing one of them for building the u grid ----", file)
		print_and_write("Initial Params: ", file)
		print_and_write(f"K: {K}, sigma: {sigma}, T: {T}, r: {r}, N: {N}, L: {L}", file)
		print_and_write("Execution time using two for loops: {}".format(end_n-init_n), file)
		print_and_write("Execution time vectorizing one of the loops: {}".format(end_f-init_f), file)


def experiment1():

	def profit_vs_spotprice() -> None:

		arr_St = np.linspace((1-np.sqrt(r)/1.5)*K, (1+np.sqrt(r)/1.5)*K, 20)
		arr_Vt = np.empty_like(arr_St)

		S0 = 1
		
		V0 = bs.get_V(S=S0, t=0, interpolation=True)
		V0_an = bs.get_V_analytical(S=S0, t=0)

		u0 = bs.get_u(S=S0, t=0)
		u0_an = bs.get_u_analytical(S=S0, t=0)	

		fig = plt.figure(figsize=(14, 6))

		t = 0.5
		for i in range(len(arr_St)):
			St = arr_St[i]
			arr_Vt[i] = bs.get_V(St, t, interpolation=True)

		
		profit = 1000*(arr_Vt - V0)

		plt.plot(arr_St, profit)

		plt.title('Profit x Spot Price on t=0.5*T for an Call Option bought on t=0*T')
		plt.xlabel('Spot Price') ; plt.ylabel('Profit')
		plt.axhline(y=0, color='red', linestyle='--')
		plt.grid()

		with open(f"results/exp1/profit_vs_spotprice/sigma={sigma}_r={r}.txt", "w") as file:
			print_and_write("---- Profit vs Spot Price ----", file)
			print_and_write("Initial Params: ", file)
			print_and_write(f"K: {bs.K}, sigma: {bs.sigma}, T: {bs.T}, r: {bs.r}, N: {bs.N}, L: {bs.L}, M: {bs.M}", file)
			print_and_write(f"S0 = {S0}", file)
			print_and_write(f"numerical V0 = {V0}", file)
			print_and_write(f"analytical V0 = {V0_an}", file)
			print_and_write(f"numerical u0 = {u0}", file)
			print_and_write(f"analytical u0 = {u0_an}", file)
			print_and_write(f"t={t}", file)
			for i in range(len(arr_St)):
				print_and_write(f"S(t):{arr_St[i]:.5f} V(t):{arr_Vt[i]:.5f} Profit:{profit[i]:.5f}", file)
			print_and_write("generating plots...", file)

		with open(f"results/exp1/profit_vs_spotprice/sigma={sigma}_r={r}.json", "w") as file:
			json.dump({f"S({t})": list(arr_St), f"V({t})": list(arr_Vt)}, file)

		fig.savefig(f"results/exp1/profit_vs_spotprice/sigma={sigma}_r={r}.png")


	def callprice_vs_spotprice() -> None:

		arr_t = np.array([0, 0.25, 0.5, 0.75, 1]) 
		
		arr_St = np.linspace((1-np.sqrt(r)/1.5)*K, (1+np.sqrt(r)/1.5)*K, 20)
		arr_Vt = np.empty_like(arr_St)

		fig = plt.figure(figsize=(14, 6))

		dict_ = {}

		for t in arr_t:

			for i in range(len(arr_St)):
				St = arr_St[i]
				arr_Vt[i] = bs.get_V(S=St, t=t, interpolation=True)

			dict_[t] = {"S(t)": list(arr_St), "V(t)": list(arr_Vt)}		

			plt.plot(arr_St, arr_Vt, alpha=0.9)

		plt.legend(['t = ' + str(t) for t in arr_t])
		plt.title('Call Option Price x Spot Price for multiple times to expiration')
		plt.xlabel('Spot Price') ; plt.ylabel('Call Option Price')
		plt.grid()

		with open(f"results/exp1/callprice_vs_spotprice/sigma={sigma}_r={r}.txt", "w") as file:
			print_and_write("---- Call Price vs Spot Price ----", file)
			print_and_write("Initial Params: ", file)
			print_and_write(f"K: {bs.K}, sigma: {bs.sigma}, T: {bs.T}, r: {bs.r}, N: {bs.N}, L: {bs.L}, M: {bs.M}", file)
			print_and_write("saving t, S(t), and V(t) in a json...", file)
			print_and_write("generating plots...", file)

		with open(f"results/exp1/callprice_vs_spotprice/sigma={sigma}_r={r}.json", "w") as file:
			json.dump(dict_, file)

		fig.savefig(f"results/exp1/callprice_vs_spotprice/sigma={sigma}_r={r}.png")


	K = 1 ; sigma = 0.01 ; T = 1 ; r = 0.01 ; 
	N = 10000 ; L = 10
	bs = BlackScholes(N, K, L, sigma, r, T)
	profit_vs_spotprice()
	callprice_vs_spotprice()

	sigma = 0.02
	bs = BlackScholes(N, K, L, sigma, r, T)
	profit_vs_spotprice()
	callprice_vs_spotprice()

	sigma = 0.1 ; r = 0.1
	bs = BlackScholes(N, K, L, sigma, r, T)
	profit_vs_spotprice()
	callprice_vs_spotprice()


def experiment2():

	K = 5.7 ; sigma = 0.1692 ; T = 3/12 ; r = 0.1075  ; N = 10000 ; L = 10
	bs = BlackScholes(N, K, L, sigma, r, T)

	S0 = 5.6376
	V0 = bs.get_V(S0, 0, interpolation=True)
	V0_an = bs.get_V_analytical(S0, 0)

	ST = 5.1604718519
	return_1 = np.maximum(ST-K,0) - V0
	return_1pct = (np.maximum(ST-K,0) - V0)/V0

	arr_St = np.linspace(4, 7, 20)
	arr_Vt = np.empty_like(arr_St)
	for i in range(len(arr_St)):
		arr_Vt[i] = bs.get_V(arr_St[i], 0, interpolation=True)

	profit_pct = (arr_Vt-V0)/V0

	fig = plt.figure(figsize=(14, 6))
	plt.plot(arr_St, profit_pct)

	# format y axis as percentage
	plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
	plt.axhline(y=0, color='red', linestyle='--')
	plt.xlabel('Spot Price') ; plt.ylabel('Profit (%)')
	plt.title("Profit x Spot Price for a Mar/22 Exp. Call Option on Jan/22")
	plt.grid()

	#plt.axvline(x=S0, color='red', linestyle='--')

	with open(f"results/exp2/report.txt", "w") as file:
		print_and_write("---- BRL/USD analysis ----", file)
		print_and_write("Initial Params: ", file)
		print_and_write(f"K: {bs.K}, sigma: {bs.sigma}, T: {bs.T}, r: {bs.r}, N: {bs.N}, L: {bs.L}, M: {bs.M}", file)
		print_and_write("---- BRL/USD analysis - until expiration ----", file)
		print_and_write(f"S0 = {S0}", file)
		print_and_write(f"numerical V0 = {V0}", file)
		print_and_write(f"analytical V0 = {V0_an}", file)
		print_and_write(f"The spot price on March/22 was {ST}...", file)
		print_and_write(f"Which means that the return on March/22 was {return_1} per unit of option...", file)
		print_and_write(f"This is equivalent to a {return_1pct*100}% return...", file)
		print_and_write("---- BRL/USD analysis - Jan/22, before expiration ----", file)
		for i in range(len(arr_St)):
			print_and_write(f"S(t):{arr_St[i]:.5f} V(t):{arr_Vt[i]:.5f} Profit:{profit_pct[i]:.5f}", file)
		print_and_write("generating plots...", file)

	fig.savefig(f"results/exp2/report.png")


def experiment3():
	# Asset: IBOV
	K = 110_000 # strike price
	T = 0.5 # 6 months
	r = 0.1175 # https://www.bcb.gov.br/controleinflacao/taxaselic
	sigma = 0.1905 # https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-a-vista/volatilidades-dos-ativos/desvio-padrao/
	N = 10000 ; L = 10
	bs = BlackScholes(N, K, L, sigma, r, T)

	S0 = 116_782 # https://www.infomoney.com.br/cotacoes/b3/indice/ibovespa/
	V0 = bs.get_V(S0, 0, interpolation=True)
	V0_an = bs.get_V_analytical(S0, 0)

	arr_St = np.linspace(70_000, 150_000, 30)
	arr_Vt = np.empty_like(arr_St)

	for i in range(len(arr_St)):
		arr_Vt[i] = bs.get_V(arr_St[i], 0.5, interpolation=True)

	profit_pct = (arr_Vt - V0)/V0

	fig = plt.figure(figsize=(14, 6))

	plt.plot(arr_St, profit_pct)
	plt.title('Call Option Price x Spot Price for a IBOV 6 Months Call Option')
	plt.xlabel('Spot Price in 6 months') ; plt.ylabel('Profit (%)')
	plt.grid()
	plt.axhline(y=0, color='red', linestyle='--')

	with open(f"results/exp3/report.txt", "w") as file:
		print_and_write("---- IBOV analysis ----", file)
		print_and_write("Initial Params: ", file)
		print_and_write(f"K: {bs.K}, sigma: {bs.sigma}, T: {bs.T}, r: {bs.r}, N: {bs.N}, L: {bs.L}, M: {bs.M}", file)
		print_and_write(f"S0 = {S0}", file)
		print_and_write(f"numerical V0 = {V0}", file)
		print_and_write(f"analytical V0 = {V0_an}", file)
		print_and_write("---- Profit vs Spot Price given the strike and the S0 ----", file)
		for i in range(len(arr_St)):
			print_and_write(f"S(t):{arr_St[i]:.5f} V(t):{arr_Vt[i]:.5f} Profit:{profit_pct[i]:.5f}", file)
		print_and_write("generating plots...", file)
	
	fig.savefig(f"results/exp3/report.png")


def main():
	print("Running experiment 0...")
	experiment0()
	print("Running experiment 1...")
	experiment1()
	print("Running experiment 2...")
	experiment2()
	print("Running experiment 3...")
	experiment3()

if __name__ == '__main__':
	main()