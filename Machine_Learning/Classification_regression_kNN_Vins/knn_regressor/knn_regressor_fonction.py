import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from math import * 


def Validation_croisee(X, y, params, cv):

	y_train = y
	X_train = X
	N_train = y_train.size
	Nk = ceil(N_train / cv)
	Nk = int(Nk)

	# Liste contenant le "numéro" de chaque fold
	i_list = np.arange(cv, dtype=int)

	# Validation croisée "manuelle"

	knn_mse_mean_list = [] # Liste afin de stocker l'erreur moyenne des différents "regressor"
	knn_r2_mean_list = [] # Liste afin de stocker lle r2 moyen des différents "regressor"

	for K in params:
	    print("=="*30)
	    print("Nombre de voisins : {}".format(K))
	    model = KNeighborsRegressor(n_neighbors = K)
	    knn_mse_val = [] # stocker la valeur de la MSE pour chaque fold
	    knn_r2_val = [] # stocker la valeur du R² pour chaque fold
	    for i in i_list:
	        print("Numéro de fold de test : {}".format(i+1))
	        X_validation = X_train[i*Nk:(i+1)*Nk]
	        y_validation = y_train[i*Nk:(i+1)*Nk]
	        X_train_cv = np.array([])
	        y_train_cv = np.array([])
	        for j in i_list:
	            if j != i:
	                X_train_cv = np.append(X_train_cv, X_train[j*Nk:(j+1)*Nk])
	                y_train_cv = np.append(y_train_cv, y_train[j*Nk:(j+1)*Nk])       
	        X_train_cv = X_train_cv.reshape(-1, 11)
	        model.fit(X_train_cv, y_train_cv)
	        train_predictions = model.predict(X_validation)
	        knn_mse = mean_squared_error(y_validation, train_predictions)
	        knn_r2 = r2_score(y_validation, train_predictions)
	        print("MSE = {:.4f} ; R² = {:.4f}".format(knn_mse, knn_r2))
	        knn_mse_val.append(knn_mse) # stocker les valeurs de mse de chaque fold
	        knn_r2_val.append(knn_r2) # stocker les valeurs de R² de chaque fold
	    knn_mse_mean = sum(knn_mse_val) / cv # Moyenne de la MSE pour chaque k-voisins
	    knn_r2_mean = sum(knn_r2_val) / cv # moyenne du R² pour chaque k-voisins
	    print("MSE = %.4f ; R² = %.4f" % (knn_mse_mean, knn_r2_mean))
	    knn_mse_mean_list.append(knn_mse_mean)
	    knn_r2_mean_list.append(knn_r2_mean)

	# Visualisation de l'évolution de l'erreur quadratique moyenne et 
	# du coefficient de détermination en fonction du nombre de voisins
	fig, ax1 = plt.subplots()

	plt.grid()

	ax1.plot(params, knn_mse_mean_list,
	         'b+', markersize=10, markeredgewidth=2,
	         label="MSE")

	ax1.set_xlabel('Nombre de voisins', fontsize=14)
	ax1.xaxis.set_ticks(params)

	ax1.set_ylabel('Mean Squared Error', color='b', fontsize=14)
	for tl in ax1.get_yticklabels():
	    tl.set_color('b')
	    
	ax2 = ax1.twinx()
	ax2.plot(params, knn_r2_mean_list, 
	         'xr', markersize=10, markeredgewidth=2,
	         label="R²")
	ax2.set_ylabel('Coefficient de détermination R²', color='r', fontsize=14)
	for tl in ax2.get_yticklabels():
	    tl.set_color('r')

	plt.title('Evolution de la MSE et de R² \nen fonction du nombre de voisins', 
	          fontsize = 16)

	plt.show()


	# Choix du modèle le plus performant
	print("=="*30)
	print("Liste MSE : {}".format([i for i in knn_mse_mean_list]))
	print("Liste R² : {}".format([i for i in knn_r2_mean_list]))
	print("=="*30)
	knn_mse_min = min(knn_mse_mean_list)
	index_min_MSE = knn_mse_mean_list.index(knn_mse_min)
	knn_r2_max = max(knn_r2_mean_list)
	print("Meilleur hyperparamètre : {}".format(params[index_min_MSE]))
	print("MSE = {:.4f} ; R² = {:.4f}".format(knn_mse_min, knn_r2_max))
	K_opt = params[index_min_MSE]

	return K_opt