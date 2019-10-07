import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from math import * 


def Validation_croisee(X, y, params, cv):

    N_train_total = y.size
    Nk = ceil(N_train_total / cv)
    Nk = int(Nk)

    # Liste contenant le "numéro" de chaque fold
    i_list = np.arange(cv, dtype=int)

    # Liste afin de stocker la précision moyenne des différents classifieurs
    knn_acc_mean_list = []

    for K in params:
        print("=="*30)
        print("Nombre de voisins : {}".format(K))
        model = KNeighborsClassifier(n_neighbors = K)
        knn_acc_val = [] # stocker les valeurs moyennes de l'accuracy
        for i in i_list:
            print("Numéro de fold de test : {}".format(i+1))
            X_validation = X[i*Nk:(i+1)*Nk]
            y_validation = y[i*Nk:(i+1)*Nk]
            X_train = np.array([])
            y_train = np.array([])
            for j in i_list:
                if j != i:
                    X_train = np.append(X_train, X[j*Nk:(j+1)*Nk])
                    y_train = np.append(y_train, y[j*Nk:(j+1)*Nk])       
            X_train = X_train.reshape(-1, 11)
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_validation)
            knn_acc = accuracy_score(y_validation, train_predictions)
            print("Accurracy du fold : {:.2f} %".format(knn_acc*100))
            knn_acc_val.append(knn_acc) # stocker les valeurs moyennes de l'accuracy de chaque fold
        knn_acc_mean = sum(knn_acc_val) / cv # moyenne de l'accuracy pour un k-voisins
        print("Accurracy moyenne du modèle \'{} voisins\' : {:.2f} %".format(K,knn_acc_mean*100))
        knn_acc_mean_list.append(knn_acc_mean)


    # Choix du modèle le plus performant
    print("=="*30)
    print("Liste accuracy : {}".format([i*100 for i in knn_acc_mean_list]))
    knn_acc_max = max(knn_acc_mean_list)
    index_max = knn_acc_mean_list.index(knn_acc_max)
    print("Meilleur hyperparamètre : {}".format(params[index_max]))
    print("Accuracy : {:.2f} %".format(knn_acc_max*100))
    K_opt = params[index_max]


    return K_opt