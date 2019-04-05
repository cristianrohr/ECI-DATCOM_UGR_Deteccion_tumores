# Cargo las librerías necesarias
import scipy.io
import gpflow
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import metrics


def get_AUC_ROC(y, p):
    # calculo AUC
    aucr = roc_auc_score(y, p)
    # calculo roc curve
    fpr, tpr, thresholds = roc_curve(y, p)
    return fpr, tpr, thresholds, aucr

def get_AUC_PR(y, p):
        # calculate precision-recall curve
        precisionf, recallf, thresholds2 = precision_recall_curve(y, p)
        # calculate F1 score
        #f1 = f1_score(testy, yhat)
        # calculate precision-recall AUC
        aucpr = metrics.auc(recallf, precisionf)
        # calculate average precision score
        ap = metrics.average_precision_score(y, p)
        return aucpr, ap


# Cargo los datos
mat = scipy.io.loadmat('Datos.mat')

# Reformateo los datos para manipularlos con Python
# Healthy_folds
hf1 = np.asarray(mat['Healthy_folds'][0,0][0]).reshape(203, 10) # Fold1
yhf1 = np.ones(hf1.shape[0])*-1
hf2 = np.asarray(mat['Healthy_folds'][0,1][0]).reshape(210, 10) # Fold2
yhf2 = np.ones(hf2.shape[0])*-1
hf3 = np.asarray(mat['Healthy_folds'][0,2][0]).reshape(206, 10) # Fold3
yhf3 = np.ones(hf3.shape[0])*-1
hf4 = np.asarray(mat['Healthy_folds'][0,3][0]).reshape(196, 10) # Fold4
yhf4 = np.ones(hf4.shape[0])*-1
hf5 = np.asarray(mat['Healthy_folds'][0,4][0]).reshape(199, 10) # Fold5
yhf5 = np.ones(hf5.shape[0])*-1

# Malignant_folds
mf1 = np.asarray(mat['Malign_folds'][0,0][0]).reshape(54, 10) # Fold1
ymf1 = np.ones(mf1.shape[0])
mf2 = np.asarray(mat['Malign_folds'][0,1][0]).reshape(72, 10) # Fold2
ymf2 = np.ones(mf2.shape[0])
mf3 = np.asarray(mat['Malign_folds'][0,2][0]).reshape(53, 10) # Fold3
ymf3 = np.ones(mf3.shape[0])
mf4 = np.asarray(mat['Malign_folds'][0,3][0]).reshape(50, 10) # Fold4
ymf4 = np.ones(mf4.shape[0])
mf5 = np.asarray(mat['Malign_folds'][0,4][0]).reshape(69, 10) # Fold5
ymf5 = np.ones(mf5.shape[0])

# Reorganizo los datos para la cross validation
Healthy_folds = [hf1, hf2, hf3, hf4, hf5]
Malign_folds = [mf1, mf2, mf3, mf4, mf5]
Healthy_folds_labels = [yhf1, yhf2, yhf3, yhf4, yhf5]
Malign_folds_labels = [ymf1, ymf2, ymf3, ymf4, ymf5]

# Variables auxiliares
todos_indices = np.arange(0, 5)
promedio = []


fig1, ax1 = plt.subplots()
# plot no skill
ax1.plot([0, 1], [0, 1], linestyle='--')

fig2, ax2 = plt.subplots()
# plot no skill
ax2.plot([0, 1], [0.5, 0.5], linestyle='--')


# 5 Fold Cross Validation
for k in np.arange(0, 5):

    # Creo X_test, a partir del fold k
    X_test = np.concatenate((Healthy_folds[k], Malign_folds[k]))
    y_test = np.concatenate((Healthy_folds_labels[k], Malign_folds_labels[k]))

    # Para almacenar los resultados
    predicciones_list = []

    # Indices restantes, aparte de k
    iterar = [x for i, x in enumerate(todos_indices) if i!=k]

    # Creo arrays vacios
    enfermos = np.array([]).reshape(0,10)
    sanos = np.array([]).reshape(0,10)

    # Itero los folds que no son test
    for v in iterar:
        # Primero tomo todos los malignos
        enfermos = np.concatenate((Malign_folds[v], enfermos))
        # Guardo todos los sanos
        sanos = np.concatenate((Healthy_folds[v], sanos))

    # De los sanos tomo un partición random y entreno un modelo
    for i in np.arange(0, 4):
        # De los healthy solo me quedo con la misma cantidad que malignos para balancear
        aux = sanos[np.random.choice(sanos.shape[0], enfermos.shape[0])]

        # Creo el dataset de train final
        X_train = np.concatenate((enfermos, aux))
        y_train = np.concatenate((np.ones(enfermos.shape[0]), (np.ones(aux.shape[0])*-1)))

        # Entreno el modelo
        mGP = gpflow.models.VGP(X_train, y_train.reshape(-1,1), kern = gpflow.kernels.RBF(X_train.shape[1]),
        likelihood = gpflow.likelihoods.Bernoulli())
        # Optimizo
        gpflow.train.ScipyOptimizer().minimize(mGP, maxiter=300)
        # Realizo las predicciones sobre test
        pred = mGP.predict_y(X_test)
        pred = np.array(pred[0])
        print("Valor de K: {}".format(k))
        predicciones_list.append(pred)

    # Calculo el promedio de los 4 modelos
    promedio = np.average(predicciones_list, axis=0)
    # Recodifico el -1 a 0
    y = y_test
    y[y == -1] = 0


    # Calculo curva ROC y AUC
    #fpr, tpr, thresholds, aucr = get_AUC_ROC(y, promedio)
    # calculo AUC
    aucr = roc_auc_score(y, promedio)
    # calculo roc curve
    fpr, tpr, thresholds = roc_curve(y, promedio)

    print('AUC: %.3f' % aucr)

    # plot the roc curve for the model
    ax1.plot(fpr, tpr, marker='.', label = "Fold {}".format(k))

    #aucpr, ap = get_AUC_PR(y, promedio)
    # calculate precision-recall curve
    precisionf, recallf, thresholds2 = precision_recall_curve(y, promedio)
    # calculate F1 score
    #f1 = f1_score(testy, yhat)
    # calculate precision-recall AUC
    aucpr = metrics.auc(recallf, precisionf)
    # calculate average precision score
    ap = metrics.average_precision_score(y, promedio)

    print('auc=%.3f ap=%.3f' % (aucpr, ap))
    # plot the roc curve for the model
    ax2.plot(recallf, precisionf, marker='.', label = "Fold {}".format(k))




# show the plot

fig1.show()
fig1.legend(loc='lower right')
fig1.savefig("ROC.png")

fig2.show()
fig2.legend(loc='lower left')
fig2.savefig("PR.png")

# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
