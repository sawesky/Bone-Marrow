import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%%###################################################
# Analiza i obrada seta podataka                     #
######################################################

pd.set_option('display.width', 400) # da se sve vidi u konzoli
pd.set_option('display.max_columns', 10) # i ovo

#### pretprocesiranje
data = pd.read_csv('03_bone_marrow_dataset.csv')
data = data.replace("?", np.nan) 
data = data.drop([128, 129])  #krvna grupa sve nepoznato
data = data.drop([186]) #alele i antigene nema
data.info(verbose=True)
disdata = data.Disease #kat. obelezje
data = data.loc[:, data.columns!='Disease'].astype(float)
# data.HLAmatch = data.HLAmatch.astype(int)
# data.Antigen = data.Antigen.astype(int)
# data.Alel = data.Alel.astype(int)
# data.ANCrecovery = data.ANCrecovery.astype(int)
# data.survival_time = data.survival_time.astype(int)


data = pd.concat([data,disdata], axis=1)

### ANC, Platelet
data['ANCrecovery'] = data['ANCrecovery'].replace(1000000, np.NaN)
data.drop(data.index[pd.isna(data.ANCrecovery)], axis=0, inplace=True) #5 losih ispitanika (nije mnogo)
data['PLTrecovery'] = np.where(data['PLTrecovery'] < 1000000, 1, 0) #prebaceno u kat. prom 1-uspesno 0-neuspesno

#### GvHD 
data.pop('time_to_aGvHD_III_IV') #nepotreban atribut predstavlja vreme kad je doslo do aGvHDa
data.pop('IIIV') #sadrzano u aGvHD i extcGvHD
#ako ima aGvHD nema extcGvHD ->
data.loc[(np.isnan(data['extcGvHD'])) & (data['aGvHDIIIIV'] == 0), 'extcGvHD'] = 1
#ako nema aGvHD i nije ziv i plateleti se nisu recoverovali onda nema extcGvHD ->
data.loc[(np.isnan(data['extcGvHD'])) & (data['aGvHDIIIIV'] == 1) &
         (data['survival_status'] == 1) & (data['PLTrecovery'] == 0), 'extcGvHD'] = 1
#ako nema aGvHD i nije ziv i plateleti su se recoverovali onda ima extcGvHD ->
data.loc[(np.isnan(data['extcGvHD'])) & (data['aGvHDIIIIV'] == 1) & 
         (data['survival_status'] == 1) & (data['PLTrecovery'] == 1), 'extcGvHD'] = 0
#sledeca obelezja su bila meni neinituitivna yes - 0 no - 1 pa sam promenio u yes - 1 no - 0
data['extcGvHD'] = np.where(data['extcGvHD'] == 1, 0, 1)
data['aGvHDIIIIV'] = np.where(data['aGvHDIIIIV'] == 1, 0, 1)
data['survival_status'] = np.where(data['survival_status'] == 1, 0, 1)

#### Recipient/donor age 
data.pop('Recipientageint') #nepotrebno 
data.pop('Recipientage10') #veoma slicno Recipientageu
data.pop('Donorage35') #veoma slicno Donorageu

#### HLA (human leukocyte antigens)
data.pop('HLAmismatch') #kategoricka promenljiva nedovoljno info daje
data.pop('HLAgrI') #kategoricka prom sklopljena od ostalih
data['Antigen'] += 1 #posle ovoga kompletno uklapanje antigena znaci 0; jedna razlika 1 itd
data['Alel'] += 1 #isto i ovde

#### CMV (cytomegalovirus)
####(NaN uzima 15 vrednosti tj. oko 10% baze, posto su obelezja kat. onda sam izbacio sve iz ove grupe) 
data.pop('CMVstatus')
data.pop('DonorCMV')
data.pop('RecipientCMV')

#### ABO,Rh (krvna grupa) 
data['ABOmatch'] = np.where(data['ABOmatch'] == 1, 0, 1) #sad je match 1
data.pop('DonorABO') #sve nepotrebno
data.pop('RecipientABO')
data.pop('RecipientRh')

#### ostalo
data['Rbodymass'].fillna(data['Rbodymass'].median(), inplace = True)
data['CD3dkgx10d8'].fillna(data['CD3dkgx10d8'].median(), inplace = True)
data.pop('CD3dCD34') #odnos cd3 i cd4 celija nepotreban
data['Gendermatch'] = np.where(data['Gendermatch'] == 1, 0, 1) #sad je match 1

print(data.isna().sum())
print(data.describe().T)
data.info(verbose=True)
data.hist(figsize=(20,15))
plt.show()

#%%###################################################
# Korelacija/Information Gain                        #
######################################################

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")

#### inicijalno Pearson
pearson_R = data.corr(method = 'pearson')
plt.figure(figsize=(20,15))
sns.heatmap(pearson_R, annot = True)
plt.show()

#### inicijalno Spearman
spearman_R = data.corr(method = 'spearman')
plt.figure(figsize=(20,15))
sns.heatmap(spearman_R, annot = True)
plt.show()

#### Information gain
def calculateInfoD(col):
    un = np.unique(col)
    infoD = 0
    for u in un:
        p = sum(col == u)/len(col)
        infoD -= p*np.log2(p)
    return infoD

klasa = data.iloc[:, -2] # -2 jer je poslednja kolona Disease
infoD = calculateInfoD(klasa)
print('Info(D) = ' + str(infoD))

for bins in [10, 25, 50]:
    data.hist(bins = bins, figsize=(20, 15))

#izabrano 25 binova
new_data = data.copy(deep = True)
new_data.pop('Disease')
def limit_feature(col, numSteps = 25):
    step = (max(col) - min(col))/numSteps
    new_col = np.floor(col/step)*step
    return new_col

for att in range (1, data.shape[1] - 2):
    temp = data.iloc[:, att]
    new_data.iloc[:, att] = limit_feature(temp)

IG = np.zeros((new_data.shape[1] - 1, 2))
for att in range (new_data.shape[1] - 1):
    f = np.unique(new_data.iloc[:, att])
    infoDA = 0
    for i in f:
        temp = klasa[new_data.iloc[:, att] == i]
        infoDi = calculateInfoD(temp)
        Di = sum(new_data.iloc[:, att] == i)
        D = len(new_data.iloc[:, att])
        infoDA += Di*infoDi/D
    IG[att, 0] = att + 1
    IG[att, 1] = infoD - infoDA

print('IG = ' + str(IG))
IGsorted = IG[IG[:, 1].argsort()]
np.set_printoptions(suppress=True)
print('IG = ' + str(IGsorted))

data.pop('Disease')
#selekcija? 
datap = data.filter(['Donorage', 'CD34kgx10d6', 'Recipientage', 'Rbodymass', 
                      'ANCrecovery','CD3dkgx10d8', 'Relapse', 'extcGvHD',
                      'PLTrecovery','survival_time','survival_status'], axis=1)

spearman_R = datap.corr(method = 'spearman')
plt.figure(figsize = (20, 15))
sns.heatmap(spearman_R, annot = True)
plt.show()    
pearson_R = datap.corr(method = 'pearson')
plt.figure(figsize = (20, 15))
sns.heatmap(pearson_R, annot = True)
plt.show()
#%%###################################################
# LDA/PCA                                            #
######################################################    

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

X = data.iloc[:, :-2];
y = data.iloc[:, -1]
X_norm = (X - np.mean(X, axis = 0))/np.max(X, axis = 0)

numcomp = 1
lda = LDA(solver = 'eigen', n_components = numcomp)
ldaComponents = lda.fit_transform(X_norm, y)
ldaComponents = pd.DataFrame(data = ldaComponents)
  
lda_df = pd.concat([ldaComponents, y] , axis = 1)
lda_df.columns = ['LDA', 'Outcome']

plt.figure()
sns.scatterplot(data = lda_df, x = 'LDA', y = 1, hue = 'Outcome', alpha=0.35, palette = 'deep')
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pcaComponents = pca.fit_transform(X_norm)
pcaComponents = pd.DataFrame(data = pcaComponents)
pca_df = pd.concat([pcaComponents, y] , axis = 1)
pca_df.columns = ['PCA1', 'PCA2', 'Outcome']

plt.figure()
sns.scatterplot(data = pca_df, x = 'PCA1', y = 'PCA2' , hue = 'Outcome')
plt.show()

#%%###################################################
# Klasifikator                                       #
######################################################  

dataklas = data.filter(['Donorage', 'CD34kgx10d6', 'Recipientage', 'Rbodymass', 
                      'CD3dkgx10d8', 'Relapse', 'extcGvHD','survival_time'], axis=1)
dataklas_norm = (dataklas - np.mean(dataklas, axis = 0))/np.max(dataklas, axis = 0)

numcomp = 1
lda = LDA(solver = 'eigen', n_components = numcomp)
ldaComponents = lda.fit_transform(dataklas_norm, y)
ldaComponents = pd.DataFrame(data = ldaComponents)
  
lda_df = pd.concat([ldaComponents, y] , axis = 1)
lda_df.columns = ['LDA', 'Outcome']

plt.figure()
sns.scatterplot(data = lda_df, x = 'LDA', y = 1, hue = 'Outcome', alpha=0.4, palette = 'deep')
plt.show()

pca = PCA(n_components = 2)
pcaComponents = pca.fit_transform(dataklas_norm)
pcaComponents = pd.DataFrame(data = pcaComponents)
pca_df = pd.concat([pcaComponents, y] , axis = 1)
pca_df.columns = ['PCA1', 'PCA2', 'Outcome']

plt.figure()
sns.scatterplot(data = pca_df, x = 'PCA1', y = 'PCA2' , hue = 'Outcome', alpha=0.4, palette = 'deep')
plt.show()
#ne valja lda pca graficki nikad,  radim samo konf. matricu 

###

def izracunaj_fgv(x, m, s):
    det = np.linalg.det(s)
    inv = np.linalg.inv(s)
    x_mu = x - m
    fgv_const = 1/np.sqrt(2*np.pi*det)
    fgv_rest = np.exp(-0.5*x_mu.T@inv@x_mu)
    return fgv_const*fgv_rest


K0 = dataklas.loc[y == 0, :] #umrli
K1 = dataklas.loc[y == 1, :] #preziveli
N0trening = int(0.7*K0.shape[0])
N1trening = int(0.7*K1.shape[0])
K0trening = K0.iloc[:N0trening, :]
K1trening = K1.iloc[:N1trening, :]

N0test = int(0.3*K0.shape[0])
N1test = int(0.3*K1.shape[0])
K0test = K0.iloc[N0trening:, :]
K1test = K1.iloc[N1trening:, :]

M0 = K0trening.mean()
S0 = K0trening.cov()
#print(M0)
#print(S0)
M1 = K1trening.mean()
S1 = K1trening.cov()
#print(M1)
#print(S1)

p0 = N0trening / (N0trening + N1trening)
p1 = N1trening / (N0trening + N1trening)
T = np.log(p0/p1)

decision = np.zeros((N0test + N1test, 1))
conf_mat = np.zeros((2,2))

for i in range(N0test):
    x0 = K0test.iloc[i, :]
    f0 = izracunaj_fgv(x0, M0, S0)
    f1 = izracunaj_fgv(x0, M1, S1)
    h0 = -np.log(f0) + np.log(f1)
    if h0 > T:
        decision[i] = 1
    else:
        decision[i] = 0

for i in range(N1test):
    x1 = K1test.iloc[i, :]
    f0 = izracunaj_fgv(x1, M0, S0)
    f1 = izracunaj_fgv(x1, M1, S1)
    h1 = -np.log(f0) + np.log(f1)
    if h1 > T:
        decision[N0test + i] = 1
    else:
        decision[N0test + i] = 0

Xtest = np.append(K0test, K1test, axis = 0)
Ytest = np.append(0*np.ones((N0test, 1)), np.ones((N1test, 1)))

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(Ytest, decision, labels=[0, 1])

plt.figure()
sns.heatmap(conf_mat, annot=True, fmt='g', cbar=False, xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.show()
acc = np.trace(conf_mat)/np.sum(conf_mat)
print(acc)

#%%###################################################
# Neuralna mreza                                     #
######################################################  

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras import regularizers
from keras.callbacks import EarlyStopping

def create_model():
    model = Sequential()
    model.add(Dense(200, input_dim = 21, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

X = data.drop('survival_status', axis = 1);
# X = X.drop('survival_time', axis = 1)
Y = data.iloc[:, -1];
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, train_size = 0.6, shuffle = True)

model = KerasClassifier(build_fn = create_model, verbose = 0)

# es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 20, verbose = 1)
batch_size = 10;
epochs = 1000;
# param_grid = dict(batch_size = batch_size, epochs = epochs)
# grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1, cv = 3)
# grid_result = grid.fit(X, Y)

# best_batch_size = grid_result.best_params_['batch_size']
# best_epochs = grid_result.best_params_['epochs']
#print(grid_result.cv_results_)


hstry = model.fit(Xtrain, Ytrain, epochs = epochs, validation_data = (Xtest, Ytest),
                  batch_size = batch_size, verbose = 0)

# _, train_acc = model.evaluate(Xtrain, Ytrain, verbose = 0)
# print('Train acc = ' + str(train_acc*100) + '%.')

# _, test_acc = model.evaluate(Xtest, Ytest, verbose = 0)
# print('Test acc = ' + str(test_acc*100) + '%.')


plt.figure()
plt.plot(hstry.history['loss'])
plt.plot(hstry.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

Ypred = model.predict(Xtest)
conf_mat = confusion_matrix(Ytest, Ypred)
sns.heatmap(conf_mat, annot = True, fmt = 'g', cbar = False)

acc = np.trace(conf_mat)/np.sum(conf_mat)
print(acc)

# print(hstry.history['accuracy'])
# print(hstry.history['val_accuracy'])












