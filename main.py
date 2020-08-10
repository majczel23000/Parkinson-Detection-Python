import numpy as np
#np.set_printoptions(threshold=np.inf)
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

class Project(object):
    def __init__(self, datasetURL):									#constructor with dataset url	
        dataset=pd.read_csv(datasetURL)
        self.data = dataset.loc[0:,["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ",      #load data
         "Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA",
         "NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"]].values.astype(np.float)
        self.target = dataset.loc[0:, ['status']].values.astype(np.float)                                                                       #load status
        print("Dane zosta³y pomyslnie wczytane.")
        patiens_p = pd.DataFrame(data=self.data, columns=["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ",      #load data
         "Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA",
         "NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print('')
            print("-------------------head()-------------------")
            print('')
            print(patiens_p.head())
            print('')
            print("-------------------describe()-------------------")
            print('')
            print(patiens_p.describe())
    
    def splitData(self,ts):                                                                                                           #to to rozmiar zbioru testuj¹cego
        self.data_train, self.data_test, self.target_train, self.target_test = \
        train_test_split(self.data, self.target, test_size=ts, random_state=5)
        print(f"Dane zosta³y pomyslnie podzielone. Rozmiar danych trenuj¹cych: {np.size(self.target_train)}. Rozmiar danych testuj¹cych: {np.size(self.target_test)}.")
    
    def classifier(self):
        option = 0
        clf = 0
        while True:
            print('')
            print('-----------------MENU---------------------')
            print("0.Generuj nowy podzia³ danych")
            print("1.Klasyfikacja DecisionTreeClassifier")
            print("2.Klasyfikacja GaussianNB")
            print("3.Klasyfikacja SVC")
            print("4.Klasyfikacja KNeighborsClassifier (n=3)")
            print("5.Klasyfikacja GradientBoostingClassifier")
            print("6.ZnajdŸ najlepsze wartosci parametrow")
            print("Wybierz q, aby zakoñczyæ")
            option = input('>>Podaj kod operacji, któr¹ chcesz wykonaæ: ')
            print('')
            if option=='1':
                print("=======Wybrana Klasyfikacja: DecisionTreeClassifier=======")
                clf=DecisionTreeClassifier()
            elif option=='2':
                print("=======Wybrana Klasyfikacja: GaussianNB=======")
                clf=GaussianNB()
            elif option=='3':
                print("=======Wybrana Klasyfikacja: SVC=======")
                clf=SVC()
            elif option=='4':
                print("=======Wybrana Klasyfikacja: KNeighborsClassifier (n=3)=======")
                clf=KNeighborsClassifier(n_neighbors=3)
            elif option=='5':
                print("=======Wybrana Klasyfikacja: GradientBoostingClassifier=======")
                clf=GradientBoostingClassifier()
            elif option=='6':
                tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
                score = 'precision'
                print("Szukanie najlepszych wartosci parametrów")
                clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
                clf.fit(self.data_train, self.target_train)
            
                print("Najlepsze parametry:")
                print(clf.best_params_)
                print()
                print("Wyniki dla poszczególnych wartosci parametrów:")
                print()
                means = clf.cv_results_['mean_test_score']
                for mean, params in zip(means, clf.cv_results_['params']):
                    print("%0.3f dla %r"
                          % (mean, params))
                
            elif option=='0':
                print('Generacja nowego podzia³u.')
                size = input('>>Podaj ilosc elementów w zbiorze testuj¹cym: ')
                self.splitData(int(size))
            elif option=='q':
                break;
            else:
                print("Niepoprawny kod. Spróbuj ponownie.")        
           
            if option>'0' and option <= '5':
                print(f"Rozmiar danych trenuj¹cych: {np.size(self.target_train)}. Rozmiar danych testuj¹cych: {np.size(self.target_test)}.")
                clf.fit(self.data_train,self.target_train)
                
                predicted_train_val = clf.predict(self.data_train)
                conf_matrix = confusion_matrix(self.target_train, predicted_train_val)
                print("Macierz konfuzji dla danych trenuj¹cych: ")
                print(conf_matrix)
                p_score = precision_score(self.target_train, predicted_train_val, average='micro')
                print("Precision_score dla danych trenuj¹cych: {0:0.3f}".format(p_score))
                
                predicted_test_val = clf.predict(self.data_test)
                conf_matrix = confusion_matrix(self.target_test, predicted_test_val)     
                print("Macierz konfuzji dla danych testuj¹cych: ")
                print(conf_matrix)
                p_score = precision_score(self.target_test, predicted_test_val, average='micro')
                print("Precision_score dla danych testuj¹cych: {0:0.3f}".format(p_score))                                                                                                                        
            

project=Project("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")
size = input('>>Podaj ilosc elementów w zbiorze testuj¹cym: ')
project.splitData(int(size))
project.classifier()
